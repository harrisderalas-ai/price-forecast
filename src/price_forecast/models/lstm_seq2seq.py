from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error


@dataclass(frozen=True)
class LSTMSeq2SeqConfig:
    """
    Immutable configuration for a seq2seq LSTM forecaster.

    Shapes (expected):
        X: (batch, input_steps, n_features)
        y: (batch, output_steps, output_dim)
    """
    input_steps: int                 # e.g., n_lookback_days * 24
    n_features: int                  # number of input features
    output_steps: int = 24           # hours to predict (fixed to 24 here)
    output_dim: int = 1              # 1 -> dam_price_eur_mwh
    lstm_units: int = 96
    dropout: float = 0.2             # between LSTM blocks
    recurrent_dropout: float = 0.0   # >0 slows training; keep 0 unless needed
    optimizer: str = "adam"
    loss: str = "mse"
    metrics: Tuple[str, ...] = ("mae",)
    seed: Optional[int] = 42         # reproducibility (where possible)


class LSTMSeq2SeqModel:
    """
    LSTM encoder-decoder model for day-ahead (24-step) forecasting.

    Responsibilities (SRP):
      - Build & compile the model (constructor)
      - Train with best practices for time series (fit)
      - Predict sequences (predict)
      - Evaluate (evaluate)
      - Persist & restore (save / load)

    Notes:
      - Decoder length equals `config.output_steps` (24), independent of input length.
      - No shuffling during training by default (preserves temporal structure).
    """

    def __init__(self, config: LSTMSeq2SeqConfig):
        self.config = config
        self._set_seed(config.seed)
        self.model: Model = self._build_model()

    # ------------------------ Build ------------------------

    def _build_model(self) -> Model:
        """
        Encoder-decoder LSTM:
          - Encoder consumes (input_steps × n_features) and yields a latent vector.
          - RepeatVector expands latent to (output_steps).
          - Decoder returns a sequence; TimeDistributed(Dense(output_dim)) maps each hour.
        """
        c = self.config
        model = Sequential(name="lstm_seq2seq_day_ahead")

        # Encoder
        model.add(LSTM(
            c.lstm_units,
            input_shape=(c.input_steps, c.n_features),
            recurrent_dropout=c.recurrent_dropout,
            name="encoder_lstm",
        ))
        model.add(Dropout(c.dropout, name="encoder_dropout"))

        # Bridge latent -> decoder time axis (exactly output_steps = 24)
        model.add(RepeatVector(c.output_steps, name="repeat_output_steps"))

        # Decoder
        model.add(LSTM(
            c.lstm_units,
            return_sequences=True,
            recurrent_dropout=c.recurrent_dropout,
            name="decoder_lstm",
        ))
        model.add(Dropout(c.dropout, name="decoder_dropout"))

        # Per-hour regression head
        model.add(TimeDistributed(Dense(c.output_dim), name="hourly_output"))

        model.compile(optimizer=c.optimizer, loss=c.loss, metrics=list(c.metrics))
        model.summary()
        return model

    @staticmethod
    def _set_seed(seed: Optional[int]) -> None:
        """Set TF seed for reproducibility where feasible."""
        if seed is not None:
            tf.keras.utils.set_random_seed(seed)

    # ------------------------ Training ------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 60,
        batch_size: int = 64,
        patience: int = 10,
        min_delta: float = 1e-4,
        reduce_lr_patience: int = 5,
        checkpoint_path: Optional[str] = "best_lstm.keras",  # full model
        verbose: int = 1,
        shuffle: bool = False,
    ):
        """
        Train with:
          - EarlyStopping (restore best weights),
          - ReduceLROnPlateau,
          - ModelCheckpoint saving the **entire model** to `.keras`.
        If no (X_val, y_val) provided, uses the last 10% of the training data as validation.
        """
        # Derive a tail validation set if none provided (time-ordered)
        if X_val is None or y_val is None:
            n = X_train.shape[0]
            val_n = max(1, int(0.1 * n))
            X_val, y_val = X_train[-val_n:], y_train[-val_n:]
            X_train, y_train = X_train[:-val_n], y_train[:-val_n]

        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=patience, min_delta=min_delta, restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=reduce_lr_patience, min_lr=1e-6, verbose=verbose
            ),
        ]

        if checkpoint_path:
            # Option B: save full model (NOT weights-only) -> use `.keras`
            callbacks.append(
                ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor="val_loss",
                    save_best_only=True,
                    save_weights_only=False,  # full SavedModel in a single .keras file
                    verbose=verbose,
                )
            )

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,  # keep temporal order
            callbacks=callbacks,
            verbose=verbose,
        )
        return history

    # ------------------------ Inference ------------------------

    def predict(self, X: np.ndarray, batch_size: int = 256, verbose: int = 0) -> np.ndarray:
        """Predict 24-step sequences: returns (batch, output_steps, output_dim)."""
        return self.model.predict(X, batch_size=batch_size, verbose=verbose)

    # ------------------------ Evaluation ------------------------

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Flattened regression metrics across all hours and samples.
        Pass **unscaled** arrays if you want metrics in original units.
        """
        mse = mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1))
        mae = mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1))
        return {"mse": float(mse), "mae": float(mae)}

    # ------------------------ Persistence ------------------------

    def save(self, path: str = "final_model.keras") -> None:
        """Save the entire model (architecture + weights) to a `.keras` file."""
        self.model.save(path)

    @staticmethod
    def load(path: str) -> "LSTMSeq2SeqModel":
        """
        Load a full model saved with `model.save('*.keras')`.
        Returns a wrapper with the loaded Keras model and inferred config.
        """
        loaded = load_model(path)
        # Infer shapes from the loaded model to rebuild a compatible config
        # (input steps, features) from input layer; (output steps, dim) from output
        input_shape = loaded.input_shape  # (None, input_steps, n_features)
        output_shape = loaded.output_shape  # (None, output_steps, output_dim)
        cfg = LSTMSeq2SeqConfig(
            input_steps=input_shape[1],
            n_features=input_shape[2],
            output_steps=output_shape[1],
            output_dim=output_shape[2],
        )
        obj = LSTMSeq2SeqModel(cfg)
        obj.model = loaded
        return obj
