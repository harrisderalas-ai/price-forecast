from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True)
class LSTMSeq2SeqConfig:
    """
    Immutable configuration for a seq2seq LSTM forecaster (encoder–decoder).

    Expected array shapes
    ---------------------
    X: (batch, input_steps, n_features)
       - 'input_steps' is the length of the encoder lookback window.
       - 'n_features' is the number of covariates/features per time step.

    y: (batch, output_steps, output_dim)
       - 'output_steps' is the decoder horizon (e.g., 24 hours).
       - 'output_dim' is typically 1 for univariate forecasting.

    Notes
    -----
    - This config is typically constructed automatically via
      `LSTMSeq2SeqModel.from_training_data(X_train, y_train)`.
    """
    # Inferred from X: (batch, input_steps, n_features)
    input_steps: int
    n_features: int

    # Inferred from y: (batch, output_steps, output_dim)
    output_steps: int = 24
    output_dim: int = 1

    # Model & training hyperparameters
    lstm_units: int = 96
    dropout: float = 0.2
    recurrent_dropout: float = 0.0  # > 0 can slow training; use sparingly.
    optimizer: str = "adam"
    loss: str = "mse"
    metrics: Tuple[str, ...] = ("mae",)

    # Reproducibility (best-effort)
    seed: Optional[int] = 42


# ============================================================================
# Model
# ============================================================================

class LSTMSeq2SeqModel:
    """
    LSTM encoder–decoder for multi-step forecasting (e.g., day-ahead 24 steps).

    Design (SRP)
    ------------
    - Build & compile the model (constructor)
    - Train with time-series-safe defaults (fit)
    - Predict sequences (predict)
    - Evaluate predictions with standard regression metrics (evaluate)
    - Persist & restore full models (save / load)

    Key points
    ----------
    - The decoder length equals `config.output_steps` (e.g., 24), independent of the
      encoder input length—handy for varying lookback windows.
    - Shuffling is disabled by default during training to preserve temporal structure.
    """

    # ---------------------------------------------------------------------
    # Constructors
    # ---------------------------------------------------------------------

    def __init__(self, config: LSTMSeq2SeqConfig):
        """
        Initialize the wrapper and build/compile the internal Keras model.
        Use `from_training_data` to auto-infer shapes from arrays.
        """
        self.config = config
        self._set_seed(config.seed)
        self.model: Model = self._build_model()

    @classmethod
    def from_training_data(
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        lstm_units: int = 96,
        dropout: float = 0.2,
        recurrent_dropout: float = 0.0,
        optimizer: str = "adam",
        loss: str = "mse",
        metrics: Tuple[str, ...] = ("mae",),
        seed: Optional[int] = 42,
    ) -> "LSTMSeq2SeqModel":
        """
        Convenience constructor that infers `input_steps`, `n_features`,
        `output_steps`, and `output_dim` directly from the training arrays.

        Parameters
        ----------
        X_train : np.ndarray
            Shape (batch, input_steps, n_features).
        y_train : np.ndarray
            Shape (batch, output_steps, output_dim).

        Other keyword-only parameters
        -----------------------------
        lstm_units, dropout, recurrent_dropout, optimizer, loss, metrics, seed
            Standard hyperparameters and training configuration. See config.

        Returns
        -------
        LSTMSeq2SeqModel
            A fully initialized model ready to train.
        """
        cls._validate_shapes(X_train, y_train)

        input_steps, n_features = X_train.shape[1], X_train.shape[2]
        output_steps, output_dim = y_train.shape[1], y_train.shape[2]

        cfg = LSTMSeq2SeqConfig(
            input_steps=input_steps,
            n_features=n_features,
            output_steps=output_steps,
            output_dim=output_dim,
            lstm_units=lstm_units,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            seed=seed,
        )
        return cls(cfg)

    # ---------------------------------------------------------------------
    # Validation / seeding
    # ---------------------------------------------------------------------

    @staticmethod
    def _validate_shapes(X: np.ndarray, y: np.ndarray) -> None:
        """
        Defensive checks to catch common shape mismatches early.
        """
        if X.ndim != 3:
            raise ValueError(f"`X` must be 3D (batch, input_steps, n_features); got {X.shape!r}")
        if y.ndim != 3:
            raise ValueError(f"`y` must be 3D (batch, output_steps, output_dim); got {y.shape!r}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Batch sizes differ: X={X.shape[0]} vs y={y.shape[0]}")

    @staticmethod
    def _set_seed(seed: Optional[int]) -> None:
        """
        Set seeds for best-effort reproducibility. (Exact determinism is not guaranteed.)
        """
        if seed is not None:
            tf.keras.utils.set_random_seed(seed)

    # ---------------------------------------------------------------------
    # Build / compile
    # ---------------------------------------------------------------------

    def _build_model(self) -> Model:
        """
        Build and compile an encoder–decoder LSTM:

        Encoder
        -------
        - LSTM consumes (input_steps × n_features) and produces a latent vector.
        - Dropout regularizes between encoder/decoder.

        Bridge
        ------
        - RepeatVector expands the latent vector across `output_steps`,
          providing one latent state per decoder time step.

        Decoder
        -------
        - LSTM (return_sequences=True) converts the latent sequence to hidden states.

        Head
        ----
        - TimeDistributed(Dense(output_dim)) predicts per-time-step outputs.

        Returns
        -------
        tf.keras.Model
            A compiled Keras model with specified optimizer/loss/metrics.
        """
        c = self.config
        model = Sequential(name="lstm_seq2seq_day_ahead")

        # ---------------- Encoder ----------------
        model.add(
            LSTM(
                c.lstm_units,
                input_shape=(c.input_steps, c.n_features),
                recurrent_dropout=c.recurrent_dropout,
                name="encoder_lstm",
            )
        )
        model.add(Dropout(c.dropout, name="encoder_dropout"))

        # ------- Latent -> time axis (decoder steps) -------
        model.add(RepeatVector(c.output_steps, name="repeat_output_steps"))

        # ---------------- Decoder ----------------
        model.add(
            LSTM(
                c.lstm_units,
                return_sequences=True,
                recurrent_dropout=c.recurrent_dropout,
                name="decoder_lstm",
            )
        )
        model.add(Dropout(c.dropout, name="decoder_dropout"))

        # --------- Per-step regression head ---------
        model.add(TimeDistributed(Dense(c.output_dim), name="hourly_output"))

        # Compile
        model.compile(optimizer=c.optimizer, loss=c.loss, metrics=list(c.metrics))

        # Optional: print a helpful summary to stdout for quick inspection.
        model.summary()

        return model

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------

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
        checkpoint_path: Optional[str] = "best_lstm.keras",  # save full model
        verbose: int = 1,
        shuffle: bool = False,  # keep temporal ordering by default
    ):
        """
        Train the model with time-series-friendly defaults.

        Validation splitting
        --------------------
        - If (X_val, y_val) is not provided, we automatically split off the
          last 10% of the provided (X_train, y_train) as a *tail* validation set
          to honor temporal ordering.
duceLROnPlateau
        - ModelCheckpoint (optional; saves the *entire* model to a `.keras` file)

        Parameters
        ----------
        X_train, y_train : np.ndarray
            Training arrays. Shapes must match the model config.
        X_val, y_val : Optional[np.ndarray]
            Optional explicit validation arrays.
        epochs, batch_size, patience, min_delta, reduce_lr_patience, checkpoint_path, verbose, shuffle
            Standard Keras training arguments.

        Returns
        -------
        tf.ke
        Callbacks
        ---------
        - EarlyStopping (restore best)
        - Reras.callbacks.History
            The Keras training history object.
        """
        X_train = np.ascontiguousarray(np.asarray(X_train, dtype=np.float32))
        y_train = np.ascontiguousarray(np.asarray(y_train, dtype=np.float32))
        X_val   = np.ascontiguousarray(np.asarray(X_val,   dtype=np.float32)) if X_val is not None else None
        y_val   = np.ascontiguousarray(np.asarray(y_val,   dtype=np.float32)) if y_val is not None else None
       
        # Basic sanity: arrays are 3D and have matching batch sizes
        self._validate_shapes(X_train, y_train)

        # Ensure provided arrays are compatible with the model's config
        if (X_train.shape[1], X_train.shape[2]) != (self.config.input_steps, self.config.n_features):
            raise ValueError(
                "X_train shape does not match model config: "
                f"expected (input_steps, n_features)=({self.config.input_steps}, {self.config.n_features}) "
                f"but got ({X_train.shape[1]}, {X_train.shape[2]})."
            )
        if (y_train.shape[1], y_train.shape[2]) != (self.config.output_steps, self.config.output_dim):
            raise ValueError(
                "y_train shape does not match model config: "
                f"expected (output_steps, output_dim)=({self.config.output_steps}, {self.config.output_dim}) "
                f"but got ({y_train.shape[1]}, {y_train.shape[2]})."
            )

        # Derive a tail validation split if none provided (time-ordered)
        if X_val is None or y_val is None:
            n = X_train.shape[0]
            val_n = max(1, int(0.1 * n))
            X_val, y_val = X_train[-val_n:], y_train[-val_n:]
            X_train, y_train = X_train[:-val_n], y_train[:-val_n]

        # Standard, sensible callbacks for sequence forecasting
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=patience,
                min_delta=min_delta,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-6,
                verbose=verbose,
            ),
        ]

        # Save the *entire* model (architecture + weights) to a single `.keras` file
        if checkpoint_path:
            callbacks.append(
                ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor="val_loss",
                    save_best_only=True,
                    save_weights_only=False,  # save full model
                    verbose=verbose,
                )
            )

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,  # do not shuffle time series by default
            callbacks=callbacks,
            verbose=verbose,
        )
        return history

    # ---------------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------------

    def predict(self, X: np.ndarray, batch_size: int = 256, verbose: int = 0) -> np.ndarray:
        """
        Predict sequences for the given batch of inputs.

        Parameters
        ----------
        X : np.ndarray
            Shape (batch, input_steps, n_features).
        batch_size : int
            Inference batch size.
        verbose : int
            Keras verbosity during prediction.

        Returns
        -------
        np.ndarray
            Predicted sequences with shape (batch, output_steps, output_dim).
        """
        return self.model.predict(X, batch_size=batch_size, verbose=verbose)

    # ---------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute flattened regression metrics across all samples and time steps.

        Pass *unscaled* arrays if you want metrics in original units.

        Parameters
        ----------
        y_true, y_pred : np.ndarray
            Both with shape (batch, output_steps, output_dim).

        Returns
        -------
        dict
            {"mse": float, "mae": float}
        """
        mse = mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1))
        mae = mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1))
        return {"mse": float(mse), "mae": float(mae)}

    # ---------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------

    def save(self, path: str = "final_model.keras") -> None:
        """
        Save the entire model (architecture + weights) to a single `.keras` file.
        Use `LSTMSeq2SeqModel.load(path)` to restore later.
        """
        self.model.save(path)

    @staticmethod
    def load(path: str) -> "LSTMSeq2SeqModel":
        """
        Load a full model saved via `.save('*.keras')` and wrap it with a compatible config.

        We infer:
        - (input_steps, n_features) from the model input shape
        - (output_steps, output_dim) from the model output shape

        Parameters
        ----------
        path : str
            Path to a `.keras` file saved by this class.

        Returns
        -------
        LSTMSeq2SeqModel
            A wrapper instance whose `model` is the loaded Keras model.
        """
        loaded = load_model(path)

        # Keras model shapes:
        # input_shape  -> (None, input_steps, n_features)
        # output_shape -> (None, output_steps, output_dim)
        input_shape = loaded.input_shape
        output_shape = loaded.output_shape

        cfg = LSTMSeq2SeqConfig(
            input_steps=input_shape[1],
            n_features=input_shape[2],
            output_steps=output_shape[1],
            output_dim=output_shape[2],
        )

        obj = LSTMSeq2SeqModel(cfg)
        obj.model = loaded  # replace the freshly built model with the loaded one
        return obj
