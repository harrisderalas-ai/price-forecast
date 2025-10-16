
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    LSTM,
    AdditiveAttention,
    Bidirectional,
    Concatenate,
    Cropping1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LayerNormalization,
    RepeatVector,
    TimeDistributed,
)
from tensorflow.keras.models import load_model


# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True)
class LSTMSeq2SeqPlusConfig:
    """
    Configuration for a Seq2Seq LSTM with optional attention and residual last-24h skip.

    Expected array shapes
    ---------------------
    X: (batch, input_steps, n_features)
       - 'input_steps' is the encoder lookback window length.
       - 'n_features' is the number of covariates per time step.

    y: (batch, output_steps, output_dim)
       - 'output_steps' is the decoder horizon (e.g., 24 hours).
       - 'output_dim' is typically 1 for univariate forecasting.

    Notes
    -----
    - This config is typically created automatically via
      `LSTMSeq2SeqPlus.from_training_data(X_train, y_train, ...)`.
    """

    # Inferred from X: (batch, input_steps, n_features)
    input_steps: int
    n_features: int

    # Inferred from y: (batch, output_steps, output_dim)
    output_steps: int = 24
    output_dim: int = 1

    # Architecture
    enc_units: Tuple[int, int] = (128, 64)   # two encoder LSTM layers by default
    dec_units: int = 128
    bidirectional: bool = True               # if True, encoder LSTMs are bidirectional
    attention: bool = True                   # if True, use AdditiveAttention over encoder seq
    dropout: float = 0.2
    recurrent_dropout: float = 0.0           # > 0 slows training; use sparingly
    layernorm: bool = True                   # layer norm after encoder/decoder
    residual_last24: bool = True             # concat projected last 24 input steps into head

    # Training
    optimizer: str = "adam"
    loss: str = "mse"
    metrics: Tuple[str, ...] = ("mae",)

    # Reproducibility (best-effort)
    seed: Optional[int] = 42


# ============================================================================
# Model
# ============================================================================

class LSTMSeq2SeqPlus:
    """
    Seq2Seq LSTM with attention + residual last-24h skip for day-ahead forecasting.

    Input : (batch, input_steps, n_features)
    Output: (batch, output_steps, output_dim)  -> typically (batch, 24, 1)

    Design (SRP)
    ------------
    - Build & compile the model (constructor)
    - Train with time-series-safe defaults (fit)
    - Predict sequences (predict)
    - Evaluate predictions (evaluate)
    - Save / load full models (persistence)
    """

    # ---------------------------------------------------------------------
    # Constructors
    # ---------------------------------------------------------------------

    def __init__(self, config: LSTMSeq2SeqPlusConfig):
        """
        Initialize and build the internal Keras model.
        Prefer `from_training_data` for auto-inferred shapes.
        """
        self.config = config
        self._set_seed(config.seed)
        self.model: Model = self._build()

    @classmethod
    def from_training_data(
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        enc_units: Tuple[int, int] = (128, 64),
        dec_units: int = 128,
        bidirectional: bool = True,
        attention: bool = True,
        dropout: float = 0.2,
        recurrent_dropout: float = 0.0,
        layernorm: bool = True,
        residual_last24: bool = True,
        optimizer: str = "adam",
        loss: str = "mse",
        metrics: Tuple[str, ...] = ("mae",),
        seed: Optional[int] = 42,
    ) -> "LSTMSeq2SeqPlus":
        """
        Convenience constructor that infers shape parameters from arrays.

        Parameters
        ----------
        X_train : np.ndarray
            Shape (batch, input_steps, n_features)
        y_train : np.ndarray
            Shape (batch, output_steps, output_dim)

        Other keyword-only parameters
        -----------------------------
        enc_units, dec_units, bidirectional, attention, dropout, recurrent_dropout,
        layernorm, residual_last24, optimizer, loss, metrics, seed

        Returns
        -------
        LSTMSeq2SeqPlus
            A fully initialized model with an inferred configuration.
        """
        cls._validate_shapes(X_train, y_train)

        input_steps, n_features = X_train.shape[1], X_train.shape[2]
        output_steps, output_dim = y_train.shape[1], y_train.shape[2]

        cfg = LSTMSeq2SeqPlusConfig(
            input_steps=input_steps,
            n_features=n_features,
            output_steps=output_steps,
            output_dim=output_dim,
            enc_units=enc_units,
            dec_units=dec_units,
            bidirectional=bidirectional,
            attention=attention,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            layernorm=layernorm,
            residual_last24=residual_last24,
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
        Set seeds for best-effort reproducibility. (Not perfectly deterministic.)
        """
        if seed is not None:
            tf.keras.utils.set_random_seed(seed)

    # ---------------------------------------------------------------------
    # Build / compile
    # ---------------------------------------------------------------------

    def _build(self) -> Model:
        """
        Build and compile the LSTM encoder–decoder with optional attention and residual skip.

        Architecture overview
        ---------------------
        Encoder
          - One or more LSTM layers (optionally bidirectional)
          - Optional LayerNorm after each encoder layer
          - Final encoder states are projected to match decoder size

        Decoder
          - Global average pooling over encoder sequence -> repeated to output_steps
          - Single LSTM decoder layer initializes from projected encoder states
          - Optional LayerNorm on decoder outputs
          - Optional AdditiveAttention between decoder sequence (query) and encoder sequence (key/value)

        Residual skip (optional)
          - Crops the last `output_steps` time steps from inputs
          - Projects them and concatenates with decoder features

        Head
          - TimeDistributed Dense -> Dropout -> TimeDistributed Dense (linear)

        Returns
        -------
        tf.keras.Model
            A compiled Keras model with the specified optimizer, loss, and metrics.
        """
        c = self.config
        x_in = Input(shape=(c.input_steps, c.n_features), name="inputs")

        # ----- Residual last-24h covariates (no Lambda closures) -----
        # Crop from the left so we keep the final `output_steps` time steps.
        last24_proj = None
        if c.residual_last24:
            left_crop = max(0, c.input_steps - c.output_steps)
            last24 = Cropping1D(cropping=(left_crop, 0), name="last24_slice")(x_in)  # (B, output_steps, F)
            last24_proj = TimeDistributed(Dense(16, activation="linear"), name="last24_proj")(last24)

        # ---------------- Encoder ----------------
        enc_seq = x_in
        enc_states = []
        for i, units in enumerate(c.enc_units):
            lstm = LSTM(
                units,
                return_sequences=True,
                return_state=True,
                dropout=c.dropout,
                recurrent_dropout=c.recurrent_dropout,
                name=f"enc_lstm_{i + 1}",
            )
            if c.bidirectional:
                # Bidirectional wrapper returns: outputs, h_f, c_f, h_b, c_b
                bi = Bidirectional(lstm, name=f"bidir_enc_{i + 1}")
                enc_seq, h_f, c_f, h_b, c_b = bi(enc_seq)
                h = Concatenate(name=f"enc_{i + 1}_h_concat")([h_f, h_b])
                cs = Concatenate(name=f"enc_{i + 1}_c_concat")([c_f, c_b])
            else:
                enc_seq, h, cs = lstm(enc_seq)

            # Keep the *last* layer's states for initializing the decoder.
            enc_states = [h, cs]

            if c.layernorm:
                enc_seq = LayerNormalization(name=f"enc_ln_{i + 1}")(enc_seq)

        # Project encoder final states to the decoder hidden size
        init_h = Dense(c.dec_units, activation="tanh", name="init_h")(enc_states[0])
        init_c = Dense(c.dec_units, activation="tanh", name="init_c")(enc_states[1])

        # Global context (mean pool) repeated to output_steps as decoder input seed
        global_ctx = GlobalAveragePooling1D(name="enc_avg_pool")(enc_seq)
        rep = RepeatVector(c.output_steps, name="repeat_output_steps")(global_ctx)

        # ---------------- Decoder ----------------
        dec_seq, _, _ = LSTM(
            c.dec_units,
            return_sequences=True,
            return_state=True,
            dropout=c.dropout,
            recurrent_dropout=c.recurrent_dropout,
            name="dec_lstm",
        )(rep, initial_state=[init_h, init_c])

        if c.layernorm:
            dec_seq = LayerNormalization(name="dec_ln")(dec_seq)

        # ---------------- Attention (optional) ----------------
        if c.attention:
            # AdditiveAttention: [query=dec_seq, value=enc_seq] => context aligned to decoder steps
            context = AdditiveAttention(name="additive_attention")([dec_seq, enc_seq])
            dec_aug = Concatenate(name="concat_dec_context")([dec_seq, context])
        else:
            dec_aug = dec_seq

        # ---------------- Residual concat (optional) ----------------
        if last24_proj is not None:
            dec_aug = Concatenate(name="concat_with_last24")([dec_aug, last24_proj])

        # ---------------- Head ----------------
        head = TimeDistributed(Dense(128, activation="relu"), name="head_dense1")(dec_aug)
        head = Dropout(c.dropout, name="head_dropout")(head)
        out = TimeDistributed(Dense(c.output_dim, activation="linear"), name="hourly_output")(head)

        model = Model(inputs=x_in, outputs=out, name="lstm_seq2seq_plus")
        model.compile(optimizer=c.optimizer, loss=c.loss, metrics=list(c.metrics))

        # Helpful model summary for quick inspection
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
        epochs: int = 80,
        batch_size: int = 64,
        patience: int = 10,
        min_delta: float = 1e-4,
        reduce_lr_patience: int = 5,
        checkpoint_path: Optional[str] = "best_lstm_plus.keras",  # save full model
        verbose: int = 1,
        shuffle: bool = False,  # preserve temporal order by default
    ):
        """
        Train with time-series-friendly defaults.

        Validation splitting
        --------------------
        - If (X_val, y_val) is not given, we automatically split off the
          last 10% of (X_train, y_train) as a *tail* validation set.

        Callbacks
        ---------
        - EarlyStopping (restore best weights)
        - ReduceLROnPlateau
        - ModelCheckpoint (optional; saves the **entire** model to `.keras`)
        """
        # Basic shape checks
        self._validate_shapes(X_train, y_train)

        # Ensure provided arrays match the model config (defensive programming)
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

        cbs = [
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

        # Save the *entire* model (architecture + weights) to `.keras`
        if checkpoint_path:
            cbs.append(
                ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor="val_loss",
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=verbose,
                )
            )

        return self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            callbacks=cbs,
            verbose=verbose,
        )

    # ---------------------------------------------------------------------
    # Inference / Evaluation
    # ---------------------------------------------------------------------

    def predict(self, X: np.ndarray, batch_size: int = 256, verbose: int = 0) -> np.ndarray:
        """
        Predict sequences for the given inputs.

        Parameters
        ----------
        X : np.ndarray
            Shape (batch, input_steps, n_features)
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

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute flattened regression metrics across all samples and time steps.

        Pass *unscaled* arrays if you want metrics in original units.

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

    def save(self, path: str = "final_lstm_plus.keras") -> None:
        """
        Save the entire model (architecture + weights) to a single `.keras` file.
        """
        self.model.save(path)

    @staticmethod
    def load(path: str) -> "LSTMSeq2SeqPlus":
        """
        Load a full model saved via `.save('*.keras')` and wrap it with a compatible config.

        We infer:
        - (input_steps, n_features) from the model input shape
        - (output_steps, output_dim) from the model output shape
        """
        loaded = load_model(path, compile=True)

        # Keras model shapes:
        # input_shape  -> (None, input_steps, n_features)
        # output_shape -> (None, output_steps, output_dim)
        inp = loaded.input_shape
        out = loaded.output_shape

        cfg = LSTMSeq2SeqPlusConfig(
            input_steps=inp[1],
            n_features=inp[2],
            output_steps=out[1],
            output_dim=out[2],
        )

        obj = LSTMSeq2SeqPlus(cfg)
        obj.model = loaded  # replace freshly built model with the loaded one
        return obj
