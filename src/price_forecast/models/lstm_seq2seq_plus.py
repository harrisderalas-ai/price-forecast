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


@dataclass(frozen=True)
class LSTMSeq2SeqPlusConfig:
    # Shapes
    input_steps: int  # n_lookback_days * 24
    n_features: int
    output_steps: int = 24  # 24h day-ahead
    output_dim: int = 1

    # Architecture
    enc_units: Tuple[int, int] = (128, 64)
    dec_units: int = 128
    bidirectional: bool = True
    attention: bool = True
    dropout: float = 0.2
    recurrent_dropout: float = 0.0
    layernorm: bool = True
    residual_last24: bool = True

    # Training
    optimizer: str = "adam"
    loss: str = "mse"
    metrics: Tuple[str, ...] = ("mae",)
    seed: Optional[int] = 42


class LSTMSeq2SeqPlus:
    """
    Seq2Seq LSTM with attention + residual last-24h skip for day-ahead forecasting.
    Input:  (batch, input_steps, n_features)
    Output: (batch, 24, output_dim)
    """

    def __init__(self, config: LSTMSeq2SeqPlusConfig):
        self.config = config
        self._set_seed(config.seed)
        self.model: Model = self._build()

    # ---------------- Build ----------------

    def _build(self) -> Model:
        c = self.config
        x_in = Input(shape=(c.input_steps, c.n_features), name="inputs")

        # ----- Residual last 24h covariates without Lambda closures -----
        if c.residual_last24:
            left_crop = max(
                0, c.input_steps - c.output_steps
            )  # crop from the left to keep last 24
            last24 = Cropping1D(cropping=(left_crop, 0), name="last24_slice")(
                x_in
            )  # (B,24,F)
            last24_proj = TimeDistributed(
                Dense(16, activation="linear"), name="last24_proj"
            )(last24)
        else:
            last24_proj = None

        # ----- Encoder -----
        enc_seq = x_in
        enc_states = []
        for i, units in enumerate(c.enc_units):
            layer = LSTM(
                units,
                return_sequences=True,
                return_state=True,
                dropout=c.dropout,
                recurrent_dropout=c.recurrent_dropout,
                name=f"enc_lstm_{i + 1}",
            )
            if c.bidirectional:
                layer = Bidirectional(layer, name=f"bidir_enc_{i + 1}")
                enc_seq, h_f, c_f, h_b, c_b = layer(enc_seq)
                h = Concatenate(name=f"enc_{i + 1}_h_concat")([h_f, h_b])
                cs = Concatenate(name=f"enc_{i + 1}_c_concat")([c_f, c_b])
            else:
                enc_seq, h, cs = layer(enc_seq)
            enc_states = [h, cs]
            if c.layernorm:
                enc_seq = LayerNormalization(name=f"enc_ln_{i + 1}")(enc_seq)

        # Map encoder final states to decoder init size
        dec_units = c.dec_units
        init_h = Dense(dec_units, activation="tanh", name="init_h")(enc_states[0])
        init_c = Dense(dec_units, activation="tanh", name="init_c")(enc_states[1])

        # Global context (mean pool) repeated to 24 steps as decoder input seed
        global_ctx = GlobalAveragePooling1D(name="enc_avg_pool")(enc_seq)
        rep = RepeatVector(c.output_steps, name="repeat_output_steps")(global_ctx)

        # Decoder over 24 steps
        dec_seq, _, _ = LSTM(
            dec_units,
            return_sequences=True,
            return_state=True,
            dropout=c.dropout,
            recurrent_dropout=c.recurrent_dropout,
            name="dec_lstm",
        )(rep, initial_state=[init_h, init_c])
        if c.layernorm:
            dec_seq = LayerNormalization(name="dec_ln")(dec_seq)

        # Attention over encoder sequence
        if c.attention:
            context = AdditiveAttention(name="additive_attention")([dec_seq, enc_seq])
            dec_aug = Concatenate(name="concat_dec_context")([dec_seq, context])
        else:
            dec_aug = dec_seq

        # Residual covariates from last 24h of inputs
        if last24_proj is not None:
            dec_aug = Concatenate(name="concat_with_last24")([dec_aug, last24_proj])

        # Head
        head = TimeDistributed(Dense(128, activation="relu"), name="head_dense1")(
            dec_aug
        )
        head = Dropout(c.dropout, name="head_dropout")(head)
        out = TimeDistributed(
            Dense(c.output_dim, activation="linear"), name="hourly_output"
        )(head)

        model = Model(inputs=x_in, outputs=out, name="lstm_seq2seq_plus")
        model.compile(optimizer=c.optimizer, loss=c.loss, metrics=list(c.metrics))
        model.summary()
        return model

    @staticmethod
    def _set_seed(seed: Optional[int]) -> None:
        if seed is not None:
            tf.keras.utils.set_random_seed(seed)

    # ---------------- Train ----------------

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
        checkpoint_path: Optional[str] = "best_lstm_plus.keras",
        verbose: int = 1,
        shuffle: bool = False,
    ):
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
        if checkpoint_path:
            cbs.append(
                ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor="val_loss",
                    save_best_only=True,
                    save_weights_only=False,  # save full model -> .keras
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

    # ---------------- Inference / Eval ----------------

    def predict(
        self, X: np.ndarray, batch_size: int = 256, verbose: int = 0
    ) -> np.ndarray:
        return self.model.predict(X, batch_size=batch_size, verbose=verbose)

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mse = mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1))
        mae = mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1))
        return {"mse": float(mse), "mae": float(mae)}

    # ---------------- Persistence ----------------

    def save(self, path: str = "final_lstm_plus.keras") -> None:
        self.model.save(path)

    @staticmethod
    def load(path: str) -> "LSTMSeq2SeqPlus":
        loaded = load_model(path, compile=True)
        inp = loaded.input_shape
        out = loaded.output_shape
        cfg = LSTMSeq2SeqPlusConfig(
            input_steps=inp[1],
            n_features=inp[2],
            output_steps=out[1],
            output_dim=out[2],
        )
        obj = LSTMSeq2SeqPlus(cfg)
        obj.model = loaded
        return obj
