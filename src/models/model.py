# src/models/model.py

import math
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Permute, Lambda, Dense, Activation, Concatenate, Average,
    Conv1D, Conv2D, DepthwiseConv2D, BatchNormalization, Dropout,
    Add, LayerNormalization, MultiHeadAttention, AveragePooling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import L2
from tensorflow.keras.constraints import max_norm

SEED = 1

# -------- Attention blocks --------

def mha_block(x, key_dim=8, num_heads=2, dropout=0.5, vanilla=True):
    x_norm = LayerNormalization(epsilon=1e-6)(x)
    if vanilla:
        attn = MultiHeadAttention(key_dim=key_dim, num_heads=num_heads, dropout=dropout)(x_norm, x_norm)
    else:
        # locality self-attention
        attn = MultiHeadAttention_LSA(key_dim=key_dim, num_heads=num_heads, dropout=dropout)(x_norm, x_norm)
    attn = Dropout(0.3, seed=SEED)(attn)
    return Add()([x, attn])

class MultiHeadAttention_LSA(tf.keras.layers.MultiHeadAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tau = tf.Variable(math.sqrt(float(self._key_dim)), trainable=True)

    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        query = tf.multiply(query, 1.0 / self.tau)
        attention_scores = tf.einsum(self._dot_product_equation, key, query)
        attention_scores = self._masked_softmax(attention_scores, attention_mask)
        attention_scores_dropout = self._dropout_layer(attention_scores, training=training)
        attention_output = tf.einsum(self._combine_equation, attention_scores_dropout, value)
        return attention_output, attention_scores

def attention_block(x, mode):
    if mode == "mha":
        return mha_block(x, vanilla=True)
    if mode == "mhla":
        return mha_block(x, vanilla=False)
    if mode in (None, "", "none"):
        return x
    raise ValueError(f"Unsupported attention '{mode}'")

# -------- EEGNet-like 2D stem --------

def Conv_block_(x, F1=16, kernLength=64, poolSize=7, D=2, in_chans=22,
                weightDecay=0.009, maxNormV=0.6, dropout=0.3):
    F2 = F1 * D
    x = Conv2D(
        F1, (kernLength, 1), padding="same", data_format="channels_last",
        kernel_regularizer=L2(weightDecay),
        kernel_constraint=max_norm(maxNormV, axis=[0, 1, 2]),
        use_bias=False
    )(x)
    x = BatchNormalization(axis=-1)(x)
    x = DepthwiseConv2D(
        (1, in_chans), depth_multiplier=D, data_format="channels_last",
        depthwise_regularizer=L2(weightDecay),
        depthwise_constraint=max_norm(maxNormV, axis=[0, 1, 2]),
        use_bias=False
    )(x)  # width → 1
    x = BatchNormalization(axis=-1)(x)
    x = Activation("elu")(x)
    x = AveragePooling2D((8, 1), data_format="channels_last")(x)
    x = Dropout(dropout, seed=SEED)(x)
    x = Conv2D(
        F2, (16, 1), padding="same", data_format="channels_last", use_bias=False,
        kernel_regularizer=L2(weightDecay),
        kernel_constraint=max_norm(maxNormV, axis=[0, 1, 2])
    )(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("elu")(x)
    x = AveragePooling2D((poolSize, 1), data_format="channels_last")(x)
    x = Dropout(dropout, seed=SEED)(x)
    return x  # [B, T', 1, F2]

# -------- Temporal Conv Net --------

def TCN_block_(x, input_dim, depth, kernel_size, filters, dropout,
               weightDecay=0.009, maxNormV=0.6, activation="elu"):
    def conv(a, dil):
        a = Conv1D(
            filters, kernel_size=kernel_size, dilation_rate=dil, activation="linear",
            kernel_regularizer=L2(weightDecay),
            kernel_constraint=max_norm(maxNormV, axis=[0, 1]),
            padding="causal", kernel_initializer="he_uniform"
        )(a)
        a = BatchNormalization()(a); a = Activation(activation)(a); a = Dropout(dropout, seed=SEED)(a)
        a = Conv1D(
            filters, kernel_size=kernel_size, dilation_rate=dil, activation="linear",
            kernel_regularizer=L2(weightDecay),
            kernel_constraint=max_norm(maxNormV, axis=[0, 1]),
            padding="causal", kernel_initializer="he_uniform"
        )(a)
        a = BatchNormalization()(a); a = Activation(activation)(a); a = Dropout(dropout, seed=SEED)(a)
        return a

    y = conv(x, 1)
    skip = x if input_dim == filters else Conv1D(filters, 1, padding="same")(x)
    y = Activation(activation)(Add()([y, skip]))
    for i in range(depth - 1):
        z = conv(y, 2 ** (i + 1))
        y = Activation(activation)(Add()([z, y]))
    return y

# -------- Model --------

def build_atcnet(
    n_classes: int,
    in_chans: int = 22,
    in_samples: int = 1000,
    n_windows: int = 5,
    attention: str = "mha",
    eegn_F1: int = 16,
    eegn_D: int = 2,
    eegn_kernel: int = 64,
    eegn_pool: int = 7,
    eegn_dropout: float = 0.3,
    tcn_depth: int = 2,
    tcn_kernel: int = 4,
    tcn_filters: int = 32,
    tcn_dropout: float = 0.3,
    tcn_activation: str = "elu",
    fuse: str = "average",              # "average" or "concat"
    from_logits: bool = False,
    return_ssl_feat: bool = False       # True → return [out, ssl_feat]
) -> tf.keras.Model:
    # input: [B, 1, C, T] → [B, T, C, 1]
    inp = Input(shape=(1, in_chans, in_samples))
    x = Permute((3, 2, 1))(inp)

    # stem → [B, T', 1, F2] → squeeze width
    x = Conv_block_(x, F1=eegn_F1, kernLength=eegn_kernel, poolSize=eegn_pool,
                    D=eegn_D, in_chans=in_chans, dropout=eegn_dropout)
    x = Lambda(lambda z: tf.squeeze(z, axis=2))(x)  # [B, T', F2]

    F2 = eegn_F1 * eegn_D
    per_win_feats = []
    feats_or_logits = []

    for i in range(n_windows):
        # dynamic crop kept inside graph
        xi = Lambda(lambda z, i=i, n_windows=n_windows: z[:, i:(tf.shape(z)[1] - n_windows + i + 1), :])(x)
        xi = attention_block(xi, attention)
        xi = TCN_block_(xi, input_dim=F2, depth=tcn_depth, kernel_size=tcn_kernel,
                        filters=tcn_filters, dropout=tcn_dropout, activation=tcn_activation)
        xi = Lambda(lambda z: z[:, -1, :])(xi)  # last timestep
        per_win_feats.append(xi)

        if fuse == "average":
            logits_i = Dense(n_classes, kernel_regularizer=L2(0.5))(xi)
            feats_or_logits.append(logits_i)
        elif fuse == "concat":
            feats_or_logits = xi if i == 0 else Concatenate(axis=-1)([feats_or_logits, xi])
        else:
            raise ValueError("fuse must be 'average' or 'concat'")

    if fuse == "average":
        out_pre = feats_or_logits[0] if len(feats_or_logits) == 1 else Average()(feats_or_logits)
    else:
        out_pre = Dense(n_classes, kernel_regularizer=L2(0.5))(feats_or_logits)

    out = Activation("linear" if from_logits else "softmax",
                     name="linear" if from_logits else "softmax")(out_pre)

    if return_ssl_feat:
        ssl_feat = per_win_feats[0] if len(per_win_feats) == 1 else Average(name="ssl_feat")(per_win_feats)
        return Model(inputs=inp, outputs=[out, ssl_feat], name="ATCNet")

    return Model(inputs=inp, outputs=out, name="ATCNet")