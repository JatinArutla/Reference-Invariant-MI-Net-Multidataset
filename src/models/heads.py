from tensorflow.keras.layers import Dense, Activation, Lambda
from tensorflow.keras.models import Model
import tensorflow as tf

def build_classifier_head(x, n_classes: int, from_logits: bool = False):
    logits = Dense(n_classes)(x)
    act = "linear" if from_logits else "softmax"
    return Activation(act)(logits)

def build_projection_head(ssl_feat, proj_dim: int = 128, out_dim: int = 64, *, l2norm: bool = True):
    h = Dense(proj_dim, activation="relu", name="proj_dense")(ssl_feat)
    z = Dense(out_dim, activation=None, name="proj_out")(h)
    if l2norm:
        z = Lambda(lambda t: tf.math.l2_normalize(t, axis=-1), name="l2norm")(z)
    return z