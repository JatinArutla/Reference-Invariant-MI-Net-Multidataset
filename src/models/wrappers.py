from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from .heads import build_projection_head

def build_ssl_projector(
    encoder_with_tap: Model,
    proj_dim: int = 128,
    out_dim: int = 64,
    *,
    l2norm: bool = True,
) -> Model:
    inp = Input(shape=encoder_with_tap.input_shape[1:], name="ssl_in")
    out_pred, ssl_feat = encoder_with_tap(inp)
    z = build_projection_head(ssl_feat, proj_dim=proj_dim, out_dim=out_dim, l2norm=l2norm)
    return Model(inp, z, name="SSL_Projector")