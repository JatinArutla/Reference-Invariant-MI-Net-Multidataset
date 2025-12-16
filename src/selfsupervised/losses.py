import tensorflow as tf

@tf.function
def nt_xent_loss(z1: tf.Tensor, z2: tf.Tensor, temperature: float = 0.5) -> tf.Tensor:
    # z1,z2: [B,d] assumed L2-normalized
    B = tf.shape(z1)[0]
    Z = tf.concat([z1, z2], axis=0)                 # [2B,d]
    sim = tf.matmul(Z, Z, transpose_b=True)         # cosine since z are normed
    mask = tf.eye(2 * B, dtype=tf.bool)
    sim = tf.where(mask, tf.zeros_like(sim), sim)   # remove self-sim
    pos = tf.concat([tf.range(B, 2 * B), tf.range(0, B)], axis=0)
    logits = sim / temperature
    labels = tf.one_hot(pos, depth=2 * B)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(loss)


def _off_diagonal(x: tf.Tensor) -> tf.Tensor:
    # Returns a 1D tensor of off-diagonal elements of a square matrix.
    n = tf.shape(x)[0]
    mask = tf.logical_not(tf.eye(n, dtype=tf.bool))
    return tf.boolean_mask(x, mask)


@tf.function
def barlow_twins_loss(
    z1: tf.Tensor,
    z2: tf.Tensor,
    *,
    lambd: float = 5e-3,
) -> tf.Tensor:
    """Barlow Twins loss.

    z1,z2: [B,d]
    """
    z1 = tf.cast(z1, tf.float32)
    z2 = tf.cast(z2, tf.float32)
    # batch norm (zero mean, unit variance per dimension)
    z1 = (z1 - tf.reduce_mean(z1, axis=0, keepdims=True)) / (tf.math.reduce_std(z1, axis=0, keepdims=True) + 1e-9)
    z2 = (z2 - tf.reduce_mean(z2, axis=0, keepdims=True)) / (tf.math.reduce_std(z2, axis=0, keepdims=True) + 1e-9)

    B = tf.cast(tf.shape(z1)[0], tf.float32)
    c = tf.matmul(z1, z2, transpose_a=True) / B  # [d,d]

    on_diag = tf.reduce_sum(tf.square(tf.linalg.diag_part(c) - 1.0))
    off_diag = tf.reduce_sum(tf.square(_off_diagonal(c)))
    return on_diag + lambd * off_diag


@tf.function
def vicreg_loss(
    z1: tf.Tensor,
    z2: tf.Tensor,
    *,
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    eps: float = 1e-4,
) -> tf.Tensor:
    """VICReg loss (variance-invariance-covariance regularization).

    z1,z2: [B,d]
    """
    z1 = tf.cast(z1, tf.float32)
    z2 = tf.cast(z2, tf.float32)

    # invariance
    sim = tf.reduce_mean(tf.square(z1 - z2))

    # variance
    def _std_loss(z):
        std = tf.math.reduce_std(z, axis=0)
        return tf.reduce_mean(tf.nn.relu(1.0 - std + eps))
    std = _std_loss(z1) + _std_loss(z2)

    # covariance
    def _cov_loss(z):
        z = z - tf.reduce_mean(z, axis=0, keepdims=True)
        B = tf.cast(tf.shape(z)[0], tf.float32)
        cov = tf.matmul(z, z, transpose_a=True) / (B - 1.0)
        return tf.reduce_mean(tf.square(_off_diagonal(cov)))
    cov = _cov_loss(z1) + _cov_loss(z2)

    return sim_coeff * sim + std_coeff * std + cov_coeff * cov