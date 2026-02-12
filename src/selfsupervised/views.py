import numpy as np
import tensorflow as tf

from src.datamodules.transforms import apply_reference

# -----------------------------------------------------------------------------
# NumPy augmentations (input [C, T])
#
# IMPORTANT:
# EEG SSL is sensitive to augmentations that destroy label-relevant structure.
# The previous defaults were intentionally "stress-test" aggressive. That is
# useful for debugging but is a liability for a paper narrative.
#
# We now expose an explicit aug policy:
#   - light (default): small, label-preserving perturbations
#   - aggressive: stronger but still avoids time-reversal and heavy permutation
#   - legacy: prior behavior (kept for reproducibility)
# -----------------------------------------------------------------------------


def _gaussian_noise(x: np.ndarray, sigma_low: float = 0.005, sigma_high: float = 0.03) -> np.ndarray:
    # Noise scaled to per-channel std.
    std = np.std(x, axis=1, keepdims=True) + 1e-6
    sigma = np.random.uniform(sigma_low, sigma_high)
    return x + (np.random.randn(*x.shape).astype(np.float32) * (sigma * std).astype(np.float32))


def _amp_scale_mild(x: np.ndarray, low: float = 0.90, high: float = 1.10) -> np.ndarray:
    s = np.random.uniform(low, high)
    return x * s


def _time_shift_zero_pad(x: np.ndarray, max_frac: float = 0.05) -> np.ndarray:
    # Shift up to +/- max_frac of the window, padding with zeros.
    C, T = x.shape
    max_shift = int(round(T * max_frac))
    if max_shift <= 0:
        return x
    sh = np.random.randint(-max_shift, max_shift + 1)
    if sh == 0:
        return x
    y = np.zeros_like(x)
    if sh > 0:
        y[:, sh:] = x[:, : T - sh]
    else:
        sh = -sh
        y[:, : T - sh] = x[:, sh:]
    return y


def _time_mask_zero(x: np.ndarray, frac_low: float = 0.05, frac_high: float = 0.20) -> np.ndarray:
    C, T = x.shape
    L = int(round(np.random.uniform(frac_low, frac_high) * T))
    L = max(1, min(L, T))
    st = np.random.randint(0, T - L + 1)
    y = x.copy()
    y[:, st: st + L] = 0.0
    return y


def _crop_resize(x: np.ndarray, ratio_low: float = 0.85, ratio_high: float = 1.00) -> np.ndarray:
    C, T = x.shape
    r = np.random.uniform(ratio_low, ratio_high)
    L = max(1, int(T * r))
    st = np.random.randint(0, T - L + 1)
    crop = x[:, st:st + L]
    grid = np.linspace(0, L - 1, T)
    return np.stack([np.interp(grid, np.arange(L), crop[c]) for c in range(C)], 0)


def _time_warp(x: np.ndarray, seg_min: int = 4, seg_max: int = 8, scale: float = 0.10) -> np.ndarray:
    # Mild piecewise-linear time warp.
    C, T = x.shape
    m = np.random.randint(seg_min, seg_max + 1)
    cuts = np.linspace(0, T, m + 1, dtype=int)
    parts = []
    for i in range(m):
        seg = x[:, cuts[i]:cuts[i + 1]]
        if seg.shape[1] <= 1:
            parts.append(seg)
            continue
        f = np.random.uniform(1.0 - scale, 1.0 + scale)
        L = max(1, int(seg.shape[1] * f))
        grid = np.linspace(0, seg.shape[1] - 1, L)
        parts.append(np.stack([np.interp(grid, np.arange(seg.shape[1]), seg[c]) for c in range(C)], 0))
    warp = np.concatenate(parts, 1)
    if warp.shape[1] == T:
        return warp
    gridT = np.linspace(0, warp.shape[1] - 1, T)
    return np.stack([np.interp(gridT, np.arange(warp.shape[1]), warp[c]) for c in range(C)], 0)


def _permute(x: np.ndarray, seg_min: int = 4, seg_max: int = 10) -> np.ndarray:
    # Legacy-style segment permutation (kept for reproducibility only).
    C, T = x.shape
    m = np.random.randint(seg_min, seg_max + 1)
    cuts = np.linspace(0, T, m + 1, dtype=int)
    segs = [x[:, cuts[i]:cuts[i + 1]] for i in range(m)]
    np.random.shuffle(segs)
    return np.concatenate(segs, 1)


def _flip_time(x: np.ndarray) -> np.ndarray:
    # Legacy only: time reversal is usually not label-preserving for MI.
    return x[:, ::-1]


def _amp_add(x: np.ndarray, low: float = 1.0, high: float = 4.0) -> np.ndarray:
    # Legacy only: huge DC offsets are unrealistic and can dominate.
    a = np.random.uniform(low, high)
    return x + a


def _amp_scale(x: np.ndarray, low: float = 2.0, high: float = 4.0) -> np.ndarray:
    # Legacy only: extreme gain changes are unrealistic.
    s = np.random.uniform(low, high)
    return x * s


def _cutout_resize(x: np.ndarray, seg_min: int = 4, seg_max: int = 10) -> np.ndarray:
    # Legacy only.
    C, T = x.shape
    m = np.random.randint(seg_min, seg_max + 1)
    cuts = np.linspace(0, T, m + 1, dtype=int)
    p = np.random.randint(0, m)
    keep = [x[:, cuts[i]:cuts[i + 1]] for i in range(m) if i != p]
    y = np.concatenate(keep, 1)
    grid = np.linspace(0, y.shape[1] - 1, T)
    return np.stack([np.interp(grid, np.arange(y.shape[1]), y[c]) for c in range(C)], 0)


def _cutout_zero(x: np.ndarray, seg_min: int = 4, seg_max: int = 10) -> np.ndarray:
    # Legacy only.
    C, T = x.shape
    m = np.random.randint(seg_min, seg_max + 1)
    cuts = np.linspace(0, T, m + 1, dtype=int)
    p = np.random.randint(0, m)
    y = x.copy()
    y[:, cuts[p]:cuts[p + 1]] = 0.0
    return y


LEGACY_AUGS = [_amp_add, _amp_scale, _time_warp, _cutout_resize, _cutout_zero, _crop_resize, _flip_time, _permute]

LIGHT_AUGS = [_gaussian_noise, _amp_scale_mild, _time_shift_zero_pad, _time_mask_zero, _crop_resize]

AGGRESSIVE_AUGS = [_gaussian_noise, _amp_scale_mild, _time_shift_zero_pad, _time_mask_zero, _crop_resize, _time_warp]


def _resolve_aug_policy(policy: str | None) -> list:
    p = (policy or "light").strip().lower()
    if p in ("light", "mild"):
        return LIGHT_AUGS
    if p in ("aggressive", "strong"):
        return AGGRESSIVE_AUGS
    if p in ("legacy", "old"):
        return LEGACY_AUGS
    raise ValueError(f"Unknown aug_policy: {policy}. Use one of: light, aggressive, legacy")


def two_random_augs(x: np.ndarray, *, aug_policy: str = "light") -> tuple[np.ndarray, np.ndarray]:
    ops = _resolve_aug_policy(aug_policy)
    if len(ops) < 2:
        raise RuntimeError("Aug policy must include at least 2 ops")
    op1, op2 = np.random.choice(ops, size=2, replace=False)
    v1 = op1(x.astype(np.float32, copy=False)).astype(np.float32, copy=False)
    v2 = op2(x.astype(np.float32, copy=False)).astype(np.float32, copy=False)
    return v1, v2


def two_reference_views(
    x: np.ndarray,
    *,
    ref_modes: list[str],
    ref_idx: int | None = None,
    lap_neighbors: list[list[int]] | None = None,
    with_augs: bool = False,
    aug_policy: str = "light",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two SSL views using different reference transforms.

    If with_augs is True, apply standard augs after re-referencing.
    """
    if len(ref_modes) < 2:
        raise ValueError("ref_modes must contain at least 2 modes")
    m1, m2 = np.random.choice(ref_modes, size=2, replace=False)
    v1 = apply_reference(x, mode=str(m1), ref_idx=ref_idx, lap_neighbors=lap_neighbors)
    v2 = apply_reference(x, mode=str(m2), ref_idx=ref_idx, lap_neighbors=lap_neighbors)
    if with_augs:
        v1, _ = two_random_augs(v1, aug_policy=aug_policy)
        v2, _ = two_random_augs(v2, aug_policy=aug_policy)
    return v1.astype(np.float32, copy=False), v2.astype(np.float32, copy=False)

# tf.data builder
def _two_views_np(x: np.ndarray, aug_policy: str) -> tuple[np.ndarray, np.ndarray]:
    return two_random_augs(x, aug_policy=aug_policy)

def make_ssl_dataset(
    X: np.ndarray,
    *,
    n_channels: int,
    in_samples: int,
    batch_size: int = 256,
    shuffle: bool = True,
    seed: int = 1,
    deterministic: bool = True,
    view_mode: str = "aug",              # "aug" (default) or "ref" or "ref+aug"
    aug_policy: str = "light",
    ref_modes: list[str] | None = None,
    ref_idx: int | None = None,
    lap_neighbors: list[list[int]] | None = None,
) -> tf.data.Dataset:
    N, C, T = X.shape
    assert C == n_channels and T == in_samples

    Xf = X.astype(np.float32, copy=False)
    ds = tf.data.Dataset.from_tensor_slices(Xf)
    if shuffle:
        ds = ds.shuffle(buffer_size=N, seed=seed, reshuffle_each_iteration=True)

    view_mode_l = (view_mode or "aug").lower()
    if view_mode_l in ("aug", "augs"):
        mapper = lambda x: tf.numpy_function(lambda xx: _two_views_np(xx, aug_policy), [x], Tout=(tf.float32, tf.float32))
    elif view_mode_l in ("ref", "reference", "ref_only"):
        if not ref_modes:
            raise ValueError("ref_modes must be provided when view_mode='ref'")
        def _two_ref(x_np):
            return two_reference_views(x_np, ref_modes=ref_modes, ref_idx=ref_idx, lap_neighbors=lap_neighbors, with_augs=False)
        mapper = lambda x: tf.numpy_function(_two_ref, [x], Tout=(tf.float32, tf.float32))
    elif view_mode_l in ("ref+aug", "ref_aug", "reference+aug"):
        if not ref_modes:
            raise ValueError("ref_modes must be provided when view_mode='ref+aug'")
        def _two_ref_aug(x_np):
            return two_reference_views(
                x_np,
                ref_modes=ref_modes,
                ref_idx=ref_idx,
                lap_neighbors=lap_neighbors,
                with_augs=True,
                aug_policy=aug_policy,
            )
        mapper = lambda x: tf.numpy_function(_two_ref_aug, [x], Tout=(tf.float32, tf.float32))
    else:
        raise ValueError(f"Unknown view_mode: {view_mode}")

    if deterministic:
        opts = tf.data.Options()
        opts.experimental_deterministic = True
        ds = ds.with_options(opts)
        num_calls = 1
        prefetch = 1
    else:
        num_calls = tf.data.AUTOTUNE
        prefetch = tf.data.AUTOTUNE

    ds = ds.map(mapper, num_parallel_calls=num_calls)
    ds = ds.map(
        lambda v1, v2: (
            tf.ensure_shape(v1, (n_channels, in_samples)),
            tf.ensure_shape(v2, (n_channels, in_samples))
        ),
        num_parallel_calls=num_calls
    )
    ds = ds.map(lambda v1, v2: (tf.expand_dims(v1, 0), tf.expand_dims(v2, 0)),  # -> [1,C,T]
                num_parallel_calls=num_calls)
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(prefetch)
    return ds
