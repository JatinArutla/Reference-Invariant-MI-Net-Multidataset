"""Self-supervised pretraining for the multi-dataset benchmark.

This version fixes the original script's broken imports, bad SSL dataset call,
multi-output encoder handling, and path / CLI incompatibilities.

Typical usage:
  python train_ssl_multi.py \
    --dataset iv2a \
    --data_root /path/to/data \
    --subject 1 \
    --out_dir outputs/iv2a/sub001/ssl_refaug \
    --view_mode ref+aug \
    --view_ref_modes native,car,laplacian,bipolar,median \
    --ssl_loss vicreg

The saved weights can be used by gate_reference_multi.py via:
  --ssl_weights <that out_dir>
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["TF_DISABLE_LAYOUT_OPTIMIZER"] = "1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from src.models.model import build_atcnet
from src.models.wrappers import build_ssl_projector
from src.datamodules.datasets import get_dataset
from src.datamodules.datasets.base import SplitSpec
from src.datamodules.channels import CANON_CHS_18, BCI2A_CH_NAMES, neighbors_to_index_list, name_to_index, parse_keep_channels
from src.datamodules.transforms import apply_reference, fit_standardizer, standardize_instance
from src.datamodules.datasets.protocol_defaults import HARMONIZED_BASELINE, get_protocol_preset
from src.selfsupervised.views import make_ssl_dataset
from src.selfsupervised.losses import vicreg_loss, barlow_twins_loss, nt_xent_loss


def set_seed(seed: int = 1):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _reshape_for_model(X: np.ndarray) -> np.ndarray:
    n, c, t = X.shape
    return X.reshape(n, 1, c, t).astype(np.float32, copy=False)


def _to_b1ct(x: tf.Tensor) -> tf.Tensor:
    return x if x.shape.rank == 4 else tf.expand_dims(x, 1)


def _split_csv(s: str | None) -> list[str]:
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def _resolve_subject_out_dir(base_out_dir: str, dataset: str, subject: int) -> str:
    subj_token = f"sub{subject:03d}"
    parts = list(Path(base_out_dir).parts)
    if subj_token in parts:
        return str(Path(base_out_dir))
    if dataset in parts:
        return str(Path(base_out_dir) / subj_token)
    return str(Path(base_out_dir) / dataset / subj_token)


def _build_encoder(args) -> tf.keras.Model:
    fuse = args.fuse
    if fuse == "avg":
        fuse = "average"
    return build_atcnet(
        n_classes=2,
        in_chans=args.n_channels,
        in_samples=args.in_samples,
        n_windows=args.n_windows,
        attention=args.attention,
        eegn_F1=args.eegn_F1,
        eegn_D=args.eegn_D,
        eegn_kernel=args.eegn_kernel,
        eegn_pool=args.eegn_pool,
        eegn_dropout=args.eegn_dropout,
        tcn_depth=args.tcn_depth,
        tcn_kernel=args.tcn_kernel,
        tcn_filters=args.tcn_filters,
        tcn_dropout=args.tcn_dropout,
        tcn_activation=args.tcn_activation,
        fuse=fuse,
        from_logits=False,
        return_ssl_feat=True,
    )


def _resolve_loss(args):
    name = (args.ssl_loss or "vicreg").lower()
    if name in ("nt_xent", "ntxent", "simclr"):
        temperature = tf.constant(args.temperature, tf.float32)

        def _loss(z1, z2):
            return nt_xent_loss(z1, z2, temperature=temperature)

        return _loss, True
    if name in ("barlow", "barlow_twins"):

        def _loss(z1, z2):
            return barlow_twins_loss(z1, z2, lambd=float(args.barlow_lambda))

        return _loss, False
    if name == "vicreg":

        def _loss(z1, z2):
            return vicreg_loss(
                z1,
                z2,
                sim_coeff=float(args.vicreg_sim),
                std_coeff=float(args.vicreg_std),
                cov_coeff=float(args.vicreg_cov),
            )

        return _loss, False
    raise ValueError(f"Unknown ssl_loss: {args.ssl_loss}")


def _current_channel_names(args) -> list[str]:
    keep_idx = parse_keep_channels(args.keep_channels, all_names=BCI2A_CH_NAMES)
    return [BCI2A_CH_NAMES[i] for i in keep_idx] if keep_idx is not None else list(BCI2A_CH_NAMES)


def _resolve_ref_params(args, view_modes: list[str]):
    cur_names = _current_channel_names(args)

    ref_idx = None
    need_ref = any((m or "").lower() in ("ref", "cz_ref", "channel_ref") for m in view_modes)
    if need_ref:
        ch_to_idx = name_to_index(cur_names)
        if args.ref_channel not in ch_to_idx:
            raise ValueError(f"ref_channel '{args.ref_channel}' not in current channel set")
        ref_idx = ch_to_idx[args.ref_channel]

    need_lap = args.laplacian or any(
        (m or "").lower() in (
            "laplacian", "lap", "local",
            "bipolar", "bip", "bipolar_like",
            "bipolar_edges", "bip_edges", "edges_bipolar",
        )
        for m in view_modes
    )
    lap_neighbors = (
        neighbors_to_index_list(
            all_names=BCI2A_CH_NAMES,
            keep_names=cur_names,
            sort_by_distance=True,
        )
        if need_lap
        else None
    )
    return ref_idx, lap_neighbors


def _prepare_ssl_standardizer(args, X_tr: np.ndarray, *, view_mode: str, view_modes: list[str], ref_idx, lap_neighbors):
    std_mode = (args.standardize_mode or "none").lower()
    if std_mode == "none":
        return None, None
    if std_mode == "instance":
        return None, None
    if std_mode != "train":
        raise ValueError("standardize_mode must be one of: train, instance, none")

    vm = (view_mode or "aug").lower()
    if vm in ("ref", "reference", "ref_only", "ref+aug", "ref_aug", "reference+aug"):
        stats_pool = np.concatenate(
            [apply_reference(X_tr, mode=m, ref_idx=ref_idx, lap_neighbors=lap_neighbors) for m in view_modes],
            axis=0,
        )
    else:
        stats_pool = X_tr
    return fit_standardizer(stats_pool)


def _band_arg(args):
    if args.band_lo is None or args.band_hi is None:
        return None
    if float(args.band_lo) <= 0.0 or float(args.band_hi) <= 0.0:
        return None
    return (float(args.band_lo), float(args.band_hi))


def _maybe_apply_protocol_presets(args):
    preset = get_protocol_preset(args.dataset, args.protocol)
    args.resolved_protocol_preset = None
    if preset is None:
        return

    changed = {}

    def maybe_set(attr: str, value, *, only_if_current=None):
        if value is None:
            return
        cur = getattr(args, attr)
        if only_if_current is not None and cur != only_if_current:
            return
        if cur != value:
            setattr(args, attr, value)
            changed[attr] = value

    maybe_set('tmin', preset.tmin, only_if_current=HARMONIZED_BASELINE.tmin)
    maybe_set('tmax', preset.tmax, only_if_current=HARMONIZED_BASELINE.tmax)

    if (args.protocol or '').lower() == 'native' and (args.dataset or '').lower() == 'iv2a':
        maybe_set('resample_hz', preset.resample_hz, only_if_current=HARMONIZED_BASELINE.resample_hz)
        maybe_set('keep_channels', preset.keep_channels, only_if_current=','.join(CANON_CHS_18))
        maybe_set('lr', preset.lr, only_if_current=HARMONIZED_BASELINE.lr)
        maybe_set('eegn_pool', preset.eegn_pool, only_if_current=HARMONIZED_BASELINE.eegn_pool)
        maybe_set('tcn_kernel', preset.tcn_kernel, only_if_current=HARMONIZED_BASELINE.tcn_kernel)
        if args.band_lo == HARMONIZED_BASELINE.band_lo and args.band_hi == HARMONIZED_BASELINE.band_hi:
            args.band_lo = 0.0
            args.band_hi = 0.0
            changed['band'] = None

    if changed:
        args.resolved_protocol_preset = changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["iv2a", "openbmi_local", "physionet_local", "cho2017_local", "dreyer2023_local", "lee2019", "physionet"])
    ap.add_argument("--data_root", default="")
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--split_mode", default="random", choices=["random", "session"])
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--task", default="all", choices=["all", "lr", "4class"], help="Dataset task subset. For SSL, this only affects which trials are loaded before unlabeled pretraining.")
    ap.add_argument("--protocol", default="native", choices=["native", "harmonized"], help="native uses dataset-specific task-window presets. harmonized keeps the shared cross-dataset defaults.")

    ap.add_argument("--tmin", type=float, default=0.0)
    ap.add_argument("--tmax", type=float, default=3.0)
    ap.add_argument("--resample_hz", type=float, default=160.0)
    ap.add_argument("--band_lo", type=float, default=8.0)
    ap.add_argument("--band_hi", type=float, default=32.0)

    ap.add_argument("--view_mode", default="ref+aug", choices=["ref", "aug", "ref+aug"])
    ap.add_argument("--view_ref_modes", default="native,car,laplacian,bipolar,gs,median")
    ap.add_argument("--ref_channel", default="Cz")
    ap.add_argument("--laplacian", action="store_true")
    ap.add_argument("--aug_policy", default="light", choices=["none", "light", "aggressive", "legacy"])
    ap.add_argument("--standardize_mode", default="train", choices=["train", "instance", "none"])
    ap.add_argument("--instance_robust", action="store_true")

    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", "--batch_size", dest="batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ssl_loss", default="vicreg", choices=["vicreg", "barlow", "ntxent", "nt_xent"])
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--barlow_lambda", type=float, default=5e-3)
    ap.add_argument("--vicreg_sim", type=float, default=25.0)
    ap.add_argument("--vicreg_std", type=float, default=25.0)
    ap.add_argument("--vicreg_cov", type=float, default=1.0)

    ap.add_argument("--n_channels", type=int, default=len(CANON_CHS_18))
    ap.add_argument("--in_samples", type=int, default=int(3.0 * 160.0))
    ap.add_argument("--n_windows", type=int, default=5)
    ap.add_argument("--attention", default="mha")
    ap.add_argument("--eegn_F1", type=int, default=16)
    ap.add_argument("--eegn_D", type=int, default=2)
    ap.add_argument("--eegn_kernel", type=int, default=64)
    ap.add_argument("--eegn_pool", type=int, default=8)
    ap.add_argument("--eegn_dropout", type=float, default=0.3)
    ap.add_argument("--tcn_depth", type=int, default=2)
    ap.add_argument("--tcn_kernel", type=int, default=8)
    ap.add_argument("--tcn_filters", type=int, default=32)
    ap.add_argument("--tcn_dropout", type=float, default=0.3)
    ap.add_argument("--tcn_activation", default="elu")
    ap.add_argument("--fuse", default="average", choices=["average", "concat", "avg"])

    # Compatibility no-ops for older notebook cells.
    ap.add_argument("--keep_channels", default=",".join(CANON_CHS_18))
    ap.add_argument("--no_ea", action="store_true")

    args = ap.parse_args()
    set_seed(args.seed)

    _maybe_apply_protocol_presets(args)

    if float(args.resample_hz) <= 0.0:
        raise ValueError("resample_hz must be > 0. Use the dataset-native sampling rate instead of 0.")
    args.in_samples = int(round((args.tmax - args.tmin) * args.resample_hz))

    split = SplitSpec(mode=args.split_mode, train_frac=args.train_frac, seed=args.seed)
    ds = get_dataset(args.dataset, data_root=args.data_root or None)
    if getattr(ds, "name", "") == "iv2a" and hasattr(ds, "keep_channels"):
        ds.keep_channels = args.keep_channels

    (X_tr, _y_tr), _ = ds.load_subject_native(
        args.subject,
        split=split,
        tmin=args.tmin,
        tmax=args.tmax,
        resample_hz=args.resample_hz,
        band=_band_arg(args),
        cache_root=args.cache_root or None,
        task=("all" if args.task == "4class" else args.task),
    )

    # Fix common off-by-one epoch length mismatch.
    if X_tr.shape[-1] == args.in_samples + 1:
        X_tr = X_tr[..., : args.in_samples]
    elif X_tr.shape[-1] != args.in_samples:
        if X_tr.shape[-1] > args.in_samples:
            X_tr = X_tr[..., : args.in_samples]
        else:
            pad = args.in_samples - X_tr.shape[-1]
            X_tr = np.pad(X_tr, ((0, 0), (0, 0), (0, pad)), mode="constant")

    if X_tr.shape[1] != args.n_channels:
        args.n_channels = int(X_tr.shape[1])

    view_modes = _split_csv(args.view_ref_modes)
    ref_idx, lap_neighbors = _resolve_ref_params(args, view_modes)
    mu, sd = _prepare_ssl_standardizer(
        args,
        X_tr,
        view_mode=args.view_mode,
        view_modes=view_modes,
        ref_idx=ref_idx,
        lap_neighbors=lap_neighbors,
    )

    ssl_ds = make_ssl_dataset(
        X_tr,
        n_channels=args.n_channels,
        in_samples=args.in_samples,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        deterministic=True,
        view_mode=args.view_mode,
        aug_policy=args.aug_policy,
        ref_modes=view_modes,
        ref_idx=ref_idx,
        lap_neighbors=lap_neighbors,
        standardize_mode=args.standardize_mode,
        mu=mu,
        sd=sd,
        instance_robust=args.instance_robust,
    )

    encoder = _build_encoder(args)
    loss_fn, need_l2norm = _resolve_loss(args)
    ssl_model = build_ssl_projector(encoder, proj_dim=128, out_dim=64, l2norm=need_l2norm)
    opt = Adam(args.lr)

    @tf.function(reduce_retracing=True)
    def train_step(v1, v2):
        with tf.GradientTape() as tape:
            z1 = ssl_model(_to_b1ct(v1), training=True)
            z2 = ssl_model(_to_b1ct(v2), training=True)
            loss = loss_fn(z1, z2)
        grads = tape.gradient(loss, ssl_model.trainable_variables)
        grads_vars = [(g, v) for g, v in zip(grads, ssl_model.trainable_variables) if g is not None]
        opt.apply_gradients(grads_vars)
        return loss

    warm = next(iter(ssl_ds))
    _ = train_step(warm[0], warm[1])

    run_dir = _resolve_subject_out_dir(args.out_dir, args.dataset, args.subject)
    os.makedirs(run_dir, exist_ok=True)
    ssl_weights_path = os.path.join(run_dir, "weights.h5")
    encoder_weights_path = os.path.join(run_dir, "encoder.weights.h5")

    for ep in range(1, args.epochs + 1):
        losses = []
        for v1, v2 in ssl_ds:
            losses.append(float(train_step(v1, v2).numpy()))
        print(f"epoch {ep:03d}/{args.epochs} | ssl_loss={np.mean(losses):.4f}")

    ssl_model.save_weights(ssl_weights_path)
    encoder.save_weights(encoder_weights_path)

    meta = {
        "dataset": args.dataset,
        "subject": args.subject,
        "protocol": args.protocol,
        "resolved_protocol_preset": getattr(args, "resolved_protocol_preset", None),
        "run_dir": os.path.abspath(run_dir),
        "weights": {
            "ssl_model": os.path.abspath(ssl_weights_path),
            "encoder": os.path.abspath(encoder_weights_path),
        },
        "view_mode": args.view_mode,
        "view_ref_modes": view_modes,
        "standardize_mode": args.standardize_mode,
        "instance_robust": bool(args.instance_robust),
        "band": _band_arg(args),
        "keep_channels": args.keep_channels,
        "resample_hz": args.resample_hz,
        "t": [args.tmin, args.tmax],
        "ssl_loss": args.ssl_loss,
        "batch_size": args.batch_size,
    }
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved SSL weights:", ssl_weights_path)
    print("Saved encoder weights:", encoder_weights_path)


if __name__ == "__main__":
    main()
