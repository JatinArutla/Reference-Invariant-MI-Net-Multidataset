"""Multi-dataset reference-mismatch benchmark.

This is the multi-dataset counterpart of gate_reference.py.

Key differences:
  - Works with iv2a (manual), lee2019 (MOABB), physionet (MOABB)
  - Enforces a fixed cross-dataset input contract (CANON18, 0-3s, 8-32Hz, 160Hz)
  - Supports leave-one-reference-family-out training and evaluation

The task is always binary: left vs right.
"""

import os
import argparse
import json
from typing import List, Dict, Tuple

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

tf.keras.backend.set_image_data_format("channels_last")
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, cohen_kappa_score

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from src.models.model import build_atcnet
from src.datamodules.transforms import fit_standardizer, apply_standardizer, apply_reference, standardize_instance
from src.datamodules.channels import CANON_CHS_18, BCI2A_CH_NAMES, neighbors_to_index_list, name_to_index
from src.datamodules.ref_jitter import RefJitterSequence
from src.datamodules.array_sequence import ArraySequence
from src.datamodules.datasets import get_dataset
from src.datamodules.datasets.base import SplitSpec
from src.datamodules.datasets.ref_families import all_ref_modes, train_modes_excluding_family


def set_seed(seed: int = 1):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _reshape_for_model(X: np.ndarray) -> np.ndarray:
    """[N,C,T] -> [N,1,C,T]"""
    N, C, T = X.shape
    return X.reshape(N, 1, C, T).astype(np.float32, copy=False)


def build_model(args) -> tf.keras.Model:
    return build_atcnet(
        n_classes=args.n_classes,
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
        fuse=args.fuse,
        from_logits=args.from_logits,
        return_ssl_feat=False,
    )


def maybe_load_ssl_weights(model: tf.keras.Model, *, ssl_weights: str | None, subject: int):
    if not ssl_weights:
        return
    w = ssl_weights
    if "{sub" in w or "{sub:" in w:
        w = w.format(sub=subject)
    if os.path.isdir(w):
        # Common convention: directory contains weights.h5
        cand = os.path.join(w, "weights.h5")
        if os.path.exists(cand):
            w = cand
    if not os.path.exists(w):
        raise FileNotFoundError(f"ssl_weights not found: {w}")
    # by_name + skip_mismatch so we can load encoder layers even if heads differ.
    model.load_weights(w, by_name=True, skip_mismatch=True)


def _binary_label_frac(X: np.ndarray, y: np.ndarray, frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if frac >= 0.999:
        return X, y
    sss = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=seed)
    idx, _ = next(sss.split(X, y))
    return X[idx], y[idx]


def _compute_lap_neighbors() -> list[list[int]]:
    # Use the repo's neighbor graph projected onto CANON18.
    return neighbors_to_index_list(
        all_names=BCI2A_CH_NAMES,
        keep_names=list(CANON_CHS_18),
        sort_by_distance=True,
    )


def _prepare_ref_args(args):
    # ref_idx for ref-to-channel mode
    ref_idx = None
    if (args.ref_channel or "").strip():
        m = name_to_index(list(CANON_CHS_18))
        if args.ref_channel in m:
            ref_idx = m[args.ref_channel]
    lap_neighbors = None
    if args.need_neighbors:
        lap_neighbors = neighbors_to_index_list(
            all_names=BCI2A_CH_NAMES,
            keep_names=list(CANON_CHS_18),
            sort_by_distance=True,
        )
    return ref_idx, lap_neighbors


def run_one_subject(args, subject: int) -> Dict:
    ds = get_dataset(args.dataset, data_root=args.data_root)
    split = SplitSpec(mode=args.split_mode, train_frac=args.train_frac, seed=args.seed)

    (X_tr0, y_tr0), (X_te0, y_te0) = ds.load_subject_native(
        subject,
        split=split,
        tmin=args.tmin,
        tmax=args.tmax,
        resample_hz=args.resample_hz,
        band=(args.band_lo, args.band_hi),
        cache_root=args.cache_root,
    )

    # Low-label
    X_tr0, y_tr0 = _binary_label_frac(X_tr0, y_tr0, args.label_frac, seed=args.seed)

    # Train/val split within train.
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=args.seed)
    tr_idx, va_idx = next(sss.split(X_tr0, y_tr0))
    X_tr, y_tr = X_tr0[tr_idx], y_tr0[tr_idx]
    X_va, y_va = X_tr0[va_idx], y_tr0[va_idx]

    # Determine which refs to train with.
    if args.holdout_family.lower() not in ("none", ""):
        train_modes = train_modes_excluding_family(args.holdout_family)
    else:
        train_modes = [m.strip() for m in args.train_modes.split(",") if m.strip()] if args.train_modes else [args.train_mode]

    eval_modes = [m.strip() for m in args.eval_modes.split(",") if m.strip()] if args.eval_modes else all_ref_modes()

    # Determine if any mode needs neighbor graph
    need_neighbors = any(m in ("laplacian", "bipolar") for m in (train_modes + eval_modes))
    args.need_neighbors = need_neighbors
    ref_idx = None
    if (args.train_mode or "") in ("ref", "cz_ref", "channel_ref") or any(m in ("ref", "cz_ref", "channel_ref") for m in eval_modes):
        m = name_to_index(list(CANON_CHS_18))
        if args.ref_channel not in m:
            raise ValueError(f"ref_channel '{args.ref_channel}' not in CANON18")
        ref_idx = m[args.ref_channel]
    lap_neighbors = None
    if need_neighbors:
        lap_neighbors = neighbors_to_index_list(
            all_names=BCI2A_CH_NAMES,
            keep_names=list(CANON_CHS_18),
            sort_by_distance=True,
        )

    # Standardization is applied AFTER reference transform.
    # - train   : global z-score using mu/sd fit on training split (leakage-safe)
    # - instance: per-trial, per-channel z-score over time (optionally robust)
    # - none    : no standardization
    def apply_mode(X, mode: str):
        return apply_reference(X, mode=mode, ref_idx=ref_idx, lap_neighbors=lap_neighbors)

    # Build training arrays/sequence
    if args.train_strategy == "fixed":
        X_tr_ref = apply_mode(X_tr, args.train_mode)
        X_va_ref = apply_mode(X_va, args.train_mode)
        if args.standardize_mode == "train":
            mu, sd = fit_standardizer(X_tr_ref)
            X_tr_ref = apply_standardizer(X_tr_ref, mu, sd)
            X_va_ref = apply_standardizer(X_va_ref, mu, sd)
        elif args.standardize_mode == "instance":
            X_tr_ref = standardize_instance(X_tr_ref, robust=args.instance_robust)
            X_va_ref = standardize_instance(X_va_ref, robust=args.instance_robust)
            mu, sd = None, None
        elif args.standardize_mode == "none":
            mu, sd = None, None
        else:
            raise ValueError("standardize_mode must be one of: train, instance, none")
        seq_tr = ArraySequence(_reshape_for_model(X_tr_ref), to_categorical(y_tr, 2), batch_size=args.batch)
        seq_va = ArraySequence(_reshape_for_model(X_va_ref), to_categorical(y_va, 2), batch_size=args.batch, shuffle=False)
    elif args.train_strategy in ("jitter", "concat"):
        # For train-mode standardization, fit mu/sd on a single deterministic reference (native)
        # and keep it fixed. This avoids leaking held-out families into statistics.
        if args.standardize_mode == "train":
            X_tr_nat = apply_mode(X_tr, "native")
            mu, sd = fit_standardizer(X_tr_nat)
        else:
            mu, sd = None, None
        seq_tr = RefJitterSequence(
            X_tr,
            y_tr,
            batch_size=args.batch,
            ref_modes=train_modes,
            ref_channel=args.ref_channel,
            laplacian=need_neighbors,
            keep_channels=",".join(list(CANON_CHS_18)),
            mu=mu,
            sd=sd,
            seed=args.seed,
            strategy=args.train_strategy,
            standardize_mode=args.standardize_mode,
            instance_robust=args.instance_robust,
        )
        X_va_nat = apply_mode(X_va, "native")
        if args.standardize_mode == "train":
            X_va_nat = apply_standardizer(X_va_nat, mu, sd)
        elif args.standardize_mode == "instance":
            X_va_nat = standardize_instance(X_va_nat, robust=args.instance_robust)
        seq_va = ArraySequence(_reshape_for_model(X_va_nat), to_categorical(y_va, 2), batch_size=args.batch, shuffle=False)
    else:
        raise ValueError(f"Unknown train_strategy {args.train_strategy}")

    model = build_model(args)
    maybe_load_ssl_weights(model, ssl_weights=args.ssl_weights, subject=subject)
    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss=CategoricalCrossentropy(from_logits=args.from_logits),
        metrics=[],
    )

    out_dir = os.path.join(args.out_dir, args.dataset, f"sub{subject:03d}")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "best.weights.h5")

    callbacks = [
        ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, args.patience // 2), verbose=0),
    ]
    if args.early_stop:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True, verbose=0))

    model.fit(
        seq_tr,
        validation_data=seq_va,
        epochs=args.epochs,
        verbose=args.verbose,
        callbacks=callbacks,
    )

    # Load weights for evaluation
    if args.eval_weights == "best" and os.path.exists(ckpt_path):
        model.load_weights(ckpt_path)

    # Evaluate across eval modes
    results = {
        "dataset": args.dataset,
        "subject": int(subject),
        "train_strategy": args.train_strategy,
        "train_mode": args.train_mode,
        "train_modes": train_modes,
        "eval_modes": eval_modes,
        "label_frac": float(args.label_frac),
        "holdout_family": args.holdout_family,
        "metrics": {},
    }

    # Standardizer used at eval time.
    # - train   : recompute mu/sd on the training split under a leakage-safe deterministic ref
    # - instance: apply per-trial standardization
    # - none    : no standardization
    if args.standardize_mode == "train":
        if args.train_strategy == "fixed":
            mu, sd = fit_standardizer(apply_mode(X_tr, args.train_mode))
        else:
            mu, sd = fit_standardizer(apply_mode(X_tr, "native"))
    else:
        mu, sd = None, None

    for m in eval_modes:
        X_te_m = apply_mode(X_te0, m)
        if args.standardize_mode == "train":
            X_te_m = apply_standardizer(X_te_m, mu, sd)
        elif args.standardize_mode == "instance":
            X_te_m = standardize_instance(X_te_m, robust=args.instance_robust)
        y_true = y_te0
        y_hat = model.predict(_reshape_for_model(X_te_m), batch_size=args.batch, verbose=0)
        y_pred = np.argmax(y_hat, axis=1)
        acc = float(accuracy_score(y_true, y_pred))
        kappa = float(cohen_kappa_score(y_true, y_pred))
        results["metrics"][m] = {"acc": acc, "kappa": kappa, "n": int(len(y_true))}

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        required=True,
        choices=["iv2a", "openbmi_local", "physionet_local", "lee2019", "physionet"],
        help=(
            "Dataset id. Prefer *_local when you already have files on disk (Kaggle). "
            "MOABB-based ids (lee2019, physionet) may download/cached via the internet."
        ),
    )
    ap.add_argument(
        "--data_root",
        default="",
        help=(
            "Dataset root on disk. Required for iv2a, openbmi_local, physionet_local. "
            "For MOABB datasets, this can be empty if using cache_root."
        ),
    )
    ap.add_argument("--cache_root", default="", help="Optional MOABB cache root")
    ap.add_argument("--out_dir", default="outputs_multi")
    ap.add_argument("--subject", type=int, default=1, help="Single subject id (ignored if --subjects is set)")
    ap.add_argument("--subjects", default="", help="Comma list of subjects to run (e.g. '1,2,3')")
    ap.add_argument("--split_mode", default="random", choices=["random", "session"])
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--label_frac", type=float, default=1.0)

    # Preprocessing contract
    ap.add_argument("--tmin", type=float, default=0.0)
    ap.add_argument("--tmax", type=float, default=3.0)
    ap.add_argument("--resample_hz", type=float, default=160.0)
    ap.add_argument("--band_lo", type=float, default=8.0)
    ap.add_argument("--band_hi", type=float, default=32.0)

    # Reference training/eval
    ap.add_argument("--train_strategy", default="fixed", choices=["fixed", "jitter", "concat"])
    ap.add_argument("--train_mode", default="native")
    ap.add_argument("--train_modes", default="", help="Comma list (overrides train_mode for jitter/concat)")
    ap.add_argument("--eval_modes", default="", help="Comma list, default=all")
    ap.add_argument("--holdout_family", default="none", choices=["none", "native", "global", "local"])
    ap.add_argument("--ref_channel", default="Cz")

    ap.add_argument("--standardize_mode", default="train", choices=["train", "instance", "none"], help="Standardization applied after referencing")
    ap.add_argument("--instance_robust", action="store_true", help="Use median/MAD for instance standardization")
    ap.add_argument("--eval_weights", default="best", choices=["best", "last"], help="Evaluate using best checkpoint or last epoch weights")

    # Model/training
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--early_stop", action="store_true")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--verbose", type=int, default=1)

    ap.add_argument(
        "--ssl_weights",
        default=None,
        help=(
            "Optional SSL init weights. Either a file path or a template containing '{sub:02d}'. "
            "Loaded with by_name=True, skip_mismatch=True."
        ),
    )

    # ATCNet params (kept consistent with existing repo)
    ap.add_argument("--n_classes", type=int, default=2)
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
    ap.add_argument("--fuse", default="avg")
    ap.add_argument("--from_logits", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)

    # Sanity: match in_samples to (tmax-tmin)*resample_hz
    expected = int(round((args.tmax - args.tmin) * args.resample_hz))
    args.in_samples = expected

    # Run one or many subjects
    subs = []
    if args.subjects.strip():
        subs = [int(s) for s in args.subjects.split(",") if s.strip()]
    else:
        subs = [int(args.subject)]

    all_res = []
    for sub in subs:
        r = run_one_subject(args, sub)
        all_res.append(r)
        print(json.dumps(r, indent=2))

    # Lightweight aggregate (mean acc per eval mode across subjects)
    if len(all_res) > 1:
        agg = {"dataset": args.dataset, "subjects": subs, "mean_acc": {}}
        modes = all_res[0]["metrics"].keys()
        for m in modes:
            agg["mean_acc"][m] = float(np.mean([x["metrics"][m]["acc"] for x in all_res]))
        out_dir = os.path.join(args.out_dir, args.dataset)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "aggregate_mean_acc.json"), "w") as f:
            json.dump(agg, f, indent=2)


if __name__ == "__main__":
    main()
