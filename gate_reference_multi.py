"""Multi-dataset reference-mismatch benchmark.

This version fixes the main issues in the original multi-dataset runner:
  - jitter/concat now use correct one-hot labels
  - concat is a real concatenation baseline, not jitter in disguise
  - LOFO family validation is leakage-safe by default
  - SSL weights paths are resolved more robustly
  - splits, best/last weights, and training metadata are saved per run
  - old notebook CLI flags are accepted for compatibility
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from src.models.model import build_atcnet
from src.datamodules.transforms import (
    apply_reference,
    apply_standardizer,
    fit_standardizer,
    standardize_instance,
)
from src.datamodules.channels import CANON_CHS_18, BCI2A_CH_NAMES, neighbors_to_index_list, name_to_index
from src.datamodules.ref_jitter import RefJitterSequence
from src.datamodules.array_sequence import ArraySequence
from src.datamodules.datasets import get_dataset
from src.datamodules.datasets.base import SplitSpec
from src.datamodules.datasets.ref_families import all_ref_modes, train_modes_excluding_family


tf.keras.backend.set_image_data_format("channels_last")
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


def set_seed(seed: int = 1):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _reshape_for_model(X: np.ndarray) -> np.ndarray:
    n, c, t = X.shape
    return X.reshape(n, 1, c, t).astype(np.float32, copy=False)


def _pretty_table_row(label: str, values: list[float]) -> str:
    parts = [f"{v:>10.2f}" for v in values]
    return f"{label:<10s} " + " ".join(parts)


def print_avg_table(title: str, modes: list[str], row_name: str, accs: list[float]):
    print("\n" + title)
    header = "train\\test".ljust(10) + " " + " ".join([f"{m:>10s}" for m in modes])
    print("\nAveraged accuracy (%):")
    print(header)
    print("-" * len(header))
    print(_pretty_table_row(row_name, [a * 100.0 for a in accs]))


def build_model(args) -> tf.keras.Model:
    fuse = args.fuse
    if fuse == "avg":
        fuse = "average"
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
        fuse=fuse,
        from_logits=args.from_logits,
        return_ssl_feat=False,
    )


def _path_parts(path_str: str) -> list[str]:
    return list(Path(path_str).parts)


def _resolve_subject_out_dir(base_out_dir: str, dataset: str, subject: int) -> str:
    subj_token = f"sub{subject:03d}"
    parts = _path_parts(base_out_dir)
    if subj_token in parts:
        return str(Path(base_out_dir))
    if dataset in parts:
        return str(Path(base_out_dir) / subj_token)
    return str(Path(base_out_dir) / dataset / subj_token)


def _resolve_aggregate_out_dir(base_out_dir: str, dataset: str, subjects: list[int]) -> str:
    parts = _path_parts(base_out_dir)
    if any(p.startswith("sub") for p in parts):
        return str(Path(base_out_dir).parent)
    if dataset in parts:
        return str(Path(base_out_dir))
    return str(Path(base_out_dir) / dataset)


def _find_weights_in_dir(d: str) -> str | None:
    root = Path(d)
    preferred = [
        root / "encoder.weights.h5",
        root / "weights.h5",
        root / "best.weights.h5",
        root / "last.weights.h5",
    ]
    for p in preferred:
        if p.exists():
            return str(p)
    recursive = []
    for pat in ("encoder.weights.h5", "weights.h5", "best.weights.h5", "last.weights.h5"):
        recursive.extend(root.rglob(pat))
    if not recursive:
        return None
    recursive = sorted(set(recursive), key=lambda p: (len(p.parts), str(p)))
    return str(recursive[0])


def maybe_load_ssl_weights(model: tf.keras.Model, *, ssl_weights: str | None, subject: int):
    if not ssl_weights:
        return None
    w = ssl_weights
    if "{sub" in w or "{sub:" in w:
        w = w.format(sub=subject)
    if os.path.isdir(w):
        cand = _find_weights_in_dir(w)
        if cand is None:
            raise FileNotFoundError(f"No weights file found under ssl_weights dir: {w}")
        w = cand
    if not os.path.exists(w):
        raise FileNotFoundError(f"ssl_weights not found: {w}")
    model.load_weights(w, by_name=True, skip_mismatch=True)
    return w


def _label_frac(X: np.ndarray, y: np.ndarray, frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if frac >= 0.999:
        idx = np.arange(len(y), dtype=np.int64)
        return X, y, idx
    sss = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=seed)
    idx, _ = next(sss.split(X, y))
    idx = np.asarray(idx, dtype=np.int64)
    return X[idx], y[idx], idx


def _class_counts(y: np.ndarray) -> dict:
    u, c = np.unique(y, return_counts=True)
    return {int(uu): int(cc) for uu, cc in zip(u, c)}

def _per_class_metric_dict(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {int(i): float(v) for i, v in enumerate(values.tolist())}

def _compute_eval_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    labels = np.arange(n_classes, dtype=np.int64)
    per_class_recall = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "n": int(len(y_true)),
        "class_counts_true": _class_counts(y_true),
        "class_counts_pred": _class_counts(y_pred),
        "per_class_recall": _per_class_metric_dict(per_class_recall),
        "confusion_matrix": cm.astype(int).tolist(),
    }

def _fix_T(X: np.ndarray, target_T: int) -> np.ndarray:
    if X.ndim != 3:
        raise ValueError(f"Expected [N,C,T], got shape {X.shape}")
    T = int(X.shape[-1])
    if T == target_T:
        return X
    if T == target_T + 1:
        return X[..., :target_T]
    if T > target_T:
        return X[..., :target_T]
    pad = target_T - T
    return np.pad(X, ((0, 0), (0, 0), (0, pad)), mode="constant")


def _resolve_train_modes(args) -> list[str]:
    if args.holdout_family.lower() not in ("none", ""):
        return train_modes_excluding_family(args.holdout_family)
    if args.train_strategy == "fixed":
        return [args.train_mode]
    if args.train_modes:
        modes = [m.strip() for m in args.train_modes.split(",") if m.strip()]
        if modes:
            return modes
    return [args.train_mode]


def _resolve_eval_modes(args) -> list[str]:
    return [m.strip() for m in args.eval_modes.split(",") if m.strip()] if args.eval_modes else all_ref_modes()


def _resolve_val_modes(args, train_modes: list[str], eval_modes: list[str]) -> tuple[list[str], str]:
    user_val_modes = [m.strip() for m in (args.val_modes or "").split(",") if m.strip()]
    if user_val_modes:
        val_modes = user_val_modes
    elif args.train_strategy == "fixed":
        val_modes = [args.train_mode]
    elif args.holdout_family.lower() not in ("none", ""):
        val_modes = list(train_modes)
    else:
        val_modes = list(train_modes if train_modes else eval_modes)

    val_single_mode = (args.val_single_mode or "auto").strip().lower()
    if val_single_mode == "auto":
        if args.train_strategy == "fixed":
            val_single_mode = args.train_mode
        elif "native" in val_modes:
            val_single_mode = "native"
        else:
            val_single_mode = val_modes[0]
    if val_single_mode not in val_modes:
        val_modes = [val_single_mode] + [m for m in val_modes if m != val_single_mode]
    return val_modes, val_single_mode


def _resolve_ref_params(args, modes: list[str]):
    need_ref = any((m or "").lower() in ("ref", "cz_ref", "channel_ref") for m in modes)
    ref_idx = None
    if need_ref:
        m = name_to_index(list(CANON_CHS_18))
        if args.ref_channel not in m:
            raise ValueError(f"ref_channel '{args.ref_channel}' not in CANON_CHS_18")
        ref_idx = m[args.ref_channel]

    need_neighbors = args.laplacian or any((m or "").lower() in ("laplacian", "lap", "local", "bipolar", "bip", "bipolar_like") for m in modes)
    lap_neighbors = None
    if need_neighbors:
        lap_neighbors = neighbors_to_index_list(
            all_names=BCI2A_CH_NAMES,
            keep_names=list(CANON_CHS_18),
            sort_by_distance=True,
        )
    return ref_idx, lap_neighbors, need_neighbors


class ValRefMeanAccFromInputs(Callback):
    def __init__(self, val_inputs_by_mode: Dict[str, np.ndarray], y_val_onehot: np.ndarray, from_logits: bool, metric_name: str = "val_refmean_acc"):
        super().__init__()
        self.val_inputs_by_mode = val_inputs_by_mode
        self.y_true = y_val_onehot.argmax(-1).astype(int)
        self.from_logits = bool(from_logits)
        self.metric_name = metric_name

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        accs = []
        for Xm in self.val_inputs_by_mode.values():
            y_pred = self.model.predict(Xm, verbose=0)
            if self.from_logits:
                y_hat = tf.nn.softmax(y_pred).numpy().argmax(-1)
            else:
                y_hat = y_pred.argmax(-1)
            accs.append((y_hat.astype(int) == self.y_true).mean())
        logs[self.metric_name] = float(np.mean(accs)) if accs else 0.0


def _apply_standardization(X: np.ndarray, *, standardize_mode: str, mu=None, sd=None, instance_robust: bool = False) -> np.ndarray:
    if standardize_mode == "train":
        if mu is None or sd is None:
            raise ValueError("mu and sd must be provided for standardize_mode='train'")
        return apply_standardizer(X, mu, sd)
    if standardize_mode == "instance":
        return standardize_instance(X, robust=instance_robust)
    if standardize_mode == "none":
        return X.astype(np.float32, copy=False)
    raise ValueError("standardize_mode must be one of: train, instance, none")


def _train_model(args, *, seq_tr, seq_va, subject: int, run_dir: str, monitor: str, extra_callbacks: list[Callback] | None = None):
    os.makedirs(run_dir, exist_ok=True)
    model = build_model(args)
    ssl_loaded = maybe_load_ssl_weights(model, ssl_weights=args.ssl_weights, subject=subject)
    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss=CategoricalCrossentropy(from_logits=args.from_logits),
        metrics=["accuracy"],
    )

    ckpt_path = os.path.join(run_dir, "best.weights.h5")
    last_path = os.path.join(run_dir, "last.weights.h5")
    callbacks: list[Callback] = []
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    monitor_mode = "min" if monitor.endswith("loss") else "max"
    callbacks.extend([
        ModelCheckpoint(ckpt_path, monitor=monitor, mode=monitor_mode, save_best_only=True, save_weights_only=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, args.patience // 2), verbose=0),
    ])
    if args.early_stop:
        callbacks.append(EarlyStopping(monitor=monitor, patience=args.patience, mode=monitor_mode, restore_best_weights=False, verbose=0))

    model.fit(
        seq_tr,
        validation_data=seq_va,
        epochs=args.epochs,
        verbose=args.verbose,
        callbacks=callbacks,
    )
    model.save_weights(last_path)

    chosen = ckpt_path if args.eval_weights == "best" and os.path.exists(ckpt_path) else last_path
    if os.path.exists(chosen):
        model.load_weights(chosen)

    with open(os.path.join(run_dir, "weights_meta.json"), "w") as f:
        json.dump(
            {
                "best_weights": os.path.abspath(ckpt_path),
                "last_weights": os.path.abspath(last_path),
                "eval_weights": os.path.abspath(chosen),
                "ssl_loaded": ssl_loaded,
                "monitor": monitor,
                "monitor_mode": monitor_mode,
            },
            f,
            indent=2,
        )

    return model, chosen, ssl_loaded


def run_one_subject(args, subject: int) -> Dict:
    ds = get_dataset(args.dataset, data_root=args.data_root or None)
    split = SplitSpec(mode=args.split_mode, train_frac=args.train_frac, seed=args.seed)
    (X_tr0, y_tr0), (X_te0, y_te0) = ds.load_subject_native(
        subject,
        split=split,
        tmin=args.tmin,
        tmax=args.tmax,
        resample_hz=args.resample_hz,
        band=(args.band_lo, args.band_hi),
        cache_root=args.cache_root or None,
        task=args.task,
    )

    args.n_channels = int(X_tr0.shape[1])
    class_ids = sorted(set(np.unique(np.concatenate([y_tr0, y_te0])).tolist()))
    class_remap = {old: new for new, old in enumerate(class_ids)}
    if any(class_remap[k] != k for k in class_remap):
        y_tr0 = np.array([class_remap[int(v)] for v in y_tr0], dtype=np.int64)
        y_te0 = np.array([class_remap[int(v)] for v in y_te0], dtype=np.int64)
    args.n_classes = int(len(class_ids))
    fs_eff = float(args.resample_hz)
    target_T = int(round((float(args.tmax) - float(args.tmin)) * fs_eff))
    if target_T > 0:
        X_tr0 = _fix_T(X_tr0, target_T)
        X_te0 = _fix_T(X_te0, target_T)
        args.in_samples = target_T

    X_pool, y_pool, labeled_pool_idx = _label_frac(X_tr0, y_tr0, args.label_frac, seed=args.seed)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=args.seed)
    tr_rel, va_rel = next(sss.split(X_pool, y_pool))
    tr_idx = labeled_pool_idx[tr_rel]
    va_idx = labeled_pool_idx[va_rel]
    X_tr, y_tr = X_pool[tr_rel], y_pool[tr_rel]
    X_va, y_va = X_pool[va_rel], y_pool[va_rel]

    y_tr_oh = to_categorical(y_tr, args.n_classes)
    y_va_oh = to_categorical(y_va, args.n_classes)

    train_modes = _resolve_train_modes(args)
    eval_modes = _resolve_eval_modes(args)
    val_modes, val_single_mode = _resolve_val_modes(args, train_modes, eval_modes)
    ref_idx, lap_neighbors, need_neighbors = _resolve_ref_params(args, list({*train_modes, *eval_modes, *val_modes, val_single_mode}))

    def apply_mode(X: np.ndarray, mode: str) -> np.ndarray:
        return apply_reference(X, mode=mode, ref_idx=ref_idx, lap_neighbors=lap_neighbors)

    run_dir = _resolve_subject_out_dir(args.out_dir, args.dataset, subject)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "splits.json"), "w") as f:
        json.dump(
            {
                "subject": int(subject),
                "label_frac": float(args.label_frac),
                "labeled_pool_idx": labeled_pool_idx.tolist(),
                "train_idx": tr_idx.tolist(),
                "val_idx": va_idx.tolist(),
            },
            f,
            indent=2,
        )

    mu = sd = None
    extra_callbacks = None
    monitor = "val_accuracy"

    if args.train_strategy == "fixed":
        X_tr_raw = apply_mode(X_tr, args.train_mode)
        X_va_raw = apply_mode(X_va, args.train_mode)
        if args.standardize_mode == "train":
            mu, sd = fit_standardizer(X_tr_raw)
        X_tr_fit = _apply_standardization(X_tr_raw, standardize_mode=args.standardize_mode, mu=mu, sd=sd, instance_robust=args.instance_robust)
        X_va_fit = _apply_standardization(X_va_raw, standardize_mode=args.standardize_mode, mu=mu, sd=sd, instance_robust=args.instance_robust)

        seq_seed = int(args.seed) + int(subject) * 1000
        seq_tr = ArraySequence(_reshape_for_model(X_tr_fit), y_tr_oh, batch_size=args.batch, shuffle=True, seed=seq_seed)
        seq_va = ArraySequence(_reshape_for_model(X_va_fit), y_va_oh, batch_size=args.batch, shuffle=False, seed=seq_seed)

    elif args.train_strategy == "jitter":
        if args.standardize_mode == "train":
            X_stats = np.concatenate([apply_mode(X_tr, m) for m in train_modes], axis=0)
            mu, sd = fit_standardizer(X_stats)
        seq_tr = RefJitterSequence(
            X_tr,
            y_tr_oh,
            batch_size=args.batch,
            ref_modes=train_modes,
            ref_channel=args.ref_channel,
            laplacian=need_neighbors,
            keep_channels=",".join(list(CANON_CHS_18)),
            mu=mu,
            sd=sd,
            seed=args.seed,
            standardize_mode=args.standardize_mode,
            instance_robust=args.instance_robust,
        )
        X_va_single = apply_mode(X_va, val_single_mode)
        X_va_fit = _apply_standardization(X_va_single, standardize_mode=args.standardize_mode, mu=mu, sd=sd, instance_robust=args.instance_robust)
        val_inputs_by_mode = {
            m: _reshape_for_model(_apply_standardization(apply_mode(X_va, m), standardize_mode=args.standardize_mode, mu=mu, sd=sd, instance_robust=args.instance_robust))
            for m in val_modes
        }
        seq_seed = int(args.seed) + int(subject) * 1000
        seq_va = ArraySequence(_reshape_for_model(X_va_fit), y_va_oh, batch_size=args.batch, shuffle=False, seed=seq_seed)
        extra_callbacks = [ValRefMeanAccFromInputs(val_inputs_by_mode, y_va_oh, args.from_logits)]
        monitor = "val_refmean_acc"

    elif args.train_strategy == "concat":
        X_tr_concat_raw = np.concatenate([apply_mode(X_tr, m) for m in train_modes], axis=0)
        y_tr_concat = np.concatenate([y_tr_oh for _ in train_modes], axis=0)
        if args.standardize_mode == "train":
            mu, sd = fit_standardizer(X_tr_concat_raw)
        X_tr_fit = _apply_standardization(X_tr_concat_raw, standardize_mode=args.standardize_mode, mu=mu, sd=sd, instance_robust=args.instance_robust)

        X_va_single = apply_mode(X_va, val_single_mode)
        X_va_fit = _apply_standardization(X_va_single, standardize_mode=args.standardize_mode, mu=mu, sd=sd, instance_robust=args.instance_robust)
        val_inputs_by_mode = {
            m: _reshape_for_model(_apply_standardization(apply_mode(X_va, m), standardize_mode=args.standardize_mode, mu=mu, sd=sd, instance_robust=args.instance_robust))
            for m in val_modes
        }

        seq_seed = int(args.seed) + int(subject) * 1000
        seq_tr = ArraySequence(_reshape_for_model(X_tr_fit), y_tr_concat, batch_size=args.batch, shuffle=True, seed=seq_seed)
        seq_va = ArraySequence(_reshape_for_model(X_va_fit), y_va_oh, batch_size=args.batch, shuffle=False, seed=seq_seed)
        extra_callbacks = [ValRefMeanAccFromInputs(val_inputs_by_mode, y_va_oh, args.from_logits)]
        monitor = "val_refmean_acc"
    else:
        raise ValueError(f"Unknown train_strategy {args.train_strategy}")

    model, chosen_weights, ssl_loaded = _train_model(
        args,
        seq_tr=seq_tr,
        seq_va=seq_va,
        subject=subject,
        run_dir=run_dir,
        monitor=monitor,
        extra_callbacks=extra_callbacks,
    )

    results = {
        "dataset": args.dataset,
        "subject": int(subject),
        "task": args.task,
        "train_strategy": args.train_strategy,
        "train_mode": args.train_mode,
        "train_modes": train_modes,
        "eval_modes": eval_modes,
        "val_modes": val_modes,
        "val_single_mode": val_single_mode,
        "label_frac": float(args.label_frac),
        "holdout_family": args.holdout_family,
        "standardize_mode": args.standardize_mode,
        "ssl_loaded": ssl_loaded,
        "weights_used_for_eval": chosen_weights,
        "n_classes": int(args.n_classes),
        "data": {
            "n_train": int(len(y_tr)),
            "n_val": int(len(y_va)),
            "n_test": int(len(y_te0)),
            "class_counts_train": _class_counts(y_tr),
            "class_counts_val": _class_counts(y_va),
            "class_counts_test": _class_counts(y_te0),
        },
        "metrics": {},
    }

    for m in eval_modes:
        X_te_m = _apply_standardization(
            apply_mode(X_te0, m),
            standardize_mode=args.standardize_mode,
            mu=mu,
            sd=sd,
            instance_robust=args.instance_robust,
        )
        y_hat = model.predict(_reshape_for_model(X_te_m), batch_size=args.batch, verbose=0)
        y_pred = tf.nn.softmax(y_hat).numpy().argmax(axis=1) if args.from_logits else np.argmax(y_hat, axis=1)

        metrics_m = _compute_eval_metrics(y_te0, y_pred, args.n_classes)
        results["metrics"][m] = metrics_m

    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    tf.keras.backend.clear_session()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["iv2a", "openbmi_local", "physionet_local", "cho2017_local", "dreyer2023_local", "lee2019", "physionet"])
    ap.add_argument("--data_root", default="")
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--out_dir", default="outputs_multi")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--subjects", default="")
    ap.add_argument("--split_mode", default="random", choices=["random", "session"])
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--label_frac", type=float, default=1.0)

    ap.add_argument("--task", default="all", choices=["all", "lr", "4class"], help="Task definition. all=dataset-native classes in this repo, lr=left-vs-right subset, 4class=iv2a alias for backward compatibility.")

    ap.add_argument("--tmin", type=float, default=0.0)
    ap.add_argument("--tmax", type=float, default=3.0)
    ap.add_argument("--resample_hz", type=float, default=160.0)
    ap.add_argument("--band_lo", type=float, default=8.0)
    ap.add_argument("--band_hi", type=float, default=32.0)

    ap.add_argument("--train_strategy", default="fixed", choices=["fixed", "jitter", "concat"])
    ap.add_argument("--train_mode", default="native")
    ap.add_argument("--train_modes", default="")
    ap.add_argument("--eval_modes", default="")
    ap.add_argument("--holdout_family", default="none", choices=["none", "native", "global", "local"])
    ap.add_argument("--ref_channel", default="Cz")
    ap.add_argument("--val_modes", default="", help="Validation reference modes for checkpoint selection. Empty -> safe defaults.")
    ap.add_argument("--val_single_mode", default="auto", help="Single seen validation mode used for val_loss. Default auto.")

    ap.add_argument("--standardize_mode", default="train", choices=["train", "instance", "none"])
    ap.add_argument("--instance_robust", action="store_true")
    ap.add_argument("--eval_weights", default="best", choices=["best", "last"])

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", "--batch_size", dest="batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--early_stop", action="store_true")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--verbose", type=int, default=1)
    ap.add_argument("--print_json", action="store_true")
    ap.add_argument("--ssl_weights", default=None)

    # Compatibility args from older notebook cells / single-dataset runner.
    ap.add_argument("--keep_channels", default=",".join(CANON_CHS_18))
    ap.add_argument("--laplacian", action="store_true")
    ap.add_argument("--no_ea", action="store_true")

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
    ap.add_argument("--fuse", default="average", choices=["average", "concat", "avg"])
    ap.add_argument("--from_logits", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)

    if args.task == "4class":
        if args.dataset != "iv2a":
            raise ValueError("--task 4class is only supported for --dataset iv2a in this repo")
        args.task = "all"
    elif args.task not in ("all", "lr"):
        raise ValueError(args.task)

    args.in_samples = int(round((args.tmax - args.tmin) * args.resample_hz))

    if args.subjects.strip():
        subs = [int(s) for s in args.subjects.split(",") if s.strip()]
    else:
        subs = [int(args.subject)]

    all_res = []
    for sub in subs:
        r = run_one_subject(args, sub)
        all_res.append(r)
        if args.print_json:
            print(json.dumps(r, indent=2))
        else:
            native_metrics = r["metrics"].get("native", None)
            if native_metrics is None:
                native_metrics = list(r["metrics"].values())[0]

            print(
                f"done: dataset={args.dataset} task={args.task} subject={sub} "
                f"train={args.train_strategy}:{args.train_mode} "
                f"native_acc={native_metrics['acc']*100.0:.2f}% "
                f"native_bal_acc={native_metrics['bal_acc']*100.0:.2f}% "
                f"native_macro_f1={native_metrics['macro_f1']*100.0:.2f}%"
            )

    if all_res:
        modes = list(all_res[0]["metrics"].keys())
        mean_acc = [float(np.mean([x["metrics"][m]["acc"] for x in all_res])) for m in modes]
        mean_bal_acc = [float(np.mean([x["metrics"][m]["bal_acc"] for x in all_res])) for m in modes]
        mean_macro_f1 = [float(np.mean([x["metrics"][m]["macro_f1"] for x in all_res])) for m in modes]
        row = args.train_strategy if args.train_strategy != "fixed" else args.train_mode
        title = f"Dataset={args.dataset} | task={args.task} | subjects={len(all_res)} | train={args.train_strategy}"
        print_avg_table(title, modes, row, mean_acc)

        agg = {
            "dataset": args.dataset,
            "task": args.task,
            "subjects": subs,
            "train_strategy": args.train_strategy,
            "train_mode": args.train_mode,
            "train_modes": all_res[0].get("train_modes", []),
            "eval_modes": modes,
            "label_frac": float(args.label_frac),
            "holdout_family": args.holdout_family,
            "mean_acc": {m: a for m, a in zip(modes, mean_acc)},
            "mean_bal_acc": {m: a for m, a in zip(modes, mean_bal_acc)},
            "mean_macro_f1": {m: a for m, a in zip(modes, mean_macro_f1)},
        }
        agg_dir = _resolve_aggregate_out_dir(args.out_dir, args.dataset, subs)
        os.makedirs(agg_dir, exist_ok=True)
        with open(os.path.join(agg_dir, "aggregate_mean_acc.json"), "w") as f:
            json.dump(agg, f, indent=2)


if __name__ == "__main__":
    main()
