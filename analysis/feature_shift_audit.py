#!/usr/bin/env python
"""Feature shift audit for reference transforms.

Runs on the data only. No model needed.

Goal: show what re-referencing changes in the dataset using simple, mechanistic statistics:
- bandpower (delta/theta/alpha/beta)
- 1/f-like PSD slope in 4-30 Hz
- temporal gradient energy
- common-mode ratio (energy of channel-mean / total energy)
- mean absolute inter-channel correlation

Important:
- For a "mechanistic" audit, you usually want --standardize_mode none.
  Standardization can erase or distort amplitude and spectral shifts.

Outputs in out_dir:
- meta.json
- summary.csv  (one row per subject x mode with feature means)
- per_trial_features_subXX_modeYY.npz  (features per trial, labels)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from src.datamodules.channels import (
    BCI2A_FS,
    BCI2A_CH_NAMES,
    parse_keep_channels,
    neighbors_to_index_list,
    name_to_index,
)
from src.datamodules.bci2a import load_bci2a_session
from src.datamodules.transforms import (
    apply_reference,
    fit_standardizer,
    apply_standardizer,
    standardize_instance,
)


def _parse_list(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def _parse_subjects(s: str, n_sub: int) -> List[int]:
    s2 = (s or "").strip().lower()
    if s2 in ("all", "*"):
        return list(range(1, int(n_sub) + 1))
    out: List[int] = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    if not out:
        raise ValueError("--subjects must be non-empty (or use 'all')")
    return out


def _ensure_dir(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d


def _periodogram_psd(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Simple periodogram PSD."""
    x = np.asarray(x, dtype=np.float64)
    x = x - float(np.mean(x))
    n = int(x.shape[0])
    if n < 8:
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        return freqs, np.zeros_like(freqs, dtype=np.float64)

    win = np.hanning(n)
    xw = x * win
    xf = np.fft.rfft(xw, n=n)
    psd = (np.abs(xf) ** 2) / (fs * np.sum(win ** 2) + 1e-12)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, psd


def _bandpowers_trial(
    X_ct: np.ndarray, fs: float, bands: List[Tuple[str, float, float]], eps: float = 1e-12
) -> Dict[str, float]:
    C, _ = X_ct.shape
    band_sums = {name: 0.0 for (name, _, _) in bands}

    for c in range(C):
        freqs, psd = _periodogram_psd(X_ct[c], fs)
        if freqs.size < 2:
            continue
        df = float(freqs[1] - freqs[0])
        for name, lo, hi in bands:
            m = (freqs >= lo) & (freqs < hi)
            p = float(np.sum(psd[m]) * df)
            band_sums[name] += np.log10(p + eps)

    for k in band_sums:
        band_sums[k] /= max(C, 1)
    return band_sums


def _psd_slope_trial(X_ct: np.ndarray, fs: float, f_lo: float = 4.0, f_hi: float = 30.0, eps: float = 1e-12) -> float:
    C, _ = X_ct.shape
    psd_sum = None
    freqs = None
    for c in range(C):
        freqs, psd = _periodogram_psd(X_ct[c], fs)
        psd_sum = psd if psd_sum is None else (psd_sum + psd)
    if psd_sum is None or freqs is None:
        return float("nan")
    psd_mean = psd_sum / max(C, 1)

    m = (freqs >= f_lo) & (freqs <= f_hi)
    f = freqs[m]
    p = psd_mean[m]
    if f.size < 5:
        return float("nan")
    x = np.log10(f + eps)
    y = np.log10(p + eps)
    a, _ = np.polyfit(x, y, deg=1)
    return float(a)


def _temporal_grad_energy_trial(X_ct: np.ndarray) -> float:
    d = np.diff(X_ct, axis=1)
    return float(np.mean(d * d))


def _common_mode_ratio_trial(X_ct: np.ndarray, eps: float = 1e-12) -> float:
    cm = np.mean(X_ct, axis=0, keepdims=False)
    num = float(np.sum(cm * cm))
    den = float(np.sum(X_ct * X_ct)) + eps
    return num / den


def _mean_abs_corr_trial(X_ct: np.ndarray, eps: float = 1e-12) -> float:
    X = X_ct.astype(np.float64, copy=False)
    X = X - X.mean(axis=1, keepdims=True)
    denom = np.sqrt(np.sum(X * X, axis=1, keepdims=True)) + eps
    Xn = X / denom
    corr = Xn @ Xn.T
    C = corr.shape[0]
    if C <= 1:
        return 0.0
    mask = ~np.eye(C, dtype=bool)
    return float(np.mean(np.abs(corr[mask])))


def _extract_features(
    X: np.ndarray,
    y: np.ndarray,
    fs: float,
    bands: List[Tuple[str, float, float]],
    max_trials: int | None = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    N = int(X.shape[0])
    idx = np.arange(N)
    if max_trials is not None and N > int(max_trials):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(idx, size=int(max_trials), replace=False)
        idx = np.sort(idx)

    Xs = X[idx]
    ys = y[idx]

    names: List[str] = []
    for name, _, _ in bands:
        names.append(f"log10_bandpower_{name}")
    names += [
        "psd_slope_4_30",
        "temporal_grad_energy",
        "common_mode_ratio",
        "mean_abs_corr",
    ]

    F = len(names)
    feats = np.zeros((Xs.shape[0], F), dtype=np.float32)

    for i in range(Xs.shape[0]):
        trial = Xs[i]
        bp = _bandpowers_trial(trial, fs, bands)
        col = 0
        for name, _, _ in bands:
            feats[i, col] = float(bp[name])
            col += 1
        feats[i, col] = float(_psd_slope_trial(trial, fs)); col += 1
        feats[i, col] = float(_temporal_grad_energy_trial(trial)); col += 1
        feats[i, col] = float(_common_mode_ratio_trial(trial)); col += 1
        feats[i, col] = float(_mean_abs_corr_trial(trial)); col += 1

    return feats, ys, names


def _apply_standardize(mode: str, X: np.ndarray, mu_sd: Tuple[np.ndarray, np.ndarray] | None, instance_robust: bool) -> np.ndarray:
    mode = (mode or "none").lower()
    if mode == "none":
        return X.astype(np.float32, copy=False)
    if mode == "train":
        if mu_sd is None:
            raise ValueError("standardize_mode=train but mu/sd is None")
        return apply_standardizer(X, *mu_sd).astype(np.float32, copy=False)
    if mode == "instance":
        return standardize_instance(X, robust=bool(instance_robust)).astype(np.float32, copy=False)
    if mode == "robust_instance":
        return standardize_instance(X, robust=True).astype(np.float32, copy=False)
    raise ValueError(f"Unknown standardize_mode: {mode}")


def main():
    p = argparse.ArgumentParser("Feature shift audit for reference transforms")

    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--subjects", type=str, default="1", help="Comma-separated subject ids or 'all'")
    p.add_argument("--n_sub", type=int, default=9)

    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--ref_modes", type=str, required=True, help="Comma-separated modes to evaluate")
    p.add_argument("--keep_channels", type=str, default="", help="Preset name or comma-separated channel names")
    p.add_argument("--ref_channel", type=str, default="Cz")

    p.add_argument("--standardize_mode", type=str, default="none", choices=["none", "train", "instance", "robust_instance"])
    p.add_argument("--standardize_fit_mode", type=str, default="native", help="Ref mode used to fit mu/sd when standardize_mode=train")
    p.add_argument("--max_trials", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--instance_robust", action="store_true")

    p.add_argument("--out_dir", type=str, required=True)

    args = p.parse_args()

    out_dir = _ensure_dir(args.out_dir)

    subjects = _parse_subjects(args.subjects, args.n_sub)
    ref_modes = _parse_list(args.ref_modes)
    if not ref_modes:
        raise ValueError("--ref_modes must be non-empty")

    keep_idx = parse_keep_channels(args.keep_channels, all_names=BCI2A_CH_NAMES)
    keep_names = [BCI2A_CH_NAMES[i] for i in keep_idx] if keep_idx is not None else list(BCI2A_CH_NAMES)

    lap_neighbors = neighbors_to_index_list(
        all_names=BCI2A_CH_NAMES,
        keep_names=keep_names,
        sort_by_distance=False,
    )
    ref_map = name_to_index(keep_names)
    ref_idx = ref_map.get(args.ref_channel, None)

    bands = [
        ("delta", 1.0, 4.0),
        ("theta", 4.0, 8.0),
        ("alpha", 8.0, 13.0),
        ("beta", 13.0, 30.0),
    ]

    summary_rows: List[Dict[str, object]] = []
    feature_names: List[str] | None = None

    for sub in subjects:
        X_raw, y = load_bci2a_session(
            args.data_root,
            int(sub),
            training=(args.split == "train"),
            ref_mode="native",
            keep_channels=args.keep_channels,
            ref_channel=args.ref_channel,
            laplacian=False,
        )

        mu_sd = None
        if (args.standardize_mode or "none").lower() == "train":
            fit_mode = (args.standardize_fit_mode or "native").strip()
            X_fit = apply_reference(X_raw, mode=fit_mode, ref_idx=ref_idx, lap_neighbors=lap_neighbors)
            mu_sd = fit_standardizer(X_fit)

        for m in ref_modes:
            X_m = apply_reference(X_raw, mode=m, ref_idx=ref_idx, lap_neighbors=lap_neighbors)
            X_m = _apply_standardize(args.standardize_mode, X_m, mu_sd, args.instance_robust)

            feats, ys, names = _extract_features(
                X_m,
                y,
                fs=float(BCI2A_FS),
                bands=bands,
                max_trials=args.max_trials,
                seed=args.seed,
            )
            if feature_names is None:
                feature_names = names

            npz_path = os.path.join(out_dir, f"per_trial_features_sub{sub:02d}_mode_{m}.npz")
            np.savez_compressed(
                npz_path,
                features=feats,
                labels=ys,
                feature_names=np.array(names, dtype=object),
            )

            means = np.nanmean(feats, axis=0)
            stds = np.nanstd(feats, axis=0)

            row = {
                "subject": int(sub),
                "split": str(args.split),
                "mode": str(m),
                "n_trials": int(feats.shape[0]),
            }
            for j, name in enumerate(names):
                row[f"{name}__mean"] = float(means[j])
                row[f"{name}__std"] = float(stds[j])
            summary_rows.append(row)

    if not summary_rows:
        raise RuntimeError("No outputs produced. Check --subjects and --data_root.")

    fieldnames = list(summary_rows[0].keys())
    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    meta = {
        "args": vars(args),
        "subjects": subjects,
        "ref_modes": ref_modes,
        "keep_channels": args.keep_channels,
        "channels": keep_names,
        "feature_names": feature_names,
        "fs": float(BCI2A_FS),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[feature_shift_audit] wrote: {csv_path}")
    print(f"[feature_shift_audit] wrote per-trial npz for {len(subjects) * len(ref_modes)} subject-mode combos")
    print(f"[feature_shift_audit] wrote: {os.path.join(out_dir, 'meta.json')}")


if __name__ == "__main__":
    main()