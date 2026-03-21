from __future__ import annotations

"""Local loader for Dreyer2023 BIDS-style motor imagery data.

Expected root example:
  .../MNE-Dreyer2023-data/sub-01/...

This dataset is intrinsically binary (left vs right). So task="all" and task="lr"
are equivalent.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import mne

from .base import BaseLRDataset, SplitSpec
from ..channels import CANON_CHS_18
from ..transforms import bandpass_filter_trials, resample_trials


SUPPORTED_EXTS = {".vhdr", ".edf", ".bdf", ".gdf", ".set", ".fif"}


def _find_subject_files(root: str, subject: int) -> list[Path]:
    sub_dir = Path(root) / f"sub-{subject:02d}"
    if not sub_dir.exists():
        raise FileNotFoundError(f"Missing Dreyer2023 subject directory: {sub_dir}")
    files = []
    for p in sorted(sub_dir.rglob("*")):
        if not p.is_file():
            continue
        name = p.name.lower()
        if "_eeg" not in name:
            continue
        if p.suffix.lower() not in SUPPORTED_EXTS:
            continue
        if "baseline" in name or "rest" in name:
            continue
        if subject == 59 and (("r5online" in name) or ("r6online" in name)):
            continue
        files.append(p)
    if not files:
        raise FileNotFoundError(f"No Dreyer2023 EEG files found for subject {subject} under {sub_dir}")
    return files


def _read_raw_any(path: Path):
    suf = path.suffix.lower()
    if suf == ".vhdr":
        return mne.io.read_raw_brainvision(str(path), preload=True, verbose="ERROR")
    if suf == ".edf":
        return mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
    if suf == ".bdf":
        return mne.io.read_raw_bdf(str(path), preload=True, verbose="ERROR")
    if suf == ".gdf":
        return mne.io.read_raw_gdf(str(path), preload=True, verbose="ERROR")
    if suf == ".set":
        return mne.io.read_raw_eeglab(str(path), preload=True, verbose="ERROR")
    if suf == ".fif":
        return mne.io.read_raw_fif(str(path), preload=True, verbose="ERROR")
    raise ValueError(f"Unsupported Dreyer2023 file type: {path}")


def _epoch_one_raw(raw, *, tmin: float, tmax: float):
    if raw.annotations is not None and len(raw.annotations):
        try:
            raw.annotations.rename({"769": "left_hand", "770": "right_hand"})
        except Exception:
            pass
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    wanted = {k: v for k, v in event_id.items() if k in ("left_hand", "right_hand")}
    if not wanted:
        raise RuntimeError(f"Could not find Dreyer2023 left/right annotations. Found keys: {list(event_id.keys())[:20]}")
    epochs = mne.Epochs(raw, events, event_id=wanted, tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose="ERROR")
    missing = [c for c in CANON_CHS_18 if c not in epochs.ch_names]
    if missing:
        raise RuntimeError(f"Dreyer2023 file missing CANON18 channels: {missing}. First 30 available: {epochs.ch_names[:30]}")
    epochs = epochs.copy().pick(list(CANON_CHS_18))
    X = epochs.get_data().astype(np.float32, copy=False)
    inv = {v: k for k, v in epochs.event_id.items()}
    labs = np.array([inv[c] for c in epochs.events[:, 2]])
    y = np.where(labs == "left_hand", 0, 1).astype(np.int64)
    return X, y


def _load_dreyer_subject(root: str, subject: int, *, tmin: float, tmax: float):
    files = _find_subject_files(root, subject)
    train_parts = []
    test_parts = []
    sfreq = None
    for p in files:
        raw = _read_raw_any(p)
        sfreq = float(raw.info["sfreq"])
        X, y = _epoch_one_raw(raw, tmin=tmin, tmax=tmax)
        name = p.name.lower()
        if "online" in name:
            test_parts.append((X, y))
        else:
            train_parts.append((X, y))
    return train_parts, test_parts, {"sfreq": sfreq or 512.0, "channels": list(CANON_CHS_18)}


@dataclass
class Dreyer2023Local(BaseLRDataset):
    data_root: str

    def __post_init__(self):
        self.subject_list = list(range(1, 88))

    def load_subject_native(
        self,
        subject: int,
        *,
        split: SplitSpec,
        tmin: float,
        tmax: float,
        resample_hz: Optional[float],
        band: Optional[Tuple[float, float]],
        cache_root: Optional[str] = None,
        task: str = "all",
    ):
        if task not in ("all", "lr"):
            raise ValueError("Dreyer2023 is binary in this repo. Use --task all or --task lr.")
        train_parts, test_parts, meta = _load_dreyer_subject(self.data_root, subject, tmin=tmin, tmax=tmax)
        sfreq = float(meta.get("sfreq", 512.0))

        def _stack(parts):
            if not parts:
                return None, None
            return np.concatenate([x for x, _ in parts], axis=0), np.concatenate([y for _, y in parts], axis=0)

        X_tr, y_tr = _stack(train_parts)
        X_te, y_te = _stack(test_parts)

        if split.mode == "session" and X_tr is not None and X_te is not None and len(y_tr) > 0 and len(y_te) > 0:
            pass
        else:
            all_parts = list(train_parts) + list(test_parts)
            if not all_parts:
                raise RuntimeError(f"No Dreyer2023 usable runs found for subject {subject}")
            X_all = np.concatenate([x for x, _ in all_parts], axis=0)
            y_all = np.concatenate([y for _, y in all_parts], axis=0)
            from sklearn.model_selection import StratifiedShuffleSplit
            if band is not None:
                X_all = bandpass_filter_trials(X_all, fs=sfreq, band=band)
            if resample_hz is not None and abs(float(resample_hz) - sfreq) > 1e-6:
                X_all = resample_trials(X_all, fs_in=sfreq, fs_out=float(resample_hz))
            sss = StratifiedShuffleSplit(n_splits=1, train_size=split.train_frac, random_state=split.seed)
            tr_idx, te_idx = next(sss.split(X_all, y_all))
            return (X_all[tr_idx].astype(np.float32), y_all[tr_idx].astype(np.int64)), (X_all[te_idx].astype(np.float32), y_all[te_idx].astype(np.int64))

        if band is not None:
            X_tr = bandpass_filter_trials(X_tr, fs=sfreq, band=band)
            X_te = bandpass_filter_trials(X_te, fs=sfreq, band=band)
        if resample_hz is not None and abs(float(resample_hz) - sfreq) > 1e-6:
            X_tr = resample_trials(X_tr, fs_in=sfreq, fs_out=float(resample_hz))
            X_te = resample_trials(X_te, fs_in=sfreq, fs_out=float(resample_hz))
        return (X_tr.astype(np.float32), y_tr.astype(np.int64)), (X_te.astype(np.float32), y_te.astype(np.int64))
