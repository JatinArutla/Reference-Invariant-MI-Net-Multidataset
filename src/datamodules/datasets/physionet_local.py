from __future__ import annotations

"""Local loader for Physionet EEG Motor Movement/Imagery (eegmmidb).

This loader assumes you already have the Physionet folder on disk (e.g. Kaggle input):
  .../files/S001/S001R04.edf, ...

We mirror the standard MNE/Physionet convention:
  - Annotations are encoded as T0/T1/T2 in the EDF.
    T1 = left fist, T2 = right fist.
  - We only use imagined left/right fist runs: 4, 8, 12.

Task: binary left vs right.
Output contract: epochs [N, C, T] in CANON_CHS_18 order.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import os
import numpy as np

import mne

from .base import BaseLRDataset, SplitSpec
from ..channels import CANON_CHS_18
from ..transforms import bandpass_filter_trials, resample_trials


def _subj_dir(root: str, subject: int) -> str:
    return os.path.join(root, f"S{subject:03d}")


def _edf_path(root: str, subject: int, run: int) -> str:
    return os.path.join(_subj_dir(root, subject), f"S{subject:03d}R{run:02d}.edf")


def _load_physionet_subject_left_right(root: str, subject: int, *, tmin: float, tmax: float):
    # Imagined left/right fist runs
    runs = [4, 8, 12]
    raws = []
    for r in runs:
        p = _edf_path(root, subject, r)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing Physionet EDF for subject {subject} run {r}: {p}")
        raw = mne.io.read_raw_edf(p, preload=True, verbose="ERROR")
        raws.append(raw)

    raw = mne.concatenate_raws(raws)
    # Standard montage; Physionet uses 64-channel 10-10 style names.
    try:
        raw.set_montage("standard_1005", on_missing="ignore")
    except Exception:
        pass

    raw.pick_types(eeg=True, eog=False, stim=False, ecg=False, emg=False, misc=False)

    # Events from EDF annotations
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    # We only care about T1/T2
    wanted = {k: v for k, v in event_id.items() if k in ("T1", "T2")}
    if not wanted:
        raise RuntimeError(
            f"Could not find T1/T2 annotations for Physionet subject {subject}. "
            f"Found annotation keys: {list(event_id.keys())[:20]}"
        )

    epochs = mne.Epochs(
        raw,
        events,
        event_id=wanted,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose="ERROR",
    )

    # Pick and order CANON18
    missing = [c for c in CANON_CHS_18 if c not in epochs.ch_names]
    if missing:
        raise RuntimeError(f"Physionet subject {subject} missing CANON18 channels: {missing}")
    epochs = epochs.copy().pick_channels(list(CANON_CHS_18))

    X = epochs.get_data().astype(np.float32, copy=False)  # [N,C,T]
    # Labels based on events: map T1->0 (left), T2->1 (right)
    # epochs.events[:,2] is the event code. We use epochs.event_id.
    inv = {v: k for k, v in epochs.event_id.items()}
    labs = np.array([inv[c] for c in epochs.events[:, 2]])
    y = np.where(labs == "T1", 0, 1).astype(np.int64)

    meta = {"sfreq": float(raw.info["sfreq"]), "channels": list(CANON_CHS_18), "runs": runs}
    return X, y, meta


@dataclass
class PhysionetLocal(BaseLRDataset):
    """Local Physionet loader."""

    data_root: str
    imagined_only: bool = True

    def __post_init__(self):
        # Subjects are S001..S109 in the dataset.
        # We trust the directory layout; missing subjects will raise when accessed.
        self.subject_list = list(range(1, 110))

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
    ):
        X_all, y_all, meta = _load_physionet_subject_left_right(self.data_root, subject, tmin=tmin, tmax=tmax)
        sfreq = float(meta.get("sfreq", 160.0))

        # Post-epoch preprocessing
        if band is not None:
            X_all = bandpass_filter_trials(X_all, fs=sfreq, band=band)
        if resample_hz is not None and abs(float(resample_hz) - sfreq) > 1e-6:
            X_all = resample_trials(X_all, fs_in=sfreq, fs_out=float(resample_hz))
            sfreq = float(resample_hz)

        from sklearn.model_selection import StratifiedShuffleSplit

        sss = StratifiedShuffleSplit(n_splits=1, train_size=split.train_frac, random_state=split.seed)
        tr_idx, te_idx = next(sss.split(X_all, y_all))
        X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
        X_te, y_te = X_all[te_idx], y_all[te_idx]
        return (X_tr.astype(np.float32), y_tr.astype(np.int64)), (X_te.astype(np.float32), y_te.astype(np.int64))
