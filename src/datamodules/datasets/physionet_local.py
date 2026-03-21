from __future__ import annotations

"""Local loader for Physionet EEG Motor Movement/Imagery (eegmmidb).

This loader assumes you already have the Physionet folder on disk (e.g. Kaggle input):
  .../files/S001/S001R04.edf, ...

Tasks supported:
  - task="all": imagined 5-class set used by MOABB metadata
      {rest, left_hand, right_hand, hands, feet}
  - task="lr": imagined left/right subset only

Output contract: epochs [N, C, T] in CANON_CHS_18 order.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import os
import re
import numpy as np
import mne

from .base import BaseLRDataset, SplitSpec
from ..channels import CANON_CHS_18
from ..transforms import bandpass_filter_trials, resample_trials


def _normalize_physionet_ch_name(ch: str) -> str:
    ch0 = str(ch).strip()
    ch0 = re.sub(r"^EEG\s+", "", ch0, flags=re.IGNORECASE)
    ch0 = re.sub(r"[-_](REF|LE|RE|A1|A2)$", "", ch0, flags=re.IGNORECASE)
    ch0 = ch0.replace(" ", "")
    ch0 = ch0.rstrip(".")
    up = ch0.upper()
    if up in ("FZ", "CZ", "PZ"):
        return up[0] + "z"
    if up == "CPZ":
        return "CPz"
    if up.endswith("Z") and not any(ch.isdigit() for ch in up):
        return up[:-1] + "z"
    return up


def _subj_dir(root: str, subject: int) -> str:
    return os.path.join(root, f"S{subject:03d}")


def _edf_path(root: str, subject: int, run: int) -> str:
    return os.path.join(_subj_dir(root, subject), f"S{subject:03d}R{run:02d}.edf")


def _load_and_prepare_run(path: str):
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    raw.rename_channels(lambda ch: _normalize_physionet_ch_name(ch))
    raw.pick(picks="eeg")
    try:
        raw.set_montage("standard_1005", on_missing="ignore")
    except Exception:
        pass
    return raw


def _epoch_physionet_runs(root: str, subject: int, runs: list[int], *, tmin: float, tmax: float):
    raws = []
    for run in runs:
        p = _edf_path(root, subject, run)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing Physionet EDF for subject {subject} run {run}: {p}")
        raw = _load_and_prepare_run(p)
        stim = raw.annotations.description.astype(np.dtype("<U16"))
        if run in (4, 8, 12):
            stim[stim == "T0"] = "rest"
            stim[stim == "T1"] = "left_hand"
            stim[stim == "T2"] = "right_hand"
        elif run in (6, 10, 14):
            stim[stim == "T0"] = "rest"
            stim[stim == "T1"] = "hands"
            stim[stim == "T2"] = "feet"
        else:
            raise ValueError(f"Unexpected Physionet run in loader: {run}")
        raw.annotations.description = stim
        raws.append(raw)

    raw = mne.concatenate_raws(raws)
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    wanted_keys = [k for k in ("rest", "left_hand", "right_hand", "hands", "feet") if k in event_id]
    wanted = {k: event_id[k] for k in wanted_keys}
    epochs = mne.Epochs(raw, events, event_id=wanted, tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose="ERROR")
    epochs.rename_channels(lambda ch: _normalize_physionet_ch_name(ch))
    missing = [c for c in CANON_CHS_18 if c not in epochs.ch_names]
    if missing:
        raise RuntimeError(
            f"Physionet subject {subject} missing CANON18 channels: {missing}. First 30 available: {epochs.ch_names[:30]}"
        )
    epochs = epochs.copy().pick(list(CANON_CHS_18))
    X = epochs.get_data().astype(np.float32, copy=False)
    inv = {v: k for k, v in epochs.event_id.items()}
    labs = np.array([inv[c] for c in epochs.events[:, 2]])
    return X, labs, float(raw.info["sfreq"])


@dataclass
class PhysionetLocal(BaseLRDataset):
    data_root: str
    imagined_only: bool = True

    def __post_init__(self):
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
        task: str = "all",
    ):
        if task == "lr":
            runs = [4, 8, 12]
            class_order = ["left_hand", "right_hand"]
        elif task == "all":
            runs = [4, 8, 12, 6, 10, 14]
            class_order = ["rest", "left_hand", "right_hand", "hands", "feet"]
        else:
            raise ValueError("Physionet local loader supports only --task all or --task lr.")

        X_all, labs, sfreq = _epoch_physionet_runs(self.data_root, subject, runs, tmin=tmin, tmax=tmax)
        keep = np.isin(labs, class_order)
        X_all = X_all[keep]
        labs = labs[keep]
        y_all = np.array([class_order.index(x) for x in labs.tolist()], dtype=np.int64)

        if band is not None:
            X_all = bandpass_filter_trials(X_all, fs=sfreq, band=band)
        if resample_hz is not None and abs(float(resample_hz) - sfreq) > 1e-6:
            X_all = resample_trials(X_all, fs_in=sfreq, fs_out=float(resample_hz))
            sfreq = float(resample_hz)

        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, train_size=split.train_frac, random_state=split.seed)
        tr_idx, te_idx = next(sss.split(X_all, y_all))
        return (X_all[tr_idx].astype(np.float32), y_all[tr_idx].astype(np.int64)), (X_all[te_idx].astype(np.float32), y_all[te_idx].astype(np.int64))
