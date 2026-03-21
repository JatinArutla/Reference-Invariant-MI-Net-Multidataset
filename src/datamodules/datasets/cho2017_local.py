from __future__ import annotations

"""Local loader for Cho2017 / GigaDB motor imagery data.

Expected files:
  s01.mat ... s52.mat

This dataset is intrinsically binary (left vs right). So task="all" and task="lr"
are equivalent.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import os
import numpy as np
import mne
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from scipy.io import loadmat

from .base import BaseLRDataset, SplitSpec
from ..channels import CANON_CHS_18
from ..transforms import bandpass_filter_trials, resample_trials


def _subject_path(root: str, subject: int) -> str:
    return os.path.join(root, f"s{subject:02d}.mat")


def _load_cho_subject(root: str, subject: int, *, tmin: float, tmax: float):
    path = _subject_path(root, subject)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing Cho2017 file for subject {subject}: {path}")

    data = loadmat(
        path,
        squeeze_me=True,
        struct_as_record=False,
        verify_compressed_data_integrity=False,
    )["eeg"]

    eeg_ch_names = [
        "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1",
        "C1", "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7",
        "P9", "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz", "Fpz", "Fp2",
        "AF8", "AF4", "AFz", "Fz", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4",
        "FC2", "FCz", "Cz", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
        "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
    ]
    emg_ch_names = ["EMG1", "EMG2", "EMG3", "EMG4"]
    ch_names = eeg_ch_names + emg_ch_names + ["Stim"]
    ch_types = ["eeg"] * 64 + ["emg"] * 4 + ["stim"]

    imagery_left = data.imagery_left - data.imagery_left.mean(axis=1, keepdims=True)
    imagery_right = data.imagery_right - data.imagery_right.mean(axis=1, keepdims=True)
    eeg_data_l = np.vstack([imagery_left * 1e-6, data.imagery_event])
    eeg_data_r = np.vstack([imagery_right * 1e-6, data.imagery_event * 2])
    eeg_data = np.hstack([eeg_data_l, np.zeros((eeg_data_l.shape[0], 500)), eeg_data_r])

    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=float(data.srate))
    raw = RawArray(data=eeg_data, info=info, verbose=False)
    raw.set_montage(make_standard_montage("standard_1005"), on_missing="ignore")

    events = mne.find_events(raw, stim_channel="Stim", shortest_event=1, verbose=False)
    event_id = {"left_hand": 1, "right_hand": 2}
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose="ERROR",
    )
    epochs = epochs.copy().pick(list(CANON_CHS_18))
    X = epochs.get_data().astype(np.float32, copy=False)
    inv = {v: k for k, v in epochs.event_id.items()}
    labs = np.array([inv[c] for c in epochs.events[:, 2]])
    y = np.where(labs == "left_hand", 0, 1).astype(np.int64)
    meta = {"sfreq": float(data.srate), "channels": list(CANON_CHS_18), "source": os.path.basename(path)}
    return X, y, meta


@dataclass
class Cho2017Local(BaseLRDataset):
    data_root: str

    def __post_init__(self):
        self.subject_list = list(range(1, 53))

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
            raise ValueError("Cho2017 is binary in this repo. Use --task all or --task lr.")
        X_all, y_all, meta = _load_cho_subject(self.data_root, subject, tmin=tmin, tmax=tmax)
        sfreq = float(meta.get("sfreq", 512.0))
        if band is not None:
            X_all = bandpass_filter_trials(X_all, fs=sfreq, band=band)
        if resample_hz is not None and abs(float(resample_hz) - sfreq) > 1e-6:
            X_all = resample_trials(X_all, fs_in=sfreq, fs_out=float(resample_hz))
            sfreq = float(resample_hz)
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, train_size=split.train_frac, random_state=split.seed)
        tr_idx, te_idx = next(sss.split(X_all, y_all))
        return (X_all[tr_idx].astype(np.float32), y_all[tr_idx].astype(np.int64)), (X_all[te_idx].astype(np.float32), y_all[te_idx].astype(np.int64))
