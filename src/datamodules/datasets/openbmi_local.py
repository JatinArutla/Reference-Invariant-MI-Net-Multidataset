from __future__ import annotations

"""Local loader for OpenBMI (Lee2019 MI) .mat files.

Designed for Kaggle-style setups where the OpenBMI MI files already exist on disk.

Expected filenames (as provided by the user):
  sess01_subj01_EEG_MI.mat ... sess02_subj54_EEG_MI.mat

Task: binary left vs right.
Output contract: epochs [N, C, T] in CANON_CHS_18 order.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import os
import numpy as np

from scipy.io import loadmat

from .base import BaseLRDataset, SplitSpec
from ..channels import CANON_CHS_18
from ..transforms import bandpass_filter_trials, resample_trials


def _mat_to_dict(mat_obj):
    """Convert matlab structs loaded by scipy into nested dicts (best-effort)."""
    if isinstance(mat_obj, np.ndarray) and mat_obj.dtype == object and mat_obj.size == 1:
        mat_obj = mat_obj.item()
    if isinstance(mat_obj, np.void):
        out = {}
        for name in mat_obj.dtype.names:
            out[name] = _mat_to_dict(mat_obj[name])
        return out
    if isinstance(mat_obj, np.ndarray) and mat_obj.dtype.names is not None:
        if mat_obj.size == 1:
            return _mat_to_dict(mat_obj.reshape(-1)[0])
        return [_mat_to_dict(x) for x in mat_obj]
    return mat_obj


def _extract_trials_from_openbmi_mat(mat_path: str):
    mat = loadmat(mat_path, simplify_cells=False)
    keys = [k for k in mat.keys() if not k.startswith("__")]

    # Kaggle OpenBMI variant often stores BOTH train/test structs at the top-level.
    # We'll treat a single .mat as "one session" and concatenate train+test trials
    # within that session.
    if "EEG_MI_train" in keys and "EEG_MI_test" in keys:
        parts = []
        for k in ["EEG_MI_train", "EEG_MI_test"]:
            parts.append(_mat_to_dict(mat[k]))
        # We'll fall through to the generic parsing below, but include these
        # session-part dicts as primary candidates.
    else:
        parts = []

    # Common: an "EEG_MI" or "EEG_MI_train" struct
    preferred = ["EEG_MI", "EEG_MI_train", "EEG_MI_offline", "MI", "data"]
    root = None
    for k in preferred:
        if k in keys:
            root = _mat_to_dict(mat[k])
            break

    candidates = []
    # First try explicit train/test parts (if present)
    for p in parts:
        if isinstance(p, dict):
            candidates.append(p)
    if isinstance(root, dict):
        candidates.append(root)
    # Finally, a dict of all top-level keys
    candidates.append({k: _mat_to_dict(mat[k]) for k in keys})

    X = None
    y = None
    ch_names = None
    sfreq = None

    def try_get(d: dict, names: list[str]):
        for n in names:
            if n in d:
                return d[n]
        return None

    for d in candidates:
        X = try_get(d, ["smt", "epo", "epoched", "trials", "trial", "x", "X", "cnt", "eeg", "EEG", "data"])
        y = try_get(d, ["y_dec", "y_logic", "y", "Y", "label", "labels", "mrk"])
        ch_names = try_get(d, ["chan", "ch_names", "channels", "clab", "ch"])
        sfreq = try_get(d, ["fs", "srate", "sfreq", "sampling_rate"])
        if X is not None and y is not None:
            break

    if X is None or y is None:
        raise RuntimeError(
            "Could not locate trials/labels in OpenBMI .mat file. "
            f"Path={mat_path}. Detected top-level keys={keys}. "
            "You may need to adapt src/datamodules/datasets/openbmi_local.py."
        )

    # Channel names
    ch_list = None
    if ch_names is not None:
        arr = np.array(ch_names)
        if arr.dtype.kind in ("U", "S"):
            ch_list = [str(c) for c in arr.reshape(-1)]
        elif arr.dtype == object:
            ch_list = [str(c).strip() for c in arr.reshape(-1)]
        else:
            ch_list = [str(c) for c in arr.reshape(-1)]

    # sfreq
    try:
        sfreq_val = float(np.array(sfreq).reshape(-1)[0]) if sfreq is not None else 1000.0
    except Exception:
        sfreq_val = 1000.0

    # --- helper: turn various label encodings into {0:left,1:right} (best-effort)
    def to_lr_labels(y_raw) -> np.ndarray:
        y_arr = np.array(y_raw).squeeze()
        # one-hot [K,N] or [N,K]
        if y_arr.ndim == 2 and min(y_arr.shape) <= 6:
            if y_arr.shape[0] <= 6:
                y_arr = np.argmax(y_arr, axis=0)
            else:
                y_arr = np.argmax(y_arr, axis=1)
        if y_arr.dtype.kind in ("U", "S", "O"):
            y_str = y_arr.astype(str)
            left = np.char.lower(y_str) == "left"
            right = np.char.lower(y_str) == "right"
            keep = left | right
            return np.where(left[keep], 0, 1).astype(np.int64)
        y_num = y_arr.astype(np.int64)
        u = set(np.unique(y_num).tolist())
        if u == {1, 2}:
            return (y_num - 1).astype(np.int64)
        if u == {-1, 1}:
            return (y_num == 1).astype(np.int64)
        return y_num.astype(np.int64)

    # --- helper: find markers inside a dict-like object
    def find_markers(obj) -> tuple[np.ndarray, np.ndarray] | None:
        if not isinstance(obj, dict):
            return None
        pos = None
        lab = None
        for k in ["pos", "t", "time", "latency", "onset", "sample", "samples"]:
            if k in obj:
                pos = obj[k]
                break
        for k in ["y", "Y", "labels", "label", "class", "classes", "event", "event_id", "y_dec"]:
            if k in obj:
                lab = obj[k]
                break
        if pos is None or lab is None:
            return None
        pos = np.array(pos).squeeze().astype(np.int64)
        lab = np.array(lab).squeeze()
        return pos, lab

    def search_markers_everywhere() -> tuple[np.ndarray, np.ndarray] | None:
        """Search for marker position+label pairs in likely places."""
        # 1) y itself (if dict-like)
        if isinstance(y, dict):
            m = find_markers(y)
            if m is not None:
                return m
        # 2) root dict
        if isinstance(root, dict):
            m = find_markers(root)
            if m is not None:
                return m
            for mk in ["mrk", "marker", "markers", "event", "events", "cue", "trig", "trigger"]:
                if mk in root and isinstance(root[mk], dict):
                    m = find_markers(root[mk])
                    if m is not None:
                        return m
        # 3) candidates (including EEG_MI_train/test parts)
        for d in candidates:
            if not isinstance(d, dict):
                continue
            m = find_markers(d)
            if m is not None:
                return m
            for mk in ["mrk", "marker", "markers", "event", "events", "cue", "trig", "trigger"]:
                if mk in d and isinstance(d[mk], dict):
                    m = find_markers(d[mk])
                    if m is not None:
                        return m
        # 4) nested: if we have top-level train/test, check inside them explicitly
        for pk in ["EEG_MI_train", "EEG_MI_test"]:
            if pk in keys:
                obj = _mat_to_dict(mat[pk])
                if isinstance(obj, dict):
                    m = find_markers(obj)
                    if m is not None:
                        return m
                    for mk in ["mrk", "marker", "markers", "event", "events", "cue", "trig", "trigger"]:
                        if mk in obj and isinstance(obj[mk], dict):
                            m = find_markers(obj[mk])
                            if m is not None:
                                return m
        return None

    X_arr = np.array(X)

    # Case A: already epoched trials [N,C,T] in some permutation
    if X_arr.ndim == 3:
        y_bin = to_lr_labels(y)
        dims = X_arr.shape
        perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
        best = None
        for p in perms:
            N, C, T = dims[p[0]], dims[p[1]], dims[p[2]]
            if C >= 18 and T >= int(0.5 * sfreq_val):
                best = p
                break
        if best is None:
            best = (0, 1, 2)
        X_nct = np.transpose(X_arr, best).astype(np.float32, copy=False)

        # Align lengths
        N = X_nct.shape[0]
        if y_bin.shape[0] != N:
            min_n = min(N, y_bin.shape[0])
            X_nct = X_nct[:min_n]
            y_bin = y_bin[:min_n]

    # Case B: continuous recording [T,C] (OpenBMI Kaggle variant) -> epoch using markers
    elif X_arr.ndim == 2:
        # Determine orientation: most OpenBMI exports are [samples, channels]
        if X_arr.shape[1] >= 18 and X_arr.shape[0] > X_arr.shape[1]:
            X_tc = X_arr
        elif X_arr.shape[0] >= 18 and X_arr.shape[1] > X_arr.shape[0]:
            X_tc = X_arr.T
        else:
            X_tc = X_arr

        markers = search_markers_everywhere()

        if markers is None:
            raise RuntimeError(
                "OpenBMI .mat appears to contain continuous data but no recognizable marker structure. "
                f"Path={mat_path}. Top-level keys={keys}. "
                "This Kaggle export often stores markers inside EEG_MI_train/EEG_MI_test structs. "
                "Inspect the .mat with scipy.io.loadmat(..., simplify_cells=True) and look inside those structs "
                "for fields like x, t/pos, and y/y_dec, then adapt marker parsing in openbmi_local.py."
            )

        pos, lab = markers
        y_num = to_lr_labels(lab)
        keep = (y_num == 0) | (y_num == 1)
        pos = pos[keep]
        y_bin = y_num[keep].astype(np.int64)

        # Extract [0s, 4s] around each cue (later cropped to [tmin, tmax]).
        epoch_len = int(round(4.0 * sfreq_val))
        n_ch = X_tc.shape[1]
        trials = []
        labels = []
        for p, yy in zip(pos.tolist(), y_bin.tolist()):
            s0 = int(p)
            s1 = s0 + epoch_len
            if s0 < 0 or s1 > X_tc.shape[0]:
                continue
            seg = X_tc[s0:s1, :].T  # [C,T]
            trials.append(seg)
            labels.append(yy)
        if not trials:
            raise RuntimeError(
                f"No valid epochs could be extracted from continuous OpenBMI recording. "
                f"Check marker positions and sampling rate. Path={mat_path}"
            )
        X_nct = np.stack(trials, axis=0).astype(np.float32, copy=False)
        y_bin = np.array(labels, dtype=np.int64)

        if ch_list is not None and len(ch_list) != n_ch:
            ch_list = None

    else:
        raise RuntimeError(
            f"Expected 3D trials array or 2D continuous array, got shape {X_arr.shape} for {mat_path}"
        )

    # NOTE: X_nct, y_bin are defined in the branches above.

    # Reorder/select CANON18
    #
    # NOTE: OpenBMI channel names are well-formed strings, but we still normalize
    # to be robust to case/whitespace/unicode oddities across exports.
    if ch_list is not None and len(ch_list) == X_nct.shape[1]:
        def _unbox(v):
            """Unwrap MATLAB-ish singleton containers.

            Some exports store channel names as object arrays or singleton lists,
            which would stringify to "['FC1']" and break matching.
            """
            while True:
                if isinstance(v, (list, tuple)) and len(v) == 1:
                    v = v[0]
                    continue
                if isinstance(v, np.ndarray):
                    arr = np.asarray(v)
                    if arr.size == 1:
                        v = arr.ravel()[0]
                        continue
                break
            return v

        def _norm(v) -> str:
            v = _unbox(v)
            return str(v).strip().replace('​', '').replace('﻿', '').upper()

        name_to_idx = {_norm(n): i for i, n in enumerate(ch_list)}
        req = [_norm(c) for c in CANON_CHS_18]
        missing = [CANON_CHS_18[i] for i, rc in enumerate(req) if rc not in name_to_idx]
        if missing:
            sample = list(sorted(list(name_to_idx.keys())))[:30]
            raise RuntimeError(
                f"OpenBMI file missing CANON18 channels: {missing}. "
                f"First 30 available (normalized): {sample}. "
                "Either change keep_channels for this dataset, or provide a mapping."
            )
        idx = [name_to_idx[rc] for rc in req]
        X_nct = X_nct[:, idx, :]

    meta = {"sfreq": sfreq_val, "channels": list(CANON_CHS_18), "source": os.path.basename(mat_path)}
    return X_nct, y_bin.astype(np.int64), meta


@dataclass
class OpenBMILocal(BaseLRDataset):
    """Local OpenBMI loader supporting session split."""

    data_root: str

    def __post_init__(self):
        self.subject_list = list(range(1, 55))

    def _mat_path(self, subject: int, session: int) -> str:
        return os.path.join(self.data_root, f"sess{session:02d}_subj{subject:02d}_EEG_MI.mat")

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
        task: str = "lr",
    ):
        if task != "lr":
            raise ValueError("OpenBMI local loader in this repo supports only --task lr (binary left-vs-right).")
        p1 = self._mat_path(subject, 1)
        p2 = self._mat_path(subject, 2)
        if not os.path.exists(p1) or not os.path.exists(p2):
            raise FileNotFoundError(f"Missing OpenBMI files for subject {subject}: {p1} or {p2}")

        X1, y1, meta1 = _extract_trials_from_openbmi_mat(p1)
        X2, y2, meta2 = _extract_trials_from_openbmi_mat(p2)

        sfreq = float(meta1.get("sfreq", 1000.0))

        # crop to [tmin, tmax]
        s0 = int(round(tmin * sfreq))
        s1 = int(round(tmax * sfreq))
        X1 = X1[:, :, s0:s1]
        X2 = X2[:, :, s0:s1]

        # preprocessing (after epoch extraction)
        X_all = np.concatenate([X1, X2], axis=0)
        y_all = np.concatenate([y1, y2], axis=0)

        if band is not None:
            X_all = bandpass_filter_trials(X_all, fs=sfreq, band=band)
        if resample_hz is not None and abs(float(resample_hz) - sfreq) > 1e-6:
            X_all = resample_trials(X_all, fs_in=sfreq, fs_out=float(resample_hz))
            sfreq = float(resample_hz)

        if split.mode == "session":
            n1 = X1.shape[0]
            X_tr, y_tr = X_all[:n1], y_all[:n1]
            X_te, y_te = X_all[n1:], y_all[n1:]
            return (X_tr.astype(np.float32), y_tr.astype(np.int64)), (X_te.astype(np.float32), y_te.astype(np.int64))

        from sklearn.model_selection import StratifiedShuffleSplit

        sss = StratifiedShuffleSplit(n_splits=1, train_size=split.train_frac, random_state=split.seed)
        tr_idx, te_idx = next(sss.split(X_all, y_all))
        X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
        X_te, y_te = X_all[te_idx], y_all[te_idx]
        return (X_tr.astype(np.float32), y_tr.astype(np.int64)), (X_te.astype(np.float32), y_te.astype(np.int64))
