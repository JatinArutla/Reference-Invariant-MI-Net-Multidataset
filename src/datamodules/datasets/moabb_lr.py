from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import os
import json
import numpy as np

from .base import BaseLRDataset, SplitSpec
from ..channels import CANON_CHS_18
from ..transforms import bandpass_filter_trials, resample_trials


def _ensure_moabb():
    try:
        import moabb  # noqa: F401
        return True
    except Exception as e:
        raise ImportError(
            "MOABB is required for this dataset wrapper. Install with: pip install moabb"
        ) from e


def _cache_paths(cache_root: str, dataset_name: str, subject: int) -> Dict[str, str]:
    d = os.path.join(cache_root, dataset_name, f"sub{subject:03d}")
    return {
        "dir": d,
        "X": os.path.join(d, "X.npy"),
        "y": os.path.join(d, "y.npy"),
        "meta": os.path.join(d, "meta.json"),
    }


def _save_cache(cache_root: str, dataset_name: str, subject: int, X: np.ndarray, y: np.ndarray, meta: Dict[str, Any]):
    paths = _cache_paths(cache_root, dataset_name, subject)
    os.makedirs(paths["dir"], exist_ok=True)
    np.save(paths["X"], X.astype(np.float32))
    np.save(paths["y"], y.astype(np.int64))
    with open(paths["meta"], "w") as f:
        json.dump(meta, f, indent=2)


def _try_load_cache(cache_root: str, dataset_name: str, subject: int):
    paths = _cache_paths(cache_root, dataset_name, subject)
    if not (os.path.exists(paths["X"]) and os.path.exists(paths["y"])):
        return None
    X = np.load(paths["X"], mmap_mode=None)
    y = np.load(paths["y"], mmap_mode=None)
    meta = {}
    if os.path.exists(paths["meta"]):
        with open(paths["meta"], "r") as f:
            meta = json.load(f)
    return X, y, meta


def _extract_left_right_from_moabb(dataset, subject: int, tmin: float, tmax: float):
    """Load epochs via MOABB and return (X,y,meta) in CANON18 order.

    We avoid relying on acquisition reference details: this is "native(as-released)".
    """
    _ensure_moabb()
    import mne
    from moabb.paradigms import MotorImagery

    # Force deterministic channel set.
    # Note: some MOABB datasets include EOG. MotorImagery should handle this.
    paradigm = MotorImagery(
        n_classes=2,
        channels=list(CANON_CHS_18),
        resample=None,
        tmin=tmin,
        tmax=tmax,
    )

    X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subject])
    # MOABB returns y as labels (strings) for some datasets.
    # Convert to int {0:left, 1:right}.
    if y.dtype.kind in ("U", "S", "O"):
        y_str = np.array(y).astype(str)
        # Most MI datasets use these names.
        left = (y_str == "left_hand")
        right = (y_str == "right_hand")
        keep = left | right
        X = X[keep]
        y = np.where(left[keep], 0, 1).astype(np.int64)
    else:
        y = y.astype(np.int64)

    # X is [N,C,T] already, channels are in the requested order.
    # Make sure it's float32.
    X = X.astype(np.float32, copy=False)

    # Meta is a DataFrame; keep small info.
    meta_info = {
        "sfreq": float(meta["sfreq"].iloc[0]) if "sfreq" in meta.columns else None,
        "n_trials": int(X.shape[0]),
        "channels": list(CANON_CHS_18),
    }

    return X, y, meta_info


@dataclass
class MOABBLR(BaseLRDataset):
    """Binary left-vs-right wrapper around a MOABB dataset class."""

    dataset_ctor: object
    name: str
    moabb_kwargs: dict

    def __post_init__(self):
        _ensure_moabb()
        ds = self.dataset_ctor(**(self.moabb_kwargs or {}))
        self.subject_list = list(ds.subject_list)

    def _get_ds(self):
        return self.dataset_ctor(**(self.moabb_kwargs or {}))

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
        if cache_root:
            cached = _try_load_cache(cache_root, self.name, subject)
            if cached is not None:
                X_all, y_all, meta = cached
            else:
                ds = self._get_ds()
                X_all, y_all, meta = _extract_left_right_from_moabb(ds, subject, tmin=tmin, tmax=tmax)
                _save_cache(cache_root, self.name, subject, X_all, y_all, meta)
        else:
            ds = self._get_ds()
            X_all, y_all, meta = _extract_left_right_from_moabb(ds, subject, tmin=tmin, tmax=tmax)

        # Preprocessing contract (applied *after* epoching)
        sfreq = float(meta.get("sfreq") or 0.0)
        if band is not None and sfreq > 0:
            X_all = bandpass_filter_trials(X_all, fs=sfreq, band=band)
        if resample_hz is not None and sfreq > 0 and abs(float(resample_hz) - sfreq) > 1e-6:
            X_all = resample_trials(X_all, fs_in=sfreq, fs_out=float(resample_hz))
            meta["sfreq"] = float(resample_hz)

        # Split
        if split.mode == "random":
            from sklearn.model_selection import StratifiedShuffleSplit

            sss = StratifiedShuffleSplit(n_splits=1, train_size=split.train_frac, random_state=split.seed)
            tr_idx, te_idx = next(sss.split(X_all, y_all))
            X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
            X_te, y_te = X_all[te_idx], y_all[te_idx]
            return (X_tr.astype(np.float32), y_tr.astype(np.int64)), (X_te.astype(np.float32), y_te.astype(np.int64))

        # "session" split is not consistently exposed across MOABB datasets in a way
        # that's uniform and stable across versions. For this repo we use a
        # deterministic stratified split as the default.
        from sklearn.model_selection import StratifiedShuffleSplit

        sss = StratifiedShuffleSplit(n_splits=1, train_size=split.train_frac, random_state=split.seed)
        tr_idx, te_idx = next(sss.split(X_all, y_all))
        X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
        X_te, y_te = X_all[te_idx], y_all[te_idx]
        return (X_tr.astype(np.float32), y_tr.astype(np.int64)), (X_te.astype(np.float32), y_te.astype(np.int64))
