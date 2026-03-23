from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..bci2a import load_bci2a_session
from ..transforms import bandpass_filter_trials, resample_trials
from .base import BaseLRDataset, SplitSpec


@dataclass
class IV2aManual(BaseLRDataset):
    """BCI Competition IV-2a (BNCI2014_001) loaded from local .mat files.

    This uses the repo's existing loader (A0{s}T/A0{s}E).
    """

    data_root: str
    name: str = "iv2a"
    subject_list: list[int] = None
    keep_channels: str = "canon18"

    def __post_init__(self):
        if self.subject_list is None:
            self.subject_list = list(range(1, 10))

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
        # Session split is fixed by dataset design: T=train, E=test.
        if split.mode not in ("session", "random"):
            raise ValueError(f"IV2aManual split.mode must be 'session' or 'random', got {split.mode}")

        # Always load in native(as-released) reference. Channel subset/order is configurable.
        X_tr, y_tr = load_bci2a_session(
            self.data_root,
            subject,
            training=True,
            t1_sec=tmin,
            t2_sec=tmax,
            ref_mode="native",
            keep_channels="canon18",
            laplacian=False,
        )
        X_te, y_te = load_bci2a_session(
            self.data_root,
            subject,
            training=False,
            t1_sec=tmin,
            t2_sec=tmax,
            ref_mode="native",
            keep_channels="canon18",
            laplacian=False,
        )

        if task == "lr":
            tr_mask = (y_tr == 0) | (y_tr == 1)
            te_mask = (y_te == 0) | (y_te == 1)
            X_tr, y_tr = X_tr[tr_mask], y_tr[tr_mask]
            X_te, y_te = X_te[te_mask], y_te[te_mask]
        elif task in ("all", "4class"):
            pass
        else:
            raise ValueError(f"Unknown task='{task}'. Use 'all'/'4class' or 'lr'.")

        # Optional preprocessing contract
        if band is not None:
            X_tr = bandpass_filter_trials(X_tr, fs=250.0, band=band)
            X_te = bandpass_filter_trials(X_te, fs=250.0, band=band)

        if resample_hz is not None and float(resample_hz) != 250.0:
            X_tr = resample_trials(X_tr, fs_in=250.0, fs_out=float(resample_hz))
            X_te = resample_trials(X_te, fs_in=250.0, fs_out=float(resample_hz))

        return (X_tr.astype(np.float32), y_tr.astype(np.int64)), (X_te.astype(np.float32), y_te.astype(np.int64))
