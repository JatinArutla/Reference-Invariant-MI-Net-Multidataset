from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SplitSpec:
    """Defines how we split data for a *single subject*.

    Modes:
      - "session": use dataset-native train/test sessions or runs when available
      - "random": stratified random split (train_frac)
    """

    mode: str = "session"  # "session" | "random"
    train_frac: float = 0.8
    seed: int = 1


class BaseLRDataset:
    """Motor-imagery dataset wrapper.

    Historical name kept for backward compatibility, but wrappers may now expose:
      - task="all": dataset's native class set used in this repo
      - task="lr" : left-vs-right subset when supported

    All datasets return X as float32 [N, C, T] with C fixed to CANON_CHS_18.
    Labels must be zero-based contiguous int64 values.
    """

    name: str
    subject_list: list[int]
    sfreq: float
    n_channels: int

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
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Return ((X_tr,y_tr),(X_te,y_te)) in native(as-released) reference."""
        raise NotImplementedError
