from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SplitSpec:
    """Defines how we split data for a *single subject*.

    For multi-dataset comparability we keep the task binary (left vs right).

    Modes:
      - "session": use session_1 for train and session_2 for test (if available)
      - "random": stratified random split (train_frac)
    """

    mode: str = "session"  # "session" | "random"
    train_frac: float = 0.8
    seed: int = 1


class BaseLRDataset:
    """Binary left-vs-right motor imagery dataset wrapper."""

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
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Return ((X_tr,y_tr),(X_te,y_te)) in *native(as-released)* reference.

        X returned as float32 [N,C,T] with C fixed to the repo's canonical montage.
        """

        raise NotImplementedError
