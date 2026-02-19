from __future__ import annotations

from typing import Optional

from .base import BaseLRDataset
from .iv2a_manual import IV2aManual
from .moabb_lr import MOABBLR


def get_dataset(
    name: str,
    *,
    data_root: Optional[str] = None,
) -> BaseLRDataset:
    """Factory for dataset wrappers.

    Supported:
      - iv2a      : local BCI IV-2a .mat files
      - lee2019   : MOABB Lee2019_MI (train_run only)
      - physionet : MOABB PhysionetMI(imagined)
    """

    key = (name or "").lower().strip()
    if key in ("iv2a", "bnci2014_001", "bciiv2a", "bci_iv_2a"):
        if not data_root:
            raise ValueError("iv2a requires --data_root pointing to the A0* .mat files")
        return IV2aManual(data_root=data_root)

    if key in ("lee2019", "openbmi", "lee2019_mi"):
        from moabb.datasets import Lee2019_MI

        return MOABBLR(dataset_ctor=Lee2019_MI, name="lee2019", moabb_kwargs={"train_run": True, "test_run": None, "sessions": (1, 2)})

    if key in ("physionet", "physionetmi"):
        from moabb.datasets import PhysionetMI

        return MOABBLR(dataset_ctor=PhysionetMI, name="physionet", moabb_kwargs={"imagined": True, "executed": False})

    raise ValueError(f"Unknown dataset '{name}'. Valid: iv2a, lee2019, physionet")
