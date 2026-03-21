from __future__ import annotations

from typing import Optional

from .base import BaseLRDataset
from .cho2017_local import Cho2017Local
from .dreyer2023_local import Dreyer2023Local
from .iv2a_manual import IV2aManual
from .moabb_lr import MOABBLR
from .openbmi_local import OpenBMILocal
from .physionet_local import PhysionetLocal


def get_dataset(
    name: str,
    *,
    data_root: Optional[str] = None,
) -> BaseLRDataset:
    """Factory for dataset wrappers.

    Supported local datasets:
      - iv2a            : local BCI IV-2a .mat files
      - openbmi_local   : local OpenBMI .mat files
      - physionet_local : local Physionet eegmmidb EDF folder
      - cho2017_local   : local Cho2017 / GigaDB .mat files
      - dreyer2023_local: local Dreyer2023 BIDS-style folder

    Optional MOABB datasets:
      - lee2019   : MOABB Lee2019_MI (binary MI)
      - physionet : MOABB PhysionetMI (still binary wrapper in this repo)
    """

    key = (name or "").lower().strip()
    if key in ("iv2a", "bnci2014_001", "bciiv2a", "bci_iv_2a"):
        if not data_root:
            raise ValueError("iv2a requires --data_root pointing to the A0* .mat files")
        return IV2aManual(data_root=data_root)

    if key in ("openbmi_local", "openbmi_mat", "lee2019_local"):
        if not data_root:
            raise ValueError("openbmi_local requires --data_root pointing to the folder with sess01_subjXX_EEG_MI.mat")
        return OpenBMILocal(data_root=data_root)

    if key in ("physionet_local", "physionet_edf", "eegmmidb_local"):
        if not data_root:
            raise ValueError("physionet_local requires --data_root pointing to the folder containing S001/... EDFs")
        return PhysionetLocal(data_root=data_root)

    if key in ("cho2017_local", "cho2017", "gigadb", "gigadb_local"):
        if not data_root:
            raise ValueError("cho2017_local requires --data_root pointing to the folder with s01.mat ... s52.mat")
        return Cho2017Local(data_root=data_root)

    if key in ("dreyer2023_local", "dreyer2023", "dreyer_local"):
        if not data_root:
            raise ValueError("dreyer2023_local requires --data_root pointing to the MNE-Dreyer2023-data folder")
        return Dreyer2023Local(data_root=data_root)

    if key in ("lee2019", "openbmi", "lee2019_mi"):
        from moabb.datasets import Lee2019_MI
        return MOABBLR(dataset_ctor=Lee2019_MI, name="lee2019", moabb_kwargs={"train_run": True, "test_run": None, "sessions": (1, 2)})

    if key in ("physionet", "physionetmi"):
        from moabb.datasets import PhysionetMI
        return MOABBLR(dataset_ctor=PhysionetMI, name="physionet", moabb_kwargs={"imagined": True, "executed": False})

    raise ValueError(f"Unknown dataset '{name}'. Valid local ids: iv2a, openbmi_local, physionet_local, cho2017_local, dreyer2023_local")
