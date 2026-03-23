from __future__ import annotations

"""Protocol presets for native-vs-harmonized dataset handling.

Important distinction:
- Some loaders use absolute trial time (e.g. IV-2a)
- Some loaders epoch relative to cue/event onset (e.g. OpenBMI, Dreyer2023)

So the native presets below are expressed in the *loader semantics* used in this repo,
not necessarily the raw paper trial timeline.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ProtocolPreset:
    tmin: float
    tmax: float
    resample_hz: Optional[float] = None
    band_lo: Optional[float] = None
    band_hi: Optional[float] = None
    keep_channels: Optional[str] = None
    lr: Optional[float] = None
    eegn_pool: Optional[int] = None
    tcn_kernel: Optional[int] = None


HARMONIZED_BASELINE = ProtocolPreset(
    tmin=0.0,
    tmax=3.0,
    resample_hz=160.0,
    band_lo=8.0,
    band_hi=32.0,
    keep_channels="canon18",
    lr=3e-4,
    eegn_pool=8,
    tcn_kernel=8,
)


NATIVE_PRESETS = {
    # Loader uses absolute trial time. The task period is 2-6 s.
    "iv2a": ProtocolPreset(
        tmin=2.0,
        tmax=6.0,
        resample_hz=250.0,
        band_lo=None,
        band_hi=None,
        keep_channels="bci2a",
        lr=1e-3,
        eegn_pool=7,
        tcn_kernel=4,
    ),
    # Loader epochs relative to cue onset. Paper task window 3-7 s from trial start.
    "openbmi_local": ProtocolPreset(tmin=0.0, tmax=4.0),
    # Loader epochs relative to event onset. MOABB/native practical default is 0-3 s.
    "physionet_local": ProtocolPreset(tmin=0.0, tmax=3.0),
    # Loader epochs relative to imagery event onset. Native task window is 0-3 s.
    "cho2017_local": ProtocolPreset(tmin=0.0, tmax=3.0),
    # Loader epochs relative to cue/event onset. Paper task window 3-8 s from trial start.
    "dreyer2023_local": ProtocolPreset(tmin=0.0, tmax=5.0),
    # MOABB wrappers kept aligned with their practical event-relative defaults in this repo.
    "lee2019": ProtocolPreset(tmin=0.0, tmax=4.0),
    "physionet": ProtocolPreset(tmin=0.0, tmax=3.0),
}


def canonical_dataset_key(name: str) -> str:
    return (name or "").lower().strip()


def get_protocol_preset(dataset: str, protocol: str) -> Optional[ProtocolPreset]:
    key = canonical_dataset_key(dataset)
    prot = (protocol or "native").lower().strip()
    if prot == "harmonized":
        return HARMONIZED_BASELINE
    if prot == "native":
        return NATIVE_PRESETS.get(key)
    raise ValueError(f"Unknown protocol '{protocol}'. Use native or harmonized.")
