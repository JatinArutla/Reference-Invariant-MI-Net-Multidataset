from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


# BCI Competition IV-2a EEG channel order used by this repo.
# This matches the usual 22 EEG channels (EOG channels are excluded).
BCI2A_CH_NAMES: List[str] = [
    "Fz",
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2",
    "POz",
]


def name_to_index(names: List[str]) -> Dict[str, int]:
    return {n: i for i, n in enumerate(names)}


def parse_keep_channels(s: str | None, *, all_names: List[str]) -> Optional[List[int]]:
    """Parse a comma-separated list of channel names into indices.

    Example: "C3,Cz,C4".
    Returns None if s is None/empty.
    """
    if not s:
        return None
    req = [p.strip() for p in s.split(",") if p.strip()]
    m = name_to_index(all_names)
    missing = [r for r in req if r not in m]
    if missing:
        raise ValueError(f"Unknown channel(s): {missing}. Available: {all_names}")
    return [m[r] for r in req]


def subset_and_reorder(X, keep_idx: List[int]):
    """Subset channels of X to keep_idx.

    Supports X:
      - [N,C,T]
      - [C,T]
    """
    if X.ndim == 3:
        return X[:, keep_idx, :]
    if X.ndim == 2:
        return X[keep_idx, :]
    raise ValueError(f"Expected 2D/3D, got {X.ndim}D")


def bci2a_default_laplacian_neighbors() -> Dict[str, List[str]]:
    """A reasonable neighbor graph for a Laplacian-like local reference.

    This is intentionally simple and purely name-based so it can be projected
    onto any channel subset (intersection) cleanly.
    """
    return {
        "Fz": ["FC1", "FCz", "FC2"],
        "FC3": ["FC1", "C3", "Fz"],
        "FC1": ["FC3", "FCz", "C1", "Fz"],
        "FCz": ["FC1", "FC2", "Cz", "Fz", "CPz"],
        "FC2": ["FCz", "FC4", "C2", "Fz"],
        "FC4": ["FC2", "C4", "Fz"],
        "C5": ["C3", "CP3"],
        "C3": ["C5", "C1", "CP3", "FC3"],
        "C1": ["C3", "Cz", "FC1", "CP1"],
        "Cz": ["C1", "C2", "FCz", "CPz"],
        "C2": ["Cz", "C4", "FC2", "CP2"],
        "C4": ["C2", "C6", "FC4", "CP4"],
        "C6": ["C4", "CP4"],
        "CP3": ["C3", "CP1", "P1", "C5"],
        "CP1": ["CP3", "CPz", "C1", "P1"],
        "CPz": ["CP1", "CP2", "Cz", "Pz", "FCz"],
        "CP2": ["CPz", "CP4", "C2", "P2"],
        "CP4": ["CP2", "C4", "P2", "C6"],
        "P1": ["CP3", "CP1", "Pz", "POz"],
        "Pz": ["P1", "P2", "CPz", "POz"],
        "P2": ["CP2", "CP4", "Pz", "POz"],
        "POz": ["P1", "Pz", "P2"],
    }


def neighbors_to_index_list(
    *,
    all_names: List[str],
    keep_names: Optional[List[str]] = None,
    neighbors_by_name: Optional[Dict[str, List[str]]] = None,
) -> List[List[int]]:
    """Build neighbors list-of-lists aligned to the current channel order.

    If keep_names is provided, only those channels are assumed to exist and
    neighbors are dropped when missing.
    """
    if neighbors_by_name is None:
        neighbors_by_name = bci2a_default_laplacian_neighbors()

    current_names = keep_names if keep_names is not None else all_names
    idx = name_to_index(current_names)

    out: List[List[int]] = []
    for ch in current_names:
        nei_names = neighbors_by_name.get(ch, [])
        out.append([idx[n] for n in nei_names if n in idx and n != ch])
    return out
