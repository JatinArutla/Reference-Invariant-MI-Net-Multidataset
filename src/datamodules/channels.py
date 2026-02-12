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

# Approximate 2D electrode positions for BCI IV-2a channels.
# These are only used to sort neighbor lists by proximity (for bipolar_nn).
# Exact coordinates are not critical; relative layout is.
BCI2A_CH_POS: Dict[str, Tuple[float, float]] = {
    "Fz": (0.0, 2.0),
    "FC3": (-1.0, 1.0),
    "FC1": (-0.5, 1.0),
    "FCz": (0.0, 1.0),
    "FC2": (0.5, 1.0),
    "FC4": (1.0, 1.0),
    "C5": (-2.0, 0.0),
    "C3": (-1.0, 0.0),
    "C1": (-0.5, 0.0),
    "Cz": (0.0, 0.0),
    "C2": (0.5, 0.0),
    "C4": (1.0, 0.0),
    "C6": (2.0, 0.0),
    "CP3": (-1.0, -1.0),
    "CP1": (-0.5, -1.0),
    "CPz": (0.0, -1.0),
    "CP2": (0.5, -1.0),
    "CP4": (1.0, -1.0),
    "P1": (-0.5, -2.0),
    "Pz": (0.0, -2.0),
    "P2": (0.5, -2.0),
    "POz": (0.0, -3.0),
}


# Canonical cross-dataset motor-imagery montage (intersection-style baseline).
#
# Rationale:
# - Many MI datasets in MOABB expose slightly different montages.
# - A fixed intersection montage is the simplest, most reproducible way to ensure
#   identical network input across datasets.
# - We intentionally omit channels that are often missing (e.g., FCz in Lee2019_MI
#   as exposed by MOABB), and we keep a compact motor strip.
CANON_CHS_18: List[str] = [
    "Fz",
    "FC3", "FC1", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "Pz",
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

    # Allow preset names so notebooks stay readable.
    key = s.strip().lower()
    if key in ("canon_chs_18", "canon18", "canon_18", "canon"):
        req = list(CANON_CHS_18)
    elif key in ("bci2a", "iv2a", "bci_iv_2a", "full", "all"):
        req = list(all_names)
    else:
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
    sort_by_distance: bool = False,
    positions: Optional[Dict[str, Tuple[float, float]]] = None,
) -> List[List[int]]:
    """Build neighbors list-of-lists aligned to the current channel order.

    If keep_names is provided, only those channels are assumed to exist and
    neighbors are dropped when missing.
    """
    if neighbors_by_name is None:
        neighbors_by_name = bci2a_default_laplacian_neighbors()

    current_names = keep_names if keep_names is not None else all_names
    idx = name_to_index(current_names)

    if positions is None:
        positions = BCI2A_CH_POS

    out: List[List[int]] = []
    for ch in current_names:
        nei_names = [n for n in neighbors_by_name.get(ch, []) if n in idx and n != ch]

        if sort_by_distance and (ch in positions):
            cx, cy = positions[ch]
            def _d2(nm: str) -> float:
                if nm not in positions:
                    return float("inf")
                nx, ny = positions[nm]
                return (cx - nx) ** 2 + (cy - ny) ** 2
            nei_names = sorted(nei_names, key=_d2)

        out.append([idx[n] for n in nei_names])
    return out


def neighbors_to_edge_list(
    *,
    all_names: List[str],
    keep_names: Optional[List[str]] = None,
    neighbors_by_name: Optional[Dict[str, List[str]]] = None,
) -> List[Tuple[int, int]]:
    """Build a deterministic undirected edge list from the neighbor graph.

    Returns a sorted list of (i, j) index pairs with i < j in the *current* channel order.
    """
    neigh = neighbors_to_index_list(
        all_names=all_names,
        keep_names=keep_names,
        neighbors_by_name=neighbors_by_name,
        sort_by_distance=False,
    )
    edges = set()
    for i, ns in enumerate(neigh):
        for j in ns:
            a, b = (i, int(j))
            if a == b:
                continue
            if a > b:
                a, b = b, a
            edges.add((a, b))
    return sorted(edges)
