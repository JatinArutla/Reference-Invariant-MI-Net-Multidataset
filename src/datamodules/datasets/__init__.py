"""Multi-dataset wrappers.

These classes provide a uniform interface that returns numpy arrays in the repo's
canonical format:
  X: float32 [N, C, T]
  y: int64   [N]   (binary left vs right)

All datasets are projected onto CANON_CHS_18 in a fixed order.
"""

from .registry import get_dataset

__all__ = ["get_dataset"]
