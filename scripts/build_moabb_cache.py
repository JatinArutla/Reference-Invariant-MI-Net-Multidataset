"""Pre-download and cache MOABB datasets into numpy arrays.

Why:
  - MOABB download links can change or rate-limit.
  - Caching makes training deterministic and faster.

This writes per-subject:
  cache_root/<dataset>/subXXX/{X.npy,y.npy,meta.json}
"""

import argparse
import os

from src.datamodules.datasets import get_dataset
from src.datamodules.datasets.base import SplitSpec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["lee2019", "physionet"], help="MOABB dataset key")
    ap.add_argument("--cache_root", required=True)
    ap.add_argument("--n_subjects", type=int, default=0, help="0 = all")
    ap.add_argument("--tmin", type=float, default=0.0)
    ap.add_argument("--tmax", type=float, default=3.0)
    ap.add_argument("--resample_hz", type=float, default=160.0)
    ap.add_argument("--band_lo", type=float, default=8.0)
    ap.add_argument("--band_hi", type=float, default=32.0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.cache_root, exist_ok=True)

    ds = get_dataset(args.dataset, data_root=None)
    subs = ds.subject_list
    if args.n_subjects and args.n_subjects > 0:
        subs = list(subs)[: args.n_subjects]

    split = SplitSpec(mode="random", train_frac=0.8, seed=args.seed)
    for s in subs:
        print(f"Caching {args.dataset} subject {s}...")
        ds.load_subject_native(
            s,
            split=split,
            tmin=args.tmin,
            tmax=args.tmax,
            resample_hz=args.resample_hz,
            band=(args.band_lo, args.band_hi),
            cache_root=args.cache_root,
        )
    print("Done.")


if __name__ == "__main__":
    main()
