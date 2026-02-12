#!/usr/bin/env python
"""Linear-operator geometry for reference transforms.

Runs once per reference suite and channel set.
No training run and no data required.

This script is intentionally strict about what it calls an "operator":
- CAR, ref-to-channel, Laplacian, bipolar, bipolar_edges, randref are fixed linear maps in channel space.
- Median reference is nonlinear.
- Gram-Schmidt in this repo is data-adaptive (alpha depends on inner products within each trial),
  so it is not representable as a single fixed matrix A that you can analyze once.

Those non-fixed modes are skipped with an explanation.

Outputs:
- <out_dir>/operators.json
- <out_dir>/rowspace_similarity.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.datamodules.channels import (
    BCI2A_CH_NAMES,
    parse_keep_channels,
    neighbors_to_index_list,
    neighbors_to_edge_list,
    name_to_index,
)


def _car_matrix(C: int) -> np.ndarray:
    I = np.eye(C, dtype=np.float64)
    one = np.ones((C, 1), dtype=np.float64)
    return I - (1.0 / C) * (one @ one.T)


def _ref_to_channel_matrix(C: int, ref_idx: int) -> np.ndarray:
    I = np.eye(C, dtype=np.float64)
    one = np.ones((C, 1), dtype=np.float64)
    e = np.zeros((C, 1), dtype=np.float64)
    e[ref_idx, 0] = 1.0
    return I - (one @ e.T)


def _laplacian_matrix(neighbors: List[List[int]], C: int) -> np.ndarray:
    A = np.eye(C, dtype=np.float64)
    for i in range(C):
        nb = neighbors[i] if neighbors is not None else []
        if not nb:
            continue
        w = 1.0 / float(len(nb))
        for j in nb:
            A[i, j] -= w
    return A


def _bipolar_matrix(neighbors: List[List[int]], C: int) -> np.ndarray:
    A = np.eye(C, dtype=np.float64)
    for i in range(C):
        nb = neighbors[i] if neighbors is not None else []
        if not nb:
            continue
        j = nb[0]
        A[i, j] -= 1.0
    return A


def _bipolar_edges_matrix(edges: List[Tuple[int, int]], C: int) -> np.ndarray:
    E = len(edges)
    A = np.zeros((E, C), dtype=np.float64)
    for k, (i, j) in enumerate(edges):
        A[k, i] = 1.0
        A[k, j] = -1.0
    return A


def _randref_matrix(C: int, *, seed: int = 0, alpha: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    w = rng.dirichlet(alpha=np.full((C,), float(alpha), dtype=np.float64))
    I = np.eye(C, dtype=np.float64)
    one = np.ones((C, 1), dtype=np.float64)
    return I - (one @ w.reshape(1, -1))


def build_operator(
    mode: str,
    *,
    C: int,
    ref_idx: int | None,
    lap_neighbors: List[List[int]] | None,
    bip_neighbors: List[List[int]] | None,
    edges: List[Tuple[int, int]] | None,
    randref_seed: int,
    randref_alpha: float,
) -> Tuple[np.ndarray | None, str]:
    m = (mode or "native").lower()

    if m in ("native", "none", ""):
        return np.eye(C, dtype=np.float64), "ok"
    if m in ("car", "car_full", "car_intersection"):
        return _car_matrix(C), "ok"
    if m in ("ref", "cz_ref", "channel_ref"):
        if ref_idx is None:
            return None, "skip: ref_idx is required for mode='ref'"
        return _ref_to_channel_matrix(C, ref_idx), "ok"
    if m in ("laplacian", "lap", "local"):
        if lap_neighbors is None:
            return None, "skip: lap_neighbors is required for laplacian"
        return _laplacian_matrix(lap_neighbors, C), "ok"
    if m in ("bipolar", "bip", "bipolar_like"):
        if bip_neighbors is None:
            return None, "skip: bip_neighbors is required for bipolar"
        return _bipolar_matrix(bip_neighbors, C), "ok"
    if m in ("bipolar_edges", "bip_edges", "edges_bipolar"):
        if edges is None:
            return None, "skip: edges are required for bipolar_edges"
        return _bipolar_edges_matrix(edges, C), "ok"
    if m in ("randref", "random_ref", "random_global_ref"):
        return _randref_matrix(C, seed=randref_seed, alpha=randref_alpha), "ok"

    if m in ("median", "median_ref", "median_reference"):
        return None, "skip: median reference is nonlinear (not a fixed matrix A)"
    if m in ("gs", "gram_schmidt", "gram-schmidt"):
        return None, "skip: Gram-Schmidt here is data-adaptive (A depends on x)"
    return None, f"skip: unknown mode '{mode}'"


@dataclass
class OperatorInfo:
    mode: str
    A: np.ndarray
    rank: int
    in_dim: int
    out_dim: int
    null_dim: int
    fro_norm: float
    idempotence_rel: float | None
    symm_rel: float | None
    rowspace_basis: np.ndarray
    singular_values: List[float]


def _svd_rank(A: np.ndarray, *, eps: float = 1e-10) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    tol = float(eps) * max(A.shape) * float(s[0] if s.size else 1.0)
    r = int(np.sum(s > tol))
    return r, U, s, Vh


def _rowspace_basis_from_svd(Vh: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return np.zeros((Vh.shape[1], 0), dtype=np.float64)
    return Vh[:r, :].T.copy()


def _principal_angle_stats(Q1: np.ndarray, Q2: np.ndarray) -> Dict[str, float]:
    if Q1.size == 0 or Q2.size == 0:
        return {"mean_sv": 0.0, "min_sv": 0.0, "max_sv": 0.0}
    M = Q1.T @ Q2
    sv = np.linalg.svd(M, compute_uv=False)
    sv = np.clip(sv, 0.0, 1.0)
    return {"mean_sv": float(np.mean(sv)), "min_sv": float(np.min(sv)), "max_sv": float(np.max(sv))}


def _idempotence_rel(A: np.ndarray) -> float | None:
    if A.shape[0] != A.shape[1]:
        return None
    denom = float(np.linalg.norm(A, ord="fro")) + 1e-12
    return float(np.linalg.norm(A @ A - A, ord="fro") / denom)


def _symmetry_rel(A: np.ndarray) -> float | None:
    if A.shape[0] != A.shape[1]:
        return None
    denom = float(np.linalg.norm(A, ord="fro")) + 1e-12
    return float(np.linalg.norm(A - A.T, ord="fro") / denom)


def main():
    p = argparse.ArgumentParser("Operator geometry for reference transforms")
    p.add_argument("--ref_modes", type=str, required=True, help="Comma-separated modes to analyze")
    p.add_argument("--keep_channels", type=str, default="", help="Preset name or comma-separated channel names")
    p.add_argument("--ref_channel", type=str, default="Cz", help="Reference channel name for mode='ref'")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--randref_seed", type=int, default=0)
    p.add_argument("--randref_alpha", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=1e-10, help="SVD tolerance scale")
    args = p.parse_args()

    modes = [m.strip() for m in args.ref_modes.split(",") if m.strip()]
    if not modes:
        raise ValueError("--ref_modes must be non-empty")

    keep_idx = parse_keep_channels(args.keep_channels, all_names=BCI2A_CH_NAMES)
    keep_names = [BCI2A_CH_NAMES[i] for i in keep_idx] if keep_idx is not None else list(BCI2A_CH_NAMES)
    C = len(keep_names)

    ref_idx = None
    if any((m or "").lower() in ("ref", "cz_ref", "channel_ref") for m in modes):
        mp = name_to_index(keep_names)
        if args.ref_channel in mp:
            ref_idx = int(mp[args.ref_channel])

    need_lap = any((m or "").lower() in ("laplacian", "lap", "local") for m in modes)
    need_bip = any((m or "").lower() in ("bipolar", "bip", "bipolar_like", "bipolar_edges", "bip_edges", "edges_bipolar") for m in modes)

    lap_neighbors = None
    bip_neighbors = None
    edges = None

    if need_lap or need_bip:
        lap_neighbors = neighbors_to_index_list(all_names=BCI2A_CH_NAMES, keep_names=keep_names, sort_by_distance=False)
    if need_bip:
        bip_neighbors = neighbors_to_index_list(all_names=BCI2A_CH_NAMES, keep_names=keep_names, sort_by_distance=True)
        edges = neighbors_to_edge_list(all_names=BCI2A_CH_NAMES, keep_names=keep_names, sort_by_distance=True)

    infos: Dict[str, OperatorInfo] = {}
    skipped: Dict[str, str] = {}

    for m in modes:
        A, status = build_operator(
            m,
            C=C,
            ref_idx=ref_idx,
            lap_neighbors=lap_neighbors,
            bip_neighbors=bip_neighbors,
            edges=edges,
            randref_seed=int(args.randref_seed),
            randref_alpha=float(args.randref_alpha),
        )
        if A is None:
            skipped[m] = status
            continue

        r, U, s, Vh = _svd_rank(A, eps=float(args.eps))
        Q = _rowspace_basis_from_svd(Vh, r)

        infos[m] = OperatorInfo(
            mode=m,
            A=A,
            rank=int(r),
            in_dim=int(A.shape[1]),
            out_dim=int(A.shape[0]),
            null_dim=int(A.shape[1] - r),
            fro_norm=float(np.linalg.norm(A, ord="fro")),
            idempotence_rel=_idempotence_rel(A),
            symm_rel=_symmetry_rel(A),
            rowspace_basis=Q,
            singular_values=[float(x) for x in s.tolist()],
        )

    os.makedirs(args.out_dir, exist_ok=True)

    operators_out = {
        "meta": {
            "ref_modes_requested": modes,
            "ref_modes_analyzed": sorted(list(infos.keys())),
            "ref_modes_skipped": skipped,
            "keep_channels": args.keep_channels,
            "channels": keep_names,
            "ref_channel": args.ref_channel,
            "ref_idx": ref_idx,
            "randref_seed": int(args.randref_seed),
            "randref_alpha": float(args.randref_alpha),
            "eps": float(args.eps),
        },
        "operators": {},
    }

    for m, info in infos.items():
        operators_out["operators"][m] = {
            "in_dim": info.in_dim,
            "out_dim": info.out_dim,
            "rank": info.rank,
            "null_dim": info.null_dim,
            "fro_norm": info.fro_norm,
            "idempotence_rel": info.idempotence_rel,
            "symmetry_rel": info.symm_rel,
            "singular_values": info.singular_values,
            "A": info.A.tolist(),
        }

    with open(os.path.join(args.out_dir, "operators.json"), "w") as f:
        json.dump(operators_out, f, indent=2)

    modes_ok = sorted(list(infos.keys()))
    sim = {a: {} for a in modes_ok}
    for a in modes_ok:
        Qa = infos[a].rowspace_basis
        for b in modes_ok:
            Qb = infos[b].rowspace_basis
            sim[a][b] = _principal_angle_stats(Qa, Qb)

    with open(os.path.join(args.out_dir, "rowspace_similarity.json"), "w") as f:
        json.dump(
            {
                "meta": {
                    "modes": modes_ok,
                    "similarity_definition": "principal-angle singular values between row-space bases in input channel space",
                },
                "rowspace_similarity": sim,
            },
            f,
            indent=2,
        )

    print(f"[operator_geometry] wrote: {os.path.join(args.out_dir, 'operators.json')}")
    print(f"[operator_geometry] wrote: {os.path.join(args.out_dir, 'rowspace_similarity.json')}")
    if skipped:
        print("[operator_geometry] skipped modes:")
        for k, v in skipped.items():
            print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()