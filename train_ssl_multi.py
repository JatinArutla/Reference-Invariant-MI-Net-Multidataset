"""Self-supervised pretraining for multi-dataset setup.

This is a thin adaptation of the single-dataset train_ssl.py to work with the
multi-dataset loaders used by gate_reference_multi.py.

Output:
  out_dir/<dataset>/subXXX/weights.h5

Those weights can be used as init for supervised training via:
  gate_reference_multi.py --ssl_weights "<out_dir>/<dataset>/sub{sub:03d}/weights.h5"
"""

from __future__ import annotations

import argparse
import os
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from src.models.model import build_atcnet
from src.datamodules.datasets import get_dataset
from src.datamodules.datasets.base import SplitSpec
from src.datamodules.channels import CANON_CHS_18, BCI2A_CH_NAMES
from src.datamodules.transforms import apply_reference
from src.datamodules.neighbors import neighbors_to_index_list
from src.selfsupervised.views import make_ssl_dataset
from src.selfsupervised.losses import vicreg_loss, barlow_twins_loss, ntxent_loss


def set_seed(seed: int = 1):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _reshape_for_model(X: np.ndarray) -> np.ndarray:
    # [N,C,T] -> [N,1,C,T]
    N, C, T = X.shape
    return X.reshape(N, 1, C, T).astype(np.float32, copy=False)


def _compute_lap_neighbors() -> list[list[int]]:
    return neighbors_to_index_list(
        all_names=BCI2A_CH_NAMES,
        keep_names=list(CANON_CHS_18),
        sort_by_distance=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["iv2a", "openbmi_local", "physionet_local", "lee2019", "physionet"])
    ap.add_argument("--data_root", default="")
    ap.add_argument("--cache_root", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--split_mode", default="random", choices=["random", "session"])
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=1)

    # Preprocessing contract
    ap.add_argument("--tmin", type=float, default=0.0)
    ap.add_argument("--tmax", type=float, default=3.0)
    ap.add_argument("--resample_hz", type=float, default=160.0)
    ap.add_argument("--band_lo", type=float, default=8.0)
    ap.add_argument("--band_hi", type=float, default=32.0)

    # SSL views
    ap.add_argument("--view_mode", default="ref+aug", choices=["ref", "aug", "ref+aug"])
    ap.add_argument("--view_ref_modes", default="native,car,laplacian,bipolar,gs,median")
    ap.add_argument("--ref_channel", default="Cz")
    ap.add_argument("--laplacian", action="store_true")
    ap.add_argument("--aug_policy", default="light", choices=["none", "light"])

    # Training
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ssl_loss", default="vicreg", choices=["vicreg", "barlow", "ntxent"])
    args = ap.parse_args()

    set_seed(args.seed)

    split = SplitSpec(mode=args.split_mode, train_frac=args.train_frac, seed=args.seed)
    ds = get_dataset(args.dataset, data_root=args.data_root or None)

    (X_tr, _y_tr), _ = ds.load_subject_native(
        args.subject,
        split=split,
        tmin=args.tmin,
        tmax=args.tmax,
        resample_hz=args.resample_hz,
        band=(args.band_lo, args.band_hi),
        cache_root=args.cache_root or None,
    )

    # SSL uses unlabeled train set
    ref_idx = list(CANON_CHS_18).index(args.ref_channel) if args.ref_channel in CANON_CHS_18 else 0
    lap_neighbors = _compute_lap_neighbors() if args.laplacian else None

    view_modes = [m.strip() for m in args.view_ref_modes.split(",") if m.strip()]

    ssl_ds = make_ssl_dataset(
        X_tr,
        view_mode=args.view_mode,
        view_ref_modes=view_modes,
        ref_idx=ref_idx,
        lap_neighbors=lap_neighbors,
        aug_policy=args.aug_policy,
        batch_size=args.batch,
        seed=args.seed,
    )

    # Encoder returning SSL features
    encoder = build_atcnet(
        n_classes=2,
        in_chans=len(CANON_CHS_18),
        in_samples=X_tr.shape[-1],
        n_windows=5,
        attention="mha",
        eegn_F1=16,
        eegn_D=2,
        eegn_kernel=64,
        eegn_pool=8,
        eegn_dropout=0.3,
        tcn_depth=2,
        tcn_kernel=8,
        tcn_filters=32,
        tcn_dropout=0.3,
        tcn_activation="elu",
        fuse="avg",
        from_logits=False,
        return_ssl_feat=True,
    )

    # Simple projection head
    feat_dim = int(encoder.output_shape[-1])
    proj = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation=None),
        ],
        name="projector",
    )

    opt = Adam(args.lr)

    if args.ssl_loss == "vicreg":
        loss_fn = lambda a, b: vicreg_loss(a, b)
    elif args.ssl_loss == "barlow":
        loss_fn = lambda a, b: barlow_twins_loss(a, b)
    elif args.ssl_loss == "ntxent":
        loss_fn = lambda a, b: ntxent_loss(a, b)
    else:
        raise ValueError(f"Unknown ssl_loss: {args.ssl_loss}")

    out_dir = os.path.join(args.out_dir, args.dataset, f"sub{args.subject:03d}")
    os.makedirs(out_dir, exist_ok=True)
    wpath = os.path.join(out_dir, "weights.h5")

    @tf.function
    def train_step(v1, v2):
        with tf.GradientTape() as tape:
            z1 = encoder(v1, training=True)
            z2 = encoder(v2, training=True)
            p1 = proj(z1, training=True)
            p2 = proj(z2, training=True)
            loss = loss_fn(p1, p2)
        vars_ = encoder.trainable_variables + proj.trainable_variables
        grads = tape.gradient(loss, vars_)
        opt.apply_gradients(zip(grads, vars_))
        return loss

    for ep in range(1, args.epochs + 1):
        losses = []
        for (v1, v2) in ssl_ds:
            losses.append(float(train_step(v1, v2).numpy()))
        print(f"epoch {ep:03d}/{args.epochs} | loss={np.mean(losses):.4f}")

    # Save weights (encoder+proj) so by_name loading can initialize shared layers.
    # We save from a temporary model that includes both parts.
    inp = tf.keras.Input(shape=(1, len(CANON_CHS_18), X_tr.shape[-1]))
    tmp = tf.keras.Model(inp, proj(encoder(inp)))
    tmp.save_weights(wpath)

    meta = {
        "dataset": args.dataset,
        "subject": args.subject,
        "view_mode": args.view_mode,
        "view_ref_modes": view_modes,
        "band": [args.band_lo, args.band_hi],
        "resample_hz": args.resample_hz,
        "t": [args.tmin, args.tmax],
        "ssl_loss": args.ssl_loss,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved:", wpath)


if __name__ == "__main__":
    main()
