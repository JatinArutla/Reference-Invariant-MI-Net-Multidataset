# Reference-Invariant MI-Net

This repo studies a practical failure mode in EEG motor imagery (MI) decoding:

**A model can perform well when train and test use the same EEG reference, but degrade sharply when the test data is referenced differently, even though the task and subject are unchanged.**

## Why this matters
EEG is always recorded relative to a reference, and reference choices vary across labs, hardware, and preprocessing pipelines. That creates a silent distribution shift at deployment time. This project measures how big that shift can be and which simple training strategies make MI decoders robust to it.

## What this repo does
- **Benchmark**: train under one reference mode, evaluate under multiple reference modes (train × test matrix).
- **Baselines**: simple supervised strategies (especially reference-jitter training) that reduce mismatch without changing the backbone.
- **SSL (optional)**: tests whether self-supervised pretraining helps beyond strong supervised baselines, mainly under low-label settings.

## High-level setup
- Dataset/protocol: within-subject, cross-session MI (BCI IV-2a style).
- Backbone: fixed strong MI model (kept stable to isolate the effect of referencing).
- Reference modes include: native, CAR, monopolar re-reference, Laplacian-like (optional: bipolar, Gram–Schmidt style).

Note: monopolar re-referencing can create a constant-zero channel for the reference electrode; an ablation is included to drop this channel post-transform and keep dimensions consistent across modes.

## Where to look
- `gate_reference.py` : main benchmark + supervised baselines
- `src/datamodules/` : reference transforms + data loading
- `train_ssl.py`, `finetune.py` : optional SSL workflows

## Analysis utilities

These scripts are for mechanistic checks and sanity audits. They do not change training.

1) Operator geometry (linear structure)

```bash
python analysis/operator_geometry.py \
  --ref_modes native,car,laplacian,bipolar,gs,median \
  --keep_channels CANON_CHS_18 \
  --out_dir /kaggle/working/analysis/operator_geom
```

This writes a JSON summary and prints key diagnostics (rank, idempotence, principal angles).
It also reports a least-squares *linear* approximation error for non-linear refs (median, gs).

2) Feature shift audit (spectral + spatial summaries)

```bash
python analysis/feature_shift_audit.py \
  --data_root "$DATA_ROOT" \
  --subject 1 \
  --ref_modes native,car,laplacian,bipolar,gs,median \
  --standardize_mode none \
  --keep_channels CANON_CHS_18 \
  --split test \
  --out_dir /kaggle/working/analysis/feature_shift_sub01
```

This outputs per-mode feature arrays and a pairwise distance matrix over modes.

3) Embedding drift (post-hoc representation stability)

```bash
python analysis/embedding_drift.py \
  --data_root "$DATA_ROOT" \
  --subject 1 \
  --ref_modes native,car,laplacian,bipolar,gs,median \
  --standardize_mode instance \
  --results_dir "$RESULTS_ROOT/jittermix_stdtrain" \
  --which best \
  --keep_channels CANON_CHS_18 \
  --out_dir /kaggle/working/analysis/embedding_drift_sub01
```

This loads a checkpoint, extracts `ssl_feat` embeddings, and writes pairwise cosine
similarity statistics across reference modes.

Start with:
- `python gate_reference.py --help`

## Multi-dataset extension (MOABB + manual IV-2a)

New entrypoint: `gate_reference_multi.py`

Design goal: keep *one* fixed cross-dataset input contract so we can attribute failures to
referencing rather than changing montages or preprocessing.

Default contract:
- Channels: `CANON_CHS_18` (18-channel intersection montage)
- Window: 0–3 s post-cue
- Band: 8–32 Hz
- Resample: 160 Hz
- Task: binary left vs right

Supported datasets:
- `iv2a` (BCI IV-2a) from local `.mat` files (your current setup)
- `lee2019` (OpenBMI) via MOABB
- `physionet` (EEGMMIDB) via MOABB

### Caching MOABB datasets (recommended)

```bash
python scripts/build_moabb_cache.py --dataset lee2019 --cache_root ./cache_moabb
python scripts/build_moabb_cache.py --dataset physionet --cache_root ./cache_moabb
```

### Example: fixed-train vs train×test ref matrix

```bash
python gate_reference_multi.py \
  --dataset lee2019 \
  --cache_root ./cache_moabb \
  --subject 1 \
  --train_strategy fixed \
  --train_mode native \
  --holdout_family none
```

### Example: leave-one-reference-family-out (unseen-family generalization)

Train with jitter over all refs except the held-out family, then evaluate on *all* modes.

```bash
python gate_reference_multi.py \
  --dataset physionet \
  --cache_root ./cache_moabb \
  --subject 1 \
  --train_strategy jitter \
  --holdout_family local \
  --label_frac 0.25
```

This is the key “high-impact” experiment in the multi-dataset story.

Status: active research repo supporting a draft paper# Reference-Invariant-MI-Net-Multidataset
# Reference-Invariant-MI-Net-Multidataset
