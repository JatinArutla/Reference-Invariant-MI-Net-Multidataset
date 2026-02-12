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

Start with:
- `python gate_reference.py --help`

Status: active research repo supporting a draft paper# Reference-Invariant-MI-Net-updated
# Reference-Invariant-MI-Net-updated
