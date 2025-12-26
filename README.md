# Reference-Invariant MI-Net

Benchmark for EEG motor-imagery decoding under reference mismatch (native, CAR, ref-to-channel, Laplacian).

Quick start:
- pip install -r requirements.txt
- python gate_reference.py --data_root "$DATA_ROOT" --results_dir ./results/gate_EAoff_train_native --train_ref_modes native --test_ref_modes native,car,ref,laplacian --no-ea --epochs 200 --seed 1
