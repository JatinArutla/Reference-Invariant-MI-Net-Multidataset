import os, argparse, time, glob
os.environ["TF_DISABLE_LAYOUT_OPTIMIZER"] = "1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

tf.keras.backend.set_image_data_format("channels_last")
tf.config.optimizer.set_experimental_options({"layout_optimizer": False})
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from src.datamodules.bci2a import load_LOSO_pool, load_subject_dependent
from src.models.model import build_atcnet
from src.datamodules.transforms import standardize_instance

def set_seed(seed: int = 1):
    import random
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def _reshape_for_model(X: np.ndarray) -> np.ndarray:
    N, C, T = X.shape
    return X.reshape(N, 1, C, T).astype(np.float32, copy=False)

def _find_best_path(results_dir: str, subject: int) -> str:
    best_list = os.path.join(results_dir, "best_models.txt")
    if os.path.exists(best_list):
        with open(best_list) as f:
            lines = [l.strip() for l in f if l.strip()]
        for rel in lines:
            if rel.endswith(f"subject-{subject}.weights.h5"):
                p = os.path.join(results_dir, rel)
                if os.path.exists(p):
                    return p
    pattern = os.path.join(results_dir, "saved_models", "**", f"subject-{subject}.weights.h5")
    cands = sorted(glob.glob(pattern, recursive=True))
    return cands[-1] if cands else ""

def build_model(args) -> tf.keras.Model:
    return build_atcnet(
        n_classes=args.n_classes,
        in_chans=args.n_channels,
        in_samples=args.in_samples,
        n_windows=args.n_windows,
        attention=args.attention,
        eegn_F1=args.eegn_F1,
        eegn_D=args.eegn_D,
        eegn_kernel=args.eegn_kernel,
        eegn_pool=args.eegn_pool,
        eegn_dropout=args.eegn_dropout,
        tcn_depth=args.tcn_depth,
        tcn_kernel=args.tcn_kernel,
        tcn_filters=args.tcn_filters,
        tcn_dropout=args.tcn_dropout,
        tcn_activation=args.tcn_activation,
        fuse=args.fuse,
        from_logits=args.from_logits,
        return_ssl_feat=False,
    )

def run(args):
    set_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)
    log_path = os.path.join(args.results_dir, "log.txt")
    log = open(log_path, "a")

    acc = np.zeros(args.n_sub)
    kappa = np.zeros(args.n_sub)
    cms = np.zeros((args.n_sub, args.n_classes, args.n_classes))
    inf_ms = np.zeros(args.n_sub)

    std_mode = (getattr(args, "standardize_mode", None) or ("train" if args.standardize else "none")).lower()
    if std_mode not in ("train", "instance", "none"):
        raise ValueError("standardize_mode must be one of: train, instance, none")

    for sub in range(1, args.n_sub + 1):
        if args.loso:
            (_, _), (X_tgt, y_tgt) = load_LOSO_pool(
                args.data_root, sub,
                n_sub=args.n_sub, ea=args.ea,
                standardize=(args.standardize if std_mode == "train" else False),
                per_block_standardize=(args.per_block_standardize if std_mode == "train" else False),
                t1_sec=args.t1_sec, t2_sec=args.t2_sec,
                ref_mode=args.data_ref_mode,
                keep_channels=args.keep_channels,
                ref_channel=args.ref_channel,
                laplacian=args.laplacian,
            )
        else:
            (_, _), (X_tgt, y_tgt) = load_subject_dependent(
                args.data_root, sub,
                ea=args.ea,
                standardize=(args.standardize if std_mode == "train" else False),
                t1_sec=args.t1_sec, t2_sec=args.t2_sec,
                ref_mode=args.data_ref_mode,
                keep_channels=args.keep_channels,
                ref_channel=args.ref_channel,
                laplacian=args.laplacian,
            )

        if std_mode == "instance":
            X_tgt = standardize_instance(X_tgt, robust=bool(getattr(args, "instance_robust", False)))

        # If channel subsetting is active, update n_channels dynamically.
        args.n_channels = int(X_tgt.shape[1])

        X = _reshape_for_model(X_tgt)
        model = build_model(args)
        wpath = _find_best_path(args.results_dir, sub)
        if not wpath:
            raise FileNotFoundError(f"No weights found for subject {sub}")
        model.load_weights(wpath)

        t0 = time.time()
        y_pred = model.predict(X, verbose=0)
        dt = (time.time() - t0) / X.shape[0] * 1000.0
        inf_ms[sub - 1] = dt

        y_hat = y_pred.argmax(-1) if not args.from_logits else tf.nn.softmax(y_pred).numpy().argmax(-1)
        y_true = y_tgt.astype(int)

        acc[sub - 1] = accuracy_score(y_true, y_hat)
        kappa[sub - 1] = cohen_kappa_score(y_true, y_hat)
        cms[sub - 1] = confusion_matrix(y_true, y_hat, labels=np.arange(args.n_classes), normalize="true")

        msg = f"Subject {sub:02d} ({'LOSO' if args.loso else 'subject'}): acc={acc[sub-1]:.4f}  kappa={kappa[sub-1]:.3f}  inf={dt:.2f} ms/trial"
        print(msg); log.write(msg + "\n")

    hdr1 = "                  " + "".join([f"sub_{i:02d}   " for i in range(1, args.n_sub+1)]) + "  average"
    hdr2 = "                  " + "".join(["-----   " for _ in range(args.n_sub)]) + "  -------"
    out = "\n---------------------------------\nTest performance (acc & kappa):\n"
    out += "---------------------------------\n" + hdr1 + "\n" + hdr2
    out += "\n(acc %)   " + "".join([f"{a*100:6.2f}   " for a in acc]) + f"  {acc.mean()*100:6.2f}"
    out += "\n(kappa)   " + "".join([f"{k:6.3f}   " for k in kappa]) + f"  {kappa.mean():6.3f}"
    out += f"\n(inf ms)  " + "".join([f"{m:6.2f}   " for m in inf_ms]) + f"  {inf_ms.mean():6.2f}"
    out += "\n---------------------------------\n"
    print(out); log.write(out + "\n")

    if args.plots:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(); ax.bar(range(1, args.n_sub+1), acc); ax.set_ylim(0,1); ax.set_title("Accuracy"); ax.set_xlabel("Subject"); plt.show()
        fig, ax = plt.subplots(); ax.bar(range(1, args.n_sub+1), kappa); ax.set_ylim(0,1); ax.set_title("Kappa"); ax.set_xlabel("Subject"); plt.show()
        cm_avg = cms.mean(0)
        fig, ax = plt.subplots(); im = ax.imshow(cm_avg, interpolation="nearest"); ax.set_title("Confusion Matrix (avg)"); plt.colorbar(im); ax.set_xlabel("Pred"); ax.set_ylabel("True"); plt.show()

    log.close()

def parse_args():
    p = argparse.ArgumentParser("ATCNet evaluation")
    # data
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--results_dir", type=str, default="./results")
    p.add_argument("--n_sub", type=int, default=9)
    p.add_argument("--n_classes", type=int, default=4)
    p.add_argument("--n_channels", type=int, default=22)
    p.add_argument("--in_samples", type=int, default=1000)
    p.add_argument("--t1_sec", type=float, default=2.0)
    p.add_argument("--t2_sec", type=float, default=6.0)

    # reference / channel control (must match training)
    p.add_argument(
        "--data_ref_mode",
        type=str,
        default="native",
        choices=[
            "native", "car", "ref", "laplacian",
            "bipolar", "bipolar_edges",
            "gs", "median",
            "randref",
        ],
        help="Reference mode applied to loaded data before EA/standardization.",
    )
    p.add_argument(
        "--keep_channels",
        type=str,
        default="",
        help="Comma-separated channel names to keep (intersection baseline).",
    )
    p.add_argument(
        "--ref_channel",
        type=str,
        default="Cz",
        help="Recorded channel name used when data_ref_mode='ref' (default Cz).",
    )
    p.add_argument(
        "--laplacian",
        action="store_true",
        help="Also build Laplacian neighbors (needed when data_ref_mode='laplacian').",
    )
    p.add_argument("--ea", action="store_true"); p.add_argument("--no-ea", dest="ea", action="store_false"); p.set_defaults(ea=True)
    p.add_argument("--standardize", action="store_true"); p.add_argument("--no-standardize", dest="standardize", action="store_false"); p.set_defaults(standardize=True)
    p.add_argument(
        "--standardize_mode",
        type=str,
        default=None,
        choices=["train", "instance", "none"],
        help="Override standardization behavior. 'train' uses train-fitted z-score (default). 'instance' standardizes each trial/channel over time. 'none' disables standardization.",
    )
    p.add_argument(
        "--instance_robust",
        action="store_true",
        help="Use median/MAD for instance standardization (only when --standardize_mode=instance).",
    )
    p.add_argument("--per_block_standardize", action="store_true"); p.add_argument("--no-per_block_standardize", dest="per_block_standardize", action="store_false"); p.set_defaults(per_block_standardize=True)
    p.add_argument("--loso", action="store_true"); p.add_argument("--no-loso", dest="loso", action="store_false"); p.set_defaults(loso=True)

    # model
    p.add_argument("--n_windows", type=int, default=5)
    p.add_argument("--attention", type=str, default="mha", choices=["mha", "mhla", "none", ""])
    p.add_argument("--eegn_F1", type=int, default=16)
    p.add_argument("--eegn_D", type=int, default=2)
    p.add_argument("--eegn_kernel", type=int, default=64)
    p.add_argument("--eegn_pool", type=int, default=7)
    p.add_argument("--eegn_dropout", type=float, default=0.3)
    p.add_argument("--tcn_depth", type=int, default=2)
    p.add_argument("--tcn_kernel", type=int, default=4)
    p.add_argument("--tcn_filters", type=int, default=32)
    p.add_argument("--tcn_dropout", type=float, default=0.3)
    p.add_argument("--tcn_activation", type=str, default="elu")
    p.add_argument("--fuse", type=str, default="average", choices=["average", "concat"])
    p.add_argument("--from_logits", action="store_true")

    # misc
    p.add_argument("--plots", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()

def main():
    args = parse_args()
    run(args)

if __name__ == "__main__":
    main()