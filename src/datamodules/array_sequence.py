import numpy as np
from tensorflow.keras.utils import Sequence


class ArraySequence(Sequence):
    """Mini-batch wrapper for numpy arrays.

    Keeps shuffling repeatable across epochs for reproducible runs.
    Expects X as [N,C,T] or [N,1,C,T]. Output X is always [B,1,C,T].
    """

    def __init__(self, X, y, *, batch_size: int, shuffle: bool, seed: int):
        super().__init__()
        self.X = X.astype(np.float32, copy=False)
        self.y = y
        self.bs = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.rng = np.random.default_rng(self.seed)

        self.idx = np.arange(len(self.y), dtype=np.int64)
        if self.shuffle:
            self.rng.shuffle(self.idx)

    def __len__(self):
        return int(np.ceil(len(self.idx) / self.bs))

    def on_epoch_end(self):
        self.epoch += 1
        if self.shuffle:
            # Re-seed per epoch so shuffling is repeatable.
            self.rng = np.random.default_rng(self.seed + self.epoch)
            self.rng.shuffle(self.idx)

    def __getitem__(self, k):
        b = self.idx[k * self.bs : (k + 1) * self.bs]
        Xb = self.X[b]
        yb = self.y[b]

        if Xb.ndim == 3:
            Xb = Xb[:, None, :, :]  # [B,C,T] -> [B,1,C,T]
        return Xb, yb
