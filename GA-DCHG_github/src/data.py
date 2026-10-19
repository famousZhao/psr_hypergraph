"""Loaders for the CWRU rolling-bearing dataset."""
import numpy as np
import scipy.io as sio

from config import D_CWRU, L_WINDOW, CACHE, SEED

# ---------------------------------------------------------------- CWRU
CWRU_CLASSES = ["Normal", "IR007", "IR014", "IR021",
                "B007", "B014", "B021",
                "OR007@6", "OR014@6", "OR021@6"]


def _cwru_files(tag):
    root = D_CWRU
    if tag == "Normal":
        return sorted((root / "Normal").glob("Normal_*.mat"))
    return sorted((root / "12k_DE").glob(f"{tag}_*.mat"))


def _cwru_signal(path):
    d = sio.loadmat(str(path))
    key = next(k for k in d if k.endswith("_DE_time"))
    return np.asarray(d[key], float).ravel()


def _windows(x, L):
    n = len(x) // L
    return x[:n * L].reshape(n, L)


def _bandpass_normalize(w, fs):
    w = w - w.mean(1, keepdims=True)
    return w


def load_cwru(per_class=120, L=L_WINDOW, seed=SEED, grouped=False):
    """Return (X, y, names[, fid]).  X is (n, L), y in 0..9."""
    rng = np.random.default_rng(seed)
    X, y, fid = [], [], []
    for c, tag in enumerate(CWRU_CLASSES):
        files = _cwru_files(tag)
        for fi, p in enumerate(files):
            w = _windows(_cwru_signal(p), L)
            if len(w) == 0:
                continue
            w = _bandpass_normalize(w, 12000.0)
            need = per_class // max(1, len(files)) + 1
            step = max(1, len(w) // need)
            w = w[::step]
            X.append(w)
            y.append(np.full(len(w), c, np.int64))
            fid.append(np.full(len(w), c * 100 + fi, np.int64))
    X = np.concatenate(X, 0); y = np.concatenate(y, 0); fid = np.concatenate(fid, 0)
    keep = []
    for c in range(len(CWRU_CLASSES)):
        idx = np.where(y == c)[0]
        keep.append(rng.choice(idx, min(per_class, len(idx)), replace=False))
    keep = np.sort(np.concatenate(keep))
    X, y, fid = X[keep], y[keep], fid[keep]
    if grouped:
        return X.astype(np.float32), y.astype(np.int64), list(CWRU_CLASSES), fid
    return X.astype(np.float32), y.astype(np.int64), list(CWRU_CLASSES)


def load(name, **kw):
    return load_cwru(**kw)


def stratified_split(y, seed=SEED, test_frac=0.20, val_frac=0.15):
    rng = np.random.default_rng(seed + 7)
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_te = int(round(n * test_frac))
        n_va = int(round(n * val_frac))
        te.append(idx[:n_te])
        va.append(idx[n_te:n_te + n_va])
        tr.append(idx[n_te + n_va:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(va)), np.sort(np.concatenate(te))


def group_split(y, fid, seed=SEED, test_frac=0.20, val_frac=0.15):
    rng = np.random.default_rng(seed + 11)
    tr, va, te = [], [], []
    for c in np.unique(y):
        files = np.unique(fid[y == c])
        rng.shuffle(files)
        n = len(files)
        n_te = max(1, int(round(n * test_frac)))
        n_va = max(1, int(round(n * val_frac))) if n >= 3 else 0
        te_f = set(files[:n_te].tolist())
        va_f = set(files[n_te:n_te + n_va].tolist())
        tr_f = set(files[n_te + n_va:].tolist())
        idx_c = np.where(y == c)[0]
        for i in idx_c:
            f = int(fid[i])
            if f in te_f:
                te.append(i)
            elif f in va_f:
                va.append(i)
            elif f in tr_f:
                tr.append(i)
    return np.sort(np.array(tr)), np.sort(np.array(va)), np.sort(np.array(te))


def add_awgn(x, snr_db, rng):
    x = np.asarray(x, float)
    p_sig = np.mean(x ** 2, axis=-1, keepdims=True)
    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    return (x + rng.normal(0.0, 1.0, size=x.shape) * np.sqrt(p_noise)).astype(np.float32)


if __name__ == "__main__":
    X, y, cn = load("cwru")
    print("cwru", X.shape, np.bincount(y), cn)
