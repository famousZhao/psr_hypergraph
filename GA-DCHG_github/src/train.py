"""Datasets, training loop, and evaluation shared by every experiment."""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import config as C
import data as D
import model as M

torch.set_num_threads(int(os.environ.get("THREADS", "8")))


# ---------------------------------------------------------------- datasets
class HyperDS(Dataset):
    def __init__(self, ds, construction, tag="", use_raw=False):
        z = np.load(C.CACHE / f"hg_{ds}{tag}_{construction}.npz")
        t = np.load(C.CACHE / f"trace_{ds}{tag}.npz")
        self.mem = z["mem"]
        self.mask = z["mask"]
        self.phi = z["phi"]
        self.gf = z["gfeat"]
        self.nodef = z["nodef"]
        m = C.PSR_M
        n, n_seg, S, _ = self.nodef.shape
        tr = t["trace"].reshape(n, C.N_SEG, C.SEG_LEN, m)
        # node features: state vector, its speed, its incidence degree
        self.x = np.concatenate([tr, self.nodef], -1).astype(np.float32)
        self.y = z["y"].astype(np.int64)
        self.fid = z["fid"]
        self.raw = None
        if use_raw:
            R = t["raw"].astype(np.float32)
            self.raw = (R - R.mean(1, keepdims=True)) / (R.std(1, keepdims=True) + 1e-8)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        items = [torch.from_numpy(self.x[i]), torch.from_numpy(self.mem[i].astype(np.int64)),
                 torch.from_numpy(self.mask[i]), torch.from_numpy(self.phi[i]),
                 torch.from_numpy(self.gf[i])]
        if self.raw is not None:
            items.append(torch.from_numpy(self.raw[i]))
        items.append(int(self.y[i]))
        return tuple(items)


class RawDS(Dataset):
    def __init__(self, ds, tag=""):
        t = np.load(C.CACHE / f"trace_{ds}{tag}.npz")
        X = t["raw"]
        X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
        self.X = X.astype(np.float32)
        self.y = t["y"].astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), int(self.y[i])


# ---------------------------------------------------------------- metrics
def metrics(y, p):
    from sklearn.metrics import accuracy_score, f1_score
    return dict(acc=float(accuracy_score(y, p)),
                f1=float(f1_score(y, p, average="macro")))


def n_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def get_split(ds, tag="", grouped=False):
    t = np.load(C.CACHE / f"trace_{ds}{tag}.npz")
    y = t["y"]
    names = [str(s) for s in t["names"]]
    if grouped:
        tr, va, te = D.group_split(y, t["fid"])
    else:
        tr, va, te = D.stratified_split(y)
    return tr, va, te, y, names


# ---------------------------------------------------------------- training
def fit(model, kind, ds, construction, tr_idx, va_idx, te_idx, seed=0,
        epochs=C.EPOCHS, lr=C.LR, bs=C.BATCH, tag="", verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if kind == "raw":
        dataset = RawDS(ds, tag)
    else:
        dataset = HyperDS(ds, construction, tag, use_raw=(kind == "hybrid"))
        # global-feature standardisation from the training split
        g = dataset.gf[tr_idx].astype(np.float32)
        M.set_global_stats(torch.tensor(g.mean(0)), torch.tensor(g.std(0) + 1e-6))
    sub = torch.utils.data.Subset(dataset, tr_idx)
    val = torch.utils.data.Subset(dataset, va_idx)
    te = torch.utils.data.Subset(dataset, te_idx)
    dl = DataLoader(sub, batch_size=bs, shuffle=True, drop_last=True)
    dvl = DataLoader(val, batch_size=128)
    dtl = DataLoader(te, batch_size=128)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.CrossEntropyLoss()
    best, best_state, wait = -1, None, 0
    t0 = time.time()
    hist = []
    for ep in range(epochs):
        model.train()
        for b in dl:
            opt.zero_grad()
            logit = _forward(model, kind, b)
            loss = lossf(logit, b[-1])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        if kind == "hyper" or kind == "hybrid":
            mv = _eval(model, dvl, kind)
            hist.append((ep, mv["acc"]))
            if mv["acc"] > best:
                best = mv["acc"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
    if best_state is not None:
        model.load_state_dict(best_state)
    train_time = time.time() - t0
    mt = _eval(model, dtl, kind)
    return dict(**mt, train_time=round(train_time, 2), params=n_params(model),
                epochs=ep + 1, hist=hist, y_true=mt["y"], y_pred=mt["pred"])


def _forward(model, kind, b):
    if kind == "raw":
        return model(b[0])[0]
    if kind == "hybrid":
        x, mem, mask, phi, gf, raw = b[:6]
        return model(x, mem, mask, phi, gf, raw)[0]
    x, mem, mask, phi, gf = b[:5]
    return model(x, mem, mask, phi, gf)[0]


@torch.no_grad()
def _eval(model, loader, kind):
    model.eval()
    P, Y = [], []
    for b in loader:
        logit = _forward(model, kind, b)
        P.append(logit.argmax(1).numpy())
        Y.append(b[-1].numpy())
    P = np.concatenate(P); Y = np.concatenate(Y)
    r = metrics(Y, P)
    r["pred"] = P.tolist()
    r["y"] = Y.tolist()
    return r


# ---------------------------------------------------------------- attention audit
@torch.no_grad()
def attention_phi_corr(model, ds, construction, idx, tag="", max_n=200):
    """Pearson correlation between attention weights and a hyperedge descriptor.

    For every hyperedge the attention mass received from its members is summed
    and correlated with phi(e).  Averaged over segments and samples.
    """
    model.eval()
    dataset = HyperDS(ds, construction, tag)
    g = dataset.gf[list(idx)].astype(np.float32)
    M.set_global_stats(torch.tensor(g.mean(0)), torch.tensor(g.std(0) + 1e-6))
    layers = model.local.layers
    if layers[-1].attention == "none":
        return None
    cors = [[] for _ in range(C.PHI_DIM)]
    for i in idx[:max_n]:
        x, mem, mask, phi, gf, _ = dataset[i]
        mem, mask, phi = mem[None], mask[None], phi[None]
        z = x[None]
        for l in layers[:-1]:
            z = l(z, mem, mask, phi)
        a = _edge_attention(layers[-1], z, mem, mask, phi)
        if a is None:
            return None
        # a: (S,M,K) attention on members; sum over members -> mass per (segment, edge)
        mass = a.sum(-1).numpy().reshape(-1)
        ph = phi[0].numpy().reshape(-1, C.PHI_DIM)
        for j in range(C.PHI_DIM):
            if ph[:, j].std() > 1e-9 and mass.std() > 1e-9:
                cors[j].append(float(np.corrcoef(mass, ph[:, j])[0, 1]))
    return [float(np.nanmean(c)) if c else float("nan") for c in cors]


def _edge_attention(layer, x, mem, mask, phi):
    """Recompute the attention tensor of a hypergraph layer from its input."""
    hx = layer.lin_x(x)
    B, S, V, h = hx.shape
    Mm, K = mem.shape[2], mem.shape[3]
    MF = Mm * K
    memf = mem.reshape(B, S, MF)
    m = mask.reshape(B, S, MF, 1).to(hx.dtype)
    hx_mem = torch.gather(hx, 2, memf.unsqueeze(-1).expand(B, S, MF, h))
    e = (hx_mem * m).reshape(B, S, Mm, K, h).sum(3) / \
        m.reshape(B, S, Mm, K, 1).sum(3).clamp(min=1.0)
    he = layer.lin_e(e)
    sv = layer.wa(hx).squeeze(-1)
    score = torch.gather(sv, 2, memf)
    score = score + layer.wb(he).squeeze(-1).unsqueeze(3).expand(B, S, Mm, K).reshape(B, S, MF)
    if layer.attention == "phys":
        pe = phi.unsqueeze(3).expand(B, S, Mm, K, phi.shape[-1])
        score = score + (pe * layer.beta).sum(-1).reshape(B, S, MF)
    score = score.masked_fill(~mask.reshape(B, S, MF), -1e9)
    gmax = torch.full((B, S, V), -1e9, dtype=score.dtype)
    gmax.scatter_reduce_(2, memf, score, reduce="amax", include_self=True)
    gmax = torch.nan_to_num(gmax, neginf=0.0)
    ex = torch.exp(score - torch.gather(gmax, 2, memf)) * mask.reshape(B, S, MF)
    gsum = torch.zeros(B, S, V, dtype=ex.dtype)
    gsum.scatter_add_(2, memf, ex)
    alpha = (ex / gsum.gather(2, memf).clamp(min=1e-9)).reshape(B, S, Mm, K)
    return alpha[0]
