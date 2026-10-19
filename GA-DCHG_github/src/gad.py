"""GA-DCHG: Geometry-Anchored Dynamic Cross-Domain Hypergraph.

Four components implemented on top of the phase-space hypergraph baseline
(psr.py / model.py / train.py):

  C1  Domain-invariance screening of the edge descriptors (kappa, chi, rho).
      The descriptor(s) with the smallest source-target distribution distance
      become the *geometric anchors*.
  C2  Anchor-guided dynamic hyperedge generation with temporal state
      propagation.  A per-segment anchor state is propagated across the four
      time-ordered segments (short-term memory of trajectory-geometry drift)
      and modulates the dynamical distance so that edges are formed where the
      local filling geometry is consistent with the tracked anchor state.  Only
      the anchor descriptor propagates; the non-anchor descriptors are rebuilt
      per window independently.
  C3  Geometric alignment loss.  A differentiable "effective anchor geometry"
      readout (attention-weighted mean of the anchor descriptor over hyperedges)
      is aligned between source and target by a 1-D Wasserstein term, a cheap
      (low-dimensional) and physically interpretable alignment that acts on the
      phase-space filling geometry rather than on a black-box embedding.
  C4  Cross-domain protocol (recording- and condition-disjoint) and a test-time
      intervention audit: perturbing the anchor descriptor at inference must
      change decisions more than perturbing a non-anchor descriptor.

Sklearn-free by design so it runs on the CPU torch environment.
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
from scipy.signal import decimate
from pathlib import Path
from torch.utils.data import DataLoader, Subset

import config as C
import psr

torch.set_num_threads(int(os.environ.get("THREADS", "6")))

# ---------------------------------------------------------------------------
# configuration for the cross-domain protocol
# ---------------------------------------------------------------------------
# CWRU conditions = load (hp) in {0,1,2,3}; all 10 classes exist at every load.
CWRU_LOADS = [0, 1, 2, 3]

# anchor generation
ETA_DEFAULT = 1.0              # anchor strength in the modulated distance
TAU_DEFAULT = 0.5              # temporal smoothing on the anchor state
LAM_ALIGN_DEFAULT = 0.5        # weight of the Wasserstein alignment term
ALIGN_EPOCH = 0                # alignment active from this epoch (0 = always)

# ---------------------------------------------------------------------------
# low-level signal readers (mirror data.py without depending on it)
# ---------------------------------------------------------------------------
def _windows(x, L):
    n = len(x) // L
    return x[:n * L].reshape(n, L)


def _norm(w):
    return w - w.mean(1, keepdims=True)


def cwru_signal(path):
    d = sio.loadmat(str(path))
    key = next(k for k in d if k.endswith("_DE_time"))
    return np.asarray(d[key], float).ravel()


# ---------------------------------------------------------------------------
# condition-aware loaders
# ---------------------------------------------------------------------------
def load_cond_cwru(loads, per_class=60, seed=C.SEED):
    """All 10 CWRU classes at the given loads. Returns X, y, fid, cond, names."""
    classes = list(D_CWRU_CLASSES())
    rng = np.random.default_rng(seed)
    root = Path(C.D_CWRU)
    X, y, fid, cond = [], [], [], []
    for load in loads:
        for c, tag in enumerate(classes):
            p = root / "Normal" / f"Normal_{load}.mat" if tag == "Normal" \
                else root / "12k_DE" / f"{tag}_{load}.mat"
            if not p.exists():
                continue
            w = _norm(_windows(cwru_signal(p), C.L_WINDOW))
            if len(w) == 0:
                continue
            idx = rng.choice(len(w), min(per_class, len(w)), replace=False)
            X.append(w[idx]); y.append(np.full(len(idx), c))
            fid.append(np.full(len(idx), c * 100 + load))
            cond.append(np.full(len(idx), float(load)))
    X = np.concatenate(X, 0).astype(np.float32)
    y = np.concatenate(y, 0).astype(np.int64)
    fid = np.concatenate(fid, 0).astype(np.int64)
    cond = np.concatenate(cond, 0).astype(np.float32)
    return X, y, fid, cond, classes


def D_CWRU_CLASSES():
    return ["Normal", "IR007", "IR014", "IR021",
            "B007", "B014", "B021", "OR007@6", "OR014@6", "OR021@6"]


def load_cond(ds, cond, **kw):
    return load_cond_cwru(cond, **kw)


# ---------------------------------------------------------------------------
# trace cache per condition
# ---------------------------------------------------------------------------
def build_trace_cache_cond(ds, cond, tag, per_class=60, seed=C.SEED):
    f = C.CACHE / f"trace_{ds}_{tag}.npz"
    if f.exists():
        return f
    X, y, fid, ld, names = load_cond(ds, cond, per_class=per_class, seed=seed)
    m, tau = C.PSR_M, C.PSR_TAU[ds]
    traces = np.stack([psr.build_trace(x, m, tau, C.N_TRACE) for x in X])
    np.savez_compressed(f, raw=X, y=y, fid=fid, cond=ld, trace=traces,
                        names=np.array(names))
    print("built", f, traces.shape, flush=True)
    return f


# ---------------------------------------------------------------------------
# anchor-modulated hyperedge generation with temporal state propagation (C2)
# ---------------------------------------------------------------------------
def _seg(trace):
    return trace.reshape(C.N_SEG, C.SEG_LEN, trace.shape[1])


def _gen_edges(st, lam, mode, prev_abar, anchor_idx=None, temporal=True):
    """Hyperedge generation for one segment.

    mode == "dyn"      : static k-NN dynamical neighbourhood (baseline).
    otherwise          : anchor-modulated distance with the descriptor given
                         by anchor_idx, and temporal propagation on the anchor
                         state when temporal is True.  Setting temporal=False
                         removes the cross-segment memory (noTemporal ablation);
                         setting anchor_idx to a domain-specific descriptor
                         gives the noAnchor ablation.
    Returns (mem, mask, sizes, rad, phi, abar_s).
    """
    dm = psr.dyn_distance(st, lam)
    if mode == "dyn":
        me, mk, sz = psr.edges_knn(dm, C.K_DYN, C.KMAX)
        rad = psr.knn_radius(dm, C.K_DYN)
        ph = psr.edge_phi(st, me, mk)
        return me, mk, sz, rad, ph, float(np.mean(ph[:, 2]))
    a = 2 if anchor_idx is None else int(anchor_idx)
    eta = ETA_DEFAULT
    tau = TAU_DEFAULT if temporal else 0.0
    me, mk, sz = psr.edges_knn(dm, C.K_DYN, C.KMAX)
    rad = psr.knn_radius(dm, C.K_DYN)
    phi1 = psr.edge_phi(st, me, mk)
    rho = phi1[:, a]
    alpha_s = float(np.mean(rho))
    if prev_abar is None or tau == 0.0:
        abar_s = alpha_s
    else:
        abar_s = tau * prev_abar + (1.0 - tau) * alpha_s
    rho_i = rho
    dev = np.abs((rho_i[:, None] + rho_i[None, :]) / 2.0 - abar_s)
    dmod = dm * (1.0 + eta * dev)
    me2, mk2, sz2 = psr.edges_knn(dmod, C.K_DYN, C.KMAX)
    rad2 = psr.knn_radius(dmod, C.K_DYN)
    phi2 = psr.edge_phi(st, me2, mk2)
    return me2, mk2, sz2, rad2, phi2, abar_s


def build_cond_cache(ds, cond, tag, mode, anchor_idx=None, temporal=True,
                     per_class=60, seed=C.SEED):
    """Build the hypergraph cache for one condition under one generation mode.

    Stores mem/mask/sizes/phi, the (possibly propagated) anchor state abar, the
    node features, the global features, and the labels / recording ids / load.
    """
    key = _cache_tag(mode, anchor_idx, temporal)
    f = C.CACHE / f"hg_{ds}_{tag}_{key}.npz"
    if f.exists():
        return f
    tr = np.load(build_trace_cache_cond(ds, cond, tag, per_class, seed))
    traces = tr["trace"]
    lam = C.LAMBDA_T[ds]
    n = len(traces)
    n_seg, S, m = C.N_SEG, C.SEG_LEN, C.PSR_M
    M = S
    mem = np.zeros((n, n_seg, M, C.KMAX), np.int16)
    mask = np.zeros((n, n_seg, M, C.KMAX), bool)
    sizes = np.zeros((n, n_seg, M), np.int32)
    phi = np.zeros((n, n_seg, M, C.PHI_DIM), np.float32)
    abar = np.zeros((n, n_seg), np.float32)
    nodef = np.zeros((n, n_seg, S, C.NODEF_DIM), np.float32)
    gf = np.zeros((n, C.N_GLOBAL), np.float32)
    for i in range(n):
        segs = _seg(traces[i])
        prev = None
        for s in range(n_seg):
            st = segs[s]
            me, mk, sz, rad, ph, a = _gen_edges(st, lam, mode, prev,
                                                anchor_idx, temporal)
            prev = a
            mem[i, s] = me; mask[i, s] = mk; sizes[i, s] = sz
            phi[i, s] = ph; abar[i, s] = a
            vel = np.linalg.norm(np.diff(st, axis=0), axis=1)
            vel = np.append(vel, vel[-1])
            inf = np.zeros(S, np.float32)
            mf, kf = np.where(mk)
            np.add.at(inf, me[mf, kf].astype(np.int64), 1.0)
            nodef[i, s, :, 0] = vel
            nodef[i, s, :, 1] = rad
            nodef[i, s, :, 2] = inf
        gt = np.ascontiguousarray(traces[i][::C.GLOBAL_STRIDE])
        gdm = psr.dyn_distance(gt, lam)
        gme, gmk, gsz = psr.edges_knn(gdm, C.K_DYN, C.GLOBAL_MAX)
        grad = psr.knn_radius(gdm, C.K_DYN)
        gphi = psr.edge_phi(gt, gme, gmk)
        gf[i] = psr.global_features(gme, gmk, gsz, gphi, grad, C.N_EIG, len(gt))
        if (i + 1) % 200 == 0:
            print(f"  {key}/{ds}/{tag} {i+1}/{n}", flush=True)
    np.savez_compressed(f, mem=mem, mask=mask, sizes=sizes, phi=phi, abar=abar,
                        nodef=nodef, gfeat=gf, y=tr["y"], fid=tr["fid"],
                        cond=tr["cond"])
    print("built", f, flush=True)
    return f


# ---------------------------------------------------------------------------
# Component 1: domain-invariance screening of the descriptors
# ---------------------------------------------------------------------------
def _w1(a, b):
    a = np.sort(np.asarray(a, float)); b = np.sort(np.asarray(b, float))
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    return float(np.mean(np.abs(a[-n:] - b[-n:])))


def _mmd_rbf(a, b, sigma=1.0, max_n=2000):
    a = np.asarray(a, float)[:, None]; b = np.asarray(b, float)[:, None]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    rng = np.random.default_rng(0)
    if len(a) > max_n:
        a = a[rng.choice(len(a), max_n, replace=False)]
    if len(b) > max_n:
        b = b[rng.choice(len(b), max_n, replace=False)]
    ka = np.exp(-((a - a.T) ** 2) / (2 * sigma ** 2))
    kb = np.exp(-((b - b.T) ** 2) / (2 * sigma ** 2))
    kab = np.exp(-((a - b.T) ** 2) / (2 * sigma ** 2))
    return float(ka.mean() + kb.mean() - 2 * kab.mean())


def _edge_descriptors(ds, cond, tag, anchor_idx):
    """Per-edge descriptors of the static dyn hypergraph of a condition,
    pooled over all windows/segments."""
    tr = np.load(C.CACHE / f"trace_{ds}_{tag}.npz")
    traces = tr["trace"]
    lam = C.LAMBDA_T[ds]
    vals = []
    for i in range(len(traces)):
        for s in range(C.N_SEG):
            st = traces[i].reshape(C.N_SEG, C.SEG_LEN, C.PSR_M)[s]
            dm = psr.dyn_distance(st, lam)
            me, mk, sz = psr.edges_knn(dm, C.K_DYN, C.KMAX)
            ph = psr.edge_phi(st, me, mk)
            vals.append(ph)
    return np.concatenate(vals, 0)


def screen_anchors(ds, condS, condT, tagS, tagT, per_class=60, seed=C.SEED):
    """Component 1.  Report W1/MMD of each descriptor between source and target
    conditions (pooled over classes).  Returns a dict per descriptor."""
    build_trace_cache_cond(ds, condS, tagS, per_class, seed)
    build_trace_cache_cond(ds, condT, tagT, per_class, seed)
    names = ["kappa", "chi", "rho"]
    out = {}
    for j in range(C.PHI_DIM):
        a = _edge_descriptors(ds, condS, tagS, j)[:, j]
        b = _edge_descriptors(ds, condT, tagT, j)[:, j]
        # robust normalisation by pooled std
        pool = np.concatenate([a, b])
        sd = pool.std() + 1e-9
        out[names[j]] = dict(w1=float(_w1(a, b) / sd),
                             mmd=float(_mmd_rbf(a / sd, b / sd)),
                             meanS=float(a.mean()), meanT=float(b.mean()))
    return out


# ---------------------------------------------------------------------------
# GA-DCHG model (C2 anchor-guidance + C3 anchor readout)
# ---------------------------------------------------------------------------
class GADHyperLayer(nn.Module):
    """Hypergraph layer with anchor-guided attention.

    The attention score for edge e in segment s is
        s = w_a^T h_i + w_b^T h_e + beta^T phi_e
          + beta_a * (phi_e[anchor] - abar_s)
    so the tracked (propagated) anchor state abar_s steers which edges the
    network uses, making the hyperedge a short-term memory of the trajectory
    geometry.  The per-edge attention mass is returned for the anchor readout.
    """

    def __init__(self, d_in, d_out, anchor_idx=2, dropout=0.0):
        super().__init__()
        self.anchor_idx = anchor_idx
        self.lin_x = nn.Linear(d_in, d_out)
        self.lin_e = nn.Linear(d_out, d_out, bias=False)
        self.lin_o = nn.Linear(d_out, d_out, bias=False)
        self.drop = nn.Dropout(dropout)
        self.wa = nn.Linear(d_out, 1, bias=True)
        self.wb = nn.Linear(d_out, 1, bias=False)
        self.beta = nn.Parameter(torch.zeros(C.PHI_DIM))
        self.beta_a = nn.Parameter(torch.zeros(1))

    def forward(self, x, mem, mask, phi, abar):
        # x:(B,S,V,h0) mem/mask:(B,S,M,K) phi:(B,S,M,F) abar:(B,S)
        hx = self.lin_x(x)
        B, S, V, h = hx.shape
        M, K = mem.shape[2], mem.shape[3]
        MF = M * K
        memf = mem.reshape(B, S, MF)
        m = mask.reshape(B, S, MF, 1).to(hx.dtype)
        hx_mem = torch.gather(hx, 2, memf.unsqueeze(-1).expand(B, S, MF, h))
        e = (hx_mem * m).reshape(B, S, M, K, h).sum(3) / \
            m.reshape(B, S, M, K, 1).sum(3).clamp(min=1.0)
        he = self.lin_e(e)
        sv = self.wa(hx).squeeze(-1)
        score = torch.gather(sv, 2, memf)
        score = score + self.wb(he).squeeze(-1).unsqueeze(3).expand(B, S, M, K).reshape(B, S, MF)
        pe = phi.unsqueeze(3).expand(B, S, M, K, phi.shape[-1])
        score = score + (pe * self.beta).sum(-1).reshape(B, S, MF)
        anchor_dev = (phi[..., self.anchor_idx].unsqueeze(3) -
                      abar.unsqueeze(2).unsqueeze(3)).expand(B, S, M, K).reshape(B, S, MF)
        score = score + (self.beta_a * anchor_dev)
        score = score.masked_fill(~mask.reshape(B, S, MF), -1e9)
        gmax = torch.full((B, S, V), -1e9, device=x.device, dtype=score.dtype)
        gmax.scatter_reduce_(2, memf, score, reduce="amax", include_self=True)
        gmax = torch.nan_to_num(gmax, neginf=0.0)
        ex = torch.exp(score - torch.gather(gmax, 2, memf))
        ex = ex * mask.reshape(B, S, MF).to(ex.dtype)
        gsum = torch.zeros(B, S, V, device=x.device, dtype=ex.dtype)
        gsum.scatter_add_(2, memf, ex)
        alpha = ex / gsum.gather(2, memf).clamp(min=1e-9)   # (B,S,MF)
        msg = torch.zeros(B, S, V, h, device=x.device, dtype=hx.dtype)
        msg.scatter_add_(2, memf.unsqueeze(-1).expand(B, S, MF, h),
                         slot(self.lin_o(he), B, S, M, K, h) * alpha.unsqueeze(-1))
        # per-edge attention mass (sum over members) for the anchor readout
        emass = alpha.reshape(B, S, M, K).sum(-1)          # (B,S,M)
        return self.drop(F.relu(msg + hx)), emass


def slot(he, B, S, M, K, h):
    return he.unsqueeze(3).expand(B, S, M, K, h).reshape(B, S, M * K, h)


class GADNet(nn.Module):
    """Anchor-guided dual-channel hypergraph network."""

    def __init__(self, n_classes, d_node, anchor_idx=2, hidden=C.HIDDEN,
                 layers=C.LAYERS, g_dim=C.N_GLOBAL, dropout=C.DROPOUT):
        super().__init__()
        self.anchor_idx = anchor_idx
        dims = [d_node] + [hidden] * layers
        self.layers = nn.ModuleList([
            GADHyperLayer(dims[i], dims[i + 1], anchor_idx, dropout)
            for i in range(layers)])
        self.norm = nn.LayerNorm(hidden)
        self.global_mlp = nn.Sequential(
            nn.Linear(g_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU())
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes))

    def forward(self, x, mem, mask, phi, abar, gfeat):
        z = x
        emass = None
        for l in self.layers:
            z, emass = l(z, mem, mask, phi, abar)
        z = self.norm(z)
        zl = self.proj(torch.cat([z.mean((1, 2)), z.amax((1, 2))], -1))
        zg = self.global_mlp(_standardize(gfeat))
        # effective anchor geometry of each window: attention-weighted mean of
        # the anchor descriptor over all hyperedges (differentiable)
        rho = phi[..., self.anchor_idx]                    # (B,S,M)
        mass = emass.clamp(min=0.0)
        num = (mass * rho).sum((1, 2))
        den = mass.sum((1, 2)).clamp(min=1e-9)
        g = num / den                                      # (B,)
        return self.head(torch.cat([zl, zg], -1)), g


_G_STATS = {}


def set_global_stats(mean, std):
    _G_STATS["mean"] = mean
    _G_STATS["std"] = std


def _standardize(g):
    if "mean" not in _G_STATS:
        return g
    return (g - _G_STATS["mean"]) / _G_STATS["std"]


# ---------------------------------------------------------------------------
# sklearn-free metrics
# ---------------------------------------------------------------------------
def accuracy(y, p):
    return float(np.mean(np.asarray(y) == np.asarray(p)))


def macro_f1(y, p):
    y = np.asarray(y); p = np.asarray(p)
    classes = np.unique(np.concatenate([y, p]))
    fs = []
    for c in classes:
        tp = np.sum((y == c) & (p == c)); fp = np.sum((y != c) & (p == c))
        fn = np.sum((y == c) & (p != c))
        prec = tp / (tp + fp + 1e-12); rec = tp / (tp + fn + 1e-12)
        fs.append(2 * prec * rec / (prec + rec + 1e-12))
    return float(np.mean(fs))


# ---------------------------------------------------------------------------
# cross-domain data plumbing
# ---------------------------------------------------------------------------
def _cache_tag(mode, anchor_idx=None, temporal=True):
    if mode == "dyn":
        return "dyn"
    a = 2 if anchor_idx is None else int(anchor_idx)
    return f"g{a}{'t' if temporal else 'n'}"


def _hg_path(ds, tag, mode, anchor_idx=None, temporal=True):
    return C.CACHE / f"hg_{ds}_{tag}_{_cache_tag(mode, anchor_idx, temporal)}.npz"


class GADDS:
    """Dataset over one condition's hypergraph cache (mode-dependent)."""

    def __init__(self, ds, tag, mode, anchor_idx=None, temporal=True, use_raw=False):
        z = np.load(_hg_path(ds, tag, mode, anchor_idx, temporal))
        t = np.load(C.CACHE / f"trace_{ds}_{tag}.npz")
        self.mem = z["mem"]; self.mask = z["mask"]; self.phi = z["phi"]
        self.abar = z["abar"]; self.gf = z["gfeat"]; self.nodef = z["nodef"]
        n, n_seg, S, _ = self.nodef.shape
        tr = t["trace"].reshape(n, C.N_SEG, C.SEG_LEN, C.PSR_M)
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
        return (torch.from_numpy(self.x[i]),
                torch.from_numpy(self.mem[i].astype(np.int64)),
                torch.from_numpy(self.mask[i]),
                torch.from_numpy(self.phi[i]),
                torch.from_numpy(self.abar[i]),
                torch.from_numpy(self.gf[i]),
                int(self.y[i]))


class RawDS2:
    def __init__(self, ds, tag):
        t = np.load(C.CACHE / f"trace_{ds}_{tag}.npz")
        X = t["raw"]; X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
        self.X = X.astype(np.float32); self.y = t["y"].astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), int(self.y[i])


def split_source(y, val_frac=0.15, seed=C.SEED):
    """Stratified (train, val) split of a source condition."""
    rng = np.random.default_rng(seed + 7)
    tr, va = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]; rng.shuffle(idx)
        n = len(idx); n_va = int(round(n * val_frac))
        va.append(idx[:n_va]); tr.append(idx[n_va:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(va))


def split_target(fid, test_frac=0.35, seed=C.SEED):
    """Recording-disjoint (adapt, test) split of a target condition: whole
    recordings go to one side, so the adapt set used for alignment never shares
    a recording with the evaluated test set."""
    rng = np.random.default_rng(seed + 11)
    files = np.unique(fid); rng.shuffle(files)
    n_te = max(1, int(round(len(files) * test_frac)))
    te_f = set(files[:n_te].tolist())
    ad, te = [], []
    for i in range(len(fid)):
        (te if int(fid[i]) in te_f else ad).append(i)
    return np.sort(np.array(ad)), np.sort(np.array(te))


def w1_loss(g_s, g_t):
    gs = torch.sort(g_s)[0]; gt = torch.sort(g_t)[0]
    n = min(len(gs), len(gt))
    return torch.mean(torch.abs(gs[-n:] - gt[-n:]))


# ---------------------------------------------------------------------------
# cross-domain training loop
# ---------------------------------------------------------------------------
def _gad_forward(model, b):
    x, mem, mask, phi, abar, gf, y = b
    logit, g = model(x, mem, mask, phi, abar, gf)
    return logit, g, y


def _hyp_forward(model, b):
    x, mem, mask, phi, abar, gf, y = b
    return model(x, mem, mask, phi, gf)[0], None, y


def _raw_forward(model, b):
    x, y = b
    return model(x)[0], None, y


def fit_cross(model, ds, tag_src, tag_tgt, mode, kind, tr_src, va_src,
              adapt_tgt, te_tgt, seed=0, epochs=C.EPOCHS, lr=C.LR, bs=C.BATCH,
              lam=LAM_ALIGN_DEFAULT, use_align=True, align_epoch=ALIGN_EPOCH,
              anchor_idx=None, temporal=True, verbose=False):
    """Cross-domain fit: labeled source + (optional) unlabeled target.

    kind in {"gad","hyper","raw"}.  Source features come from tag_src, the
    unlabeled target-adapt and the evaluated target-test come from tag_tgt.
    For "gad", a 1-D Wasserstein alignment on the effective anchor geometry is
    added between source and target-adapt batches.  The best checkpoint is
    chosen on the source validation split; the returned metrics are on the
    target test split.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    src = GADDS(ds, tag_src, mode, anchor_idx, temporal) if kind != "raw" \
        else RawDS2(ds, tag_src)
    if kind != "raw":
        g = src.gf[tr_src].astype(np.float32)
        set_global_stats(torch.tensor(g.mean(0)), torch.tensor(g.std(0) + 1e-6))
    tgt = None
    if kind == "gad" and use_align and len(adapt_tgt):
        tgt = GADDS(ds, tag_tgt, mode, anchor_idx, temporal)
    fwd = {"gad": _gad_forward, "hyper": _hyp_forward, "raw": _raw_forward}[kind]

    bs_src = bs // 2 if tgt is not None else bs
    dl_s = DataLoader(Subset(src, tr_src), batch_size=bs_src, shuffle=True,
                      drop_last=True)
    dl_t = DataLoader(Subset(tgt, adapt_tgt), batch_size=bs_src, shuffle=True,
                      drop_last=True) if tgt is not None else None
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.CrossEntropyLoss()
    best, best_state, wait = -1, None, 0
    import time
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        it_s = iter(dl_s); it_t = iter(dl_t) if dl_t is not None else None
        while True:
            try:
                b_s = next(it_s)
            except StopIteration:
                break
            logit_s, g_s, y_s = fwd(model, b_s)
            ce = lossf(logit_s, y_s)
            loss = ce
            if dl_t is not None and ep >= align_epoch:
                try:
                    b_t = next(it_t)
                except StopIteration:
                    it_t = iter(dl_t); b_t = next(it_t)
                _, g_t, _ = fwd(model, b_t)
                loss = ce + lam * w1_loss(g_s, g_t)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        if kind != "raw":
            va_acc = _eval_acc(model, src, va_src, kind, mode)
            if va_acc > best:
                best = va_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    train_time = time.time() - t0
    te = GADDS(ds, tag_tgt, mode, anchor_idx, temporal) if kind != "raw" \
        else RawDS2(ds, tag_tgt)
    yt, yp, gv = _predict(model, te, te_tgt, kind, mode)
    return dict(acc=accuracy(yt, yp), f1=macro_f1(yt, yp),
                train_time=round(train_time, 2),
                params=int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                epochs=ep + 1, y_true=yt, y_pred=yp, g=gv)


def x(b): return b[0]
def mem(b): return b[1]
def mask(b): return b[2]
def phi(b): return b[3]
def abar(b): return b[4]
def gf(b): return b[5]


@torch.no_grad()
def _eval_acc(model, ds, idx, kind, mode):
    yt, yp, _ = _predict(model, ds, idx, kind, mode)
    return accuracy(yt, yp)


@torch.no_grad()
def _predict(model, ds, idx, kind, mode):
    dl = DataLoader(Subset(ds, idx), batch_size=128)
    P, Y, G = [], [], []
    for b in dl:
        if kind == "raw":
            x0, y0 = b
            logit = model(x0)[0]; p = logit.argmax(1).numpy(); g = None
        elif kind == "hyper":
            x0, m0, mk0, ph0, ab0, gf0, y0 = b
            logit, _ = model(x0, m0, mk0, ph0, gf0); p = logit.argmax(1).numpy(); g = None
        else:
            x0, m0, mk0, ph0, ab0, gf0, y0 = b
            logit, g = model(x0, m0, mk0, ph0, ab0, gf0); p = logit.argmax(1).numpy()
            g = g.numpy().tolist()
        P.append(p); Y.append(y0.numpy())
        if g is not None:
            G.extend(g)
    return (np.concatenate(Y), np.concatenate(P),
            np.asarray(G) if G else None)


# ---------------------------------------------------------------------------
# Component 4: intervention audit
# ---------------------------------------------------------------------------
@torch.no_grad()
def intervention_audit(model, ds, tag, mode, idx, anchor_idx=None,
                       temporal=True, deltas=(0.5, 1.0, 2.0), seed=0):
    """Perturb one descriptor at test time and measure how decisions change.

    For each descriptor j and perturbation scale delta we replace
    phi[...,j] <- phi[...,j] + delta * std_j * z  (z ~ N(0,1), fixed seed) and
    record (i) the flip rate against the clean decisions and (ii) the accuracy
    change.  The descriptor whose perturbation changes decisions most is the one
    the decisions depend on causally.  Returns a dict keyed by descriptor name.
    """
    rng = np.random.default_rng(seed)
    z = np.load(_hg_path(ds, tag, mode, anchor_idx, temporal))
    t = np.load(C.CACHE / f"trace_{ds}_{tag}.npz")
    n, n_seg, S, _ = z["nodef"].shape
    tr = t["trace"].reshape(n, C.N_SEG, C.SEG_LEN, C.PSR_M)
    x_all = np.concatenate([tr, z["nodef"]], -1).astype(np.float32)
    std_j = [float(z["phi"][..., j].std()) for j in range(C.PHI_DIM)]
    # clean predictions
    y_clean = []
    for i0 in range(0, len(idx), 64):
        ids = idx[i0:i0 + 64]
        xx = torch.from_numpy(x_all[ids]); mm = torch.from_numpy(z["mem"][ids].astype(np.int64))
        mk = torch.from_numpy(z["mask"][ids]); ph = torch.from_numpy(z["phi"][ids])
        ab = torch.from_numpy(z["abar"][ids]); gg = torch.from_numpy(z["gfeat"][ids])
        logit, _ = model(xx, mm, mk, ph, ab, gg)
        y_clean.append(logit.argmax(1).numpy())
    y_clean = np.concatenate(y_clean)
    y_true = z["y"][idx]
    names = ["kappa", "chi", "rho"]
    out = {}
    for j in range(C.PHI_DIM):
        for delta in deltas:
            pert = np.random.default_rng(seed + j * 100 + int(delta * 10))
            flips, accs = [], []
            for i0 in range(0, len(idx), 64):
                ids = idx[i0:i0 + 64]
                xx = torch.from_numpy(x_all[ids]); mm = torch.from_numpy(z["mem"][ids].astype(np.int64))
                mk = torch.from_numpy(z["mask"][ids])
                ph = np.array(z["phi"][ids], copy=True)
                zz = pert.normal(0, 1, size=ph[..., j].shape)
                ph[..., j] = ph[..., j] + delta * std_j[j] * zz
                ph = torch.from_numpy(ph)
                ab = torch.from_numpy(z["abar"][ids]); gg = torch.from_numpy(z["gfeat"][ids])
                logit, _ = model(xx, mm, mk, ph, ab, gg)
                yp = logit.argmax(1).numpy()
                flips.append(np.mean(yp != y_clean[i0:i0 + 64]))
                accs.append(np.mean(yp == y_true[i0:i0 + 64]))
            out[f"{names[j]}_d{delta}"] = dict(
                flip=float(np.mean(flips)), acc=float(np.mean(accs)))
    return out
