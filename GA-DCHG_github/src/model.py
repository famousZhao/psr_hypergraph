"""Models: the dynamical-neighbourhood hypergraph network and its baselines.

The hypergraph layer works on padded incidence lists, so a batch is a plain
tensor stack (no sparse assembly).  Two attention modes share the layer: a
standard content attention and the physical-constrained attention that adds a
learnable weight on the dynamical descriptor of each hyperedge.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


# ---------------------------------------------------------------- hypergraph layer
class HyperLayer(nn.Module):
    def __init__(self, d_in, d_out, attention="none", phi_dim=C.PHI_DIM, dropout=0.0):
        super().__init__()
        self.attention = attention
        self.lin_x = nn.Linear(d_in, d_out)
        self.lin_e = nn.Linear(d_out, d_out, bias=False)
        self.lin_o = nn.Linear(d_out, d_out, bias=False)
        self.drop = nn.Dropout(dropout)
        if attention != "none":
            self.wa = nn.Linear(d_out, 1, bias=True)
            self.wb = nn.Linear(d_out, 1, bias=False)
            if attention == "phys":
                self.beta = nn.Parameter(torch.zeros(phi_dim))
            else:
                self.register_parameter("beta", None)

    def forward(self, x, mem, mask, phi):
        # x: (B,S,V,d_in)  mem/mask: (B,S,M,K)  phi: (B,S,M,F)
        hx = self.lin_x(x)                              # (B,S,V,h)
        B, S, V, h = hx.shape
        M, K = mem.shape[2], mem.shape[3]
        MF = M * K
        memf = mem.reshape(B, S, MF)
        m = mask.reshape(B, S, MF, 1).to(hx.dtype)
        hx_mem = torch.gather(hx, 2, memf.unsqueeze(-1).expand(B, S, MF, h))
        e = (hx_mem * m).reshape(B, S, M, K, h).sum(3) / \
            m.reshape(B, S, M, K, 1).sum(3).clamp(min=1.0)     # (B,S,M,h)

        def slot(he):
            return he.unsqueeze(3).expand(B, S, M, K, h).reshape(B, S, MF, h)

        if self.attention == "none":
            he = self.lin_o(e)
            cnt = torch.zeros(B, S, V, device=x.device, dtype=hx.dtype)
            cnt.scatter_add_(2, memf, mask.reshape(B, S, MF).to(hx.dtype))
            msg = torch.zeros(B, S, V, h, device=x.device, dtype=hx.dtype)
            msg.scatter_add_(2, memf.unsqueeze(-1).expand(B, S, MF, h), slot(he) * m)
            msg = msg / cnt.clamp(min=1.0).unsqueeze(-1)
        else:
            he = self.lin_e(e)                                # (B,S,M,h)
            sv = self.wa(hx).squeeze(-1)                      # (B,S,V)
            score = torch.gather(sv, 2, memf)                 # (B,S,MF)
            eb = self.wb(he).squeeze(-1).unsqueeze(3).expand(B, S, M, K)
            score = score + eb.reshape(B, S, MF)
            if self.attention == "phys":
                pe = phi.unsqueeze(3).expand(B, S, M, K, phi.shape[-1])
                score = score + (pe * self.beta).sum(-1).reshape(B, S, MF)
            score = score.masked_fill(~mask.reshape(B, S, MF), -1e9)
            gmax = torch.full((B, S, V), -1e9, device=x.device, dtype=score.dtype)
            gmax.scatter_reduce_(2, memf, score, reduce="amax", include_self=True)
            gmax = torch.nan_to_num(gmax, neginf=0.0)
            ex = torch.exp(score - torch.gather(gmax, 2, memf))
            ex = ex * mask.reshape(B, S, MF).to(ex.dtype)
            gsum = torch.zeros(B, S, V, device=x.device, dtype=ex.dtype)
            gsum.scatter_add_(2, memf, ex)
            alpha = ex / gsum.gather(2, memf).clamp(min=1e-9)
            msg = torch.zeros(B, S, V, h, device=x.device, dtype=hx.dtype)
            msg.scatter_add_(2, memf.unsqueeze(-1).expand(B, S, MF, h),
                             slot(self.lin_o(he)) * alpha.unsqueeze(-1))
        return self.drop(F.relu(msg + hx))


class LocalBranch(nn.Module):
    """Shared-weight HGNN over the segmented local hypergraphs."""

    def __init__(self, d_in, hidden, layers, attention, dropout):
        super().__init__()
        dims = [d_in] + [hidden] * layers
        self.layers = nn.ModuleList([
            HyperLayer(dims[i], dims[i + 1], attention, dropout=dropout)
            for i in range(layers)])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, mem, mask, phi):
        z = x
        for l in self.layers:
            z = l(z, mem, mask, phi)
        z = self.norm(z)                                  # (B,S,V,h)
        return z.mean((1, 2))                             # pool over segments and nodes


class LocalBranchPlus(nn.Module):
    """Shared-weight HGNN with mixed mean/max pooling, for the improved model."""

    def __init__(self, d_in, hidden, layers, attention, dropout, pool="mixed"):
        super().__init__()
        dims = [d_in] + [hidden] * layers
        self.layers = nn.ModuleList([
            HyperLayer(dims[i], dims[i + 1], attention, dropout=dropout)
            for i in range(layers)])
        self.norm = nn.LayerNorm(hidden)
        self.pool = pool
        self.drop = nn.Dropout(dropout)
        if pool == "mixed":
            self.proj = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU())
        elif pool == "attn":
            self.w = nn.Linear(hidden, 1)
            self.proj = nn.Identity()

    def forward(self, x, mem, mask, phi):
        z = x
        for l in self.layers:
            z = l(z, mem, mask, phi)
        z = self.norm(z)
        if self.pool == "mean":
            return z.mean((1, 2))
        if self.pool == "attn":
            a = torch.softmax(self.w(z).squeeze(-1).flatten(1), -1).reshape(z.shape[:3])
            return self.drop((z * a.unsqueeze(-1)).sum((1, 2)))
        return self.drop(self.proj(torch.cat([z.mean((1, 2)), z.amax((1, 2))], -1)))


class RawBranch(nn.Module):
    """A light convolutional branch over the raw window, for fusion."""

    def __init__(self, hidden=C.HIDDEN, dropout=C.DROPOUT):
        super().__init__()

        def blk(i, o, k=9, s=2):
            return nn.Sequential(nn.Conv1d(i, o, k, s, k // 2),
                                 nn.BatchNorm1d(o), nn.ReLU())

        self.conv = nn.Sequential(blk(1, 16), blk(16, 32), blk(32, 64),
                                  blk(64, hidden), nn.AdaptiveAvgPool1d(1), nn.Flatten())

    def forward(self, raw):
        return self.conv(raw.unsqueeze(1))


class TemporalBranch(nn.Module):
    """An LSTM branch over the raw window, for order-sensitive fusion."""

    def __init__(self, hidden=C.HIDDEN, dropout=C.DROPOUT):
        super().__init__()
        self.down = nn.Sequential(nn.Conv1d(1, 16, 15, 4, 7), nn.ReLU(),
                                  nn.Conv1d(16, 16, 15, 4, 7), nn.ReLU())
        self.lstm = nn.LSTM(16, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, raw):
        z = self.down(raw.unsqueeze(1)).transpose(1, 2)
        o, _ = self.lstm(z)
        return self.drop(o[:, -1])


class DynHyperNetPlus(nn.Module):
    """Improved dynamical-neighbourhood network with mixed pooling and an
    optional raw-signal fusion branch."""

    def __init__(self, n_classes, d_node, g_dim=C.N_GLOBAL, hidden=C.HIDDEN,
                 layers=C.LAYERS, attention="phys", dual=True, use_raw=True,
                 pool="mixed", raw_kind="cnn", dropout=C.DROPOUT):
        super().__init__()
        self.dual = dual
        self.use_raw = use_raw
        self.local = LocalBranchPlus(d_node, hidden, layers, attention, dropout, pool)
        if dual:
            self.global_mlp = nn.Sequential(
                nn.Linear(g_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.ReLU())
        if use_raw:
            self.raw = (TemporalBranch(hidden, dropout) if raw_kind == "lstm"
                        else RawBranch(hidden, dropout))
        d_fuse = hidden * (1 + (1 if dual else 0) + (1 if use_raw else 0))
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(d_fuse, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes))

    def forward(self, x, mem, mask, phi, gfeat, raw=None):
        zl = self.local(x, mem, mask, phi)
        parts = [zl]
        if self.dual:
            parts.append(self.global_mlp(_standardize(gfeat)))
        if self.use_raw:
            parts.append(self.raw(raw))
        return self.head(torch.cat(parts, -1)), zl


# ---------------------------------------------------------------- dual-channel net
class DynHyperNet(nn.Module):
    def __init__(self, n_classes, d_node, g_dim=C.N_GLOBAL, hidden=C.HIDDEN,
                 layers=C.LAYERS, attention="phys", dual=True, dropout=C.DROPOUT):
        super().__init__()
        self.dual = dual
        self.local = LocalBranch(d_node, hidden, layers, attention, dropout)
        self.global_mlp = nn.Sequential(
            nn.Linear(g_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU())
        d_fuse = hidden * (2 if dual else 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(d_fuse, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes))

    def forward(self, x, mem, mask, phi, gfeat):
        zl = self.local(x, mem, mask, phi)
        if self.dual:
            zg = self.global_mlp(_standardize(gfeat))
            z = torch.cat([zl, zg], -1)
        else:
            z = zl
        return self.head(z), zl


_G_STATS = {}


def set_global_stats(mean, std):
    _G_STATS["mean"] = mean
    _G_STATS["std"] = std


def _standardize(g):
    if "mean" not in _G_STATS:
        return g
    return (g - _G_STATS["mean"]) / _G_STATS["std"]


class GlobalNet(nn.Module):
    """Global-channel only: an MLP over the hypergraph topological statistics."""

    def __init__(self, n_classes, g_dim=C.N_GLOBAL, hidden=C.HIDDEN, dropout=C.DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(g_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes))

    def forward(self, x, mem, mask, phi, gfeat):
        return self.net(_standardize(gfeat)), None


# ---------------------------------------------------------------- baselines
class CNN1D(nn.Module):
    def __init__(self, n_classes, L=C.L_WINDOW, hidden=C.HIDDEN, dropout=C.DROPOUT):
        super().__init__()
        def blk(i, o, k=9, s=2):
            return nn.Sequential(nn.Conv1d(i, o, k, s, k // 2),
                                 nn.BatchNorm1d(o), nn.ReLU())
        self.net = nn.Sequential(
            blk(1, 16), blk(16, 32), blk(32, 64), blk(64, hidden),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes))

    def forward(self, raw):
        return self.net(raw.unsqueeze(1)), None


class LSTMNet(nn.Module):
    def __init__(self, n_classes, hidden=C.HIDDEN, dropout=C.DROPOUT):
        super().__init__()
        self.down = nn.Sequential(nn.Conv1d(1, 16, 15, 4, 7), nn.ReLU(),
                                  nn.Conv1d(16, 16, 15, 4, 7), nn.ReLU())
        self.lstm = nn.LSTM(16, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, raw):
        z = self.down(raw.unsqueeze(1)).transpose(1, 2)
        o, _ = self.lstm(z)
        return self.head(o[:, -1]), None


class GCN(nn.Module):
    """Plain graph convolutional baseline on the same phase-space nodes."""

    def __init__(self, n_classes, d_node, hidden=C.HIDDEN, layers=C.LAYERS,
                 k=8, dropout=C.DROPOUT):
        super().__init__()
        self.k = k
        dims = [d_node] + [hidden] * layers
        self.ws = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layers)])
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)

    def _adj(self, x):
        d = torch.cdist(x, x)
        kth = d.topk(self.k + 1, largest=False).values[..., -1:]
        a = (d <= kth).float()
        a = torch.maximum(a, a.transpose(-1, -2))
        return a

    def forward(self, x, mem, mask, phi, gfeat=None):
        B, S, V, _ = x.shape
        z = x
        for l, w in enumerate(self.ws):
            a = self._adj(z) + torch.eye(V, device=z.device)
            deg = a.sum(-1, keepdim=True).clamp(min=1.0)
            a = a / deg
            z = F.relu(w(torch.einsum("bsvw,bsvh->bsvh", a, z)))
            z = self.drop(z)
        return self.head(self.norm(z).mean((1, 2))), None
