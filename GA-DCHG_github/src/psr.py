"""Phase-space reconstruction and dynamical-neighbourhood hyperedges.

Central object: the trace {X_1,...,X_N} obtained by time-delay embedding of a
vibration window.  A *dynamical* distance combines the geometric separation in
state space with a time-decay factor, and each state point spawns a hyperedge
made of the state points inside its dynamical neighbourhood.
"""
import numpy as np


# ---------------------------------------------------------------- trace
def build_trace(x, m, tau, n_trace):
    """Delay-embed one window x (L,) into (n_trace, m) state points."""
    x = np.asarray(x, float)
    n_raw = len(x) - (m - 1) * tau
    idx = np.arange(m)[None, :] * tau
    full = x[idx + np.arange(n_raw)[:, None]]          # (n_raw, m)
    sel = np.unique(np.linspace(0, n_raw - 1, n_trace).round().astype(int))
    tr = full[sel]
    return tr.astype(np.float32)


# ---------------------------------------------------------------- distances
def pairwise(tr):
    d = tr[:, None, :] - tr[None, :, :]
    return np.sqrt((d * d).sum(-1) + 1e-12)


def dyn_distance(tr, lam):
    """Time-penalised separation, ||X_i-X_j|| / w_ij with a time-decay kernel
    w_ij = exp(-|i-j|/lam).  Two state points count as close only when they are
    close in state space AND close along the trajectory; a spatially close point
    that the orbit only revisits much later is pushed away by the growing
    penalty.  lam sets the scale over which the temporal weight decays."""
    N = len(tr)
    dist = pairwise(tr)
    dt = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :]).astype(np.float32)
    return (dist * np.exp(dt / lam)).astype(np.float32)


# ---------------------------------------------------------------- hyperedges
def edges_knn(dmat, k, kmax):
    """One hyperedge per vertex: all points within the distance of its k-th
    nearest neighbour (variable size, ties included).  Returns padded members
    and a validity mask."""
    N = dmat.shape[0]
    order = np.argsort(dmat, axis=1, kind="stable")
    thr = dmat[np.arange(N), order[:, min(k, N - 1)]]
    near = dmat <= thr[:, None]
    near[np.arange(N), np.arange(N)] = True            # a point is in its own edge
    sizes = near.sum(1)
    cap = min(kmax, N)
    order2 = np.argsort(~near, axis=1, kind="stable")  # members first
    mem = order2[:, :cap].astype(np.int16)
    mask = np.take_along_axis(near, mem.astype(np.int64), axis=1)
    mem = np.where(mask, mem, 0).astype(np.int16)
    return mem, mask.astype(bool), sizes.astype(np.int32)


def edges_quantile(dmat, q, kmax):
    """One hyperedge per vertex: all points inside a distance quantile.

    The radius is the q-quantile of the pairwise distances of the trace, so the
    edge sizes are free to vary with the local density of the trajectory.  A
    stiff (uniform) flow gives edges of nearly equal size, an intermittent one
    gives a spread of sizes.
    """
    N = dmat.shape[0]
    iu = np.triu_indices(N, 1)
    r = float(np.quantile(dmat[iu], q))
    return edges_radius(dmat, r, kmax)


def knn_radius(dmat, k):
    """Distance to the k-th nearest neighbour of each point, the local
    dynamical scale of its neighbourhood."""
    return np.sort(dmat, axis=1)[:, min(k, dmat.shape[1] - 1)].astype(np.float32)


def edges_radius(dmat, r, kmax):
    """One hyperedge per vertex: all points within the fixed radius r."""
    N = dmat.shape[0]
    near = dmat <= r
    near[np.arange(N), np.arange(N)] = True
    sizes = near.sum(1)
    cap = min(kmax, N)
    order2 = np.argsort(~near, axis=1, kind="stable")
    mem = order2[:, :cap].astype(np.int16)
    mask = np.take_along_axis(near, mem.astype(np.int64), axis=1)
    mem = np.where(mask, mem, 0).astype(np.int16)
    return mem, mask.astype(bool), sizes.astype(np.int32)


def edges_cluster(tr, n_clusters, kmax, seed=0):
    """KMeans partition of state points, one hyperedge per cluster."""
    from sklearn.cluster import KMeans
    N = len(tr)
    n_clusters = min(n_clusters, N)
    lab = KMeans(n_clusters=n_clusters, n_init=4, random_state=seed).fit_predict(tr)
    mem = np.zeros((n_clusters, kmax), np.int16)
    mask = np.zeros((n_clusters, kmax), bool)
    sizes = np.zeros(n_clusters, np.int32)
    for c in range(n_clusters):
        pts = np.where(lab == c)[0]
        s = min(len(pts), kmax)
        mem[c, :s] = pts[:s]
        mask[c, :s] = True
        sizes[c] = len(pts)
    return mem, mask, sizes


# ---------------------------------------------------------------- edge physics
def _ordered(tr, mem, mask):
    """Members sorted by trace index, invalid slots pushed to the right."""
    order = np.lexsort((mem.astype(np.int64), ~mask), axis=1)
    ms = np.take_along_axis(mem.astype(np.int64), order, axis=1)
    mk = np.take_along_axis(mask, order, axis=1)
    return tr[ms], mk


def edge_phi(tr, mem, mask):
    """Dynamical descriptors of each hyperedge, vectorised over all edges,
    (0) mean discrete (Menger) curvature of the time-ordered members,
    (1) local expansion index (a finite-time Lyapunov proxy),
    (2) filling ratio (compactness of the member point cloud)."""
    M, K = mem.shape
    out = np.zeros((M, 3), np.float32)
    if K < 3:
        return out
    P, mk = _ordered(tr, mem, mask)                        # (M,K,m)
    n = mk.sum(1)
    cnt = np.clip(n, 1, None)[:, None]
    cent = (P * mk[..., None]).sum(1) / cnt                # (M,m)
    Pc = (P - cent[:, None, :]) * mk[..., None]
    # (0) Menger curvature over consecutive time-ordered triples
    a, b, c = P[:, :-2], P[:, 1:-1], P[:, 2:]
    v3 = mk[:, :-2] & mk[:, 1:-1] & mk[:, 2:]
    ab = np.linalg.norm(b - a, axis=2) + 1e-12
    bc = np.linalg.norm(c - b, axis=2) + 1e-12
    ca = np.linalg.norm(c - a, axis=2) + 1e-12
    cr = np.linalg.norm(np.cross(b - a, c - a), axis=2) + 1e-12
    curv = 2.0 * cr / (ab * bc * ca)
    n3 = np.clip(v3.sum(1), 1, None)
    out[:, 0] = (curv * v3).sum(1) / n3
    # (1) expansion index
    step = np.linalg.norm(P[:, 1:] - P[:, :-1], axis=2)
    v2 = mk[:, 1:] & mk[:, :-1]
    s_mean = (step * v2).sum(1) / np.clip(v2.sum(1), 1, None)
    spread = np.linalg.norm(Pc, axis=2).sum(1) / cnt[:, 0]
    out[:, 1] = np.log((s_mean + 1e-12) / (spread + 1e-12))
    # (2) filling ratio from the covariance spectrum
    cov = np.einsum("mkp,mkq->mpq", Pc, Pc) / cnt[:, :, None]
    ev = np.linalg.eigvalsh(cov)
    ev = np.clip(ev, 1e-14, None)
    ok = n >= 4
    out[ok, 2] = np.exp(np.log(ev[ok]).mean(1)) / np.sqrt(ev[ok].max(1))
    return out


# ---------------------------------------------------------------- global features
def global_features(mem, mask, sizes, phi, radius, n_eig, V):
    """Spectral and distributional statistics of one hypergraph.

    The vertex count of the global hypergraph is small (a few hundred), so the
    normalised Laplacian is diagonalised densely.
    """
    V = int(V)
    M = mem.shape[0]
    rows = np.repeat(np.arange(M), mask.sum(1))
    cols = mem[mask].astype(np.int64)
    H = np.zeros((V, M), np.float32)
    H[cols, rows] = 1.0
    dv = H.sum(1)
    de = H.sum(0)
    di = (1.0 / np.sqrt(np.clip(dv, 1e-9, None)))
    dei = (1.0 / np.clip(de, 1e-9, None))
    S = (di[:, None] * H * dei[None, :]) @ (H.T * di[None, :])
    L = np.eye(V) - S
    lam, _ = np.linalg.eigh(L)
    lam = np.clip(lam, 0, None)
    n_eig = min(n_eig, V)

    def moments(v):
        v = np.asarray(v, float)
        m, s = v.mean(), v.std()
        z = (v - m) / (s + 1e-12)
        p = np.bincount(np.clip(v, 0, None).astype(int), minlength=1).astype(float)
        p = p / (p.sum() + 1e-12)
        ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
        return [float(m), float(s), float((z ** 3).mean()), float((z ** 4).mean()), ent]

    feats = []
    feats += list(lam[:n_eig])                            # smallest eigenvalues
    feats += [float(lam[-1])]                             # largest eigenvalue
    feats += moments(np.log(radius + 1e-12))              # neighbourhood scale
    feats += moments(sizes)                               # hyperedge size moments
    feats += moments(dv)                                  # vertex degree moments
    feats += moments(phi[:, 0])
    feats += moments(phi[:, 1])
    feats += moments(phi[:, 2])
    return np.asarray(feats, np.float32)
