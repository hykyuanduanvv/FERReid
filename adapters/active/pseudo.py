"""Pseudo identities of the unlabeled pool: k-reciprocal Jaccard graph + DBSCAN, constrained by the answers.
Everything heavy runs in torch on the features' device (the GPU in practice); only the bookkeeping of the
answered clusters (a few hundred images) is done in numpy.

  PoolGraph(feats)       k-NN of every pool image, k-reciprocal encodings V (Zhong et al., CVPR 2017: k1-reciprocal
                         sets expanded by their k1/2-reciprocal subsets, exp(-distance) weights, local query
                         expansion over k2 neighbours; dense (N, N) on the device) and the Jaccard distance between
                         each image and its k1 nearest neighbours. Built once per round from the current features.
  graph.cluster(store)   DBSCAN on that k-NN distance graph (core point: >= min_samples neighbours within eps, the
                         point itself included; clusters = connected components of core points; a border point
                         joins the cluster of its nearest core neighbour), then the human answers are enforced:
                           * every answered positive cluster (ConstraintStore) ends up in one pseudo cluster
                             (must-link edges at distance ~0; the pseudo clusters it spans are merged);
                           * a pseudo cluster containing two answered clusters that are known to be different
                             people is split: answered clusters are grouped greedily without cannot-links,
                             every other member joins the group with the nearest centroid.
                         Returns one label per pool image (-1: outlier, not used for training).
  graph.jaccard_sim(i, j)  1 - Jaccard distance for arbitrary pairs (the "disagree" strategy).

Labels never come from the person ids: the clustering sees features and the answers only.
"""
import numpy as np
import torch


@torch.no_grad()
def knn(feats, k, chunk=4096):
    """k nearest neighbours (self excluded) of L2-normalised feats (N, D) torch: (sims, idx) torch (N, k)."""
    n = feats.size(0)
    k = min(k, n - 1)
    sims, idx = [], []
    for s in range(0, n, chunk):
        rows = torch.arange(s, min(s + chunk, n), device=feats.device)
        S = feats[rows] @ feats.T
        S[torch.arange(len(rows), device=feats.device), rows] = -2
        v, i = S.topk(k, dim=1)
        sims.append(v.float())
        idx.append(i)
    return torch.cat(sims), torch.cat(idx)


def _reciprocal(idx, k, n):
    """(N, N) bool: j in the k-NN of i and i in the k-NN of j, plus the diagonal."""
    A = torch.zeros(n, n, dtype=torch.bool, device=idx.device)
    A.scatter_(1, idx[:, :k], True)
    R = A & A.T
    R.fill_diagonal_(True)
    return R


class PoolGraph:

    def __init__(self, feats, k1=30, k2=6, eps=0.6, min_samples=4, chunk=1024):
        t = feats if torch.is_tensor(feats) else torch.from_numpy(np.asarray(feats, np.float32))
        self.Xt = t.float()
        self.X = self.Xt.cpu().numpy()
        self.n = len(self.X)
        self.k1, self.k2 = min(k1, self.n - 1), max(1, min(k2, self.n - 1))
        self.eps, self.min_samples, self.chunk = eps, min_samples, chunk
        # large pools: half-precision encodings (weights are normalised per row, ~1e-3 relative error)
        self.dtype = torch.float32 if self.n <= 16000 else torch.float16
        with torch.no_grad():
            self.nn_sim_t, self.nn_idx_t = knn(self.Xt, self.k1)
            self.nn_idx = self.nn_idx_t.cpu().numpy()
            self.V = self._encodings()
            self._dist = self._knn_distance()

    # ---------------------------------------------------------------- k-reciprocal encodings

    def _encodings(self):
        n, k1, X, idx, c = self.n, self.k1, self.Xt, self.nn_idx_t, self.chunk
        R = _reciprocal(idx, k1, n)
        Rh = _reciprocal(idx, max(1, int(round(k1 / 2))), n)
        Rh_h = Rh.half()
        size_h = Rh.sum(1).float()
        V = torch.empty(n, n, dtype=self.dtype, device=X.device)
        for s in range(0, n, c):
            r = R[s:s + c]
            overlap = r.half() @ Rh_h.T                        # |R(i) & R_half(j)| (small integers: exact)
            add = r & (overlap.float() > 2 / 3 * size_h[None, :])
            rs = r | ((add.half() @ Rh_h) > 0)                 # R*(i) = R(i) + qualifying R_half(j)
            w = torch.exp(-(2 - 2 * (X[s:s + c] @ X.T))) * rs  # exp(-squared euclidean distance) on R*(i)
            V[s:s + c] = (w / w.sum(1, keepdim=True)).to(self.dtype)
        del R, Rh, Rh_h
        if self.k2 > 1:  # local query expansion: average over the k2 nearest images (self included)
            nb = torch.cat([torch.arange(n, device=X.device)[:, None], idx[:, :self.k2 - 1]], dim=1)
            V2 = torch.empty_like(V)
            for s in range(0, n, max(1, c // 4)):
                e = min(s + max(1, c // 4), n)
                V2[s:e] = V[nb[s:e]].float().mean(1).to(self.dtype)
            V = V2
        return V

    @torch.no_grad()
    def _jaccard_t(self, a, b, chunk=256):
        out = torch.empty(len(a), dtype=torch.float32, device=self.V.device)
        for s in range(0, len(a), chunk):
            m = torch.minimum(self.V[a[s:s + chunk]], self.V[b[s:s + chunk]]).float().sum(1)
            out[s:s + chunk] = m / (2 - m)
        return out

    def jaccard_sim(self, a, b):
        """1 - Jaccard distance of the encodings of pairs (a[t], b[t]); rows of V sum to one. numpy in/out."""
        dev = self.V.device
        a = torch.as_tensor(np.asarray(a, np.int64), device=dev)
        b = torch.as_tensor(np.asarray(b, np.int64), device=dev)
        return self._jaccard_t(a, b).double().cpu().numpy()

    def _knn_distance(self):
        a = torch.arange(self.n, device=self.V.device).repeat_interleave(self.k1)
        b = self.nn_idx_t.reshape(-1)
        d = 1 - self._jaccard_t(a, b)
        return a, b, d.clamp_min(1e-6)

    # ---------------------------------------------------------------- clustering

    def cluster(self, store=None):
        a, b, d = self._dist
        groups = [] if store is None else [c for c in store.clusters() if len(c) >= 2]
        if groups:  # must-links: a chain inside every answered cluster at distance ~0
            dev = a.device
            ml_a = torch.as_tensor(np.concatenate([np.asarray(g[:-1]) for g in groups]), device=dev)
            ml_b = torch.as_tensor(np.concatenate([np.asarray(g[1:]) for g in groups]), device=dev)
            a, b = torch.cat([a, ml_a]), torch.cat([b, ml_b])
            d = torch.cat([d, torch.full((len(ml_a),), 1e-6, device=dev)])
        labels = dbscan_graph(a, b, d, self.n, self.eps, self.min_samples).cpu().numpy()
        if store is not None:
            labels = self._enforce(labels, store)
        return _relabel(labels)

    def _enforce(self, labels, store):
        labels = labels.copy()
        nxt = labels.max() + 1
        roots = {}  # answered cluster (root) -> members
        for c in store.clusters():
            roots[store.find(c[0])] = c
        for members in roots.values():  # an answered cluster merges every pseudo cluster it touches
            if len(members) < 2:
                continue
            lab = np.unique(labels[members])
            lab = lab[lab >= 0]
            if len(lab):
                labels[np.isin(labels, lab)] = lab[0]
                labels[members] = lab[0]
            else:
                labels[members] = nxt
                nxt += 1
        answered = np.asarray(store.images(), np.int64)
        touched = np.unique(labels[answered]) if len(answered) else np.zeros(0, np.int64)
        for lab in touched[touched >= 0]:  # only pseudo clusters holding answered images can conflict
            members = np.flatnonzero(labels == lab)
            rs = sorted({store.find(int(i)) for i in members if store.labeled(int(i))},
                        key=lambda r: -len(roots[r]))
            groups = []
            for r in rs:  # greedy: first group without a cannot-link to r
                for g in groups:
                    if not any(store.infer(roots[r][0], roots[o][0]) is False for o in g):
                        g.append(r)
                        break
                else:
                    groups.append([r])
            if len(groups) < 2:
                continue
            fixed = {}
            for gi, g in enumerate(groups):
                for r in g:
                    for i in roots[r]:
                        fixed[i] = gi
            cent = np.stack([self.X[[i for i, gi in fixed.items() if gi == g]].mean(0) for g in range(len(groups))])
            new = np.array([nxt + g - 1 if g else lab for g in range(len(groups))])
            nxt += len(groups) - 1
            free = np.array([i for i in members if i not in fixed], np.int64)
            if len(free):
                labels[free] = new[np.argmax(self.X[free] @ cent.T, axis=1)]
            for i, gi in fixed.items():
                labels[i] = new[gi]
        return labels


@torch.no_grad()
def dbscan_graph(a, b, d, n, eps, min_samples, max_iter=10000):
    """DBSCAN on a sparse distance graph given as edges (a, b, d) (any direction, duplicates allowed), in torch.
    Neighbours of i: the points joined to i by an edge with distance <= eps, and i itself. Core points have at
    least min_samples neighbours; clusters are the connected components of core points over core-core edges
    (min-label propagation with pointer jumping); a non-core point joins the cluster of its nearest core
    neighbour, else it is an outlier (-1). Returns (N,) int64 labels (component ids, not consecutive)."""
    dev = a.device
    keep = d <= eps
    ea, eb, ed = torch.cat([a[keep], b[keep]]), torch.cat([b[keep], a[keep]]), torch.cat([d[keep], d[keep]])
    nz = ea != eb
    ea, eb, ed = ea[nz], eb[nz], ed[nz]
    if len(ea):  # unique (i, j) edges, smallest distance kept
        order = torch.argsort(ed)
        ea, eb, ed = ea[order], eb[order], ed[order]
        key = ea * n + eb
        order = torch.argsort(key, stable=True)
        ea, eb, ed, key = ea[order], eb[order], ed[order], key[order]
        first = torch.ones_like(key, dtype=torch.bool)
        first[1:] = key[1:] != key[:-1]
        ea, eb, ed = ea[first], eb[first], ed[first]
    deg = torch.bincount(ea, minlength=n) + 1
    core = deg >= min_samples
    lab = torch.arange(n, device=dev)
    cc = core[ea] & core[eb]
    ca, cb = ea[cc], eb[cc]
    for _ in range(max_iter):
        new = lab.clone()
        new.scatter_reduce_(0, ca, lab[cb], reduce="amin")
        new = new[new]  # pointer jumping
        if torch.equal(new, lab):
            break
        lab = new
    out = torch.full((n,), -1, dtype=torch.int64, device=dev)
    out[core] = lab[core]
    border = (~core[ea]) & core[eb]  # edges from a non-core point to a core point
    if border.any():
        ba, bb, bd = ea[border], eb[border], ed[border]
        best = torch.full((n,), float("inf"), device=dev)
        best.scatter_reduce_(0, ba, bd, reduce="amin")
        hit = bd == best[ba]
        out[ba[hit]] = lab[bb[hit]]  # ties: any nearest core neighbour
    return out


def _relabel(labels):
    """Consecutive labels 0..C-1; -1 stays."""
    out = np.full(len(labels), -1, np.int64)
    ok = labels >= 0
    _, inv = np.unique(labels[ok], return_inverse=True)
    out[ok] = inv
    return out


def label_clusters(labels):
    """Lists of members per pseudo label (label order)."""
    order = np.argsort(labels, kind="stable")
    lab = labels[order]
    keep = lab >= 0
    order, lab = order[keep], lab[keep]
    cuts = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
    return [order[s:e].tolist() for s, e in zip(cuts, np.r_[cuts[1:], len(order)])]
