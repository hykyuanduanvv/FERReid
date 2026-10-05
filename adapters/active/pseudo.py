"""Pseudo identities of the unlabeled pool: k-reciprocal Jaccard graph + DBSCAN, constrained by the answers.

  PoolGraph(feats)       k-NN of every pool image, k-reciprocal encodings V (Zhong et al., CVPR 2017, with
                         local query expansion) and the sparse Jaccard distance between each image and its k1
                         nearest neighbours. Built once per round from the current features.
  graph.cluster(store)   DBSCAN on that sparse distance (as in the usual cluster-then-train UDA recipe),
                         then the human answers are enforced:
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
import scipy.sparse as sp
import torch


@torch.no_grad()
def knn(feats, k, chunk=4096):
    """k nearest neighbours (self excluded) of L2-normalised feats (N, D): (sims, idx), numpy (N, k)."""
    n = feats.size(0)
    k = min(k, n - 1)
    sims, idx = [], []
    for s in range(0, n, chunk):
        rows = torch.arange(s, min(s + chunk, n), device=feats.device)
        S = feats[rows] @ feats.T
        S[torch.arange(len(rows), device=feats.device), rows] = -2
        v, i = S.topk(k, dim=1)
        sims.append(v.float().cpu())
        idx.append(i.cpu())
    return torch.cat(sims).numpy(), torch.cat(idx).numpy()


class PoolGraph:

    def __init__(self, feats, k1=30, k2=6, eps=0.6, min_samples=4):
        self.X = feats.float().cpu().numpy() if torch.is_tensor(feats) else np.asarray(feats, np.float32)
        self.n = len(self.X)
        self.k1, self.k2 = min(k1, self.n - 1), max(1, min(k2, self.n - 1))
        self.eps, self.min_samples = eps, min_samples
        t = feats if torch.is_tensor(feats) else torch.from_numpy(self.X)
        self.nn_sim, self.nn_idx = knn(t, self.k1)
        self.V = self._encodings()
        self._dist = self._knn_distance()

    # ---------------------------------------------------------------- k-reciprocal encodings

    def _encodings(self):
        n, k1, X, idx = self.n, self.k1, self.X, self.nn_idx
        half = max(1, int(round(k1 / 2)))
        full_sets = [set(r.tolist()) for r in idx]
        half_sets = [set(r[:half].tolist()) for r in idx]
        recip = [[i] + [j for j in idx[i].tolist() if i in full_sets[j]] for i in range(n)]
        recip_half = [[i] + [j for j in idx[i, :half].tolist() if i in half_sets[j]] for i in range(n)]
        rows, cols, vals = [], [], []
        for i in range(n):
            R = set(recip[i])
            for j in recip[i]:
                c = recip_half[j]
                if len(R.intersection(c)) > 2 / 3 * len(c):
                    R.update(c)
            R = np.fromiter(R, np.int64, len(R))
            w = np.exp(-(2 - 2 * (X[R] @ X[i])))   # exp(-squared euclidean distance)
            rows.append(np.full(len(R), i))
            cols.append(R)
            vals.append(w / w.sum())
        V = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        if self.k2 > 1:  # local query expansion: average over the k2 nearest images (self included)
            nb = np.concatenate([np.arange(n)[:, None], idx[:, :self.k2 - 1]], axis=1)
            Q = sp.csr_matrix((np.full(nb.size, 1.0 / nb.shape[1]), (np.repeat(np.arange(n), nb.shape[1]),
                                                                      nb.reshape(-1))), shape=(n, n))
            V = (Q @ V).tocsr()
        return V

    def jaccard_sim(self, a, b, chunk=50000):
        """1 - Jaccard distance of the encodings of pairs (a[t], b[t]); rows of V sum to one."""
        a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
        out = np.empty(len(a), np.float64)
        for s in range(0, len(a), chunk):
            m = np.asarray(self.V[a[s:s + chunk]].minimum(self.V[b[s:s + chunk]]).sum(1)).reshape(-1)
            out[s:s + chunk] = m / (2 - m)
        return out

    def _knn_distance(self):
        a = np.repeat(np.arange(self.n), self.k1)
        b = self.nn_idx.reshape(-1)
        d = 1 - self.jaccard_sim(a, b)
        return a, b, np.maximum(d, 1e-6)  # explicit entries only: keep zero distances stored

    # ---------------------------------------------------------------- clustering

    def cluster(self, store=None):
        from sklearn.cluster import DBSCAN
        a, b, d = self._dist
        groups = [] if store is None else [c for c in store.clusters() if len(c) >= 2]
        if groups:  # must-links: a chain inside every answered cluster at distance ~0
            ml_a = np.concatenate([np.asarray(g[:-1]) for g in groups])
            ml_b = np.concatenate([np.asarray(g[1:]) for g in groups])
            a, b, d = np.r_[a, ml_a], np.r_[b, ml_b], np.r_[d, np.full(len(ml_a), 1e-6)]
        # symmetric; i in N(j) and j in N(i) give duplicate entries -> keep the smaller distance
        D = _min_duplicates(np.r_[a, b], np.r_[b, a], np.r_[d, d], self.n)
        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="precomputed").fit_predict(D)
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
        for lab in np.unique(labels[labels >= 0]):
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


def _min_duplicates(rows, cols, vals, n):
    """Sparse symmetric distance matrix keeping the smallest value of duplicate entries."""
    key = rows.astype(np.int64) * n + cols
    order = np.lexsort((vals, key))
    key, rows, cols, vals = key[order], rows[order], cols[order], vals[order]
    first = np.r_[True, key[1:] != key[:-1]]
    return sp.csr_matrix((vals[first], (rows[first], cols[first])), shape=(n, n))


def _relabel(labels):
    """Consecutive labels 0..C-1 by first appearance; -1 stays."""
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
