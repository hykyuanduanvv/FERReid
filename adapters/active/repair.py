"""Cluster-repair queries: yes/no questions that merge or split pseudo identities (pseudo.PoolGraph).

With pseudo labels, one answer can relabel many images, and how many differs strongly between questions --
this is where a selection strategy can beat random order. Two kinds of question, each one image pair:

  merge  medoids of two nearby pseudo clusters (k nearest centroids), or an outlier image and the medoid of
         its nearest cluster: "yes" merges them (fixes an identity split, typically across cameras);
  split  the medoid of a cluster and the medoid of its far part (members closer to the cluster's farthest
         member than to its medoid): "no" splits the far part off (fixes an impure cluster).

Answers go into the ConstraintStore like every other answer; pseudo.PoolGraph.cluster enforces them, so a
"yes" merges the two pseudo clusters and a "no" separates them.

Ranking (p = calibrated probability that the pair is one person, Calibrator):
  repair         expected number of images whose pseudo label changes:  merge p * min(|A|, |B|),
                 split (1 - p) * |far part|                                                     (ours)
  repair_unc     p (1 - p): uncertainty only, the same questions                 (ablation: no impact term)
  repair_random  random order of the same questions                         (random baseline, same pool)
In each case a cluster takes part in at most `per_cluster` questions of a round (no redundant answers).
"""
import numpy as np


class Calibrator:
    """P(same person | cosine similarity) = sigmoid(a * s + b): logistic regression on the answered pairs
    (current features), MAP with a Gaussian prior around the label-free curve: centre tau (std centre_std in
    cosine units), slope 1 / temp (log-normal, std 1).
    Works from the first answer on, also when every answer so far is "yes" (merge questions mostly are); with
    many answers the data dominate the prior."""

    def __init__(self, sims, labels, tau, temp=0.05, centre_std=0.1, prior_weight=1.0):
        from scipy.optimize import minimize
        sims, labels = np.asarray(sims, np.float64), np.asarray(labels, np.float64)
        self.a, self.b = 1.0 / temp, -tau / temp
        if len(sims) == 0:
            return
        theta0 = np.array([0.0, 0.0])  # offsets from the prior, in units of the prior slope

        def unpack(t):
            a = self.a * np.exp(t[0])           # slope stays positive
            return a, -(tau + t[1]) * a         # centre tau + t[1]

        def loss(t):
            a, b = unpack(t)
            z = a * sims + b
            nll = np.sum(np.logaddexp(0, z) - labels * z)
            return nll + prior_weight * (t[0] ** 2 + (t[1] / centre_std) ** 2) / 2

        res = minimize(loss, theta0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 400})
        self.a, self.b = unpack(res.x)

    def __call__(self, s):
        return 1 / (1 + np.exp(-(self.a * np.asarray(s, np.float64) + self.b)))


def repair_candidates(X, labels, k_merge=5, min_split=4, cams=None, k_overall=None):
    """Merge and split questions for pseudo labels (-1: outlier). X: (N, D) L2-normalised numpy features.
    cams (pool camera ids, or None): merge partners are the k_merge nearest clusters whose cameras are disjoint
    from the cluster's own (an identity split by the clustering is mostly split by camera) plus the
    k_overall (default k_merge // 2) nearest clusters of any camera; without cameras, the k_merge nearest.
    Returns a dict of numpy arrays: i, j (pool images), sim, impact (images relabelled by the answer that
    changes the clustering), kind (0 merge, 1 split), ga, gb (cluster ids, for the per-cluster quota; outliers
    get ids >= number of clusters)."""
    n_cl = int(labels.max()) + 1 if (labels >= 0).any() else 0
    members = [np.flatnonzero(labels == c) for c in range(n_cl)]
    out = {k: [] for k in ("i", "j", "impact", "kind", "ga", "gb")}

    def add(i, j, impact, kind, ga, gb):
        out["i"].append(min(i, j)); out["j"].append(max(i, j))
        out["impact"].append(impact); out["kind"].append(kind); out["ga"].append(ga); out["gb"].append(gb)

    if n_cl:
        cent = np.stack([X[m].mean(0) for m in members])
        cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12
        medoid = np.array([m[np.argmax(X[m] @ cent[c])] for c, m in enumerate(members)])
        size = np.array([len(m) for m in members])
        if n_cl > 1:  # merges between clusters
            S = cent @ cent.T
            np.fill_diagonal(S, -np.inf)
            nb = [_top(S, k_merge)]
            if cams is not None:
                cam_idx = np.unique(np.asarray(cams), return_inverse=True)[1].reshape(-1)
                mask = np.zeros((n_cl, cam_idx.max() + 1), bool)
                for c, m in enumerate(members):
                    mask[c, cam_idx[m]] = True
                disjoint = ~(mask.astype(np.int32) @ mask.T.astype(np.int32)).astype(bool)
                nb = [_top(np.where(disjoint, S, -np.inf), k_merge),
                      _top(S, k_merge // 2 if k_overall is None else k_overall)]
            seen = set()
            for a in range(n_cl):
                for b in [x for t in nb for x in t[a]]:
                    key = (min(a, b), max(a, b))
                    if key in seen:
                        continue
                    seen.add(key)
                    add(int(medoid[a]), int(medoid[b]), int(min(size[a], size[b])), 0, key[0], key[1])
        outliers = np.flatnonzero(labels < 0)
        if len(outliers):  # an outlier joins its nearest cluster
            near = np.argmax(X[outliers] @ cent.T, axis=1)
            for t, (o, c) in enumerate(zip(outliers.tolist(), near.tolist())):
                add(o, int(medoid[c]), 1, 0, c, n_cl + t)
        for c, m in enumerate(members):  # splits of a cluster's far part
            if len(m) < min_split:
                continue
            far = m[np.argmin(X[m] @ X[medoid[c]])]
            part = m[(X[m] @ X[far]) > (X[m] @ X[medoid[c]])]
            if len(part) == 0 or len(part) >= len(m):
                continue
            pc = X[part].mean(0)
            q = part[np.argmax(X[part] @ pc)]
            add(int(medoid[c]), int(q), int(len(part)), 1, c, c)
    res = {k: np.asarray(v, np.int64) for k, v in out.items()}
    res["sim"] = (X[res["i"]] * X[res["j"]]).sum(1) if len(res["i"]) else np.zeros(0, np.float32)
    res["mutual"] = np.zeros(len(res["i"]), bool)
    res["rank"] = np.zeros(len(res["i"]), np.int64)
    return res


def _top(S, k):
    """Per row, the columns of the k largest finite entries (lists)."""
    k = min(k, S.shape[1] - 1)
    if k <= 0:
        return [[] for _ in range(len(S))]
    nb = np.argpartition(-S, k - 1, axis=1)[:, :k]
    return [[int(b) for b in row if np.isfinite(S[a, b])] for a, row in enumerate(nb)]


def _quota(order, cand, per_cluster):
    """Order with each cluster in at most per_cluster questions first; the rest after (fallback)."""
    used, first, rest = {}, [], []
    for p in order:
        ga, gb = int(cand["ga"][p]), int(cand["gb"][p])
        if used.get(ga, 0) < per_cluster and used.get(gb, 0) < per_cluster:
            used[ga] = used.get(ga, 0) + 1
            if gb != ga:
                used[gb] = used.get(gb, 0) + 1
            first.append(p)
        else:
            rest.append(p)
    return np.array(first + rest, dtype=np.int64)


def expected_change(cand, p):
    return np.where(cand["kind"] == 0, p, 1 - p) * cand["impact"]


def rank_repair(cand, p, rng, mode="repair", per_cluster=1):
    n = len(cand["i"])
    tie = rng.rand(n)
    if mode == "repair":
        order = np.lexsort((tie, -expected_change(cand, p)))
    elif mode == "repair_unc":
        order = np.lexsort((tie, -(p * (1 - p))))
    elif mode == "repair_random":
        order = rng.permutation(n)
    else:
        raise ValueError(mode)
    return _quota(order, cand, per_cluster)


REPAIR_STRATEGIES = ("repair", "repair_unc", "repair_random")
