"""Pairs the model proposes as "probably one person", and the label-free same-person threshold."""
import numpy as np
import torch


@torch.no_grad()
def candidate_pairs(feats, camids, has_cameras=True, k=10, chunk=4096):
    """Cross-camera k-nearest-neighbour pairs of the pool (any other image when the domain has no cameras).

    feats: (N, D) L2-normalised torch tensor. Returns a dict of numpy arrays, one entry per undirected pair
    (i < j): i, j, sim (cosine), mutual (each is among the other's k nearest), rank (best of the two
    neighbour ranks, 0 = nearest)."""
    n = feats.size(0)
    k = min(k, n - 1)
    cams = torch.as_tensor(np.asarray(camids), device=feats.device)
    src, dst, sims, ranks = [], [], [], []
    for s in range(0, n, chunk):
        rows = torch.arange(s, min(s + chunk, n), device=feats.device)
        S = feats[rows] @ feats.T
        S[torch.arange(len(rows), device=feats.device), rows] = -2
        if has_cameras:
            S = S.masked_fill(cams[rows][:, None] == cams[None, :], -2)
        val, idx = S.topk(k, dim=1)
        ok = val > -2
        src.append(rows[:, None].expand_as(idx)[ok].cpu())
        dst.append(idx[ok].cpu())
        sims.append(val[ok].float().cpu())
        ranks.append(torch.arange(k, device=feats.device)[None, :].expand_as(idx)[ok].cpu())
    a, b = torch.cat(src).numpy(), torch.cat(dst).numpy()
    s, r = torch.cat(sims).numpy(), torch.cat(ranks).numpy()
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = lo.astype(np.int64) * n + hi
    order = np.lexsort((r, key))  # by pair, best rank first
    key, lo, hi, s, r = key[order], lo[order], hi[order], s[order], r[order]
    first = np.r_[True, key[1:] != key[:-1]]
    count = np.diff(np.r_[np.flatnonzero(first), len(key)])
    return {"i": lo[first], "j": hi[first], "sim": s[first], "mutual": count == 2, "rank": r[first]}


def random_pair_quantile(feats, q=0.99, n_pairs=200000, seed=0):
    """Similarity above which a pair is "probably one person": the q-quantile of random pairs, which are
    almost always two different people (label-free; the same rule as adapters/selectors.py dup_tau)."""
    X = feats.float().cpu().numpy() if torch.is_tensor(feats) else np.asarray(feats, np.float32)
    n = len(X)
    if n * (n - 1) // 2 <= n_pairs:
        iu = np.triu_indices(n, 1)
        sims = (X @ X.T)[iu]
    else:
        g = np.random.RandomState(seed)
        a, b = g.randint(n, size=n_pairs), g.randint(n, size=n_pairs)
        keep = a != b
        sims = (X[a[keep]] * X[b[keep]]).sum(1)
    return float(np.quantile(sims, q))


def fit_threshold(sims, labels, default):
    """Decision threshold from answered pairs: the cut maximising balanced accuracy (positives above).
    Falls back to `default` until both answers have been seen."""
    sims, labels = np.asarray(sims, np.float64), np.asarray(labels, bool)
    if labels.all() or not labels.any():
        return float(default)
    cuts = np.unique(sims)
    cuts = (cuts[:-1] + cuts[1:]) / 2 if len(cuts) > 1 else cuts
    pos, neg = sims[labels], sims[~labels]
    tpr = (pos[None, :] > cuts[:, None]).mean(1)
    tnr = (neg[None, :] <= cuts[:, None]).mean(1)
    return float(cuts[np.argmax(tpr + tnr)])
