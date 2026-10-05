"""Which proposed pairs to show the annotator.

A strategy ranks the candidate pairs (candidates.candidate_pairs); loop.ActiveRun walks the ranking and
asks about each pair whose answer is not already implied (constraints.ConstraintStore.infer) until the
round's budget is spent. Every strategy sees only label-free information: features, camera ids, the
threshold tau and the answers given so far.

  random      random order of the proposed pairs
  confident   most similar first (the pairs the model is surest are one person; little to learn from)
  uncertain   similarity closest to tau first (decision boundary)
  balanced    alternately just above and just below tau (LBAS-style positive / negative balance)
  cover       ours: uncertainty, plus coverage of the domain --
                * part of the budget (--expand_ratio) goes to pairs touching annotated images (grow or
                  link the identity clusters), the rest to pairs of unseen images (new identities);
                * no two pairs in a round whose images look like the same person (redundant answers);
                * a quota per camera pair, so every camera combination of the network is asked about.
"""
import math

import numpy as np


class SelectionContext:
    def __init__(self, store, budget, rng, feats, camids, tau, dup_tau, has_cameras=True, expand_ratio=0.5):
        self.store, self.budget, self.rng = store, budget, rng
        self.feats, self.camids, self.has_cameras = feats, np.asarray(camids), has_cameras
        self.tau, self.dup_tau, self.expand_ratio = tau, dup_tau, expand_ratio


def _uncertainty(sim, tau):
    d = np.abs(sim - tau)
    return 1 - d / (d.max() + 1e-9)


def select_random(cand, ctx):
    return ctx.rng.permutation(len(cand["sim"]))


def select_confident(cand, ctx):
    return np.lexsort((ctx.rng.rand(len(cand["sim"])), -cand["sim"]))


def select_uncertain(cand, ctx):
    return np.lexsort((ctx.rng.rand(len(cand["sim"])), np.abs(cand["sim"] - ctx.tau)))


def select_balanced(cand, ctx):
    sim = cand["sim"]
    above = np.flatnonzero(sim > ctx.tau)
    below = np.flatnonzero(sim <= ctx.tau)
    above = above[np.argsort(sim[above])]    # closest above tau first
    below = below[np.argsort(-sim[below])]   # closest below tau first
    out = []
    for a, b in zip(above, below):
        out += [a, b]
    n = min(len(above), len(below))
    return np.array(out + list(above[n:]) + list(below[n:]), dtype=np.int64)


def _greedy(order, cand, ctx, n_target, group_key=None, per_group=2):
    """Walk `order`, keeping a pair unless (a) one of its images looks like the same person as an image
    already kept this round, (b) its camera pair has used its quota, or (c) its existing cluster (group_key)
    has used its quota. Returns the kept positions (at most n_target)."""
    X, cams = ctx.feats, ctx.camids
    i_all, j_all = cand["i"], cand["j"]
    if ctx.has_cameras:
        cam_key = np.minimum(cams[i_all], cams[j_all]) * 100000 + np.maximum(cams[i_all], cams[j_all])
        n_keys = max(1, len(np.unique(cam_key[order]))) if len(order) else 1
        cam_cap = max(2, math.ceil(2 * n_target / n_keys))
    kept, cam_used, grp_used = [], {}, {}
    near = np.full(len(X), -np.inf, np.float32)  # max similarity of every image to the images kept so far
    for p in order:
        if len(kept) >= n_target:
            break
        i, j = int(i_all[p]), int(j_all[p])
        if max(near[i], near[j]) > ctx.dup_tau:
            continue
        if ctx.has_cameras:
            c = int(cam_key[p])
            if cam_used.get(c, 0) >= cam_cap:
                continue
        if group_key is not None:
            g = group_key(i, j)
            if grp_used.get(g, 0) >= per_group:
                continue
            grp_used[g] = grp_used.get(g, 0) + 1
        if ctx.has_cameras:
            cam_used[c] = cam_used.get(c, 0) + 1
        kept.append(int(p))
        near = np.maximum(near, np.maximum(X @ X[i], X @ X[j]))
    return kept


def select_cover(cand, ctx):
    n = len(cand["sim"])
    u = _uncertainty(cand["sim"], ctx.tau)
    store = ctx.store
    lab_i = np.array([store.labeled(int(x)) for x in cand["i"]], bool)
    lab_j = np.array([store.labeled(int(x)) for x in cand["j"]], bool)
    touch = np.flatnonzero(lab_i | lab_j)
    fresh = np.flatnonzero(~(lab_i | lab_j))
    tie = ctx.rng.rand(n)
    touch = touch[np.lexsort((tie[touch], -u[touch]))]
    fresh = fresh[np.lexsort((tie[fresh], -u[fresh]))]
    n_target = 3 * ctx.budget  # reserve: some kept pairs turn out to be implied by earlier answers

    def cluster_of(i, j):  # the existing cluster(s) a pair touches
        ri, rj = store.find(i), store.find(j)
        return (min(x for x in (ri, rj) if x is not None), max(x for x in (ri, rj) if x is not None))

    a = _greedy(touch, cand, ctx, int(round(ctx.expand_ratio * n_target)), group_key=cluster_of)
    b = _greedy(fresh, cand, ctx, n_target)
    order, ia, ib = [], 0, 0
    while ia < len(a) or ib < len(b):  # interleave in the ratio expand_ratio : 1 - expand_ratio
        want_a = ia < len(a) and (ib >= len(b) or ia < ctx.expand_ratio * (ia + ib + 1))
        if want_a:
            order.append(a[ia]); ia += 1
        else:
            order.append(b[ib]); ib += 1
    used = set(order)
    rest = [p for p in np.argsort(-u) if p not in used]  # fallback when the reserve runs out
    return np.array(order + rest, dtype=np.int64)


STRATEGIES = {
    "random": select_random,
    "confident": select_confident,
    "uncertain": select_uncertain,
    "balanced": select_balanced,
    "cover": select_cover,
}
