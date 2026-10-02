"""Label-free context selectors (direction B) for --selection_unit image.

A selector sees only the unlabeled pool: image paths, camera ids and features computed without labels
(attached by attach_features). It returns k distinct anchor-image indices; annotation is simulated
afterwards (adapters/context_selection.annotate). Person ids are never visible here.

Features (extract_selector_features): L2-normalized CLS features of the model's own source-trained encoder
without prompts (label-free on the target; the frozen pretrained copy is available as an option but
is a much weaker person-similarity) -- and "style" statistics (channel mean
and std of the patch tokens after the first blocks), z-scored per dimension.

Families (see docs/RUN_PLAN_20261001.md section 9):
  efficiency      dedup, pairable
  representative  typical
  coverage        kcenter, camera_balanced, style_cover
  informative     hard_negative
  combined        facility, facility_camera (facility location = representativeness + coverage,
                  + camera-coverage bonus, + de-duplication; greedy, monotone submodular objective)
"""
import numpy as np
import torch
import torch.nn.functional as F


# ============================================================================ features

@torch.no_grad()
def extract_selector_features(model, images, device, batch_size=256, style_blocks=4, source="trained"):
    """images: an (N, 3, H, W) tensor, or an iterable of image batches (e.g. a DataLoader yielding
    (images, ...) in pool order). Returns (feats (N, D) L2-normalized, style (N, 2D) z-scored),
    both float32 numpy arrays.

    source="trained" (default): the model's own retrieval encoder (source-trained LoRA / weights) without
    any prompt -- label-free on the target, and a far better person-similarity than the raw pretrained
    backbone (needed for the "probably the same person" de-duplication);
    source="pretrained": VICP's frozen pretrained copy (encoder_copy)."""
    enc = model.encoder
    if source == "pretrained" and hasattr(getattr(model, "encoder_copy", None), "prepare_tokens_with_masks"):
        enc = model.encoder_copy
    dtype = enc.patch_embed.proj.weight.dtype
    if torch.is_tensor(images):
        batches = (images[i:i + batch_size] for i in range(0, len(images), batch_size))
    else:
        batches = (b[0] if isinstance(b, (tuple, list)) else b for b in images)
    feats, styles = [], []
    for x in batches:
        x = x.to(device=device, dtype=dtype)
        feats.append(F.normalize(enc.forward_features(x)["x_norm_clstoken"].float(), dim=1).cpu())
        t = enc.prepare_tokens_with_masks(x, None)
        for blk in enc.blocks[:style_blocks]:
            t = blk(t)
        p = t[:, 1:].float()
        styles.append(torch.cat([p.mean(1), p.std(1)], dim=1).cpu())
    feats = torch.cat(feats).numpy()
    style = torch.cat(styles).numpy()
    style = (style - style.mean(0)) / (style.std(0) + 1e-6)
    return feats.astype(np.float32), style.astype(np.float32)


# ============================================================================ pool statistics (cached)

def _stats(pool):
    """Label-free per-image statistics, computed once per pool:
    nn_sim      cosine to the nearest other image
    typicality  1 / mean (1 - cos) to the 20 nearest neighbours (high = dense, typical region)
    pairability best cosine to an image from another camera (any other image without cameras)
    dup_tau     de-duplication threshold = 99th percentile of random-pair similarity (pairs above it
                are probably one person)
    partner     mutual cross-camera nearest neighbour (i and j are each other's best match in another
                camera: probably two views of one person), -1 if none"""
    if getattr(pool, "_sel_stats", None) is not None:
        return pool._sel_stats
    X = torch.from_numpy(pool.feats)
    cams = torch.tensor(pool.camids)
    n, m = len(X), min(20, len(X) - 1)
    nn_sim, typ, pair = np.empty(n, np.float32), np.empty(n, np.float32), np.empty(n, np.float32)
    pair_idx = np.empty(n, np.int64)
    for i in range(0, n, 2048):
        S = X[i:i + 2048] @ X.T
        rows = torch.arange(i, min(i + 2048, n))
        S[torch.arange(len(rows)), rows] = -2  # exclude self
        top = S.topk(m, dim=1).values
        nn_sim[rows] = top[:, 0].numpy()
        typ[rows] = (1.0 / (1 - top).clamp(min=1e-4).mean(1)).numpy()
        if pool.has_cameras:
            S = S.masked_fill(cams[rows][:, None] == cams[None, :], -2)
        best = S.max(1)
        pair[rows], pair_idx[rows] = best.values.numpy(), best.indices.numpy()
    # de-duplication threshold: random pairs are almost always two different people, so a pair more
    # similar than the top 1% of random pairs is treated as "probably the same person" (a person has many
    # images -- ~17 per person in Market -- so the 99.9th percentile can already fall among same-person pairs)
    # (a nearest-neighbour statistic would not do: an image's nearest neighbour is often the same person)
    if n * (n - 1) // 2 <= 200000:  # small pools (e.g. a training batch): every pair, exactly
        iu = np.triu_indices(n, 1)
        rand_sim = (pool.feats @ pool.feats.T)[iu]
    else:
        g = np.random.RandomState(0)
        a, b = g.randint(n, size=200000), g.randint(n, size=200000)
        keep = a != b
        rand_sim = (pool.feats[a[keep]] * pool.feats[b[keep]]).sum(1)
    mutual = pair_idx[pair_idx] == np.arange(n)  # i and pair_idx[i] are each other's best cross-camera match
    pool._sel_stats = {"nn_sim": nn_sim, "typicality": typ, "pairability": pair,
                       "partner": np.where(mutual, pair_idx, -1),
                       "dup_tau": float(np.quantile(rand_sim, 0.99))}
    return pool._sel_stats


def _need(pool, style=False):
    if getattr(pool, "feats", None) is None or (style and getattr(pool, "style", None) is None):
        raise ValueError("this selector needs label-free features: call attach_features() on the pool "
                         "(the evaluation scripts do this automatically)")


def _is_dup(pool, chosen, i):
    """Label-free "probably a person already chosen": near-identical to a chosen image, or the mutual
    cross-camera best match of a chosen image."""
    if not chosen:
        return False
    st = _stats(pool)
    return (int(i) in chosen or float((pool.feats[chosen] @ pool.feats[i]).max()) > st["dup_tau"]
            or int(st["partner"][i]) in chosen)


def _dedup_fill(pool, order, k, chosen=None):
    """Take indices from `order`, skipping probable duplicates of already chosen images."""
    chosen = list(chosen or [])
    for i in order:
        if len(chosen) == k:
            break
        if _is_dup(pool, chosen, i):
            continue
        chosen.append(int(i))
    for i in order:  # pool exhausted under the constraint: fill without it
        if len(chosen) == k:
            break
        if i not in chosen:
            chosen.append(int(i))
    return chosen


def _kmeans(X, k, rng):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=3, random_state=int(rng.randint(2 ** 31 - 1)))
    labels = km.fit_predict(X)
    return labels, km.cluster_centers_


# ============================================================================ selectors

def select_dedup(pool, k, rng, **kw):
    """Random order, but never two anchors that look like the same person (efficiency)."""
    _need(pool)
    return _dedup_fill(pool, rng.permutation(len(pool)), k)


def select_pairable(pool, k, rng, **kw):
    """Images with a confident match in another camera first (fewer failed annotations), de-duplicated.
    Ties broken randomly."""
    _need(pool)
    st = _stats(pool)
    order = np.lexsort((rng.rand(len(pool)), -st["pairability"]))
    return _dedup_fill(pool, order, k)


def select_typical(pool, k, rng, **kw):
    """TypiClust-style: k clusters in feature space, the most typical (densest) image of each."""
    _need(pool)
    st = _stats(pool)
    labels, _ = _kmeans(pool.feats, k, rng)
    chosen = []
    for c in range(k):  # most typical member of each cluster that is not a probable duplicate
        members = np.where(labels == c)[0]
        for i in members[np.argsort(-st["typicality"][members])]:
            if not _is_dup(pool, chosen, i):
                chosen.append(int(i))
                break
    return _dedup_fill(pool, np.argsort(-st["typicality"]), k, chosen=chosen)


def select_kcenter(pool, k, rng, **kw):
    """Greedy farthest-point (k-center / core-set) from a random start: maximal coverage of the space."""
    _need(pool)
    X = pool.feats
    chosen = [int(rng.randint(len(pool)))]
    dist = 1 - X @ X[chosen[0]]
    while len(chosen) < k:
        i = int(np.argmax(dist))
        chosen.append(i)
        dist = np.minimum(dist, 1 - X @ X[i])
    return chosen


def select_camera_balanced(pool, k, rng, **kw):
    """Anchors spread evenly over cameras (round robin, random within a camera), de-duplicated.
    Falls back to dedup when the domain has no real cameras."""
    _need(pool)
    if not pool.has_cameras:
        return select_dedup(pool, k, rng)
    cams = np.array(pool.camids)
    per_cam = {c: list(rng.permutation(np.where(cams == c)[0])) for c in rng.permutation(np.unique(cams))}
    order = []
    while any(per_cam.values()):
        for c in list(per_cam):
            if per_cam[c]:
                order.append(per_cam[c].pop(0))
    return _dedup_fill(pool, order, k)


def select_style_cover(pool, k, rng, **kw):
    """k clusters of the style statistics (early-layer channel mean / std), the image closest to each
    centroid: one anchor per appearance condition (lighting, resolution, camera look)."""
    _need(pool, style=True)
    labels, centers = _kmeans(pool.style, k, rng)
    chosen = []
    for c in range(k):  # closest-to-centre member of each style cluster that is not a probable duplicate
        members = np.where(labels == c)[0]
        d = ((pool.style[members] - centers[c]) ** 2).sum(1)
        for i in members[np.argsort(d)]:
            if not _is_dup(pool, chosen, i):
                chosen.append(int(i))
                break
    return _dedup_fill(pool, rng.permutation(len(pool)), k, chosen=chosen)


def select_hard_negative(pool, k, rng, **kw):
    """Pairs of look-alike images that are probably *different* people (similar, but below the
    same-person threshold): their cross-pairs become hard negative questions for the generator."""
    _need(pool)
    st = _stats(pool)
    X, tau = pool.feats, st["dup_tau"]
    chosen = []
    for a in rng.permutation(len(pool)):
        if len(chosen) >= k:
            break
        if _is_dup(pool, chosen, a):
            continue
        chosen.append(int(a))
        sims = X @ X[a]
        sims[a] = -2
        sims[sims > tau] = -2  # probably the same person
        if st["partner"][a] >= 0:
            sims[st["partner"][a]] = -2
        for j in np.argsort(-sims):  # the most similar image that is probably someone else
            if len(chosen) >= k or sims[j] <= -2:
                break
            if not _is_dup(pool, chosen, j):
                chosen.append(int(j))
                break
    return _dedup_fill(pool, rng.permutation(len(pool)), k, chosen=chosen[:k])


def _facility(pool, k, rng, camera_bonus=0.0, n_eval=4000, n_cand=8000):
    """Greedy facility location F(S) = sum_x max_{s in S} sim(x, s) (sim = clipped cosine) over a random
    subset of evaluation points, + camera_bonus * (normalised) for a candidate from a camera not yet
    covered, subject to the de-duplication constraint. F is monotone submodular: plain greedy is a
    (1 - 1/e)-approximation of the facility term."""
    _need(pool)
    st = _stats(pool)
    X, n = pool.feats, len(pool)
    ev = rng.choice(n, size=min(n, n_eval), replace=False)
    cand = rng.choice(n, size=min(n, n_cand), replace=False)
    S = np.clip(X[cand] @ X[ev].T, 0, None)            # (cand, eval)
    cur = np.zeros(len(ev), np.float32)
    cams = np.array(pool.camids)
    covered, chosen = set(), []
    alive = np.ones(len(cand), bool)
    while len(chosen) < k and alive.any():
        gain = np.maximum(S - cur, 0).sum(1)
        score = gain / max(gain[alive].max(), 1e-9)
        if camera_bonus and pool.has_cameras:
            score = score + camera_bonus * np.array([cams[c] not in covered for c in cand])
        score[~alive] = -np.inf
        j = int(np.argmax(score))
        i = int(cand[j])
        chosen.append(i)
        covered.add(cams[i])
        cur = np.maximum(cur, S[j])
        alive &= (X[cand] @ X[i]) <= st["dup_tau"]   # de-duplication: near-identical images
        alive &= cand != st["partner"][i]             # and the mutual cross-camera match of i
        alive[j] = False
    return _dedup_fill(pool, rng.permutation(n), k, chosen=chosen)


def select_facility(pool, k, rng, **kw):
    return _facility(pool, k, rng, camera_bonus=0.0)


def select_facility_camera(pool, k, rng, **kw):
    return _facility(pool, k, rng, camera_bonus=kw.get("camera_bonus", 0.5))


SELECTORS = {
    "dedup": select_dedup,
    "pairable": select_pairable,
    "typical": select_typical,
    "kcenter": select_kcenter,
    "camera_balanced": select_camera_balanced,
    "style_cover": select_style_cover,
    "hard_negative": select_hard_negative,
    "facility": select_facility,
    "facility_camera": select_facility_camera,
}
FEATURE_FREE = {"random", "first"}


# ============================================================================ properties of a selection

def selection_properties(pool, idx):
    """Label-free description of a selection, for the "what makes a context good" analysis."""
    if getattr(pool, "feats", None) is None:
        return {}
    st = _stats(pool)
    X = pool.feats[idx]
    sims = X @ X.T
    off = sims[~np.eye(len(idx), dtype=bool)] if len(idx) > 1 else np.array([1.0])
    cams = [pool.camids[i] for i in idx]
    return {
        "p_n_cams": len(set(cams)) if pool.has_cameras else float("nan"),
        "p_diversity": float(1 - off.mean()),
        "p_max_sim": float(off.max()),
        "p_typicality": float(st["typicality"][idx].mean()),
        "p_pairability": float(st["pairability"][idx].mean()),
        "p_style_spread": float(pool.style[idx].std(0).mean()) if getattr(pool, "style", None) is not None else float("nan"),
    }
