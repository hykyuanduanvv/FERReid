"""VPT active selection: main-branch heuristics, anonymous view, isolated annotation RNG.

Facility variants remain heuristics: their exclusion and dynamic camera bonus do
not inherit the unconstrained facility-location approximation guarantee.
"""
import hashlib
import numpy as np
from adapters.selectors import SELECTORS, _stats

METHODS = ('random', 'dedup', 'kcenter', 'typical', 'hard_negative',
           'facility', 'facility_camera')


class AnonymousPool:
    __slots__ = ('feats', 'camids', 'has_cameras', 'style', '_sel_stats')

    def __init__(self, features, cameras, has_cameras):
        x = np.asarray(features, dtype=np.float32).copy()
        assert x.ndim == 2 and len(x) >= 2 and np.isfinite(x).all()
        assert np.allclose(np.linalg.norm(x, axis=1), 1, atol=2e-5)
        assert len(cameras) == len(x)
        self.feats = x
        self.camids = tuple(map(int, cameras))
        self.has_cameras = bool(has_cameras)
        self.style = None
        self._sel_stats = None

    def __len__(self):
        return len(self.feats)


def safe_kcenter(pool, k, rng):
    # Main's farthest-point rule, explicitly mask selected rows on ties.
    chosen = [int(rng.randint(len(pool)))]
    distance = np.maximum(0, 1 - pool.feats @ pool.feats[chosen[0]])
    while len(chosen) < k:
        distance[chosen] = -np.inf
        i = int(np.argmax(distance))
        chosen.append(i)
        distance = np.minimum(distance, np.maximum(0, 1 - pool.feats @ pool.feats[i]))
    return chosen


def choose(pool, method, k, seed):
    if method not in METHODS or not 1 <= k <= len(pool):
        raise ValueError((method, k, len(pool)))
    rng = np.random.RandomState(seed)
    if method == 'random':
        idx = list(map(int, rng.choice(len(pool), k, replace=False)))
    elif method == 'kcenter':
        idx = safe_kcenter(pool, k, rng)
    else:
        idx = list(map(int, SELECTORS[method](pool, k, rng)))
    assert len(idx) == k and len(set(idx)) == k
    assert all(0 <= i < len(pool) for i in idx)
    return idx


def anchor_seed(domain, split, annotation_seed, anchor):
    key = f'{domain}:{split}:{annotation_seed}:{anchor}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], 'little')


def annotate_selected(records, anchors, *, has_cameras, domain, split, annotation_seed):
    """Only called AFTER choose returns. Labels never enter AnonymousPool.

    Each anchor has its own deterministic annotation stream, shared across methods.
    No replacement on duplicates/failures. This is simulated pair annotation, not
    a claim that finding a cross-camera partner is free.
    """
    by_pid = {}
    for i, row in enumerate(records):
        by_pid.setdefault(row[1], []).append(i)
    pairs, indices, revealed_ids, seen, outcomes = [], [], [], set(), []
    for a in anchors:
        path, pid, cam = records[a][:3]
        if pid in seen:
            outcomes.append(dict(anchor=a, result='duplicate'))
            continue
        seen.add(pid)
        eligible = [j for j in by_pid[pid]
                    if j != a and (not has_cameras or records[j][2] != cam)]
        if not eligible:
            outcomes.append(dict(anchor=a, result='no_partner'))
            continue
        rng = np.random.RandomState(anchor_seed(domain, split, annotation_seed, a))
        b = int(eligible[rng.randint(len(eligible))])
        pairs.append((path, records[b][0]))
        indices.append((a, b)); revealed_ids.append(int(pid))
        outcomes.append(dict(anchor=a, result='pair', partner=b))
    info = dict(k=len(anchors), n_pairs=len(pairs),
                n_dup=sum(r['result'] == 'duplicate' for r in outcomes),
                n_fail=sum(r['result'] == 'no_partner' for r in outcomes),
                outcomes=outcomes, pair_indices=indices)
    assert info['k'] == info['n_pairs'] + info['n_dup'] + info['n_fail']
    assert len(set(revealed_ids)) == len(pairs)
    return dict(anchors=list(anchors), pairs=pairs, annotation=info,
                effective_identities=len(pairs), revealed_ids=revealed_ids)


def selection_metrics(pool, anchors):
    st = _stats(pool)
    x = pool.feats[anchors]
    s = x @ x.T
    mask = ~np.eye(len(x), dtype=bool)
    return dict(cameras=len(set(pool.camids[i] for i in anchors)) if pool.has_cameras else None,
                mean_pairwise_distance=float(1 - s[mask].mean()) if mask.any() else None,
                typicality=float(st['typicality'][anchors].mean()),
                pairability=float(st['pairability'][anchors].mean()))


def unit_checks():
    x = np.tile(np.array([[1, 0, 0]], np.float32), (12, 1))
    pool = AnonymousPool(x, [0, 1] * 6, True)
    assert not hasattr(pool, 'paths') and not hasattr(pool, '_oracle')
    assert not hasattr(pool, 'pids') and not hasattr(pool, '__dict__')
    for method in METHODS:
        first = choose(pool, method, 8, 1600)
        assert first == choose(pool, method, 8, 1600)
    # All-identical features exercise kcenter's repeated-index and empty-cluster cases.
    assert len(set(choose(pool, 'kcenter', 12, 1))) == 12
    try:
        choose(pool, 'random', 13, 1)
    except ValueError:
        pass
    else:
        raise AssertionError('oversized budget accepted')
    records = [('a', 0, 0), ('b', 0, 1), ('c', 0, 1),
               ('d', 1, 0), ('e', 2, 0), ('f', 2, 1)]
    kw = dict(has_cameras=True, domain='synthetic', split=0, annotation_seed=9)
    a = annotate_selected(records, [0, 1, 3, 4], **kw)
    assert (a['annotation']['n_pairs'], a['annotation']['n_dup'], a['annotation']['n_fail']) == (2, 1, 1)
    b = annotate_selected(records, [4, 0], **kw)
    mapping_a = dict(a['annotation']['pair_indices'])
    assert dict(b['annotation']['pair_indices']) == {k: mapping_a[k] for k in [4, 0]}
    no_cam = AnonymousPool(np.eye(5, dtype=np.float32), [0]*5, False)
    assert choose(no_cam, 'facility', 3, 2) == choose(no_cam, 'facility_camera', 3, 2)
    return dict(all_methods_unique_reproducible=True, anonymous_interface=True,
                duplicate_feature_kcenter=True, empty_cluster_fill=True,
                budget_failure_accounting=True, annotation_order_independent=True,
                no_camera_equivalence=True)
