"""Context (support set) selection for in-context ReID.

A selector sees the unlabeled target-domain candidate pool and returns k identities
to annotate. Annotation is simulated by revealing, for each chosen identity, one
cross-camera positive pair. New selection methods go into SELECTORS.

Selector signature:
    fn(pool, k, rng, **kwargs) -> list of pids
    pool: CandidatePool
    rng:  np.random.RandomState (use it for all randomness so runs are reproducible)
"""
from collections import OrderedDict

import numpy as np


class CandidatePool:
    """Target-domain train split grouped by identity.

    Only identities seen by >= 2 cameras are eligible, since each must yield a
    cross-camera positive pair once annotated.
    """

    def __init__(self, train_data):
        self.pid2items = OrderedDict()  # keeps torchreid's listing order
        for img_path, pid, camid, *_ in train_data:
            self.pid2items.setdefault(pid, []).append((img_path, camid))
        self.eligible_pids = [
            pid for pid, items in self.pid2items.items()
            if len({camid for _, camid in items}) >= 2
        ]
        # flat view for label-free selectors (pids must not be used for scoring)
        self.images = [(p, c) for items in self.pid2items.values() for p, c in items]

    def __len__(self):
        return len(self.eligible_pids)


def select_first(pool, k, rng, **kwargs):
    """VICP default: the first k identities in the listing order."""
    return pool.eligible_pids[:k]


def select_random(pool, k, rng, **kwargs):
    idx = rng.choice(len(pool.eligible_pids), size=k, replace=False)
    return [pool.eligible_pids[i] for i in idx]


SELECTORS = {
    "first": select_first,
    "random": select_random,
}


def select(method, pool, k, rng, **kwargs):
    if k > len(pool):
        raise ValueError("budget k={} exceeds {} eligible identities".format(k, len(pool)))
    pids = SELECTORS[method](pool, k, rng, **kwargs)
    assert len(set(pids)) == k, "selector must return k distinct identities"
    return pids


def make_pairs(pool, pids, rng):
    """One cross-camera positive pair per identity, two distinct images."""
    pairs = []
    for pid in pids:
        items = pool.pid2items[pid]
        cams = sorted({camid for _, camid in items})
        cam_a, cam_b = rng.choice(cams, size=2, replace=False)
        imgs_a = [p for p, c in items if c == cam_a]
        imgs_b = [p for p, c in items if c == cam_b]
        pairs.append((imgs_a[rng.randint(len(imgs_a))], imgs_b[rng.randint(len(imgs_b))]))
    return pairs
