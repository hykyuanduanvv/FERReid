"""Context (support set) selection for in-context ReID.

Two selection units (--selection_unit):

image (default, label-free)
    The selector sees only the unlabeled target-domain pool: image paths and camera ids (camera ids
    come with surveillance footage, no annotation needed) -- never person ids. It returns k *anchor
    images*. Annotation is simulated by an oracle that, for each anchor, returns another image of the
    same person from a different camera (any other image of that person in NO_CAMERA_DOMAINS).
    An anchor whose person has no such image, or whose person was already annotated, still consumes
    budget and yields no new pair (reported as n_fail / n_dup).
    Image selector signature:  fn(pool: ImagePool, k, rng, **kwargs) -> list of k distinct image indices

identity (historical)
    The selector picks k identities from the pid-grouped pool (only identities seen by >= 2 cameras)
    and one cross-camera positive pair is revealed per identity. Grouping the pool by pid uses the
    labels before annotation, so this is *not* label-free; kept to reproduce earlier results.
    Identity selector signature:  fn(pool: CandidatePool, k, rng, **kwargs) -> list of pids

All randomness must come from rng (np.random.RandomState) so runs are reproducible.
New selection methods go into IMAGE_SELECTORS (or SELECTORS for the identity unit).
"""
from collections import OrderedDict

import numpy as np


# ============================================================================ identity unit (historical)

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


# ============================================================================ image unit (label-free)

class _AnnotationOracle:
    """Holds the labels of the pool. Only annotate() may use it; selectors never see it."""

    def __init__(self, pids, camids):
        self._pids = list(pids)
        self._camids = list(camids)
        self._by_pid = {}
        for i, pid in enumerate(self._pids):
            self._by_pid.setdefault(pid, []).append(i)

    def pid(self, i):
        return self._pids[i]

    def partner(self, i, rng, cross_camera):
        """Index of another image of the same person (from another camera if cross_camera), or None."""
        pid, cam = self._pids[i], self._camids[i]
        cands = [j for j in self._by_pid[pid] if j != i and (not cross_camera or self._camids[j] != cam)]
        if not cands:
            return None
        return cands[rng.randint(len(cands))]


class ImagePool:
    """Unlabeled target-domain pool (the train split): what a label-free selector may look at.

    Public attributes: paths, camids (camera ids; not meaningful when has_cameras is False), has_cameras.
    Person ids live in a private oracle used only by annotate().
    """

    def __init__(self, train_data, has_cameras=True):
        self.paths = [x[0] for x in train_data]
        self.camids = [x[2] for x in train_data]
        self.has_cameras = has_cameras
        self.feats = None   # (N, D) label-free features, see attach_features()
        self.style = None   # (N, S) label-free style statistics
        self._oracle = _AnnotationOracle([x[1] for x in train_data], self.camids)
        self._sel_stats = None

    def __len__(self):
        return len(self.paths)

    def attach_features(self, feats, style=None):
        """Label-free features (row i = image i of this pool) for the feature-based selectors."""
        assert len(feats) == len(self.paths)
        self.feats, self.style, self._sel_stats = feats, style, None


def select_images_random(pool, k, rng, **kwargs):
    """Uniformly random anchor images (the label-free random baseline)."""
    return [int(i) for i in rng.choice(len(pool), size=k, replace=False)]


def select_images_first(pool, k, rng, **kwargs):
    """The first k images in the listing order (label-free analogue of VICP's 'first')."""
    return list(range(k))


IMAGE_SELECTORS = {
    "random": select_images_random,
    "first": select_images_first,
}
# label-free feature-based selectors (direction B): dedup, pairable, typical, kcenter, camera_balanced,
# style_cover, hard_negative, facility, facility_camera
from adapters.selectors import SELECTORS as _FEATURE_SELECTORS, FEATURE_FREE  # noqa: E402
IMAGE_SELECTORS.update(_FEATURE_SELECTORS)


def needs_features(method):
    return method not in FEATURE_FREE


def select_images(method, pool, k, rng, **kwargs):
    if k > len(pool):
        raise ValueError("budget k={} exceeds {} pool images".format(k, len(pool)))
    idx = IMAGE_SELECTORS[method](pool, k, rng, **kwargs)
    assert len(set(idx)) == k and all(0 <= i < len(pool) for i in idx), \
        "image selector must return k distinct valid image indices"
    return idx


def annotate(pool, anchors, rng):
    """Simulated annotation of the anchor images. Returns (pairs, info):
    pairs = [(anchor_path, partner_path)], one per newly annotated person;
    info = {k, n_pairs, n_fail (no valid partner), n_dup (person already annotated)}."""
    oracle = pool._oracle
    pairs, seen, n_fail, n_dup = [], set(), 0, 0
    for a in anchors:
        pid = oracle.pid(a)
        if pid in seen:
            n_dup += 1
            continue
        seen.add(pid)
        b = oracle.partner(a, rng, cross_camera=pool.has_cameras)
        if b is None:
            n_fail += 1
            continue
        pairs.append((pool.paths[a], pool.paths[b]))
    return pairs, {"k": len(anchors), "n_pairs": len(pairs), "n_fail": n_fail, "n_dup": n_dup}


# ============================================================================ single entry point

class ContextSampler:
    """Both pools of one target split; draw() returns the annotated context pairs for a budget k."""

    def __init__(self, train_data, has_cameras=True):
        self.identity_pool = CandidatePool(train_data)
        self.image_pool = ImagePool(train_data, has_cameras=has_cameras)

    def attach_features(self, feats, style=None):
        self.image_pool.attach_features(feats, style)

    @property
    def has_features(self):
        return self.image_pool.feats is not None

    def draw(self, unit, method, k, rng):
        """-> (pairs, info). unit "image": label-free anchors + simulated annotation;
        unit "identity": historical pid-level selection (exactly the earlier random stream)."""
        if unit == "identity":
            pids = select(method, self.identity_pool, k, rng)
            pairs = make_pairs(self.identity_pool, pids, rng)
            return pairs, {"k": k, "n_pairs": len(pairs), "n_fail": 0, "n_dup": 0,
                           "selected": " ".join(map(str, pids))}
        if unit == "image":
            anchors = select_images(method, self.image_pool, k, rng)
            pairs, info = annotate(self.image_pool, anchors, rng)
            info["selected"] = " ".join(map(str, anchors))
            from adapters.selectors import selection_properties
            info.update(selection_properties(self.image_pool, anchors))  # empty without features
            return pairs, info
        raise ValueError("--selection_unit must be image or identity, got {}".format(unit))
