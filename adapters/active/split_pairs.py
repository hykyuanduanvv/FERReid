"""Split pairs of a clustering (task list v3: E1a self-healing analysis, E2 oracle_merge_subset). Person ids are used
here, so this is oracle / analysis code: it never feeds a label-free selector.

Units: every pseudo cluster (assigned to its majority person, as oracle_merge does) and every outlier image (its own
person). A person spread over several units is split; its main unit is the one holding most of its images (ties:
the cluster with the lowest label, clusters before outliers). Every other unit of that person forms one split pair
(unit, main unit). Merging a pair = one "same person" answer between the two units' representative images (for a
cluster: its majority person's image closest to the normalised cluster centroid -- an impure cluster's plain medoid
can be another person; for an outlier: itself), kept across rounds as a must-link like any answer.
"""
import numpy as np


def split_pairs(labels, pids, feats):
    """List of dicts {pid, rep_a, rep_b, unit_a, unit_b, size_a, size_b, kind} for the clustering `labels`
    (-1 outlier), person ids `pids`, L2-normalised features `feats` (N, D) numpy. rep_b / unit_b: the main unit."""
    labels, pids = np.asarray(labels), np.asarray(pids)
    X = np.asarray(feats, np.float32)
    units = {}  # person -> list of (n_own_images, is_outlier, unit_key, rep, size)
    for c in np.unique(labels[labels >= 0]):
        m = np.flatnonzero(labels == c)
        u, n = np.unique(pids[m], return_counts=True)
        maj = u[np.argmax(n)]
        cent = X[m].mean(0)
        cent /= np.linalg.norm(cent) + 1e-12
        own = m[pids[m] == maj]  # the representative is an image of the majority person (oracle code)
        rep = int(own[np.argmax(X[own] @ cent)])
        units.setdefault(maj.item(), []).append((int(n.max()), 0, ("c", int(c)), rep, len(m)))
    for i in np.flatnonzero(labels < 0):
        units.setdefault(pids[i].item(), []).append((1, 1, ("o", int(i)), int(i), 1))
    out = []
    for p, us in units.items():
        if len(us) < 2:
            continue
        us = sorted(us, key=lambda t: (-t[0], t[1], t[2][1]))
        main = us[0]
        for u in us[1:]:
            out.append({"pid": p, "rep_a": u[3], "rep_b": main[3], "unit_a": "{}{}".format(*u[2]),
                        "unit_b": "{}{}".format(*main[2]), "size_a": u[4], "size_b": main[4],
                        "kind": "outlier" if u[1] else "cluster"})
    return out


def healed(pairs, labels):
    """For each pair (rep_a, rep_b): True when both representatives sit in the same (non-outlier) cluster."""
    labels = np.asarray(labels)
    return np.array([labels[p["rep_a"]] >= 0 and labels[p["rep_a"]] == labels[p["rep_b"]] for p in pairs], bool)
