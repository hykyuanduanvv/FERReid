"""Simulated annotator. It holds the pool's person ids; the active module only asks it questions.

  same(i, j)        "are pool images i and j the same person?"   (one pair query)
  partner(i, rng)   "find this person in another camera"          (one anchor annotation, the ID-level
                                                                    protocol of the in-context setting)
Everything else here is for reporting and never feeds back into selection.
"""
import numpy as np


class PairOracle:

    def __init__(self, pids, camids, has_cameras=True):
        self._pids = np.asarray(pids)
        self._cams = np.asarray(camids)
        self._has_cameras = has_cameras
        self._by_pid = {}
        for i, p in enumerate(self._pids.tolist()):
            self._by_pid.setdefault(p, []).append(i)
        self.n_pair_queries = 0
        self.n_anchor_queries = 0

    def same(self, i, j):
        self.n_pair_queries += 1
        return bool(self._pids[i] == self._pids[j])

    def partner(self, i, rng):
        """Another image of the anchor's person (from another camera when the domain has cameras), or None."""
        self.n_anchor_queries += 1
        p, c = self._pids[i], self._cams[i]
        cands = [j for j in self._by_pid[p] if j != i and (not self._has_cameras or self._cams[j] != c)]
        return int(cands[rng.randint(len(cands))]) if cands else None

    # ------------------------------------------------------------------ reporting only

    def report(self, clusters):
        """Quality of the positive clusters (lists of pool indices): distinct true identities covered,
        purity (share of images agreeing with the majority person of their cluster) and the number of
        clusters that are in fact the same person as another cluster (not merged yet)."""
        big = [c for c in clusters if len(c) >= 2]
        if not big:
            return {"true_ids": 0, "purity": float("nan"), "split_ids": 0}
        majority, agree, total = [], 0, 0
        for c in big:
            vals, counts = np.unique(self._pids[c], return_counts=True)
            majority.append(vals[np.argmax(counts)])
            agree += counts.max()
            total += len(c)
        return {"true_ids": len(set(majority)), "purity": agree / total,
                "split_ids": len(majority) - len(set(majority))}

    def pseudo_report(self, labels):
        """Quality of pseudo labels (-1: outlier) against the person ids: pairwise precision / recall / F1 over
        the clustered images, NMI, number of clusters, share of outliers, and the identities whose images are
        split over several clusters (e.g. by camera)."""
        from sklearn.metrics import normalized_mutual_info_score
        labels = np.asarray(labels)
        ok = labels >= 0
        out = {"pseudo_clusters": int(labels.max()) + 1 if ok.any() else 0, "pseudo_outliers": float(1 - ok.mean())}
        if ok.sum() < 2:
            return dict(out, pw_prec=float("nan"), pw_rec=float("nan"), pw_f=float("nan"), nmi=float("nan"),
                        pseudo_split_ids=0)
        y, c = self._pids[ok], labels[ok]
        pair = lambda n: (n * (n - 1) // 2).sum()
        _, joint = np.unique(np.stack([y, c]), axis=1, return_counts=True)
        tp = pair(joint)
        pred = pair(np.unique(c, return_counts=True)[1])
        true = pair(np.unique(self._pids, return_counts=True)[1])  # recall against every pool pair of a person
        prec, rec = tp / max(pred, 1), tp / max(true, 1)
        n_split = sum(len(np.unique(c[y == p])) > 1 for p in np.unique(y))
        return dict(out, pw_prec=float(prec), pw_rec=float(rec), pw_f=float(2 * prec * rec / max(prec + rec, 1e-12)),
                    nmi=float(normalized_mutual_info_score(y, c)), pseudo_split_ids=int(n_split))

    def all_identities(self, min_images=2):
        """Every pool identity as a cluster (the full-annotation upper bound)."""
        return [v for v in self._by_pid.values() if len(v) >= min_images]
