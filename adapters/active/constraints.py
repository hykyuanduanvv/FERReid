"""Answers to pair queries as must-link / cannot-link constraints.

Positive answers merge images into identity clusters (union-find); negative answers become cannot-links
between clusters. Transitivity answers further pairs without asking:
  i ~ j and j ~ k             ->  i ~ k
  i ~ j and j !~ k            ->  i !~ k
so a strategy's ranked list is consumed until the budget is spent on pairs whose answer is unknown.
"""


class ConstraintStore:

    def __init__(self):
        self._parent = {}
        self._members = {}   # root -> list of images
        self._cannot = {}    # root -> set of roots known to be other people
        self.answers = []    # (i, j, same) in query order
        self.n_inferred = 0  # pairs a strategy proposed whose answer followed from earlier answers

    # ------------------------------------------------------------------ union-find

    def _add_node(self, i):
        if i not in self._parent:
            self._parent[i] = i
            self._members[i] = [i]
            self._cannot[i] = set()

    def find(self, i):
        if i not in self._parent:
            return None
        root = i
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[i] != root:  # path compression
            self._parent[i], i = root, self._parent[i]
        return root

    def labeled(self, i):
        return i in self._parent

    def images(self):
        """Every image that appears in an answer."""
        return list(self._parent)

    def infer(self, i, j):
        """True / False when the answer follows from the stored answers, else None."""
        ri, rj = self.find(i), self.find(j)
        if ri is None or rj is None:
            return None
        if ri == rj:
            return True
        if rj in self._cannot[ri]:
            return False
        return None

    def add(self, i, j, same):
        i, j = int(i), int(j)
        self._add_node(i)
        self._add_node(j)
        self.answers.append((i, j, bool(same)))
        ri, rj = self.find(i), self.find(j)
        if same:
            if ri == rj:
                return
            if rj in self._cannot[ri]:
                raise ValueError("contradicting answers for images {} and {}".format(i, j))
            if len(self._members[ri]) < len(self._members[rj]):
                ri, rj = rj, ri
            self._parent[rj] = ri
            self._members[ri] += self._members.pop(rj)
            moved = self._cannot.pop(rj)
            self._cannot[ri] |= moved
            for r in moved:
                self._cannot[r].discard(rj)
                self._cannot[r].add(ri)
        else:
            if ri == rj:
                raise ValueError("contradicting answers for images {} and {}".format(i, j))
            self._cannot[ri].add(rj)
            self._cannot[rj].add(ri)

    # ------------------------------------------------------------------ views

    def clusters(self, min_size=1):
        """Identity clusters (lists of pool indices), largest first; singletons = images only seen in
        negative answers."""
        out = [sorted(m) for m in self._members.values() if len(m) >= min_size]
        return sorted(out, key=lambda c: (-len(c), c[0]))

    def cannot_links(self, clusters):
        """Cannot-link pairs as positions into `clusters` (as returned by clusters())."""
        pos = {self.find(c[0]): n for n, c in enumerate(clusters)}
        out = set()
        for r, others in self._cannot.items():
            for o in others:
                if r in pos and o in pos:
                    out.add((min(pos[r], pos[o]), max(pos[r], pos[o])))
        return sorted(out)

    def stats(self):
        n_pos = sum(1 for *_, s in self.answers if s)
        big = [m for m in self._members.values() if len(m) >= 2]
        return {"n_queries": len(self.answers), "n_pos": n_pos, "n_neg": len(self.answers) - n_pos,
                "n_inferred": self.n_inferred, "n_clusters": len(big),
                "n_cluster_imgs": sum(len(m) for m in big), "n_labeled_imgs": len(self._parent)}
