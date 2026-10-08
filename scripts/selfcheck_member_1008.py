"""10-08 code self-check (task order 2.5), no GPU training:
  1. off:member on a round-0 export (features + round-0 labels): the 20 / B most suspicious questions and the share
     of true "different" answers (person ids read only to score the questions, after selection);
  2. constraints: merge "different" answers (split=False) do not split clusters, member ones (split=True) do.

  python scripts/selfcheck_member_1008.py /data1/yangbin/dz/experiments/exp_1006_2_20261006/round0/cuhk03/cuhk03_epoch0.npz 121
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from adapters.active.constraints import ConstraintStore
from adapters.active.loop import ActiveRun, mix_eps
from adapters.active.oracle import PairOracle
from adapters.active.pseudo import PoolGraph


class FixedGraph:
    def __init__(self, labels):
        self.labels = labels

    def cluster(self, store=None):
        return self.labels


def member_check(path, B):
    z = np.load(path)
    run = object.__new__(ActiveRun)
    run.strategy, run.store, run.budget = "off:member", ConstraintStore(), B
    run.oracle = PairOracle(z["pids"], z["cams"])
    run.graph = FixedGraph(z["labels"])
    info = run._ask_member(torch.from_numpy(z["features"]))
    ans = run.store.answers
    pids = z["pids"]
    print("budget {}: asked {} no {} ({:.1%}), candidates {}, mean suspicion {:.3f}".format(
        B, info["n_member_q"], info["n_member_no"], info["n_member_no"] / max(info["n_member_q"], 1),
        info["member_cands"], info["member_susp_mean"]))
    X = z["features"] / np.linalg.norm(z["features"], axis=1, keepdims=True)
    print("top 20 (member, medoid, cos(member, medoid), true answer):")
    for i, j, s in ans[:20]:
        print("  {:5d} {:5d}  {:.3f}  {}".format(i, j, float(X[i] @ X[j]), "same" if s else "DIFFERENT"))
    no20 = sum(not s for *_, s in ans[:20])
    print("top-20 true 'different': {}/20 = {:.0%}".format(no20, no20 / 20))
    # reference: a random member of a random cluster (same per-cluster rule), label-free baseline
    lab = z["labels"]
    rng = np.random.RandomState(0)
    ok = np.flatnonzero(lab >= 0)
    rnd = rng.choice(ok, size=min(2000, len(ok)), replace=False)
    maj = {}
    for c in np.unique(lab[ok]):
        u, n = np.unique(pids[lab == c], return_counts=True)
        maj[c] = u[np.argmax(n)]
    print("reference: share of clustered images not of their cluster's majority id = {:.1%}".format(
        np.mean([pids[i] != maj[lab[i]] for i in rnd])))
    return no20 / 20


def split_check():
    rng = np.random.RandomState(0)
    a = rng.randn(1, 64) + 0.3 * rng.randn(30, 64)  # one tight blob of 30 images -> one cluster
    X = torch.nn.functional.normalize(torch.from_numpy(a.astype(np.float32)), dim=1)
    g = PoolGraph(X, k1=10, k2=3, eps=0.6, min_samples=3)
    base = g.cluster()
    s = ConstraintStore()
    s.add(0, 1, False, split=False)  # a merge question's "different"
    g.split_cannot, g.member_split_only = True, True
    l1 = g.cluster(s)
    s.add(2, 3, False, split=True)   # a member question's "different"
    l2 = g.cluster(s)
    g.member_split_only = False      # legacy: every "different" splits
    l3 = g.cluster(ConstraintStore())
    print("clusters: none {} | merge-no only (member_split_only) {} | + member-no {} | legacy empty {}".format(
        len(set(base[base >= 0])), len(set(l1[l1 >= 0])), len(set(l2[l2 >= 0])), len(set(l3[l3 >= 0]))))
    assert len(set(l1[l1 >= 0])) == len(set(base[base >= 0])), "merge 'different' must not split"
    assert l2[2] != l2[3], "member 'different' must split"
    assert mix_eps("mix:0.25") == 0.25 and mix_eps("off:member") == 1.0 and mix_eps("off:rule") is None
    print("constraint split check: OK")


if __name__ == "__main__":
    split_check()
    r = member_check(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 121)
    print("GATE (>= 30% true 'different' in the top 20): {}".format("PASS" if r >= 0.3 else "FAIL"))
