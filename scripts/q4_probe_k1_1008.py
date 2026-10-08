"""Q4 (task list v3), offline: choose k1 without labels from a small probe of member questions.

On a domain's round-0 export (base-model features), for each k1 in the grid (eps 0.6): cluster (PoolGraph, k2 6,
min_samples 4, no answers), then probe n random clusters (size >= 2, label-free choice): one question each,
"is the cluster's most suspicious member (lowest cosine to the normalised centroid) the same person as its medoid?".
The simulated annotator answers from the person ids (that is the only use of labels: the answers a human would
give). p_hat(k1) = share of "different" answers, averaged over `draws` random probes (its spread is reported).

Rule, fixed before looking at the data: the largest k1 with p_hat <= TAU (TAU = 0.30, the member-question gate of the
10-08 task order); none qualifies -> the smallest k1. Reported alongside: the TAU range for which the rule picks
k1 in the target set (CUHK03 15 +- 5, MSMT17 >= 25), and the true pairwise precision / recall per k1.

  python scripts/q4_probe_k1_1008.py <round0 npz> <out json> [--device cpu|cuda]

--mode pairs (Q4b, task list v5; the second attempt, after Q4 picked k1 = 30 on CUHK03 -- the estimator changed, the
rule's threshold was fixed by the App side before this run): the probe draws image pairs uniformly from all
same-cluster pairs (a cluster with n images is drawn with weight n(n-1)/2, then a random pair inside it) and asks
"same person?"; the share of "same" answers is an unbiased estimate of the pairwise precision. Rule: the largest
k1 with p_hat >= 0.50 (none qualifies -> the smallest k1).
"""
import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from adapters.active.pseudo import PoolGraph, label_clusters

GRID = (10, 12, 15, 18, 20, 22, 25, 28, 30)
TAU = 0.30
N_PROBE = 60
DRAWS = 20


def pairwise(labels, pids):
    ok = labels >= 0
    y, c = pids[ok], labels[ok]
    pair = lambda n: (n * (n - 1) // 2).sum()
    _, joint = np.unique(np.stack([y, c]), axis=1, return_counts=True)
    tp, pred = pair(joint), pair(np.unique(c, return_counts=True)[1])
    true = pair(np.unique(pids, return_counts=True)[1])
    return float(tp / max(pred, 1)), float(tp / max(true, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("out")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mode", default="member", choices=("member", "pairs", "deploy"))
    ap.add_argument("--grid", default="")  # deploy: comma-separated k1 values (default GRID)
    o = ap.parse_args()
    if o.mode == "pairs":
        return main_pairs(o)
    if o.mode == "deploy":
        return main_deploy(o)
    z = np.load(o.npz)
    X = z["features"].astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    pids = z["pids"]
    res = {"npz": o.npz, "tau": TAU, "n_probe": N_PROBE, "draws": DRAWS, "grid": {}}
    for k1 in GRID:
        g = PoolGraph(torch.from_numpy(X).to(o.device), k1, 6, 0.6, 4)
        lab = g.cluster()
        del g
        cl = [np.asarray(m) for m in label_clusters(lab) if len(m) >= 2]
        worst = []  # per cluster: (suspicious member, medoid) -> answer "different"
        for m in cl:
            cent = X[m].mean(0)
            cent /= np.linalg.norm(cent) + 1e-12
            cos = X[m] @ cent
            worst.append(bool(pids[m[np.argmin(cos)]] != pids[m[np.argmax(cos)]]))
        worst = np.array(worst)
        rng = np.random.RandomState(k1)
        est = [worst[rng.choice(len(worst), min(N_PROBE, len(worst)), replace=False)].mean() for _ in range(DRAWS)]
        prec, rec = pairwise(lab, pids)
        res["grid"][k1] = {"clusters": int(lab.max()) + 1, "p_hat_mean": float(np.mean(est)), "p_hat_std": float(np.std(est)),
                           "p_all_clusters": float(worst.mean()), "pw_prec": prec, "pw_rec": rec}
        print("k1 {:2d}: clusters {:5d}  p_hat {:.3f} +- {:.3f} (all clusters {:.3f})  pw P/R {:.2f}/{:.2f}".format(
            k1, res["grid"][k1]["clusters"], np.mean(est), np.std(est), worst.mean(), prec, rec), flush=True)

    def pick(tau):
        ok = [k for k in GRID if res["grid"][k]["p_hat_mean"] <= tau]
        return max(ok) if ok else min(GRID)
    res["picked_k1"] = pick(TAU)
    res["picked_by_tau"] = {"{:.2f}".format(t): pick(t) for t in np.arange(0.05, 0.61, 0.05)}
    print("picked k1 at tau {:.2f}: {}".format(TAU, res["picked_k1"]))
    print("picked by tau:", res["picked_by_tau"])
    json.dump(res, open(o.out, "w"), indent=1)


TAU_PAIRS = 0.50


def main_pairs(o):
    z = np.load(o.npz)
    X = z["features"].astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    pids = z["pids"]
    res = {"npz": o.npz, "mode": "pairs", "tau": TAU_PAIRS, "rule": "largest k1 with p_hat >= tau",
           "n_probe": N_PROBE, "draws": DRAWS, "grid": {}}
    for k1 in GRID:
        g = PoolGraph(torch.from_numpy(X).to(o.device), k1, 6, 0.6, 4)
        lab = g.cluster()
        del g
        cl = [np.asarray(m) for m in label_clusters(lab) if len(m) >= 2]
        w = np.array([len(m) * (len(m) - 1) / 2 for m in cl], float)
        rng = np.random.RandomState(1000 + k1)
        est = []
        for _ in range(DRAWS):
            ci = rng.choice(len(cl), N_PROBE, p=w / w.sum())
            same = 0
            for c in ci:
                a, b = rng.choice(cl[c], 2, replace=False)
                same += pids[a] == pids[b]  # the simulated annotator's answer
            est.append(same / N_PROBE)
        prec, rec = pairwise(lab, pids)
        res["grid"][k1] = {"clusters": int(lab.max()) + 1, "p_hat_mean": float(np.mean(est)),
                           "p_hat_std": float(np.std(est)), "pw_prec": prec, "pw_rec": rec,
                           "_draws": [float(e) for e in est]}
        print("k1 {:2d}: clusters {:5d}  p_hat {:.3f} +- {:.3f}  true pw P/R {:.2f}/{:.2f}".format(
            k1, res["grid"][k1]["clusters"], np.mean(est), np.std(est), prec, rec), flush=True)
    ok = [k for k in GRID if res["grid"][k]["p_hat_mean"] >= TAU_PAIRS]
    res["picked_k1"] = max(ok) if ok else min(GRID)
    # spread of the decision: share of the 20 single probes (60 pairs each) that would pick each k1
    picks = []
    for d in range(DRAWS):
        okd = [k for k in GRID if _single(res, k, d) >= TAU_PAIRS]
        picks.append(max(okd) if okd else min(GRID))
    res["picked_by_single_probe"] = {str(k): picks.count(k) for k in sorted(set(picks))}
    print("picked k1 (mean p_hat >= {:.2f}): {}".format(TAU_PAIRS, res["picked_k1"]))
    print("picked by each single 60-pair probe:", res["picked_by_single_probe"])
    json.dump(res, open(o.out, "w"), indent=1)


N_DEPLOY = 120


def _same_pairs_sample(cl, n, rng):
    """n image pairs drawn uniformly from all same-cluster pairs of the clusters `cl` (lists of members)."""
    w = np.array([len(m) * (len(m) - 1) / 2 for m in cl], float)
    out = []
    for c in rng.choice(len(cl), n, p=w / w.sum()):
        a, b = rng.choice(cl[c], 2, replace=False)
        out.append((int(a), int(b)))
    return out


def main_deploy(o):
    """Deployment protocol of task list v6: one shared probe of N_DEPLOY questions, drawn uniformly from the
    same-cluster pairs of the loosest k1 of the grid; p_hat(k1) uses the probe pairs that also share a cluster at
    k1. Reported: the k1 picked by each of DRAWS independent probes (rule: the largest k1 with p_hat >= 0.50), the
    number of probe pairs behind each p_hat, and the nesting check (share of a tighter k1's same-cluster pairs that
    are also same-cluster at the loosest k1, from 2000 pairs per k1)."""
    grid = tuple(int(x) for x in o.grid.split(",")) if o.grid else GRID
    z = np.load(o.npz)
    X = z["features"].astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    pids = z["pids"]
    labs = {}
    for k1 in grid:
        g = PoolGraph(torch.from_numpy(X).to(o.device), k1, 6, 0.6, 4)
        labs[k1] = g.cluster()
        del g
        if o.device == "cuda":
            torch.cuda.empty_cache()
    loose = max(grid)
    cl_loose = [np.asarray(m) for m in label_clusters(labs[loose]) if len(m) >= 2]
    res = {"npz": o.npz, "mode": "deploy", "tau": TAU_PAIRS, "n_probe": N_DEPLOY, "draws": DRAWS, "grid": list(grid),
           "loosest_k1": loose, "per_k1": {}, "picks": []}
    rng_n = np.random.RandomState(7)
    for k1 in grid:
        cl = [np.asarray(m) for m in label_clusters(labs[k1]) if len(m) >= 2]
        sample = _same_pairs_sample(cl, 2000, rng_n)
        nested = float(np.mean([labs[loose][a] >= 0 and labs[loose][a] == labs[loose][b] for a, b in sample]))
        prec, rec = pairwise(labs[k1], pids)
        res["per_k1"][k1] = {"clusters": int(labs[k1].max()) + 1, "pw_prec": prec, "pw_rec": rec, "nested_share": nested,
                             "p_hat": [], "n_used": []}
    for d in range(DRAWS):
        rng = np.random.RandomState(2000 + d)
        probe = _same_pairs_sample(cl_loose, N_DEPLOY, rng)
        ans = {pr: bool(pids[pr[0]] == pids[pr[1]]) for pr in probe}  # simulated annotator
        ok = []
        for k1 in grid:
            l = labs[k1]
            used = [pr for pr in probe if l[pr[0]] >= 0 and l[pr[0]] == l[pr[1]]]
            p = float(np.mean([ans[pr] for pr in used])) if used else float("nan")
            res["per_k1"][k1]["p_hat"].append(p)
            res["per_k1"][k1]["n_used"].append(len(used))
            if used and p >= TAU_PAIRS:
                ok.append(k1)
        res["picks"].append(max(ok) if ok else min(grid))
    res["pick_distribution"] = {str(k): res["picks"].count(k) for k in grid if res["picks"].count(k)}
    res["pick_mode"] = max(set(res["picks"]), key=res["picks"].count)
    for k1 in grid:
        r = res["per_k1"][k1]
        print("k1 {:2d}: clusters {:5d}  p_hat {:.3f} +- {:.3f} (n {:.0f})  true pw P/R {:.2f}/{:.2f}  nested {:.3f}".format(
            k1, r["clusters"], np.nanmean(r["p_hat"]), np.nanstd(r["p_hat"]), np.mean(r["n_used"]), r["pw_prec"],
            r["pw_rec"], r["nested_share"]), flush=True)
    print("pick distribution over {} probes of {} questions: {} (mode {})".format(
        DRAWS, N_DEPLOY, res["pick_distribution"], res["pick_mode"]))
    json.dump(res, open(o.out, "w"), indent=1)


def _single(res, k, d):
    return res["grid"][k]["_draws"][d] if "_draws" in res["grid"][k] else res["grid"][k]["p_hat_mean"]


if __name__ == "__main__":
    main()
