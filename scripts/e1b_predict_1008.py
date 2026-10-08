"""E1b (task list v3): can rounds 1-2 alone predict which round-1 split pairs persist to round 20?

Inputs: E1a outputs of one or more runs (<run>_persist.json, <run>_rounds.npz). Every predictor is label-free (it
uses features, cameras and the round-1 / round-2 clusterings; person ids only define which pairs are split and the
persist label). Units are the round-1 units of a pair (split_pairs.py): a cluster's members or an outlier image.

Directions are fixed before looking at the data (higher score = predicted persist):
  static_sim        -cos(unit centroids) in round 1                                  [static baseline]
  d_sim             -(cos in round 2 - cos in round 1), round-1 memberships           [dynamic]
  d_rank            log(rank of b's representative among a's neighbours, round 2) - same in round 1 (rank worsens) [dynamic]
  recip_r2          -share of (a, b) image pairs that are mutual k-NN (k = 20) in round 2                       [dynamic]
  same_r2           -[the two representatives share a cluster in round 2]                                       [dynamic]
  (task list v7: d_rank / same_r2 use label-free representatives -- the unit member closest to its round-1
   centroid; the oracle representatives of E1a are reported as d_rank_major / same_r2_major, not used for the call)
  shared_cam        [the units share no camera]                                       [structure, reported only]
  size_small        -min(size_a, size_b)                                              [structure, reported only]
  logreg_dynamic    5-fold cross-validated logistic regression on d_sim, d_rank, recip_r2, same_r2 (standardised)
AUC per run and pooled over runs; the task-list criterion uses the best dynamic predictor.

  python scripts/e1b_predict_1008.py <e1a out_dir> <run> [<run> ...] --out <json>
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

DYNAMIC = ("d_sim", "d_rank", "recip_r2", "same_r2")


def unit_members(unit, labels1):
    kind, idx = unit[0], int(unit[1:])
    return np.flatnonzero(labels1 == idx) if kind == "c" else np.array([idx])


def mutual_knn(X, k=20, chunk=4096):
    Xt = torch.from_numpy(X).float()
    if torch.cuda.is_available():
        Xt = Xt.cuda()
    idx = []
    for s in range(0, len(X), chunk):
        S = Xt[s:s + chunk] @ Xt.T
        S[torch.arange(S.size(0)), torch.arange(s, s + S.size(0))] = -2
        idx.append(S.topk(k, dim=1).indices.cpu().numpy())
    nn = np.concatenate(idx)
    sets = [set(r.tolist()) for r in nn]
    return nn, sets


def rank_of(X, i, j):
    s = X @ X[i]
    s[i] = -2
    return int((s > s[j]).sum()) + 1


def features_of_run(d, run):
    pj = json.load(open(os.path.join(d, run + "_persist.json")))
    z = np.load(os.path.join(d, run + "_rounds.npz"))
    L = z["labels"]
    l1, l2 = L[0], L[1]
    X1 = z["feats_r1"].astype(np.float32)
    X2 = z["feats_r2"].astype(np.float32)
    X1 /= np.linalg.norm(X1, axis=1, keepdims=True)
    X2 /= np.linalg.norm(X2, axis=1, keepdims=True)
    cams = z["cams"]
    _, sets2 = mutual_knn(X2)
    rows = []
    for p in pj["pairs"]:
        A, B = unit_members(p["unit_a"], l1), unit_members(p["unit_b"], l1)
        ca1, cb1 = X1[A].mean(0), X1[B].mean(0)
        ca2, cb2 = X2[A].mean(0), X2[B].mean(0)
        cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
        s1, s2 = cos(ca1, cb1), cos(ca2, cb2)
        # task list v7: predictors use label-free representatives (the unit member closest to its round-1 centroid,
        # person ids not looked at); the oracle representatives (majority person) are reported as *_major only
        fa = int(A[np.argmax(X1[A] @ (ca1 / (np.linalg.norm(ca1) + 1e-12)))])
        fb = int(B[np.argmax(X1[B] @ (cb1 / (np.linalg.norm(cb1) + 1e-12)))])
        Bs = set(B.tolist())
        mutual = sum(1 for a in A for b in sets2[a] if b in Bs and a in sets2[b])
        recip = mutual / max(len(A) * len(B), 1)
        row = {"persist": int(p["persist"]), "persist_majority": int(p.get("persist_majority", p["persist"])),
               "static_sim": -s1, "d_sim": -(s2 - s1), "recip_r2": -recip,
               "shared_cam": float(len(set(cams[A]) & set(cams[B])) == 0),
               "size_small": -float(min(len(A), len(B))), "kind": p["kind"]}
        for suffix, (i, j) in (("", (fa, fb)), ("_major", (p["rep_a"], p["rep_b"]))):
            row["d_rank" + suffix] = np.log(rank_of(X2, i, j)) - np.log(rank_of(X1, i, j))
            row["same_r2" + suffix] = -float(l2[i] >= 0 and l2[i] == l2[j])
        rows.append(row)
    return rows


def aucs(rows, seed=0):
    y = np.array([r["persist"] for r in rows])
    out = {"n": len(y), "n_persist": int(y.sum())}
    if y.min() == y.max():
        return dict(out, note="one class only")
    for k in ("static_sim",) + DYNAMIC + ("shared_cam", "size_small", "d_rank_major", "same_r2_major"):
        out[k] = float(roc_auc_score(y, [r[k] for r in rows]))
    ym = np.array([r["persist_majority"] for r in rows])
    if 0 < ym.sum() < len(ym):  # same predictors against the majority-image persist label (E1a, v7)
        out["vs_majority_label"] = {k: float(roc_auc_score(ym, [r[k] for r in rows])) for k in ("static_sim",) + DYNAMIC}
    F = np.array([[r[k] for k in DYNAMIC] for r in rows], float)
    pred = np.zeros(len(y))
    n_splits = int(min(5, y.sum(), len(y) - y.sum()))
    if n_splits >= 2:
        for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(F, y):
            sc = StandardScaler().fit(F[tr])
            m = LogisticRegression(max_iter=1000).fit(sc.transform(F[tr]), y[tr])
            pred[te] = m.predict_proba(sc.transform(F[te]))[:, 1]
        out["logreg_dynamic"] = float(roc_auc_score(y, pred))
    dyn = {k: out[k] for k in DYNAMIC + ("logreg_dynamic",) if k in out}
    best = max(dyn, key=dyn.get)
    out["best_dynamic"] = best
    out["best_dynamic_auc"] = dyn[best]
    out["best_minus_static"] = dyn[best] - out["static_sim"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("e1a_dir")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    o = ap.parse_args()
    res, pooled = {}, []
    for run in o.runs:
        rows = features_of_run(o.e1a_dir, run)
        res[run] = aucs(rows)
        pooled += rows
        print(run, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[run].items()}), flush=True)
    res["pooled"] = aucs(pooled)
    print("pooled", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["pooled"].items()}))
    json.dump(res, open(o.out, "w"), indent=1)


if __name__ == "__main__":
    main()
