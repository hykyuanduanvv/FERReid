"""E1c (task list v7), offline: inside the questions the rule selector actually asks in round 1, do the dynamic
predictors find the pairs that are truly split AND stay split (persist)?

For a no-question run analysed by E1a (<run>_rounds.npz: labels of every round, features of rounds 1 and 2), the
off:rule selector is called on the round-1 clustering (offline_recall_1006_2.old_queries, K=1, as the loop does)
and its first N questions (default 605, the CUHK03 budget; duplicates removed, no answers) are kept. Label of a
question (i, j): same person AND not in one cluster at round R (persist split). Most questions are two different
people (label 0), which is the deployment situation E1b did not have.
Scores (higher = predicted persist split), all label-free and on the question's own two images:
  rule_rank   -position in the rule's order          static_sim  -cos(units' round-1 centroids)
  d_sim       -(cos round 2 - cos round 1)           d_rank      log rank(j | i) round 2 - round 1
  recip_r2    -mutual 20-NN share between the units in round 2       same_r2  -[i, j share a round-2 cluster]
  logreg_dynamic  5-fold CV logistic regression on d_sim, d_rank, recip_r2, same_r2
Also reported: AUC for "same person" alone (is the predictor just a same-person detector?).

  python scripts/e1c_rule_candidates_1008.py <e1a out_dir> <run> [<run> ...] --n 605 --out <json>

Task list v8 (fixed before the real results): the main test uses the rule's questions on the ROUND-2 clustering
(--round 2, the E2 deployment time; --round 1 kept as a reference), label = same person and still apart at round
R. Main score = P_same x pct(persist):
  P_same        logistic regression "cosine of the question's two images (round-q features) -> same person",
                fitted on 60 questions drawn uniformly from the candidates and answered by the oracle (charged);
  pct(persist)  percentile, among the candidates, of the persist predictor chosen by E1b (--persist_predictor;
                d_sim while E1b has not decided).
Compared with: rule_rank, P_same alone, pct(persist) alone; secondary: AUC of pct(persist) among the same-person
candidates only. AUCs are computed on the candidates other than the 60 probe questions (main) and on all.
"""
import argparse
import importlib.util
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

DYN = ("d_sim", "d_rank", "recip_r2", "same_r2")


def _mod(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def run_one(d, run, n_q, e1b, rq=2):
    z = np.load(os.path.join(d, run + "_rounds.npz"))
    L, pids, cams = z["labels"], z["pids"], z["cams"]
    l1, l2, lR = L[0], L[1], L[-1]
    X1 = z["feats_r1"].astype(np.float32); X1 /= np.linalg.norm(X1, axis=1, keepdims=True)
    X2 = z["feats_r2"].astype(np.float32); X2 /= np.linalg.norm(X2, axis=1, keepdims=True)
    lq, Xq = (l1, X1) if rq == 1 else (l2, X2)  # the clustering / features the rule asks on
    base = _mod("offline_recall_1006_2", os.path.join(os.path.dirname(os.path.abspath(__file__)), "offline_recall_1006_2.py"))
    pool = base.Pool(Xq, lq, cams)
    order = base.old_queries(pool, 1, rule=True)
    qs, seen = [], set()
    for i, j in order:
        key = (min(int(i), int(j)), max(int(i), int(j)))
        if key in seen:
            continue
        seen.add(key)
        qs.append(key)
        if len(qs) >= n_q:
            break
    _, sets2 = e1b.mutual_knn(X2)
    unit = lambda i: np.flatnonzero(lq == lq[i]) if lq[i] >= 0 else np.array([i])  # units of the asking round
    rows = []
    for pos, (i, j) in enumerate(qs):
        A, B = unit(i), unit(j)
        ca1, cb1, ca2, cb2 = X1[A].mean(0), X1[B].mean(0), X2[A].mean(0), X2[B].mean(0)
        cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
        Bs = set(B.tolist())
        mutual = sum(1 for a in A for b in sets2[a] if b in Bs and a in sets2[b])
        same = bool(pids[i] == pids[j])
        persist = same and not (lR[i] >= 0 and lR[i] == lR[j])
        rows.append({"y": int(persist), "same": int(same), "rule_rank": -pos, "pair_cos": float(Xq[i] @ Xq[j]),
                     "static_sim": -cos(ca1, cb1),
                     "d_sim": -(cos(ca2, cb2) - cos(ca1, cb1)),
                     "d_rank": np.log(e1b.rank_of(X2, i, j)) - np.log(e1b.rank_of(X1, i, j)),
                     "recip_r2": -mutual / max(len(A) * len(B), 1), "same_r2": -float(l2[i] >= 0 and l2[i] == l2[j])})
    return rows


def main_score(rows, persist_pred, seed):
    """Task list v8: score = P_same x pct(persist); P_same from 60 oracle-answered random candidates. Adds the score
    columns to rows and returns (probe indices, fit info)."""
    from scipy.stats import rankdata
    rng = np.random.RandomState(seed)
    probe = rng.choice(len(rows), min(60, len(rows)), replace=False)
    x = np.array([[r["pair_cos"]] for r in rows])
    yp = np.array([rows[k]["same"] for k in probe])
    if yp.min() == yp.max():  # one class among the probes: constant P_same
        ps = np.full(len(rows), yp.mean())
        info = {"probe_same": int(yp.sum()), "fit": "constant (one class in the 60 probes)"}
    else:
        m = LogisticRegression(max_iter=1000).fit(x[probe], yp)
        ps = m.predict_proba(x)[:, 1]
        info = {"probe_same": int(yp.sum()), "fit": "logistic", "coef": float(m.coef_[0, 0]), "intercept": float(m.intercept_[0])}
    pct = rankdata([r[persist_pred] for r in rows]) / len(rows)
    for r, a, b in zip(rows, ps, pct):
        r["P_same"], r["pct_persist"], r["main_score"] = float(a), float(b), float(a * b)
    return set(probe.tolist()), info


def aucs_v8(rows, probe):
    out = {}
    for name, sel in (("excl_probes", [k for k in range(len(rows)) if k not in probe]), ("all", list(range(len(rows))))):
        rr = [rows[k] for k in sel]
        y = np.array([r["y"] for r in rr])
        if y.min() == y.max():
            out[name] = "one class only"
            continue
        out[name] = {k: float(roc_auc_score(y, [r[k] for r in rr])) for k in ("main_score", "rule_rank", "P_same", "pct_persist")}
        out[name]["main_minus_rule"] = out[name]["main_score"] - out[name]["rule_rank"]
        same = [r for r in rr if r["same"]]
        ys = np.array([r["y"] for r in same])
        out[name]["pct_persist_within_same_person"] = float(roc_auc_score(ys, [r["pct_persist"] for r in same])) \
            if len(same) and 0 < ys.sum() < len(ys) else "n/a"
        out[name]["n"], out[name]["n_pos"] = len(rr), int(y.sum())
    return out


def bootstrap_v9(rows, n_boot=1000, seed=0):
    """Task list v9: pooled main analysis (rows of all runs, probes excluded): AUC of main_score and rule_rank and
    their difference with bootstrap 95% CIs (resampling candidates); call: undecidable when positives < 20 or the
    CI of the difference is wider than 0.2, else support if main - rule >= 0.05, refute if <= 0."""
    rr = [r for r in rows if not r["_probe"]]
    y = np.array([r["y"] for r in rr])
    s = {k: np.array([r[k] for r in rr]) for k in ("main_score", "rule_rank")}
    out = {"n": len(rr), "n_pos": int(y.sum())}
    if y.min() == y.max():
        return dict(out, call="undecidable (one class)")
    point = {k: float(roc_auc_score(y, v)) for k, v in s.items()}
    rng = np.random.RandomState(seed)
    bs = {"main_score": [], "rule_rank": [], "diff": []}
    for _ in range(n_boot):
        idx = rng.randint(len(y), size=len(y))
        if y[idx].min() == y[idx].max():
            continue
        a = roc_auc_score(y[idx], s["main_score"][idx]); b = roc_auc_score(y[idx], s["rule_rank"][idx])
        bs["main_score"].append(a); bs["rule_rank"].append(b); bs["diff"].append(a - b)
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in bs.items()}
    diff = point["main_score"] - point["rule_rank"]
    width = ci["diff"][1] - ci["diff"][0]
    if out["n_pos"] < 20 or width > 0.2:
        call = "undecidable (n_pos {} / CI width {:.3f})".format(out["n_pos"], width)
    else:
        call = "support" if diff >= 0.05 else ("refute" if diff <= 0 else "undecidable (0 < diff < 0.05)")
    return dict(out, auc=point, diff=diff, ci95=ci, ci_diff_width=width, call=call)


def space_v9(rows, budgets=(100, 121)):
    """Task list v9 'room' diagnostic: share of same-person candidates, share of persist among them, and for the
    first B questions: persist splits found by the rule's order, by the main score, and the ceiling min(B, all)."""
    n = len(rows)
    same = sum(r["same"] for r in rows)
    pos = sum(r["y"] for r in rows)
    out = {"n_candidates": n, "same_person_share": same / max(n, 1), "persist_share_among_same": pos / max(same, 1),
           "n_persist_split": pos}
    for B in budgets:
        top = lambda key: sum(r["y"] for r in sorted(rows, key=lambda r: -r[key])[:B])
        out["B{}".format(B)] = {"rule_order": top("rule_rank"), "main_score_order": top("main_score"),
                                "ceiling": min(B, pos)}
    return out


def aucs(rows):
    out = {"n": len(rows), "n_same_person": int(sum(r["same"] for r in rows)), "n_persist_split": int(sum(r["y"] for r in rows))}
    for target in ("y", "same"):
        y = np.array([r[target] for r in rows])
        if y.min() == y.max():
            out[target] = "one class only"
            continue
        res = {k: float(roc_auc_score(y, [r[k] for r in rows])) for k in ("rule_rank", "static_sim") + DYN}
        F = np.array([[r[k] for k in DYN] for r in rows], float)
        pred = np.zeros(len(y))
        n_splits = int(min(5, y.sum(), len(y) - y.sum()))
        if n_splits >= 2:
            for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=0).split(F, y):
                sc = StandardScaler().fit(F[tr])
                pred[te] = LogisticRegression(max_iter=1000).fit(sc.transform(F[tr]), y[tr]).predict_proba(sc.transform(F[te]))[:, 1]
            res["logreg_dynamic"] = float(roc_auc_score(y, pred))
        out["target_persist_split" if target == "y" else "target_same_person"] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("e1a_dir")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--n", type=int, default=605)
    ap.add_argument("--round", type=int, default=2, choices=(1, 2))
    ap.add_argument("--persist_predictor", default="d_sim", choices=DYN)
    ap.add_argument("--out", required=True)
    o = ap.parse_args()
    e1b = _mod("e1b_predict_1008", os.path.join(os.path.dirname(os.path.abspath(__file__)), "e1b_predict_1008.py"))
    res = {"round": o.round, "persist_predictor": o.persist_predictor, "runs": {}}
    pooled = []
    for k, run in enumerate(o.runs):
        rows = run_one(o.e1a_dir, run, o.n, e1b, o.round)
        probe, info = main_score(rows, o.persist_predictor, seed=100 + k)
        for idx, r in enumerate(rows):
            r["_probe"] = idx in probe
        pooled += rows
        res["runs"][run] = {"n_candidates": len(rows), "p_same_fit": info, "v8_main": aucs_v8(rows, probe),
                            "space": space_v9(rows), "single_predictors": aucs(rows)}
        print(run, json.dumps(res["runs"][run]["v8_main"]), flush=True)
        print("  space", json.dumps(res["runs"][run]["space"]), flush=True)
    res["pooled_v9"] = bootstrap_v9(pooled)
    res["pooled_space"] = space_v9(pooled, budgets=(100 * len(o.runs), 121 * len(o.runs)))
    print("POOLED", json.dumps(res["pooled_v9"]))
    print("POOLED space", json.dumps(res["pooled_space"]))
    json.dump(res, open(o.out, "w"), indent=1)


if __name__ == "__main__":
    main()
