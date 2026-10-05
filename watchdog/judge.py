"""Verdict on the pilot: prints 'GOOD <reason>', 'BAD <reason>' or 'WAIT <reason>'.

  python judge.py sim    experiments/sim_cuhk03/sim.csv
  python judge.py pilot  experiments/pilot/active.csv
"""
import csv
import sys
from collections import defaultdict

import numpy as np


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def judge_sim(path):
    rs = rows(path)
    last = max(int(r["round"]) for r in rs)
    f = defaultdict(list)
    for r in rs:
        if int(r["round"]) == last:
            f[r["strategy"]].append(num(r.get("pw_f")))
    if not f.get("repair") or not f.get("random"):
        return "WAIT", "sim: repair / random rows missing"
    rep, rnd = np.nanmean(f["repair"]), np.nanmean(f["random"])
    detail = "sim round {}: pairwise F1 ".format(last) + ", ".join(
        "{} {:.4f}".format(k, np.nanmean(v)) for k, v in sorted(f.items()))
    if not rep > rnd:
        return "BAD", detail + " -> repair does not beat random"
    return "GOOD", detail


def aulc(points, base):
    pts = sorted([(0.0, base)] + points)
    x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2) / x[-1])


def judge_pilot(path):
    rs = rows(path)
    base = [num(r["mAP"]) for r in rs if r["strategy"] == "base"]
    if not base:
        return "WAIT", "pilot: no base row"
    base = float(np.mean(base))
    curves = defaultdict(lambda: defaultdict(list))  # strategy -> seed -> [(queries, mAP)]
    last = defaultdict(dict)                          # strategy -> seed -> (round, mAP)
    for r in rs:
        m = num(r["mAP"])
        if r["strategy"] in ("base", "oracle_all") or m != m:
            continue
        s, seed, rnd = r["strategy"], r["seed"], int(r["round"])
        curves[s][seed].append((num(r.get("n_queries", 0)), m))
        if rnd >= last[s].get(seed, (-1, 0))[0]:
            last[s][seed] = (rnd, m)
    need = ("none", "repair", "random")
    if any(s not in last for s in need):
        return "WAIT", "pilot: missing strategies {}".format([s for s in need if s not in last])
    final = {s: float(np.mean([m for _, m in last[s].values()])) for s in last}
    # AULC over answers (none asks nothing: no curve)
    au = {s: float(np.mean([aulc(p, base) for p in curves[s].values()])) for s in ("repair", "random", "repair_random")
          if s in curves}
    detail = "base {:.2f} | final mAP ".format(base) + ", ".join("{} {:.2f}".format(k, v) for k, v in sorted(final.items())) \
        + " | AULC " + ", ".join("{} {:.2f}".format(k, v) for k, v in sorted(au.items()))
    if not final["repair"] > final["none"]:
        return "BAD", detail + " -> repair <= none (answers add nothing over unsupervised)"
    if not au["repair"] > au["random"]:
        return "BAD", detail + " -> repair AULC <= random"
    return "GOOD", detail


if __name__ == "__main__":
    kind, path = sys.argv[1], sys.argv[2]
    try:
        v, why = judge_sim(path) if kind == "sim" else judge_pilot(path)
    except FileNotFoundError:
        v, why = "WAIT", "{} not found".format(path)
    print(v, why)
