"""E1a (task list v3): which round-1 split pairs of a no-question run heal by themselves by round 20.

The loop's round-r clustering uses the features under the prompt after round r-1 (round 1: the base model's default
prompt). For r = 1..R this recomputes the pool features under that prompt, re-runs the label-free clustering
(PoolGraph, same k1 / k2 / eps / min_samples, no answers: a no-question run has none) and checks the number of
clusters against the run's active.csv. Round-1 split pairs (split_pairs.py, outliers included) are labelled healed
when both representatives share a cluster in round R's clustering, else persist. Person ids are used only to
define the pairs and to report.

Outputs (out_dir): <run>_summary.json (counts, persist share, per-round healed share, cluster-count check),
<run>_persist.json (round-1 pairs with the label, for E2 --persist_file), <run>_rounds.npz (labels of every round,
features of rounds 1 and 2 as float16, for E1b).

  python scripts/e1a_selfheal_1008.py --run_dir <exp dir> --domain cuhk03 --k1 15 --checkpoint <base ckpt> --out_dir <dir>
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import glob
import importlib.util
import json
import re
import tempfile

import numpy as np
import torch
import transformers

from adapters.active.image_store import features
from adapters.active.loop import default_prompt
from adapters.active.pseudo import PoolGraph
from adapters.active.split_pairs import healed, split_pairs
from adapters.args_reid import ReIDTrainingArguments
from adapters.baseline_model import load_checkpoint_model


def _eval_active():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_active.py")
    s = importlib.util.spec_from_file_location("eval_active_e1a", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--k1", type=int, default=30)
    ap.add_argument("--k2", type=int, default=6)
    ap.add_argument("--eps", type=float, default=0.6)
    ap.add_argument("--min_samples", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--out_dir", required=True)
    o = ap.parse_args()
    run = os.path.basename(o.run_dir.rstrip("/"))
    snaps = {int(re.search(r"round(\d+)\.pt$", p).group(1)): p
             for p in glob.glob(os.path.join(o.run_dir, "prompts", "rounds_*", "round*.pt"))}
    need = list(range(1, o.rounds))
    missing = [r for r in need if r not in snaps]
    if missing:
        raise FileNotFoundError("{}: no per-round prompt for rounds {}".format(run, missing))
    ref = {}
    acsv = os.path.join(o.run_dir, "active.csv")
    if os.path.exists(acsv):
        for row in csv.DictReader(open(acsv)):
            if row.get("round") not in (None, "", "0") and row.get("pseudo_clusters") not in (None, ""):
                ref[int(row["round"])] = int(float(row["pseudo_clusters"]))
    ea = _eval_active()
    parser = transformers.HfArgumentParser((ReIDTrainingArguments, ea.ActiveArguments))
    args, a = parser.parse_args_into_dataclasses(["--output_dir", tempfile.mkdtemp(dir=os.environ.get("TMPDIR", "/tmp")),
                                                  "--checkpoint", o.checkpoint, "--fp16", "True", "--report_to", "none",
                                                  "--eval_num_workers", "8"])
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    model = load_checkpoint_model(args, device, o.checkpoint)
    split = ea.load_split(o.domain, device, a.cache_max, args.eval_num_workers)
    pids = split.pool_pids
    labels, feats12, check = {}, {}, {}
    for r in range(1, o.rounds + 1):
        prompt = default_prompt(model) if r == 1 else torch.load(snaps[r - 1], map_location="cpu")["prompt"].to(device)
        f = features(model, split.pool, prompt)
        g = PoolGraph(f, o.k1, o.k2, o.eps, o.min_samples)
        labels[r] = g.cluster()
        del g
        torch.cuda.empty_cache()
        if r <= 2:
            feats12[r] = f.cpu().numpy()
        n = int(labels[r].max()) + 1
        check[r] = {"recomputed": n, "csv": ref.get(r), "match": ref.get(r) == n if r in ref else None}
        print("{} round {:2d}: clusters {} (csv {})".format(run, r, n, ref.get(r)), flush=True)
    X1 = feats12[1] / np.linalg.norm(feats12[1], axis=1, keepdims=True)
    pairs = split_pairs(labels[1], pids, X1)
    curve = {r: float(healed(pairs, labels[r]).mean()) if pairs else float("nan") for r in labels}
    persist = ~healed(pairs, labels[o.rounds])
    first = []  # first round from which the pair stays healed through round R (None: persist)
    H = np.stack([healed(pairs, labels[r]) for r in range(1, o.rounds + 1)])  # (R, P)
    for k in range(len(pairs)):
        col = H[:, k]
        if not col[-1]:
            first.append(None)
            continue
        last_bad = np.flatnonzero(~col)
        first.append(int(last_bad[-1] + 2) if len(last_bad) else 1)
    # task list v7: a second label by majority images (the pair's person's images in unit A and in the main unit:
    # healed when their most frequent round-R labels coincide and are not outliers), against the representative-
    # based label, which calls a pair persist when one representative happens to be an outlier in round R
    lR, l1 = labels[o.rounds], labels[1]

    def own_images(unit, p):
        if unit[0] == "o":
            return np.array([int(unit[1:])])
        m = np.flatnonzero(l1 == int(unit[1:]))
        return m[pids[m] == p]

    def modal(idx):
        v, n = np.unique(lR[idx], return_counts=True)
        return v[np.argmax(n)]
    healed_major = np.array([(lambda a, b: a >= 0 and a == b)(modal(own_images(p["unit_a"], p["pid"])),
                                                             modal(own_images(p["unit_b"], p["pid"]))) for p in pairs], bool)
    for p, ps, fr, hm in zip(pairs, persist, first, healed_major):
        p["persist"] = bool(ps)
        p["healed_from_round"] = fr
        p["persist_majority"] = bool(not hm)
    os.makedirs(o.out_dir, exist_ok=True)
    kinds = {k: {"n": int(sum(p["kind"] == k for p in pairs)),
                 "persist": int(sum(p["persist"] for p in pairs if p["kind"] == k))} for k in ("cluster", "outlier")}
    seed = int(re.search(r"_s(\d+)$", run).group(1))
    summary = {"run": run, "domain": o.domain, "seed": seed, "k1": o.k1, "eps": o.eps, "rounds": o.rounds,
               "n_pairs": len(pairs), "n_persist": int(persist.sum()), "persist_share": float(persist.mean()),
               "by_kind": kinds, "healed_share_by_round": curve, "cluster_count_check": check,
               "n_persist_majority": int((~healed_major).sum()) if pairs else 0,
               "label_agreement_rep_vs_majority": float(np.mean(persist == ~healed_major)) if pairs else float("nan"),
               "all_cluster_counts_match": all(v["match"] for v in check.values() if v["match"] is not None)}
    json.dump(summary, open(os.path.join(o.out_dir, run + "_summary.json"), "w"), indent=1)
    json.dump({"run": run, "domain": o.domain, "seed": seed, "k1": o.k1, "pairs": pairs},
              open(os.path.join(o.out_dir, run + "_persist.json"), "w"))
    np.savez_compressed(os.path.join(o.out_dir, run + "_rounds.npz"),
                        labels=np.stack([labels[r] for r in range(1, o.rounds + 1)]).astype(np.int32),
                        feats_r1=feats12[1].astype(np.float16), feats_r2=feats12[2].astype(np.float16),
                        pids=pids, cams=split.pool_cams)
    print("{}: {} split pairs, persist {} ({:.1%}); cluster counts match csv: {}".format(
        run, len(pairs), int(persist.sum()), persist.mean(), summary["all_cluster_counts_match"]), flush=True)


if __name__ == "__main__":
    main()
