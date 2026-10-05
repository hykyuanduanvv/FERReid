"""Offline screening of question strategies: no prompt training, no retrieval evaluation (minutes per fold).

The pool features of the base model are extracted once; every strategy then asks `rounds` x `budget`
questions on those fixed features (the simulated annotator answers), and the effect of the answers is
measured directly:
  n_pos, true_ids        positive answers, identities covered by the answered clusters (pair setting)
  pw_prec / pw_rec / pw_f, nmi, pseudo_split_ids
                         quality of the constrained pseudo labels (--pseudo True): what the answers repair
A strategy that does not beat random here will not beat it after training; only the promising ones need the
full runs (scripts/eval_active.py). Writes sim.csv (one row per strategy / seed / round) and sim_summary.csv
(mean / std over seeds; paired difference to --paired_ref).

  python scripts/sim_selection.py --output_dir experiments/sim_cuhk03 --checkpoint experiments/base_md_cuhk03/checkpoint-12000 \\
      --domains cuhk03 --pseudo True --strategies none,repair,repair_unc,repair_random,random,cover,disagree \\
      --rounds 5 --budget 200 --n_seeds 5 --fp16 True --report_to none
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import time
from collections import defaultdict
from dataclasses import fields

import numpy as np
import torch
import transformers

from adapters.args_reid import ReIDTrainingArguments
from adapters.active.image_store import features
from adapters.active.loop import ActiveConfig, ActiveRun, default_prompt
from adapters.baseline_model import load_checkpoint_model
from scripts.eval_active import ActiveArguments, load_split, write_csv

METRICS = ("n_queries", "n_pos", "true_ids", "purity", "pw_prec", "pw_rec", "pw_f", "nmi", "pseudo_clusters",
           "pseudo_outliers", "pseudo_split_ids", "round_pos_rate", "exp_change")


def summarize(rows, ref):
    by = defaultdict(list)
    for r in rows:
        by[(r["domain"], r["strategy"], r["round"])].append(r)
    out = []
    for (d, s, rnd), rs in sorted(by.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        row = {"domain": d, "strategy": s, "round": rnd, "n": len(rs)}
        for m in METRICS:
            v = np.array([r.get(m, np.nan) for r in rs], float)
            if np.isfinite(v).any():
                row[m], row[m + "_std"] = float(np.nanmean(v)), float(np.nanstd(v))
        refs = {r["seed"]: r for r in by.get((d, ref, rnd), [])}
        for m in ("pw_f", "true_ids", "n_pos"):
            diffs = [r[m] - refs[r["seed"]][m] for r in rs if r["seed"] in refs and m in r and m in refs[r["seed"]]]
            if diffs:
                row["d_" + m] = float(np.mean(diffs))
        out.append(row)
    return out


def main():
    parser = transformers.HfArgumentParser((ReIDTrainingArguments, ActiveArguments))
    args, a = parser.parse_args_into_dataclasses()
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model = load_checkpoint_model(args, device, a.checkpoint)
    cfg = ActiveConfig(**{f.name: getattr(a, f.name) for f in fields(ActiveConfig)})
    os.makedirs(args.output_dir, exist_ok=True)
    rows, t0 = [], time.time()
    for name in a.domains.split(","):
        split = load_split(name, device, a.cache_max, args.eval_num_workers)
        feats = features(model, split.pool, default_prompt(model))
        print("[{:6.0f}s] {}: pool {} images, features {}".format(time.time() - t0, name, len(split.pool),
                                                                 tuple(feats.shape)), flush=True)
        for seed in range(a.n_seeds):
            for strategy in [s for s in a.strategies.split(",") if s]:
                run = ActiveRun(model, split, strategy, cfg, seed=a.base_seed + seed)
                run.features_cache = feats
                for row in run.run(tune=False):
                    rows.append(dict(domain=name, strategy=strategy, seed=seed, **row))
                write_csv(os.path.join(args.output_dir, "sim.csv"), rows)
                write_csv(os.path.join(args.output_dir, "sim_summary.csv"), summarize(rows, a.paired_ref))
        del split
    summ = summarize(rows, a.paired_ref)
    last = max(r["round"] for r in rows)
    print("\n{:<10} {:<16} {:>7} {:>6} {:>8} {:>7} {:>8}".format("domain", "strategy", "queries", "pos", "true_ids",
                                                                 "pw_f", "d_pw_f"))
    for s in summ:
        if s["round"] == last:
            print("{:<10} {:<16} {:>7.0f} {:>6.0f} {:>8.1f} {:>7.3f} {:>8.3f}".format(
                s["domain"], s["strategy"], s.get("n_queries", 0), s.get("n_pos", 0), s.get("true_ids", 0),
                s.get("pw_f", float("nan")), s.get("d_pw_f", float("nan"))))
    print("total time: {:.0f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
