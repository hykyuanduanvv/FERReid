"""Active pair querying + target-domain prompt: strategies x target domains x seeds.

For every target domain: round 0 = the base model (no target labels); then, for each strategy and seed,
`rounds` rounds of `budget` queries, a domain prompt tuned after every round, retrieval evaluated on the
target's query / gallery. Optionally the full-annotation upper bound (every pool identity).

Outputs (output_dir):
  active.csv   one row per domain / strategy / seed / round: queries, anchors, positive / negative /
               inferred answers, clusters, true identities covered, cluster purity, threshold, Rank-1, mAP
  summary.csv  mean / std over seeds per domain / strategy / round
  paired.csv   per domain / strategy / round: mean and std over seeds of the mAP difference to the reference
               strategy of the same seed (--paired_ref, default random), and the area under the mAP-vs-answers
               curve (rounds evaluated, mean over seeds); paired differences cancel the seed's shared noise
  prompts/     the final domain prompt of every run (--save_prompts True): a constant per-domain parameter

Image sets larger than --cache_max are streamed from disk (Market / MSMT17 pools and galleries); smaller ones
(e.g. CUHK03 query) are cached on the device.

  python scripts/eval_active.py --output_dir experiments/act_cuhk03 --checkpoint experiments/base_vpt_to_cuhk03/checkpoint-12000 \
      --domains cuhk03 --strategies cover,uncertain,balanced,random,anchor:random \
      --rounds 5 --budget 200 --n_seeds 2 --fp16 True --report_to none

Cluster repair (pseudo labels of the whole pool, merge / split questions; docs/CLUSTER_REPAIR.md):

  python scripts/eval_active.py --output_dir experiments/rep_cuhk03 --checkpoint experiments/base_md_cuhk03/checkpoint-12000       --domains cuhk03 --pseudo True --warm_start True --strategies none,repair,repair_random,random       --rounds 5 --budget 200 --n_seeds 3 --paired_ref repair_random --fp16 True --report_to none
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import csv
import time
from collections import defaultdict
from dataclasses import dataclass, field, fields

import numpy as np
import torch
import transformers

from adapters.args_reid import ReIDTrainingArguments
from adapters.active.image_store import TargetSplit
from adapters.active.loop import ActiveConfig, ActiveRun, evaluate_base
from adapters.baseline_model import load_checkpoint_model
from adapters.config_reid import DOMAIN_CONFIG, NO_CAMERA_DOMAINS
from adapters.trainer_reid import _get_dataset_cls


@dataclass
class ActiveArguments:
    checkpoint: str = field(default="")
    domains: str = field(default="cuhk03")  # the fold's target domain (comma-separated: several, same model)
    strategies: str = field(default="cover,uncertain,balanced,confident,random,anchor:random")
    n_seeds: int = field(default=3)
    base_seed: int = field(default=0)
    rounds: int = field(default=5)
    budget: int = field(default=200)
    candidate_k: int = field(default=10)
    expand_ratio: float = field(default=0.5)
    prompt_mode: str = field(default="append")
    domain_tokens: int = field(default=0)       # 0: as many as the base model's source-domain tokens (else 8)
    token_init: str = field(default="source_mean")  # source_mean | random
    init_std: float = field(default=0.02)
    steps: int = field(default=300)
    lr: float = field(default=3e-4)
    ids_per_batch: int = field(default=32)
    hn_prob: float = field(default=0.5)
    warm_start: bool = field(default=False)
    eval_rounds: str = field(default="all")
    oracle_all: bool = field(default=True)      # also the full-annotation upper bound (one run per target)
    save_prompts: bool = field(default=True)
    cache_max: int = field(default=40000)       # sets up to this size are decoded once and kept on the GPU (fp16)
    pseudo: bool = field(default=False)
    pseudo_k1: int = field(default=30)
    pseudo_k2: int = field(default=6)
    pseudo_eps: float = field(default=0.6)
    pseudo_min_samples: int = field(default=4)
    contrast_weight: float = field(default=1.0)
    contrast_temp: float = field(default=0.05)
    cross_cam: bool = field(default=True)
    repair_k: int = field(default=5)
    repair_per_cluster: int = field(default=1)
    paired_ref: str = field(default="random")   # paired.csv: differences to this strategy (same seed)
    budget_schedule: str = field(default="")    # questions per round, e.g. "250,0,0,0,0" (overrides --budget)
    strategy_suffix: str = field(default="")    # appended to the strategy name in the outputs (e.g. "@front")
    cam_norm: bool = field(default=False)       # clustering / selection on camera-normalised features
    shortlist_k: int = field(default=5)         # shortlist questions: candidate clusters per question
    shortlist_rank: str = field(default="rule")  # shortlist: rule | committee (source-token clusterings)


def load_split(name, device, cache_max, num_workers):
    ds = _get_dataset_cls(name)(root=DOMAIN_CONFIG["data_root"], verbose=False)
    pool = {p for p, *_ in ds.train}
    assert not (pool & {p for p, *_ in ds.query + ds.gallery}), "{}: pool overlaps query/gallery".format(name)
    return TargetSplit(ds, name, device, has_cameras=name not in NO_CAMERA_DOMAINS, cache_max=cache_max,
                       num_workers=num_workers)


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["domain"], r["strategy"], r["round"])].append(r)
    out = []
    for (d, s, rnd), rs in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        m = np.array([r["mAP"] for r in rs], float)
        r1 = np.array([r["rank1"] for r in rs], float)
        out.append({"domain": d, "strategy": s, "round": rnd, "n": int(np.isfinite(m).sum()),
                    "queries": float(np.mean([r.get("n_queries", 0) for r in rs])),
                    "anchors": float(np.mean([r.get("n_anchors", 0) for r in rs])),
                    "true_ids": float(np.mean([r.get("true_ids", 0) for r in rs])),
                    "mAP": float(np.nanmean(m)) if np.isfinite(m).any() else float("nan"),
                    "mAP_std": float(np.nanstd(m)) if np.isfinite(m).any() else float("nan"),
                    "rank1": float(np.nanmean(r1)) if np.isfinite(r1).any() else float("nan")})
    return out


def paired(rows, ref):
    """mAP difference to `ref` of the same domain / seed / round, mean and std over seeds; AULC per strategy."""
    by = {(r["domain"], r["strategy"], r["seed"], r["round"]): r["mAP"] for r in rows if r["strategy"] != "base"}
    base = {r["domain"]: r["mAP"] for r in rows if r["strategy"] == "base"}
    groups, curves = defaultdict(list), defaultdict(list)
    for (d, s, seed, rnd), m in by.items():
        if rnd == 0 or m != m:
            continue
        q = [r.get("n_queries", 0) for r in rows if (r["domain"], r["strategy"], r["seed"], r["round"]) == (d, s, seed, rnd)][0]
        curves[(d, s, seed)].append((q, m))
        o = by.get((d, ref, seed, rnd))
        if o is not None and o == o:
            groups[(d, s, rnd)].append(m - o)
    out = []
    for (d, s, rnd), diffs in sorted(groups.items()):
        out.append({"domain": d, "strategy": s, "round": rnd, "ref": ref, "n": len(diffs),
                    "dmAP": float(np.mean(diffs)), "dmAP_std": float(np.std(diffs)),
                    "wins": int(sum(x > 0 for x in diffs))})
    aulc = defaultdict(list)
    for (d, s, seed), pts in curves.items():  # trapezoid from (0, base mAP), normalised by the last budget
        pts = sorted([(0, base.get(d, pts[0][1]))] + pts)
        x, y = np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)
        if x[-1] > 0:
            aulc[(d, s)].append(float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2) / x[-1]))
    for (d, s), v in sorted(aulc.items()):
        out.append({"domain": d, "strategy": s, "round": "aulc", "ref": "", "n": len(v),
                    "dmAP": float(np.mean(v)), "dmAP_std": float(np.std(v)), "wins": ""})
    return out


def main():
    parser = transformers.HfArgumentParser((ReIDTrainingArguments, ActiveArguments))
    args, a = parser.parse_args_into_dataclasses()
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(int(os.environ.get("FERREID_CPU_THREADS", "8")))  # the work runs on the GPU
    model = load_checkpoint_model(args, device, a.checkpoint)
    cfg = ActiveConfig(**{f.name: getattr(a, f.name) for f in fields(ActiveConfig)})
    strategies = [s for s in a.strategies.split(",") if s]
    os.makedirs(args.output_dir, exist_ok=True)
    if a.save_prompts:
        os.makedirs(os.path.join(args.output_dir, "prompts"), exist_ok=True)
    rows, t0 = [], time.time()

    def save():
        write_csv(os.path.join(args.output_dir, "active.csv"), rows)
        write_csv(os.path.join(args.output_dir, "summary.csv"), summarize(rows))
        write_csv(os.path.join(args.output_dir, "paired.csv"), paired(rows, a.paired_ref))

    for name in a.domains.split(","):
        split_id = 0  # one fixed split per dataset
        split = load_split(name, device, a.cache_max, args.eval_num_workers)
        print("[{:6.0f}s] {}: pool {} images ({}), query {}, gallery {}".format(
            time.time() - t0, name, len(split.pool), "cached" if split.pool.cached else "streamed",
            len(split.query), len(split.gallery)), flush=True)
        base = dict(domain=name, split=split_id)
        r1, mAP = evaluate_base(model, split)
        rows.append(dict(base, strategy="base", seed=0, round=0, n_queries=0, n_anchors=0, rank1=r1, mAP=mAP))
        print("  base: mAP {:.2f} R1 {:.2f}".format(mAP, r1), flush=True)
        if a.oracle_all:
            run = ActiveRun(model, split, "random", cfg, seed=a.base_seed)
            rows.append(dict(base, strategy="oracle_all", seed=0, **run.run(oracle_all=True)[0]))
            print("  oracle_all: mAP {:.2f} R1 {:.2f}".format(rows[-1]["mAP"], rows[-1]["rank1"]), flush=True)
        for seed in range(a.n_seeds):
            s = a.base_seed + seed  # the same seed for every strategy
            for strategy in strategies:
                run = ActiveRun(model, split, strategy, cfg, seed=s)
                for row in run.run():
                    rows.append(dict(base, strategy=strategy + a.strategy_suffix, seed=s, **row))  # actual seed: runs split over tasks merge
                if a.save_prompts:
                    torch.save({"prompt": run.prompt.cpu(), "domain": name, "strategy": strategy, "seed": s,
                                "checkpoint": a.checkpoint, "config": cfg.__dict__,
                                "n_queries": run.store.stats()["n_queries"], "n_anchors": run.anchors_used},
                               os.path.join(args.output_dir, "prompts", "{}_{}_seed{}.pt".format(
                                   name, (strategy + a.strategy_suffix).replace(":", "-").replace("@", "-"), s)))
                save()
        print("[{:6.0f}s] {} done".format(time.time() - t0, name), flush=True)
        del split
        if device == "cuda":
            torch.cuda.empty_cache()
    save()

    print("\n{:<10} {:<20} {:>5} {:>8} {:>8} {:>8} {:>7} {:>7} {:>6}".format(
        "domain", "strategy", "round", "queries", "anchors", "ids", "mAP", "std", "R1"))
    for s in summarize(rows):
        print("{:<10} {:<20} {:>5} {:>8.0f} {:>8.0f} {:>8.1f} {:>7.2f} {:>7.2f} {:>6.2f}".format(
            s["domain"], s["strategy"], s["round"], s["queries"], s["anchors"], s["true_ids"], s["mAP"],
            s["mAP_std"], s["rank1"]))
    print("total time: {:.0f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
