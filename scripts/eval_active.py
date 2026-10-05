"""Active pair querying + target-domain prompt: strategies x target domains x splits x seeds.

For every target split: round 0 = the base model (no target labels); then, for each strategy and seed,
`rounds` rounds of `budget` queries, a domain prompt tuned after every round, retrieval evaluated on the
split's query / gallery. Optionally the full-annotation upper bound (every pool identity).

Outputs (output_dir):
  active.csv   one row per domain / split / strategy / seed / round: queries, anchors, positive / negative /
               inferred answers, clusters, true identities covered, cluster purity, threshold, Rank-1, mAP
  summary.csv  mean / std over splits x seeds per domain / strategy / round
  prompts/     the final domain prompt of every run (--save_prompts True): a constant per-domain parameter

Small targets (VIPeR / GRID / i-LIDS) are cached on the device; large ones (Market-1501, MSMT17, CUHK03 as
targets) are streamed from disk (--cache_max).

  python scripts/eval_active.py --output_dir experiments/act_small --checkpoint experiments/base_vpt/checkpoint-12000 \
      --domains viper,grid,ilids --eval_splits 3 --strategies cover,uncertain,balanced,confident,random,anchor:random \
      --rounds 4 --budget 25 --n_seeds 3 --fp16 True --report_to none
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
from adapters.config_reid import DOMAIN_CONFIG, NUM_SPLITS, NO_CAMERA_DOMAINS
from adapters.trainer_reid import _get_dataset_cls


@dataclass
class ActiveArguments:
    checkpoint: str = field(default="")
    domains: str = field(default="viper,grid,ilids")
    strategies: str = field(default="cover,uncertain,balanced,confident,random,anchor:random")
    n_seeds: int = field(default=3)
    base_seed: int = field(default=0)
    rounds: int = field(default=4)
    budget: int = field(default=25)
    candidate_k: int = field(default=10)
    expand_ratio: float = field(default=0.5)
    prompt_mode: str = field(default="append")
    domain_tokens: int = field(default=8)
    init_std: float = field(default=0.02)
    steps: int = field(default=300)
    lr: float = field(default=3e-4)
    ids_per_batch: int = field(default=32)
    hn_prob: float = field(default=0.5)
    warm_start: bool = field(default=False)
    eval_rounds: str = field(default="all")
    oracle_all: bool = field(default=True)      # also the full-annotation upper bound (one run per split)
    save_prompts: bool = field(default=True)
    cache_max: int = field(default=6000)        # sets larger than this are streamed from disk


def load_split(name, split_id, device, cache_max, num_workers):
    kwargs = {"split_id": split_id} if name in NUM_SPLITS else {}
    ds = _get_dataset_cls(name)(root=DOMAIN_CONFIG["data_root"], verbose=False, **kwargs)
    pool = {p for p, *_ in ds.train}
    assert not (pool & {p for p, *_ in ds.query + ds.gallery}), "{} split {}: pool overlaps query/gallery".format(
        name, split_id)
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


def main():
    parser = transformers.HfArgumentParser((ReIDTrainingArguments, ActiveArguments))
    args, a = parser.parse_args_into_dataclasses()
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
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

    for name in a.domains.split(","):
        for split_id in range(min(args.eval_splits, NUM_SPLITS.get(name, 1))):
            split = load_split(name, split_id, device, a.cache_max, args.eval_num_workers)
            print("[{:6.0f}s] {} split {}: pool {} images ({}), query {}, gallery {}".format(
                time.time() - t0, name, split_id, len(split.pool), "cached" if split.pool.cached else "streamed",
                len(split.query), len(split.gallery)), flush=True)
            base = dict(domain=name, split=split_id)
            r1, mAP = evaluate_base(model, split)
            rows.append(dict(base, strategy="base", seed=0, round=0, n_queries=0, n_anchors=0, rank1=r1, mAP=mAP))
            print("  base: mAP {:.2f} R1 {:.2f}".format(mAP, r1), flush=True)
            if a.oracle_all:
                run = ActiveRun(model, split, "random", cfg, seed=a.base_seed + 1000 * split_id)
                rows.append(dict(base, strategy="oracle_all", seed=0, **run.run(oracle_all=True)[0]))
                print("  oracle_all: mAP {:.2f} R1 {:.2f}".format(rows[-1]["mAP"], rows[-1]["rank1"]), flush=True)
            for seed in range(a.n_seeds):
                s = a.base_seed + 1000 * split_id + seed  # the same seed for every strategy
                for strategy in strategies:
                    run = ActiveRun(model, split, strategy, cfg, seed=s)
                    for row in run.run():
                        rows.append(dict(base, strategy=strategy, seed=seed, **row))
                    if a.save_prompts:
                        torch.save({"prompt": run.prompt.cpu(), "domain": name, "split": split_id,
                                    "strategy": strategy, "seed": seed, "checkpoint": a.checkpoint,
                                    "config": cfg.__dict__, "n_queries": run.store.stats()["n_queries"],
                                    "n_anchors": run.anchors_used},
                                   os.path.join(args.output_dir, "prompts", "{}_s{}_{}_seed{}.pt".format(
                                       name, split_id, strategy.replace(":", "-"), seed)))
                    save()
            print("[{:6.0f}s] {} split {} done".format(time.time() - t0, name, split_id), flush=True)
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
