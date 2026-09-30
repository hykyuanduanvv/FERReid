"""Load a checkpoint and evaluate context selection methods on the target domains.

Loops over domain x split x method x budget k x seed, writes one CSV row per run,
and prints mean (over splits) and std (over seeds) per domain/method/k.

Example:
  python scripts/eval_context.py --output_dir experiments/x/eval \
      --checkpoint experiments/x/checkpoint-5000 --methods first,random \
      --ks 2,4,8,16,32 --eval_seeds 3 --num_icl_samples 64 --fp16 True --report_to none
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import csv
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch
import transformers

from adapters.args_reid import ReIDTrainingArguments


@dataclass
class EvalArguments:
    checkpoint: str = field(default="")
    methods: str = field(default="first,random")
    ks: str = field(default="2,4,8,16,32")
    domains: str = field(default="target")  # "target", "val", or comma-separated names
    out_csv: str = field(default="")
    # a checkpoint whose keys do not match the model is an error unless explicitly allowed
    allow_partial: bool = field(default=False)


def main():
    parser = transformers.HfArgumentParser((ReIDTrainingArguments, EvalArguments))
    args, eargs = parser.parse_args_into_dataclasses()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from adapters.trainer_reid import DGReIDTrainer
    if eargs.checkpoint:
        # rebuild the checkpoint's own architecture and source-identity count (training_args.bin)
        from adapters.reid_model import apply_checkpoint_structure
        apply_checkpoint_structure(args, eargs.checkpoint)
        saved = os.path.join(eargs.checkpoint, "training_args.bin")
        if os.path.isfile(saved):
            saved = torch.load(saved, map_location="cpu", weights_only=False)
            for name in ("source_domains", "source_all_images", "val_domains"):
                if hasattr(saved, name) and getattr(args, name) != getattr(saved, name):
                    print("[checkpoint] {} = {!r} (command line had {!r})".format(
                        name, getattr(saved, name), getattr(args, name)))
                    setattr(args, name, getattr(saved, name))
    trainer = DGReIDTrainer(args=args, device=device)

    if eargs.checkpoint:
        print("Loading checkpoint:", eargs.checkpoint)
        state = torch.load(os.path.join(eargs.checkpoint, "pytorch_model.bin"), map_location=device, weights_only=True)
        missing, unexpected = trainer.model.load_state_dict(state, strict=False)
        print("missing:", len(missing), "unexpected:", len(unexpected))
        if (missing or unexpected) and not eargs.allow_partial:
            raise RuntimeError("checkpoint does not match the model (missing {}, unexpected {}; e.g. {}). "
                               "Check --model_type/--backbone or pass --allow_partial True."
                               .format(len(missing), len(unexpected), (missing + unexpected)[:3]))
    else:
        print("WARNING: no checkpoint given, evaluating initial weights")

    if eargs.domains == "target":
        domains = trainer.target_domains
    elif eargs.domains == "val":
        domains = trainer.val_domains
    else:
        domains = eargs.domains.split(",")
    methods = eargs.methods.split(",")
    ks = [int(k) for k in eargs.ks.split(",")]

    rows = trainer.run_eval(domains, methods, ks, seeds=range(args.eval_seeds), num_splits=args.eval_splits)

    out_csv = eargs.out_csv or os.path.join(args.output_dir, "context_eval.csv")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Wrote {} rows to {}".format(len(rows), out_csv))

    # per seed: average over splits; then mean/std over seeds
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[(r["domain"], r["method"], r["k"])][r["seed"]].append((r["rank1"], r["mAP"]))
    print("\n{:10s} {:8s} {:>4s} {:>6s} {:>16s} {:>16s}".format("domain", "method", "k", "splits", "rank1", "mAP"))
    for (d, m, k), by_seed in sorted(groups.items()):
        per_seed = np.array([np.mean(v, axis=0) for v in by_seed.values()])
        n_splits = len(next(iter(by_seed.values())))
        mu, sd = per_seed.mean(0), per_seed.std(0)
        print("{:10s} {:8s} {:>4d} {:>6d} {:>9.2f} ± {:<5.2f} {:>9.2f} ± {:<5.2f}".format(
            d, m, k, n_splits, mu[0], sd[0], mu[1], sd[1]))


if __name__ == "__main__":
    main()
