"""Does *which* k identities are annotated matter, when the annotation is used by prompt tuning?

For each target domain / split, n_draws random k-identity selections. For each selection:
  vicp   prompt generated in-context by VICP from those k pairs (no gradient)
  tuned  VICP prompt then tuned on the same k pairs with the triplet loss (oracle_prompt.tune_prompt)
Spread over draws (std, worst, best = oracle@N) measures the headroom of a selection method
for each generator; both use exactly the same selections.

Example:
  python scripts/prompt_select.py --output_dir experiments/prompt_select \
      --checkpoint experiments/cv_fold1_3k/val_cuhk03/checkpoint-3000 \
      --domains viper,grid,ilids --eval_splits 1 --k 16 --n_draws 10 --steps 300 --lr 1e-3 \
      --num_icl_samples 64 --fp16 False --report_to none
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import csv
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import transformers

from adapters.args_reid import ReIDTrainingArguments
from adapters.config_reid import DOMAIN_CONFIG, NUM_SPLITS

from adapters.trainer_reid import _get_dataset_cls
from scripts.context_sensitivity import build_model, SplitCache, evaluate, context_prompts
from scripts.oracle_prompt import tune_prompt, pool_id2imgs


@dataclass
class SelectArguments:
    checkpoint: str = field(default="")
    domains: str = field(default="viper,grid,ilids")
    k: int = field(default=16)
    n_draws: int = field(default=10)
    steps: int = field(default=300)
    lr: float = field(default=1e-3)
    ids_per_batch: int = field(default=32)
    base_seed: int = field(default=0)
    # >= 0: every draw uses the SAME selection (this seed); only the question/tuning seed changes
    # -> spread over draws = noise of the generator itself, not of the selection
    selection_seed: int = field(default=-1)


def main():
    parser = transformers.HfArgumentParser((ReIDTrainingArguments, SelectArguments))
    args, a = parser.parse_args_into_dataclasses()
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = build_model(args, device, a.checkpoint)
    state = torch.load(os.path.join(a.checkpoint, "pytorch_model.bin"), map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("checkpoint:", a.checkpoint, "missing:", len(missing), "unexpected:", len(unexpected))
    for p in model.parameters():
        p.requires_grad_(False)
    os.makedirs(args.output_dir, exist_ok=True)
    rows, t0 = [], time.time()

    for name in a.domains.split(","):
        for split_id in range(min(args.eval_splits, NUM_SPLITS.get(name, 1))):
            kwargs = {"split_id": split_id} if name in NUM_SPLITS else {}
            ds = _get_dataset_cls(name)(root=DOMAIN_CONFIG["data_root"], verbose=False, **kwargs)
            assert not ({p for p, *_ in ds.train} & {p for p, *_ in ds.query + ds.gallery}), "leak"
            cache = SplitCache(ds, device, name)
            s_eval = a.base_seed + 1000 * split_id
            for draw in range(a.n_draws):
                s = a.base_seed + 1000 * split_id + 17 * draw + 5
                sel_seed = s if a.selection_seed < 0 else a.selection_seed + 1000 * split_id
                rng = np.random.RandomState(sel_seed)
                pairs, info = cache.sampler.draw(args.selection_unit, "random", a.k, rng)
                pids = info["selected"].split()  # identities (unit identity) or anchor images (unit image)
                if not pairs:
                    continue
                vicp = context_prompts(model, cache, pairs, s)
                r1v, mv = evaluate(model, cache, vicp, s_eval)
                tuned, l0, l1 = tune_prompt(model, cache, pool_id2imgs(cache, pairs=pairs), vicp, a.steps, a.lr,
                                            a.ids_per_batch, s)
                r1t, mt = evaluate(model, cache, tuned, s_eval)
                rows.append(dict(domain=name, split=split_id, draw=draw, k=a.k, vicp_rank1=r1v, vicp_mAP=mv,
                                 tuned_rank1=r1t, tuned_mAP=mt, loss_start=l0, loss_end=l1,
                                 pids=" ".join(map(str, pids))))
                print("[{:5.0f}s] {} split {} draw {}: vicp mAP {:.2f} -> tuned {:.2f}".format(
                    time.time() - t0, name, split_id, draw, mv, mt), flush=True)

    with open(os.path.join(args.output_dir, "prompt_select.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    df = pd.DataFrame(rows)
    for col in ("vicp_mAP", "tuned_mAP"):
        g = df.groupby(["domain", "split"])[col].agg(["mean", "std", "min", "max"])
        g["best-mean"] = g["max"] - g["mean"]
        g["mean-worst"] = g["mean"] - g["min"]
        print("\n" + col + "\n" + g.round(2).to_string())
    print("\ncorrelation vicp_mAP vs tuned_mAP per domain:",
          df.groupby("domain").apply(lambda x: round(np.corrcoef(x.vicp_mAP, x.tuned_mAP)[0, 1], 2)).to_dict())
    print("total time: {:.0f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
