"""Training pairs for the MLLM annotator from the labeled SOURCE domains of a fold (scripts/finetune_vlm.py).

For every source domain (train split, person ids are source labels) the base model's features (shared prompt +
source-mean tokens, as the base model sees an unknown domain) give, per identity and anchor image:
  hardpos   the least similar image of the same person from another camera (any camera without camera ids)
  hardneg   the most similar image of another person from another camera
  randpos / randneg   a random image of the same / another person
`per_id` anchors per identity; the four kinds are balanced, so half of the pairs are positives.
Writes <output_dir>/train_pairs.csv: domain, kind, path_a, path_b, same, sim.

  python scripts/make_vlm_source_pairs.py --output_dir experiments/vlm_train_msmt \\
      --checkpoint experiments/base_md_msmt17/checkpoint-12000 --domains market1501,cuhk03,cuhksysu --fp16 True --report_to none
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import csv
from dataclasses import dataclass, field

import numpy as np
import torch
import transformers

from adapters.args_reid import ReIDTrainingArguments
from adapters.active.image_store import features
from adapters.active.loop import default_prompt
from adapters.baseline_model import load_checkpoint_model
from scripts.eval_active import load_split


@dataclass
class SourcePairArgs:
    checkpoint: str = field(default="")
    domains: str = field(default="market1501,cuhk03,cuhksysu")
    per_id: int = field(default=2)
    cache_max: int = field(default=40000)


def main():
    parser = transformers.HfArgumentParser((ReIDTrainingArguments, SourcePairArgs))
    args, a = parser.parse_args_into_dataclasses()
    model = load_checkpoint_model(args, "cuda", a.checkpoint)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for name in a.domains.split(","):
        split = load_split(name, "cuda", a.cache_max, args.eval_num_workers)
        pids, cams, paths = np.asarray(split.pool_pids), np.asarray(split.pool_cams), split.pool_paths
        X = features(model, split.pool, default_prompt(model))
        pt, ct = torch.as_tensor(pids, device=X.device), torch.as_tensor(cams, device=X.device)
        n_before = len(rows)
        for p in np.unique(pids):
            idx = np.flatnonzero(pids == p)
            if len(idx) < 2:
                continue
            for a_ in rng.choice(idx, size=min(a.per_id, len(idx)), replace=False):
                s = X[a_] @ X.T
                other_cam = (ct != ct[a_]) if split.has_cameras else torch.ones_like(pt, dtype=torch.bool)
                pos = (pt == int(p)) & other_cam
                pos[a_] = False
                neg = (pt != int(p)) & other_cam
                if not pos.any() or not neg.any():
                    continue
                hp = int(torch.where(pos, s, torch.full_like(s, 9.0)).argmin())
                hn = int(torch.where(neg, s, torch.full_like(s, -9.0)).argmax())
                rp = int(rng.choice(torch.nonzero(pos).flatten().cpu().numpy()))
                rn = int(rng.choice(torch.nonzero(neg).flatten().cpu().numpy()))
                for kind, b, y in (("hardpos", hp, 1), ("hardneg", hn, 0), ("randpos", rp, 1), ("randneg", rn, 0)):
                    rows.append({"domain": name, "kind": kind, "path_a": paths[a_], "path_b": paths[b], "same": y,
                                 "sim": float(s[b])})
        print("{}: {} pairs from {} ids".format(name, len(rows) - n_before, len(np.unique(pids))), flush=True)
        del split, X
        torch.cuda.empty_cache()
    rng.shuffle(rows)
    with open(os.path.join(args.output_dir, "train_pairs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", len(rows), "pairs")


if __name__ == "__main__":
    main()
