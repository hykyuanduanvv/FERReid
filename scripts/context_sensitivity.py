"""How much does the choice of context identities matter?

For each target domain / split / budget k:
  random   N random k-identity contexts            -> spread of results (std, min, max)
  oracle@N best of those N contexts (chosen on test) -> optimistic upper bound
  first    VICP default: first k identities
  zero     all-zero prompts (no context at all)
  noise    one fixed context, only the question-sampling seed changes
           -> variance that is NOT due to which identities were chosen

Images of each split are decoded once and cached on the GPU; every evaluation is
then just forward passes.

Example:
  python scripts/context_sensitivity.py --output_dir experiments/sens_x \
      --checkpoint experiments/cv_0926/val_cuhk03/checkpoint-1000 \
      --domains viper,grid,ilids --ks 4,8,16,32 --n_contexts 30 --noise_reps 5 \
      --eval_splits 1 --num_icl_samples 64 --fp16 True --report_to none
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import csv
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from PIL import Image
from torchreid.metrics import evaluate_rank

from adapters.args_reid import ReIDTrainingArguments
from adapters.config_reid import DOMAIN_CONFIG, NUM_SPLITS
from adapters.context_selection import CandidatePool, select, make_pairs
from adapters.reid_dataset import EVAL_TRANSFORM
from adapters.trainer_reid import _get_dataset_cls, _seed_all


@dataclass
class SensArguments:
    checkpoint: str = field(default="")
    domains: str = field(default="viper,grid,ilids")
    ks: str = field(default="4,8,16,32")
    n_contexts: int = field(default=30)
    noise_reps: int = field(default=5)
    base_seed: int = field(default=0)


def build_model(args, device):
    from adapters.reid_model import ReIDModel
    model = ReIDModel(args)
    dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    model.to(dtype=dtype, device=device)
    for p in model.parameters():  # same dtype layout as DGReIDTrainer
        if p.requires_grad:
            p.data = p.to(dtype=torch.float32)
    return model.eval()


def load_images(paths, device):
    with ThreadPoolExecutor(16) as ex:
        imgs = list(ex.map(lambda p: EVAL_TRANSFORM(Image.open(p).convert("RGB")), paths))
    return torch.stack(imgs).to(device)


class SplitCache:
    """Decoded images of one split, kept on the GPU."""

    def __init__(self, ds, device):
        self.pool = CandidatePool(ds.train)
        pool_paths = sorted({p for p, *_ in ds.train})
        self.pool_idx = {p: i for i, p in enumerate(pool_paths)}
        self.pool_imgs = load_images(pool_paths, device)
        self.q_imgs = load_images([x[0] for x in ds.query], device)
        self.g_imgs = load_images([x[0] for x in ds.gallery], device)
        self.q_pids = np.array([x[1] for x in ds.query]); self.q_cams = np.array([x[2] for x in ds.query])
        self.g_pids = np.array([x[1] for x in ds.gallery]); self.g_cams = np.array([x[2] for x in ds.gallery])


@torch.no_grad()
def features(model, imgs, prompts, bs=256):
    out = []
    for i in range(0, len(imgs), bs):
        x = imgs[i:i + bs]
        f = model(x, prompts=prompts)["features"] + model(torch.flip(x, dims=(3,)), prompts=prompts)["features"]
        out.append(F.normalize(f.float(), dim=1))
    return torch.cat(out)


@torch.no_grad()
def evaluate(model, cache, prompts, seed):
    _seed_all(seed)
    qf = features(model, cache.q_imgs, prompts)
    gf = features(model, cache.g_imgs, prompts)
    distmat = (1 - qf @ gf.T).cpu().numpy()
    cmc, mAP = evaluate_rank(distmat, cache.q_pids, cache.g_pids, cache.q_cams, cache.g_cams, max_rank=10)
    return float(cmc[0]) * 100, float(mAP) * 100


@torch.no_grad()
def context_prompts(model, cache, pairs, seed):
    _seed_all(seed)
    idx = torch.tensor([[cache.pool_idx[a], cache.pool_idx[b]] for a, b in pairs], device=cache.pool_imgs.device)
    batch = {"image_crops": cache.pool_imgs[idx], "labels": torch.arange(len(pairs), device=idx.device)}
    return model(**batch)["prompts"][:1]  # identical across images when num_icl_bs == 1


def main():
    parser = transformers.HfArgumentParser((ReIDTrainingArguments, SensArguments))
    args, sargs = parser.parse_args_into_dataclasses()
    device = "cuda"
    model = build_model(args, device)
    if sargs.checkpoint:
        state = torch.load(os.path.join(sargs.checkpoint, "pytorch_model.bin"), map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("checkpoint:", sargs.checkpoint, "missing:", len(missing), "unexpected:", len(unexpected))
    else:
        print("WARNING: no checkpoint, evaluating initial weights")

    ks = [int(k) for k in sargs.ks.split(",")]
    L, V = model.num_layers, args.num_vpt_tokens
    zero_prompts = torch.zeros(1, L, V, model.hidden_size, device=device)
    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    t0 = time.time()

    for name in sargs.domains.split(","):
        n_splits = min(args.eval_splits, NUM_SPLITS.get(name, 1))
        for split_id in range(n_splits):
            kwargs = {"split_id": split_id} if name in NUM_SPLITS else {}
            ds = _get_dataset_cls(name)(root=DOMAIN_CONFIG["data_root"], verbose=False, **kwargs)
            assert not ({p for p, *_ in ds.train} & {p for p, *_ in ds.query + ds.gallery}), "leak"
            cache = SplitCache(ds, device)
            base = sargs.base_seed + 100000 * split_id
            add = lambda **r: rows.append(dict(domain=name, split=split_id, **r))

            r1, mAP = evaluate(model, cache, zero_prompts, base)
            add(k=0, kind="zero", idx=0, seed=base, rank1=r1, mAP=mAP, pids="")

            for k in ks:
                if k > len(cache.pool):
                    print("skip {} k={} (only {} eligible)".format(name, k, len(cache.pool)))
                    continue
                for kind, n in (("first", 1), ("random", sargs.n_contexts)):
                    for i in range(n):
                        seed = base + 1000 * k + i
                        rng = np.random.RandomState(seed)
                        pids = select(kind, cache.pool, k, rng)
                        pairs = make_pairs(cache.pool, pids, rng)
                        prompts = context_prompts(model, cache, pairs, seed)
                        r1, mAP = evaluate(model, cache, prompts, seed)
                        add(k=k, kind=kind, idx=i, seed=seed, rank1=r1, mAP=mAP,
                            pids=" ".join(map(str, pids)))
                # noise floor: context of random #0 fixed, only the model seed changes
                rng = np.random.RandomState(base + 1000 * k)
                pids = select("random", cache.pool, k, rng)
                pairs = make_pairs(cache.pool, pids, rng)
                for rep in range(sargs.noise_reps):
                    seed = base + 1000 * k + 500 + rep
                    prompts = context_prompts(model, cache, pairs, seed)
                    r1, mAP = evaluate(model, cache, prompts, seed)
                    add(k=k, kind="noise", idx=rep, seed=seed, rank1=r1, mAP=mAP,
                        pids=" ".join(map(str, pids)))
            print("[{:6.0f}s] {} split {} done".format(time.time() - t0, name, split_id), flush=True)
            del cache
            torch.cuda.empty_cache()

    out_csv = os.path.join(args.output_dir, "sensitivity_runs.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # summary: statistics within each split, then averaged over splits
    by = defaultdict(list)
    for r in rows:
        by[(r["domain"], r["k"], r["kind"], r["split"])].append(r)
    summary = []
    for name in sargs.domains.split(","):
        splits = sorted({r["split"] for r in rows if r["domain"] == name})
        if not splits:
            continue
        zero = np.mean([by[(name, 0, "zero", s)][0]["mAP"] for s in splits])
        for k in ks:
            if not by.get((name, k, "random", splits[0])):
                continue
            stat = defaultdict(list)
            for s in splits:
                rnd = np.array([r["mAP"] for r in by[(name, k, "random", s)]])
                rnd_r1 = np.array([r["rank1"] for r in by[(name, k, "random", s)]])
                noi = np.array([r["mAP"] for r in by[(name, k, "noise", s)]])
                stat["rand_mean"].append(rnd.mean()); stat["rand_std"].append(rnd.std())
                stat["rand_min"].append(rnd.min()); stat["oracle"].append(rnd.max())
                stat["rand_r1_mean"].append(rnd_r1.mean()); stat["oracle_r1"].append(rnd_r1.max())
                stat["first"].append(by[(name, k, "first", s)][0]["mAP"])
                stat["noise_std"].append(noi.std() if len(noi) > 1 else float("nan"))
            m = {key: float(np.mean(v)) for key, v in stat.items()}
            summary.append(dict(domain=name, k=k, splits=len(splits), zero_mAP=zero, **m,
                                gap_oracle_minus_mean=m["oracle"] - m["rand_mean"],
                                gap_oracle_minus_min=m["oracle"] - m["rand_min"]))
    sum_csv = os.path.join(args.output_dir, "sensitivity_summary.csv")
    with open(sum_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print("\nmAP (stats within a split over {} random contexts, then averaged over splits)".format(sargs.n_contexts))
    hdr = "{:7s} {:>3s} {:>6s} {:>7s} {:>7s} {:>6s} {:>7s} {:>7s} {:>7s} {:>9s} {:>8s} {:>9s}"
    print(hdr.format("domain", "k", "zero", "first", "rand", "std", "min", "oracle", "noise", "orc-mean", "orc-min", "R1 r/orc"))
    for s in summary:
        print("{:7s} {:>3d} {:6.2f} {:7.2f} {:7.2f} {:6.2f} {:7.2f} {:7.2f} {:7.2f} {:9.2f} {:8.2f} {:4.1f}/{:<4.1f}".format(
            s["domain"], s["k"], s["zero_mAP"], s["first"], s["rand_mean"], s["rand_std"], s["rand_min"],
            s["oracle"], s["noise_std"], s["gap_oracle_minus_mean"], s["gap_oracle_minus_min"],
            s["rand_r1_mean"], s["oracle_r1"]))
    print("\nnoise = std of mAP when only the question-sampling seed changes (same context)")
    print("wrote", out_csv, "and", sum_csv, "({:.0f}s)".format(time.time() - t0))


if __name__ == "__main__":
    main()
