"""Direction B, phase 1: compare label-free context selectors at test time (the generator is fixed).

For each target domain / split (small targets: images are cached on the GPU), every selector picks k
anchor images from the unlabeled pool; simulated annotation turns them into pairs; the generator
turns the pairs into a prompt; retrieval is evaluated. All selectors use the same seeds per draw.

  --generator incontext : prompt generated from the context without gradients (VICP / direction-A model)
  --generator tuned     : prompt tuned on the annotated pairs with the triplet loss (oracle_prompt.tune_prompt),
                          starting from the model's context-free prompt (VPT prompt / residual base prompt /
                          VICP's in-context prompt) -- the label-using generator that is already known to be
                          sensitive to the selection

Selector features: label-free (adapters/selectors.extract_selector_features), computed once per split.

Outputs (output_dir): selectors.csv (one row per domain/split/selector/k/draw, with n_pairs / n_fail / n_dup
and the selection properties p_*), summary.csv (mean / std / worst per selector and k, difference to
random, random oracle@N = best random draw), correlation.csv (rank correlation of each property with mAP
over the random draws, centred within each split).

  python scripts/eval_selectors.py --output_dir experiments/sel_tuned_vpt --checkpoint experiments/s1_vpt_bot/checkpoint-3000 \
      --generator tuned --domains viper,grid,ilids --eval_splits 3 --ks 4,16 --n_draws 3 \
      --num_icl_samples 64 --fp16 False --report_to none
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import csv
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch
import transformers

from adapters.args_reid import ReIDTrainingArguments
from adapters.config_reid import DOMAIN_CONFIG, NUM_SPLITS
from adapters.context_selection import IMAGE_SELECTORS, needs_features
from adapters.selectors import extract_selector_features
from adapters.trainer_reid import _get_dataset_cls
from scripts.context_sensitivity import build_model, SplitCache, evaluate, context_prompts, budget_ok
from scripts.oracle_prompt import tune_prompt, pool_id2imgs

ALL = "random,first,dedup,pairable,typical,kcenter,camera_balanced,style_cover,hard_negative,facility,facility_camera"


@dataclass
class SelArguments:
    checkpoint: str = field(default="")
    generator: str = field(default="incontext")      # incontext | tuned
    selectors: str = field(default=ALL)
    domains: str = field(default="viper,grid,ilids")
    ks: str = field(default="4,8,16")
    n_draws: int = field(default=5)
    random_draws: int = field(default=10)             # random gets more draws: baseline, oracle@N, correlations
    steps: int = field(default=300)                   # tuned generator
    lr: float = field(default=3e-4)                   # tuned generator (chosen on CUHK03 on 2026-09-30)
    ids_per_batch: int = field(default=32)
    base_seed: int = field(default=0)


def context_free_prompt(model):
    if hasattr(model, "base_prompts") and model.base_prompts() is not None:
        return model.base_prompts()
    if hasattr(model, "prompt"):
        return model.prompt
    return None


def spearman(a, b):
    from scipy.stats import spearmanr
    ok = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[ok], b[ok]).correlation) if ok.sum() > 3 else float("nan")


def main():
    parser = transformers.HfArgumentParser((ReIDTrainingArguments, SelArguments))
    args, a = parser.parse_args_into_dataclasses()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    model = build_model(args, device, a.checkpoint)
    state = torch.load(os.path.join(a.checkpoint, "pytorch_model.bin"), map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("checkpoint:", a.checkpoint, "missing:", len(missing), "unexpected:", len(unexpected))
    if missing or unexpected:
        raise RuntimeError("checkpoint does not match the model")
    for p in model.parameters():
        p.requires_grad_(False)
    selectors = [s for s in a.selectors.split(",") if s]
    unknown = [s for s in selectors if s not in IMAGE_SELECTORS]
    if unknown:
        raise SystemExit("unknown selectors {}; available: {}".format(unknown, sorted(IMAGE_SELECTORS)))
    if a.generator not in ("incontext", "tuned"):
        raise SystemExit("--generator must be incontext or tuned")
    if a.generator == "incontext" and args.model_type != "vicp":
        raise SystemExit("--generator incontext needs a VICP-type checkpoint (vpt / plain ignore the context)")
    ks = [int(k) for k in a.ks.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)
    rows, t0 = [], time.time()

    for name in a.domains.split(","):
        for split_id in range(min(args.eval_splits, NUM_SPLITS.get(name, 1))):
            kwargs = {"split_id": split_id} if name in NUM_SPLITS else {}
            ds = _get_dataset_cls(name)(root=DOMAIN_CONFIG["data_root"], verbose=False, **kwargs)
            assert not ({p for p, *_ in ds.train} & {p for p, *_ in ds.query + ds.gallery}), "leak"
            cache = SplitCache(ds, device, name)
            if any(needs_features(s) for s in selectors):
                order = torch.tensor([cache.pool_idx[p] for p, *_ in ds.train], device=cache.pool_imgs.device)
                feats, style = extract_selector_features(model, cache.pool_imgs[order], device)
                cache.sampler.attach_features(feats, style)
            s_eval = a.base_seed + 1000 * split_id
            for k in ks:
                if not budget_ok(cache, "image", k):
                    continue
                for draw in range(max(a.n_draws, a.random_draws)):
                    seed = a.base_seed + 1000 * split_id + 100 * k + draw  # same seed for every selector
                    for sel in selectors:
                        if draw >= (a.random_draws if sel == "random" else a.n_draws):
                            continue
                        pairs, info = cache.sampler.draw("image", sel, k, np.random.RandomState(seed))
                        if not pairs:
                            continue
                        if a.generator == "incontext":
                            prompts = context_prompts(model, cache, pairs, seed)
                        else:
                            init = context_free_prompt(model)
                            if init is None:
                                init = context_prompts(model, cache, pairs, seed)
                            prompts, _, _ = tune_prompt(model, cache, pool_id2imgs(cache, pairs=pairs), init,
                                                        a.steps, a.lr, a.ids_per_batch, seed)
                        r1, mAP = evaluate(model, cache, prompts, s_eval)
                        rows.append(dict(domain=name, split=split_id, selector=sel, k=k, draw=draw,
                                         n_pairs=info["n_pairs"], n_fail=info["n_fail"], n_dup=info["n_dup"],
                                         rank1=r1, mAP=mAP, **{x: v for x, v in info.items() if x.startswith("p_")}))
            print("[{:6.0f}s] {} split {} done".format(time.time() - t0, name, split_id), flush=True)
            del cache
            torch.cuda.empty_cache()

    keys = ["domain", "split", "selector", "k", "draw", "n_pairs", "n_fail", "n_dup", "rank1", "mAP"]
    keys += sorted({x for r in rows for x in r if x.startswith("p_")})
    with open(os.path.join(args.output_dir, "selectors.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # ---------------------------------------------------------------- summary per domain / selector / k
    def per_split(dom, sel, k, fn):
        vals = []
        for s in sorted({r["split"] for r in rows if r["domain"] == dom}):
            v = [r["mAP"] for r in rows if r["domain"] == dom and r["selector"] == sel and r["k"] == k and r["split"] == s]
            if v:
                vals.append(fn(v))
        return float(np.mean(vals)) if vals else float("nan")

    summary = []
    for dom in sorted({r["domain"] for r in rows}):
        for k in ks:
            rnd = per_split(dom, "random", k, np.mean)
            oracle = per_split(dom, "random", k, np.max)
            for sel in selectors:
                mean = per_split(dom, sel, k, np.mean)
                pairs = [r["n_pairs"] for r in rows if r["domain"] == dom and r["selector"] == sel and r["k"] == k]
                summary.append(dict(domain=dom, k=k, selector=sel, mean=mean, std=per_split(dom, sel, k, np.std),
                                    worst=per_split(dom, sel, k, np.min), vs_random=mean - rnd,
                                    random_oracle=oracle, gap_to_oracle=oracle - mean,
                                    n_pairs=float(np.mean(pairs)) if pairs else float("nan")))
    with open(os.path.join(args.output_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print("\n{:<7} {:>3} {:<16} {:>7} {:>6} {:>7} {:>9} {:>8} {:>7}".format(
        "domain", "k", "selector", "mAP", "std", "worst", "vs_rand", "to_orc", "pairs"))
    for s in summary:
        print("{:<7} {:>3} {:<16} {:7.2f} {:6.2f} {:7.2f} {:+9.2f} {:8.2f} {:7.1f}".format(
            s["domain"], s["k"], s["selector"], s["mean"], s["std"], s["worst"], s["vs_random"],
            s["gap_to_oracle"], s["n_pairs"]))

    # ---------------------------------------------------------------- what makes a context good (random draws)
    props = sorted({x for r in rows for x in r if x.startswith("p_")} | {"n_pairs"})
    corr = []
    for dom in sorted({r["domain"] for r in rows}):
        R = [r for r in rows if r["domain"] == dom and r["selector"] == "random"]
        if len(R) < 5:
            continue
        y = np.array([r["mAP"] for r in R], float)
        groups = [(r["split"], r["k"]) for r in R]
        for g in set(groups):  # remove split / k effects
            m = np.array([x == g for x in groups])
            y[m] -= y[m].mean()
        for p in props:
            x = np.array([r.get(p, np.nan) for r in R], float)
            for g in set(groups):
                m = np.array([q == g for q in groups])
                x[m] -= np.nanmean(x[m]) if np.isfinite(x[m]).any() else 0
            corr.append(dict(domain=dom, property=p, spearman=spearman(x, y), n=len(R)))
    if corr:
        with open(os.path.join(args.output_dir, "correlation.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(corr[0]))
            w.writeheader()
            w.writerows(corr)
        print("\nrank correlation of selection properties with mAP over random draws (within split / k):")
        for c in corr:
            print("  {:<7} {:<16} {:+.2f}  (n={})".format(c["domain"], c["property"], c["spearman"], c["n"]))
    print("total time: {:.0f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
