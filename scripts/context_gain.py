"""Direction A decision test: does the in-context prompt beat a context-free prompt on unseen domains,
and is the gain specific to the domain the context comes from?

For each target domain / split (small targets: images are cached on the GPU):
  own_k<k>   n_draws contexts of k anchor images from the domain's own unlabeled pool (label-free
             image selection + simulated annotation)                    -> in-context prompt
  cross      contexts (k = max ks) from *other* target domains' pools -> prompt from the wrong domain
  base       the model's own context-free prompt (--prompt_mode residual only)
  reference  a separately trained model without context, e.g. VPT (--reference_checkpoint, optional)
  zero       all-zero prompt

Reported per domain (mean over splits): mAP of every condition, gain_vs_base = own - base,
gain_vs_ref = own - reference, specificity = own - cross, spread over draws, and prompt cosine
similarities (between own draws, own vs cross, own vs base) that show whether the prompt moves at all.
Decision rule used in docs/RUN_PLAN_20261001.md: direction A works if, on the unseen domains, the
own-domain prompt beats the context-free prompt (base / VPT) clearly (>= ~1 mAP, beyond the ~0.5
run-to-run noise) *and* beats the cross-domain prompt.

  python scripts/context_gain.py --output_dir experiments/a_gain_X --checkpoint experiments/a_X/checkpoint-3000 \
      --reference_checkpoint experiments/s1_vpt_bot/checkpoint-3000 --domains viper,grid,ilids --eval_splits 3 \
      --ks 4,16 --n_draws 5 --num_icl_samples 64 --fp16 True --report_to none
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import copy
import csv
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
import transformers

from adapters.args_reid import ReIDTrainingArguments
from adapters.config_reid import DOMAIN_CONFIG, NUM_SPLITS
from adapters.trainer_reid import _get_dataset_cls
from scripts.context_sensitivity import build_model, SplitCache, evaluate, context_prompts, budget_ok


@dataclass
class GainArguments:
    checkpoint: str = field(default="")
    reference_checkpoint: str = field(default="")  # e.g. a VPT model trained on the same data
    domains: str = field(default="viper,grid,ilids")
    ks: str = field(default="4,16")
    n_draws: int = field(default=5)
    n_cross: int = field(default=2)                # contexts per other domain
    base_seed: int = field(default=0)


def load_weights(model, checkpoint, device):
    state = torch.load(os.path.join(checkpoint, "pytorch_model.bin"), map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("checkpoint:", checkpoint, "missing:", len(missing), "unexpected:", len(unexpected))
    if missing or unexpected:
        raise RuntimeError("checkpoint does not match the model: {}".format((missing + unexpected)[:3]))


def flat(p):
    return p[:1].float().flatten()


def cos(a, b):
    return float(F.cosine_similarity(a, b, dim=0))


def main():
    parser = transformers.HfArgumentParser((ReIDTrainingArguments, GainArguments))
    args, g = parser.parse_args_into_dataclasses()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref_args = copy.deepcopy(args)
    model = build_model(args, device, g.checkpoint)
    load_weights(model, g.checkpoint, device)
    ref = None
    if g.reference_checkpoint:
        ref = build_model(ref_args, device, g.reference_checkpoint)
        load_weights(ref, g.reference_checkpoint, device)
    unit = args.selection_unit
    ks = [int(k) for k in g.ks.split(",")]
    domains = g.domains.split(",")
    L, V = model.num_layers, args.num_vpt_tokens
    zero = torch.zeros(1, L, V, model.hidden_size, device=device)
    base = model.base_prompts() if hasattr(model, "base_prompts") else None
    os.makedirs(args.output_dir, exist_ok=True)

    def load(name, split_id):
        kwargs = {"split_id": split_id} if name in NUM_SPLITS else {}
        ds = _get_dataset_cls(name)(root=DOMAIN_CONFIG["data_root"], verbose=False, **kwargs)
        assert not ({p for p, *_ in ds.train} & {p for p, *_ in ds.query + ds.gallery}), "leak"
        return SplitCache(ds, device, name)

    # split-0 pools of every domain, source of the cross-domain contexts
    cross_pool = {d: load(d, 0) for d in domains}
    rows, t0 = [], time.time()
    for name in domains:
        for split_id in range(min(args.eval_splits, NUM_SPLITS.get(name, 1))):
            cache = cross_pool[name] if split_id == 0 else load(name, split_id)
            s0 = g.base_seed + 1000 * split_id

            def add(cond, k, draw, prompts, src=name, info=None, mdl=model):
                r1, mAP = evaluate(mdl, cache, prompts, s0)
                rows.append(dict(domain=name, split=split_id, condition=cond, k=k, draw=draw, context_from=src,
                                 n_pairs=(info or {}).get("n_pairs", 0), rank1=r1, mAP=mAP))
                return mAP

            add("zero", 0, 0, zero)
            if base is not None:
                add("base", 0, 0, base)
            if ref is not None:
                ref_prompt = ref.prompt if hasattr(ref, "prompt") else None
                add("reference", 0, 0, ref_prompt, mdl=ref)
            own_prompts = defaultdict(list)
            for k in ks:
                if not budget_ok(cache, unit, k):
                    continue
                for d in range(g.n_draws):
                    seed = s0 + 100 * k + d
                    pairs, info = cache.sampler.draw(unit, "random", k, np.random.RandomState(seed))
                    if not pairs:
                        continue
                    p = context_prompts(model, cache, pairs, seed)
                    own_prompts[k].append(flat(p))
                    add("own", k, d, p, info=info)
            kmax = max(ks)
            cross_prompts = []
            for other in domains:
                if other == name or not budget_ok(cross_pool[other], unit, kmax):
                    continue
                for d in range(g.n_cross):
                    seed = s0 + 50000 + 100 * kmax + d
                    pairs, info = cross_pool[other].sampler.draw(unit, "random", kmax, np.random.RandomState(seed))
                    if not pairs:
                        continue
                    p = context_prompts(model, cross_pool[other], pairs, seed)
                    cross_prompts.append(flat(p))
                    add("cross", kmax, d, p, src=other, info=info)
            # prompt geometry (split 0 only is enough, but cheap everywhere)
            own = own_prompts.get(kmax, [])
            geo = dict(domain=name, split=split_id, condition="cosine", k=kmax, draw=-1, context_from="-",
                       n_pairs=0, rank1=float("nan"), mAP=float("nan"))
            geo["cos_own_own"] = float(np.mean([cos(a, b) for i, a in enumerate(own) for b in own[i + 1:]])) if len(own) > 1 else float("nan")
            geo["cos_own_cross"] = float(np.mean([cos(a, b) for a in own for b in cross_prompts])) if own and cross_prompts else float("nan")
            geo["cos_own_base"] = float(np.mean([cos(a, flat(base)) for a in own])) if own and base is not None else float("nan")
            rows.append(geo)
            print("[{:6.0f}s] {} split {} done".format(time.time() - t0, name, split_id), flush=True)

    keys = ["domain", "split", "condition", "k", "draw", "context_from", "n_pairs", "rank1", "mAP",
            "cos_own_own", "cos_own_cross", "cos_own_base"]
    with open(os.path.join(args.output_dir, "context_gain.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # summary: per domain, mean over splits of per-split means
    summary = {}
    for name in domains:
        R = [r for r in rows if r["domain"] == name and r["condition"] != "cosine"]
        splits = sorted({r["split"] for r in R})

        def m(cond, k=None):
            vals = []
            for s in splits:
                v = [r["mAP"] for r in R if r["split"] == s and r["condition"] == cond and (k is None or r["k"] == k)]
                if v:
                    vals.append(np.mean(v))
            return float(np.mean(vals)) if vals else float("nan")

        def spread(k):
            vals = [np.std([r["mAP"] for r in R if r["split"] == s and r["condition"] == "own" and r["k"] == k])
                    for s in splits]
            return float(np.mean(vals)) if vals else float("nan")

        C = [r for r in rows if r["domain"] == name and r["condition"] == "cosine"]
        entry = {"zero": m("zero"), "base": m("base"), "reference": m("reference"), "cross": m("cross")}
        for k in ks:
            entry["own_k{}".format(k)] = m("own", k)
            entry["own_k{}_std".format(k)] = spread(k)
        own = entry["own_k{}".format(max(ks))]
        entry.update(gain_vs_base=own - entry["base"], gain_vs_ref=own - entry["reference"],
                     specificity=own - entry["cross"])
        for c in ("cos_own_own", "cos_own_cross", "cos_own_base"):
            entry[c] = float(np.nanmean([r[c] for r in C])) if C else float("nan")
        summary[name] = entry
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("\nmAP (mean over splits); own = own-domain context with k = {}".format(max(ks)))
    print("{:<8} {:>7} {:>7} {:>7} {:>7} {:>7} {:>9} {:>9} {:>9}  {:>7} {:>7}".format(
        "domain", "own", "base", "ref", "cross", "zero", "vs_base", "vs_ref", "specific", "cos_oo", "cos_oc"))
    for name, e in summary.items():
        print("{:<8} {:7.2f} {:7.2f} {:7.2f} {:7.2f} {:7.2f} {:+9.2f} {:+9.2f} {:+9.2f}  {:7.4f} {:7.4f}".format(
            name, e["own_k{}".format(max(ks))], e["base"], e["reference"], e["cross"], e["zero"],
            e["gain_vs_base"], e["gain_vs_ref"], e["specificity"], e["cos_own_own"], e["cos_own_cross"]))
    print("total time: {:.0f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
