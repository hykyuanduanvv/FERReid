"""Does the in-context model actually use the annotated context?

Rebuilds VICP's ICL sequence + prompt generation (models.Model.forward) with control
over what the context contains, and evaluates retrieval on the target domains:

  true      annotated in-domain context (reference)
  shuffle   answers randomly permuted (same yes/no ratio)
  flip      every answer inverted
  all_yes   every answer "yes"
  all_no    every answer "no"
  cross     annotated pairs from another target domain
  source    annotated pairs from a source domain (Market1501)
  pseudo    no annotation: k random target images, each paired with an augmented copy of itself
  zero      all-zero prompts

Also reports the LLM's own yes/no accuracy / AUC on the context questions (against the
true labels) and cosine similarity of prompts within / across domains.

A consistency check first verifies that mode "true" reproduces models.Model.forward's
prompts exactly for the same seed.
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
from PIL import Image
from sklearn.metrics import roc_auc_score

from adapters.args_reid import ReIDTrainingArguments
from adapters.config_reid import DOMAIN_CONFIG, NUM_SPLITS, NO_CAMERA_DOMAINS
from adapters.context_selection import ContextSampler
from adapters.reid_dataset import TRAIN_TRANSFORM
from adapters.trainer_reid import _get_dataset_cls, _seed_all
from scripts.context_sensitivity import build_model, load_images, SplitCache, evaluate

ANSWER_MODES = ("true", "shuffle", "flip", "all_yes", "all_no")


@dataclass
class DiagArguments:
    checkpoint: str = field(default="")
    domains: str = field(default="viper,grid,ilids")
    source_domain: str = field(default="market1501")
    ks: str = field(default="4,16")
    n_contexts: int = field(default=8)
    base_seed: int = field(default=0)


# ----------------------------------------------------------------------------- prompts

@torch.no_grad()
def build_prompts(model, imgs, answer_mode, seed):
    """Replicates the ICL part of models.Model.forward (num_icl_bs = 1).

    imgs: (2k, 3, H, W) ordered [id0_a, id0_b, id1_a, id1_b, ...].
    Returns prompts (1, layers, vpt, dim) and the LLM's scores on the questions.
    """
    args = model.args
    assert args.num_icl_bs == 1
    _seed_all(seed)
    enc_dtype = model.encoder.patch_embed.proj.weight.dtype
    feats = model.encoder_copy.forward_features(imgs[:256].to(dtype=enc_dtype))["x_norm_clstoken"]
    n = feats.size(0)
    labels = torch.arange(n // 2).repeat_interleave(2)
    yes_id = model.tokenizer.convert_tokens_to_ids("yes")
    no_id = model.tokenizer.convert_tokens_to_ids("no")

    pair_feats, truth = [], []
    for _ in range(args.num_icl_samples):  # same RNG call order as models.Model.forward
        x = torch.randint(0, 2, (1,)).item()
        if x == 1:
            idx = torch.randint(0, n // 2, (1,)).item()
            pair_feats.append(feats.reshape(-1, 2, model.hidden_size)[idx])
            truth.append(1)
        else:
            i1 = torch.randint(0, n, (1,)).item()
            i2 = torch.randint(0, n, (1,)).item()
            pair_feats.append(torch.stack([feats[i1], feats[i2]]))
            truth.append(int(labels[i1] == labels[i2]))
    truth = np.array(truth)

    rng = np.random.RandomState(seed + 7)  # separate stream, keeps torch RNG order intact
    shown = {"true": truth, "shuffle": rng.permutation(truth), "flip": 1 - truth,
             "all_yes": np.ones_like(truth), "all_no": np.zeros_like(truth)}[answer_mode]

    s = []
    for a in shown:
        s.extend([-1] * args.num_id_tokens)
        s.append(yes_id if a == 1 else no_id)
    input_ids = torch.tensor([s], device=feats.device).long()
    selected = input_ids == -1
    input_ids[input_ids < 0] = 0

    pf = torch.cat(pair_feats).reshape(-1, model.hidden_size * 2)
    pf = model.mm_projector(pf.to(dtype=model.mm_projector.visual_proj.weight.dtype))
    emb = model.lm.get_input_embeddings()(input_ids).clone()
    emb[selected] = emb[selected] * 0 + pf.reshape(-1, model.lm.config.hidden_size).to(emb.dtype)

    # LLM's own answers (logit yes - logit no at the position before each answer)
    logits = model.lm(inputs_embeds=emb, use_cache=False).logits[0]
    ans_pos = torch.arange(args.num_icl_samples, device=feats.device) * (args.num_id_tokens + 1) + args.num_id_tokens
    score = (logits[ans_pos - 1, yes_id] - logits[ans_pos - 1, no_id]).float().cpu().numpy()

    q = model.query_embeddings.to(dtype=emb.dtype)
    emb2 = torch.cat([emb, q.unsqueeze(0).expand(emb.size(0), -1, -1)], dim=1)
    hidden = model.lm(inputs_embeds=emb2, use_cache=False, output_hidden_states=True).hidden_states[-1]
    prompts = hidden[:, -args.num_vpt_tokens * model.num_layers:]
    prompts = model.prompt_mlp(prompts.to(dtype=model.prompt_mlp.weight.dtype))
    prompts = prompts.reshape(1, model.num_layers, args.num_vpt_tokens, -1)
    return prompts, score, truth


def llm_metrics(score, truth):
    acc = float(((score > 0).astype(int) == truth).mean())
    auc = float(roc_auc_score(truth, score)) if 0 < truth.sum() < len(truth) else float("nan")
    return acc, auc


# ----------------------------------------------------------------------------- contexts

class PairSource:
    """Annotated cross-camera pairs from one domain's train split (context pool)."""

    unit = "image"  # set from --selection_unit in main()

    def __init__(self, name, split_id=0):
        kwargs = {"split_id": split_id} if name in NUM_SPLITS else {}
        ds = _get_dataset_cls(name)(root=DOMAIN_CONFIG["data_root"], verbose=False, **kwargs)
        self.name = name
        self.sampler = ContextSampler(ds.train, has_cameras=name not in NO_CAMERA_DOMAINS)
        self.pool = self.sampler.identity_pool  # also used by pseudo_images (unlabeled image list)

    def budget(self):
        return len(self.pool) if self.unit == "identity" else len(self.sampler.image_pool)

    def images(self, k, rng, device):
        pairs, _ = self.sampler.draw(self.unit, "random", k, rng)
        return load_images([p for pair in pairs for p in pair], device)


def pseudo_images(pool, k, rng, seed, device):
    """Label-free context: k random pool images, each paired with an augmented copy."""
    paths = [pool.images[i][0] for i in rng.choice(len(pool.images), size=k, replace=False)]
    _seed_all(seed)
    out = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        out.extend([TRAIN_TRANSFORM(img), TRAIN_TRANSFORM(img)])
    return torch.stack(out).to(device)


# ----------------------------------------------------------------------------- main

def main():
    parser = transformers.HfArgumentParser((ReIDTrainingArguments, DiagArguments))
    args, dargs = parser.parse_args_into_dataclasses()
    device = "cuda"
    model = build_model(args, device, dargs.checkpoint)
    PairSource.unit = args.selection_unit
    print("selection unit:", args.selection_unit)
    state = torch.load(os.path.join(dargs.checkpoint, "pytorch_model.bin"), map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("checkpoint:", dargs.checkpoint, "missing:", len(missing), "unexpected:", len(unexpected))

    domains = dargs.domains.split(",")
    ks = [int(k) for k in dargs.ks.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    # ---- consistency check: rebuilt prompts == models.Model.forward prompts
    src = PairSource(domains[0])
    imgs = src.images(16, np.random.RandomState(123), device)
    mine, _, _ = build_prompts(model, imgs, "true", seed=5)
    _seed_all(5)
    with torch.no_grad():
        ref = model(image_crops=imgs.reshape(16, 2, *imgs.shape[1:]), labels=torch.arange(16, device=device))["prompts"][:1]
    diff = (mine.float() - ref.float()).abs().max().item()
    print("consistency check: max |rebuilt - original| = {:.3e}".format(diff))
    assert diff < 1e-3, "rebuilt prompt generation does not match models.Model.forward"

    others = {d: PairSource(d) for d in domains}           # split-0 pools for cross-domain contexts
    source = PairSource(dargs.source_domain)
    zero_prompts = torch.zeros(1, model.num_layers, args.num_vpt_tokens, model.hidden_size, device=device)

    rows, bank = [], defaultdict(list)                     # bank: prompts for similarity (split 0, k=max)
    for name in domains:
        n_splits = min(args.eval_splits, NUM_SPLITS.get(name, 1))
        for split_id in range(n_splits):
            kwargs = {"split_id": split_id} if name in NUM_SPLITS else {}
            ds = _get_dataset_cls(name)(root=DOMAIN_CONFIG["data_root"], verbose=False, **kwargs)
            cache = SplitCache(ds, device, name)
            own = PairSource(name, split_id)
            base = dargs.base_seed + 100000 * split_id

            def record(k, cond, ctx, i, seed, prompts, score=None, truth=None):
                r1, mAP = evaluate(model, cache, prompts, seed)
                acc, auc = llm_metrics(score, truth) if score is not None else (float("nan"), float("nan"))
                rows.append(dict(domain=name, split=split_id, k=k, condition=cond, context_from=ctx, idx=i,
                                 seed=seed, rank1=r1, mAP=mAP, llm_acc=acc, llm_auc=auc))
                if split_id == 0 and k == max(ks) and cond in ("true", "shuffle", "pseudo", "cross", "source"):
                    bank[(name, cond, ctx)].append(prompts.float().flatten().cpu())

            record(0, "zero", "-", 0, base, zero_prompts)
            for k in ks:
                for i in range(dargs.n_contexts):
                    seed = base + 1000 * k + i
                    imgs = own.images(k, np.random.RandomState(seed), device)
                    for mode in ANSWER_MODES:
                        p, sc, tr = build_prompts(model, imgs, mode, seed)
                        record(k, mode, name, i, seed, p, sc, tr)
                    for other in domains:
                        if other == name or k > others[other].budget():
                            continue
                        p, sc, tr = build_prompts(model, others[other].images(k, np.random.RandomState(seed), device), "true", seed)
                        record(k, "cross", other, i, seed, p, sc, tr)
                    p, sc, tr = build_prompts(model, source.images(k, np.random.RandomState(seed), device), "true", seed)
                    record(k, "source", source.name, i, seed, p, sc, tr)
                    p, sc, tr = build_prompts(model, pseudo_images(own.pool, k, np.random.RandomState(seed), seed, device), "true", seed)
                    record(k, "pseudo", name, i, seed, p, sc, tr)
            print("[{:6.0f}s] {} split {} done".format(time.time() - t0, name, split_id), flush=True)
            del cache
            torch.cuda.empty_cache()

    out_csv = os.path.join(args.output_dir, "diagnostics_runs.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- summary: per domain / k / condition (mean over splits and contexts)
    conds = ["true", "shuffle", "flip", "all_yes", "all_no", "cross", "source", "pseudo"]
    print("\nmAP / R1 (mean over splits x contexts), delta vs 'true', LLM acc / AUC on context questions")
    print("{:7s} {:>3s} {:8s} {:>7s} {:>7s} {:>7s} {:>6s} {:>6s}".format("domain", "k", "cond", "mAP", "d_mAP", "R1", "acc", "auc"))
    summary = []
    for name in domains:
        zero = np.mean([r["mAP"] for r in rows if r["domain"] == name and r["condition"] == "zero"])
        print("{:7s} {:>3s} {:8s} {:7.2f}".format(name, "-", "zero", zero))
        for k in ks:
            ref = np.mean([r["mAP"] for r in rows if r["domain"] == name and r["k"] == k and r["condition"] == "true"])
            for c in conds:
                sub = [r for r in rows if r["domain"] == name and r["k"] == k and r["condition"] == c]
                if not sub:
                    continue
                m = lambda key: float(np.nanmean([r[key] for r in sub]))
                s = dict(domain=name, k=k, condition=c, mAP=m("mAP"), d_mAP=m("mAP") - ref, rank1=m("rank1"),
                         llm_acc=m("llm_acc"), llm_auc=m("llm_auc"), zero_mAP=zero, n=len(sub))
                summary.append(s)
                print("{:7s} {:>3d} {:8s} {:7.2f} {:+7.2f} {:7.2f} {:6.3f} {:6.3f}".format(
                    name, k, c, s["mAP"], s["d_mAP"], s["rank1"], s["llm_acc"], s["llm_auc"]))
    with open(os.path.join(args.output_dir, "diagnostics_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # ---- prompt similarity (split 0, k = max(ks))
    def mean_cos(a, b, same):
        A = torch.nn.functional.normalize(torch.stack(a), dim=1)
        B = torch.nn.functional.normalize(torch.stack(b), dim=1)
        c = A @ B.T
        if same:
            c = c[~torch.eye(len(a), dtype=torch.bool)]
        return c.mean().item()

    print("\nprompt cosine similarity (split 0, k={})".format(max(ks)))
    true_keys = {n: (n, "true", n) for n in domains}
    for n in domains:
        print("  {:6s} true vs true (same domain, different contexts): {:.4f}".format(n, mean_cos(bank[true_keys[n]], bank[true_keys[n]], True)))
    for i, a in enumerate(domains):
        for b in domains[i + 1:]:
            print("  true[{}] vs true[{}] (across domains): {:.4f}".format(a, b, mean_cos(bank[true_keys[a]], bank[true_keys[b]], False)))
    for n in domains:
        for cond, ctx in [("shuffle", n), ("pseudo", n), ("source", dargs.source_domain)]:
            if bank.get((n, cond, ctx)):
                print("  {:6s} true vs {:7s}: {:.4f}".format(n, cond, mean_cos(bank[true_keys[n]], bank[(n, cond, ctx)], False)))
    print("\nwrote", out_csv, "({:.0f}s)".format(time.time() - t0))


if __name__ == "__main__":
    main()
