"""How much are k annotated identities worth, and how much does *which* k matter?

Context generator that is guaranteed to use the annotation (no LLM): a frozen ReID
feature extractor + a Mahalanobis metric estimated from the context (KISSME form).

  raw        cosine on frozen features
  center     subtract the mean of the unlabeled candidate pool                  (no labels)
  unlab      KISSME with the within-identity covariance replaced by isotropic   (no labels)
             -> M = (lam I)^-1 - (2 S_T + lam I)^-1, S_T = pool covariance: pure unlabeled whitening
  kiss_k     KISSME from k annotated identities (one cross-camera pair each):
             M = (S_W + lam I)^-1 - (2 S_T + lam I)^-1, S_W from the k positive-pair differences,
             S_T from all unlabeled pool images                                   (k labels)
  kiss_all   same, S_W from every image pair of every pool identity           (oracle labels)

For kiss_k, n_draws random k-identity selections per split -> mean / std / min / oracle@N
(best draw, chosen on test = optimistic upper bound of any selection method).
lam = gamma * tr(S_T) / d; reported for each gamma in --gammas (fixed a priori, not tuned on test).

Features are extracted once per split (cached as .npz), everything else is linear algebra.

Example:
  python scripts/label_value.py --output_dir experiments/label_value \
      --checkpoint experiments/baseline_plain/val_cuhk03/checkpoint-1000 --model_type plain \
      --domains viper,grid,ilids --eval_splits 10 --ks 2,4,8,16,32 --n_draws 30 \
      --gammas 0.1,1.0 --fp16 False --report_to none
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
from torchreid.metrics import evaluate_rank

from adapters.args_reid import ReIDTrainingArguments
from adapters.config_reid import DOMAIN_CONFIG, NUM_SPLITS

from adapters.trainer_reid import _get_dataset_cls
from scripts.context_sensitivity import SplitCache, features, budget_ok


@dataclass
class LabelValueArguments:
    checkpoint: str = field(default="")
    domains: str = field(default="viper,grid,ilids")
    ks: str = field(default="2,4,8,16,32")
    n_draws: int = field(default=30)
    gammas: str = field(default="0.1,1.0")
    base_seed: int = field(default=0)


def build(args, device, checkpoint):
    """Feature extractor with the checkpoint's architecture. plain / vpt only: VICP needs a context."""
    from adapters.reid_model import apply_checkpoint_structure
    apply_checkpoint_structure(args, checkpoint)
    if args.model_type == "plain":
        from adapters.baseline_model import PlainReIDModel
        model = PlainReIDModel(args)
    elif args.model_type == "vpt":
        from adapters.baseline_model import VPTReIDModel
        model = VPTReIDModel(args)
    else:
        raise ValueError("label_value.py extracts context-free features: use a plain or vpt checkpoint")
    return model.to(device=device, dtype=torch.float32).eval()


# ----------------------------------------------------------------------------- metric

def kissme_transform(S_W, S_T, gamma):
    """L such that ||(x - y) L||^2 = (x - y)^T M (x - y), M = (S_W + lam I)^-1 - (2 S_T + lam I)^-1,
    negative eigenvalues clipped (standard KISSME PSD projection). All torch float64."""
    d = S_T.shape[0]
    lam = gamma * torch.trace(S_T) / d
    I = torch.eye(d, dtype=S_T.dtype, device=S_T.device)
    M = torch.linalg.inv(S_W + lam * I) - torch.linalg.inv(2 * S_T + lam * I)
    M = (M + M.T) / 2
    e, U = torch.linalg.eigh(M)
    return U * e.clamp(min=0).sqrt()


def pair_cov(diffs):
    """Covariance of pair differences (zero-mean by symmetry): (1/n) sum d d^T."""
    return diffs.T @ diffs / len(diffs)


def rank(qf, gf, cache):
    """Squared Euclidean distance on transformed features -> R1 / mAP."""
    distmat = torch.cdist(qf, gf).pow(2).cpu().numpy()
    cmc, mAP = evaluate_rank(distmat, cache.q_pids, cache.g_pids, cache.q_cams, cache.g_cams, max_rank=10)
    return float(cmc[0]) * 100, float(mAP) * 100


def main():
    parser = transformers.HfArgumentParser((ReIDTrainingArguments, LabelValueArguments))
    args, largs = parser.parse_args_into_dataclasses()
    device = "cuda"
    model = build(args, device, largs.checkpoint)
    state = torch.load(os.path.join(largs.checkpoint, "pytorch_model.bin"), map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("checkpoint:", largs.checkpoint, "model:", args.model_type, "missing:", len(missing), "unexpected:", len(unexpected))
    assert not missing, missing[:5]

    ks = [int(k) for k in largs.ks.split(",")]
    gammas = [float(g) for g in largs.gammas.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)
    rows, t0 = [], time.time()

    for name in largs.domains.split(","):
        n_splits = min(args.eval_splits, NUM_SPLITS.get(name, 1))
        for split_id in range(n_splits):
            kwargs = {"split_id": split_id} if name in NUM_SPLITS else {}
            try:
                ds = _get_dataset_cls(name)(root=DOMAIN_CONFIG["data_root"], verbose=False, **kwargs)
            except Exception as e:
                print("Skipping {} split {}: {}".format(name, split_id, e))
                break
            assert not ({p for p, *_ in ds.train} & {p for p, *_ in ds.query + ds.gallery}), "leak"
            cache = SplitCache(ds, device, name)
            # prompts=None: plain model ignores it; for VICP this would be the zero-prompt model
            pf = features(model, cache.pool_imgs, None).double()
            qf = features(model, cache.q_imgs, None).double()
            gf = features(model, cache.g_imgs, None).double()
            np.savez_compressed(os.path.join(args.output_dir, "feats_{}_{}.npz".format(name, split_id)),
                                pool=pf.cpu().numpy(), query=qf.cpu().numpy(), gallery=gf.cpu().numpy())
            add = lambda **r: rows.append(dict(domain=name, split=split_id, **r))

            add(condition="raw", k=0, gamma=-1, draw=-1, rank1=rank(qf, gf, cache)[0], mAP=rank(qf, gf, cache)[1])
            mu = pf.mean(0, keepdim=True)
            r1, mAP = rank(qf - mu, gf - mu, cache)
            add(condition="center", k=0, gamma=-1, draw=-1, rank1=r1, mAP=mAP)
            S_T = pair_cov(pf - mu)
            zero_W = torch.zeros_like(S_T)

            # oracle labels: all within-identity pairs in the pool
            diffs = []
            for pid in cache.pool.eligible_pids:
                ii = [cache.pool_idx[p] for p, _ in cache.pool.pid2items[pid]]
                for a in range(len(ii)):
                    for b in range(a + 1, len(ii)):
                        diffs.append(pf[ii[a]] - pf[ii[b]])
            S_W_all = pair_cov(torch.stack(diffs))

            for g in gammas:
                L = kissme_transform(zero_W, S_T, g)
                r1, mAP = rank((qf - mu) @ L, (gf - mu) @ L, cache)
                add(condition="unlab", k=0, gamma=g, draw=-1, rank1=r1, mAP=mAP)
                L = kissme_transform(S_W_all, S_T, g)
                r1, mAP = rank((qf - mu) @ L, (gf - mu) @ L, cache)
                add(condition="kiss_all", k=len(cache.pool), gamma=g, draw=-1, rank1=r1, mAP=mAP)

            for k in ks:
                if not budget_ok(cache, args.selection_unit, k):
                    continue
                for draw in range(largs.n_draws):
                    rng = np.random.RandomState(largs.base_seed + 1000 * split_id + 100 * draw + k)
                    pairs, info = cache.sampler.draw(args.selection_unit, "random", k, rng)
                    pids = info["selected"].split()
                    if not pairs:
                        continue
                    d = torch.stack([pf[cache.pool_idx[a]] - pf[cache.pool_idx[b]] for a, b in pairs])
                    S_W = pair_cov(d)
                    for g in gammas:
                        L = kissme_transform(S_W, S_T, g)
                        r1, mAP = rank((qf - mu) @ L, (gf - mu) @ L, cache)
                        add(condition="kiss_k", k=k, gamma=g, draw=draw, rank1=r1, mAP=mAP,
                            pids=" ".join(map(str, pids)))
            print("[{:5.0f}s] {} split {} done".format(time.time() - t0, name, split_id), flush=True)

    keys = ["domain", "split", "condition", "k", "gamma", "draw", "rank1", "mAP", "pids"]
    with open(os.path.join(args.output_dir, "label_value.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # summary (mAP, mean over splits). kiss_k: per split mean / std / min / max(=oracle@N) over draws
    print("\n{:6s} {:9s} {:>5s} {:>4s} {:>7s} {:>6s} {:>7s} {:>7s} {:>9s}".format(
        "domain", "cond", "gamma", "k", "mean", "std", "min", "oracle", "orc-mean"))
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[(r["domain"], r["condition"], r["gamma"], r["k"])][r["split"]].append(r["mAP"])
    for (d, c, g, k), per_split in sorted(by.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3])):
        v = list(per_split.values())
        mean = np.mean([np.mean(x) for x in v]); std = np.mean([np.std(x) for x in v])
        mn = np.mean([np.min(x) for x in v]); mx = np.mean([np.max(x) for x in v])
        print("{:6s} {:9s} {:5.2f} {:4d} {:7.2f} {:6.2f} {:7.2f} {:7.2f} {:9.2f}".format(d, c, g, k, mean, std, mn, mx, mx - mean))
    print("total time: {:.0f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
