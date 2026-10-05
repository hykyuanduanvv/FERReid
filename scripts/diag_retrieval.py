"""Diagnostics before running the active module: can the base model propose useful pairs?

Per target domain, with the base model's features of the unlabeled pool (labels only used to score):
  hit@k          share of pool images (whose person appears in another camera) with a same-person image
                 among their k nearest images from other cameras (k = 1, 5, 10, 20)
  mutual_prec    share of mutual cross-camera nearest neighbours that are one person, and their coverage
  cand_pos_rate  share of proposed pairs (cross-camera k-NN, --candidate_k) that are one person
  calibration    positive rate of the proposed pairs per similarity decile (is there an uncertain band?)
  auc / acc_tau  ReID similarity as a same-person verifier on the proposed pairs: ROC-AUC, and accuracy at
                 the label-free threshold tau (99th percentile of random-pair similarity)
  verify_pairs.csv  a balanced sample of proposed pairs (paths, similarity, label) for scripts/diag_vlm.py

  python scripts/diag_retrieval.py --output_dir experiments/diag_cuhk03 --checkpoint experiments/base_vpt_to_cuhk03/checkpoint-12000 \
      --domains cuhk03 --fp16 True --report_to none
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import csv
import json
from dataclasses import dataclass, field

import numpy as np
import torch
import transformers

from adapters.args_reid import ReIDTrainingArguments
from adapters.active.candidates import candidate_pairs, random_pair_quantile
from adapters.active.image_store import ImageStore, features
from adapters.baseline_model import load_checkpoint_model
from adapters.config_reid import DOMAIN_CONFIG, NO_CAMERA_DOMAINS
from adapters.trainer_reid import _get_dataset_cls


@dataclass
class DiagArguments:
    checkpoint: str = field(default="")
    domains: str = field(default="cuhk03")
    candidate_k: int = field(default=10)
    n_verify_pairs: int = field(default=200)   # per class and domain, written to verify_pairs.csv
    cache_max: int = field(default=6000)


def auc(scores, labels):
    """ROC-AUC via the rank statistic."""
    from scipy.stats import rankdata
    labels = np.asarray(labels, bool)
    if labels.all() or not labels.any():
        return float("nan")
    r = rankdata(scores)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    return float((r[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def hit_at_k(feats, pids, cams, has_cameras, ks=(1, 5, 10, 20), chunk=2048):
    pids_t = torch.as_tensor(pids, device=feats.device)
    cams_t = torch.as_tensor(cams, device=feats.device)
    n, kmax = feats.size(0), min(max(ks), feats.size(0) - 1)
    hits = {k: 0 for k in ks}
    valid = 0
    for s in range(0, n, chunk):
        rows = torch.arange(s, min(s + chunk, n), device=feats.device)
        S = feats[rows] @ feats.T
        S[torch.arange(len(rows), device=feats.device), rows] = -2
        other = (cams_t[rows][:, None] != cams_t[None, :]) if has_cameras else torch.ones_like(S, dtype=torch.bool)
        other[torch.arange(len(rows), device=feats.device), rows] = False
        S = S.masked_fill(~other, -2)
        same = (pids_t[rows][:, None] == pids_t[None, :]) & other
        has = same.any(1)
        top = S.topk(kmax, dim=1).indices
        hit = torch.gather(same, 1, top)
        for k in ks:
            hits[k] += int((hit[:, :min(k, kmax)].any(1) & has).sum())
        valid += int(has.sum())
    return {"hit@{}".format(k): hits[k] / max(valid, 1) for k in ks}, valid


def main():
    parser = transformers.HfArgumentParser((ReIDTrainingArguments, DiagArguments))
    args, a = parser.parse_args_into_dataclasses()
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model = load_checkpoint_model(args, device, a.checkpoint)
    prompts = model.prompt.detach().float() if getattr(model, "prompt", None) is not None else None
    os.makedirs(args.output_dir, exist_ok=True)
    results, verify = [], []
    for name in a.domains.split(","):
        has_cameras = name not in NO_CAMERA_DOMAINS
        for split_id in range(1):  # one fixed split per dataset
            ds = _get_dataset_cls(name)(root=DOMAIN_CONFIG["data_root"], verbose=False)
            paths = [x[0] for x in ds.train]
            pids = np.array([x[1] for x in ds.train]); cams = np.array([x[2] for x in ds.train])
            feats = features(model, ImageStore(paths, device, cache_max=a.cache_max,
                                               num_workers=args.eval_num_workers), prompts)
            res = {"domain": name, "split": split_id, "pool_images": len(paths), "pool_ids": len(set(pids))}
            hits, n_valid = hit_at_k(feats, pids, cams, has_cameras)
            res.update(hits, pairable_images=n_valid)
            cand = candidate_pairs(feats, cams, has_cameras, k=a.candidate_k)
            lab = pids[cand["i"]] == pids[cand["j"]]
            res["n_candidates"] = int(len(lab))
            res["cand_pos_rate"] = float(lab.mean())
            m = cand["mutual"] & (cand["rank"] == 0)
            res["mutual_prec"] = float(lab[m].mean()) if m.any() else float("nan")
            res["mutual_coverage"] = float(2 * m.sum() / len(paths))
            tau = random_pair_quantile(feats, 0.99)
            res["tau"] = tau
            res["auc"] = auc(cand["sim"], lab)
            res["acc_tau"] = float(((cand["sim"] > tau) == lab).mean())
            res["share_above_tau"] = float((cand["sim"] > tau).mean())
            edges = np.quantile(cand["sim"], np.linspace(0, 1, 11))
            dec = np.clip(np.searchsorted(edges, cand["sim"], side="right") - 1, 0, 9)
            res["calibration"] = [float(lab[dec == d].mean()) if (dec == d).any() else float("nan") for d in range(10)]
            results.append(res)
            print("{}: hit@1/5/10/20 = {} | candidates {} pos {:.2f} | mutual prec {:.2f} cov {:.2f} | "
                  "AUC {:.3f} acc@tau {:.3f}".format(
                      name, " / ".join("{:.2f}".format(hits[k]) for k in hits), res["n_candidates"],
                      res["cand_pos_rate"], res["mutual_prec"], res["mutual_coverage"], res["auc"], res["acc_tau"]))
            print("   positive rate per similarity decile:", " ".join("{:.2f}".format(c) for c in res["calibration"]))
            rng = np.random.RandomState(split_id)
            for want in (True, False):
                idx = np.flatnonzero(lab == want)
                for p in rng.choice(idx, size=min(a.n_verify_pairs, len(idx)), replace=False):
                    verify.append({"domain": name, "split": split_id, "path_a": paths[cand["i"][p]],
                                   "path_b": paths[cand["j"][p]], "sim": float(cand["sim"][p]),
                                   "tau": tau, "same": int(want)})
    with open(os.path.join(args.output_dir, "diag_retrieval.json"), "w") as f:
        json.dump(results, f, indent=1)
    if verify:
        with open(os.path.join(args.output_dir, "verify_pairs.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(verify[0]))
            w.writeheader()
            w.writerows(verify)
    print("wrote", os.path.join(args.output_dir, "diag_retrieval.json"), "and verify_pairs.csv")


if __name__ == "__main__":
    main()
