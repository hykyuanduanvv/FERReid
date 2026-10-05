"""Where are the identities the clustering misses? (person ids used only to score; nothing is trained)

For each feature setting -- the base model's features, the features under a saved domain prompt (e.g. after 5
rounds of unsupervised training), each with and without camera normalisation (pseudo.camera_normalize) -- the
pool is clustered as in the active loop (pseudo.PoolGraph) and:
  clustering      pairwise precision / recall / F1, NMI, identities split over several clusters
  missed pairs    same-person pairs in different clusters (outliers = own cluster): share cross-camera, share
                  inside the cross-camera k-NN candidates, their cosine as a percentile of all cross-cluster
                  candidate pairs
  candidates      cross-cluster k-NN candidate pairs: positive rate overall, per similarity decile, and of the
                  top 50 / 250 by similarity (what similarity-ranked merge questions get)
  shortlist       for every pseudo cluster whose person also lives in another cluster: is one of those clusters
                  among its K nearest clusters (centroid cosine)? K = 1, 5, 10; all clusters / clusters with no
                  camera in common ("complementary")
  retrieval       cross-camera hit@1 / hit@10 of the pool
Writes diag_missed.json.

  python scripts/diag_missed.py --output_dir experiments/diag_missed --checkpoint experiments/base_md_cuhk03/checkpoint-12000 \\
      --domains cuhk03 --prompts experiments/pilot_s0a/prompts/cuhk03_none_seed0.pt --fp16 True --report_to none
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import json
from dataclasses import dataclass, field

import numpy as np
import torch
import transformers

from adapters.args_reid import ReIDTrainingArguments
from adapters.active.candidates import candidate_pairs
from adapters.active.image_store import features
from adapters.active.loop import default_prompt
from adapters.active.oracle import PairOracle
from adapters.active.pseudo import PoolGraph, camera_normalize
from adapters.baseline_model import load_checkpoint_model
from scripts.eval_active import load_split


@dataclass
class DiagArgs:
    checkpoint: str = field(default="")
    domains: str = field(default="cuhk03")
    prompts: str = field(default="")          # comma-separated saved prompts (eval_active prompts/*.pt)
    candidate_k: int = field(default=10)
    pseudo_k1: int = field(default=30)
    pseudo_k2: int = field(default=6)
    pseudo_eps: float = field(default=0.6)
    pseudo_min_samples: int = field(default=4)
    cache_max: int = field(default=40000)


def analyse(feats, pids, cams, has_cameras, a):
    X = feats.float()
    g = PoolGraph(X, a.pseudo_k1, a.pseudo_k2, a.pseudo_eps, a.pseudo_min_samples)
    labels = g.cluster()
    out = {"clustering": PairOracle(pids, cams, has_cameras).pseudo_report(labels)}
    gid = labels.copy()
    o = np.flatnonzero(gid < 0)
    gid[o] = gid.max() + 1 + np.arange(len(o))
    Xn = X.cpu().numpy()
    # missed same-person pairs
    mi, mj = [], []
    for p in np.unique(pids):
        idx = np.flatnonzero(pids == p)
        if len(idx) < 2:
            continue
        a_, b_ = np.triu_indices(len(idx), 1)
        i, j = idx[a_], idx[b_]
        keep = gid[i] != gid[j]
        mi.append(i[keep]); mj.append(j[keep])
    mi, mj = np.concatenate(mi), np.concatenate(mj)
    msim = (Xn[mi] * Xn[mj]).sum(1)
    cand = candidate_pairs(X, cams, has_cameras, k=a.candidate_k)
    ci, cj, cs = cand["i"], cand["j"], cand["sim"]
    cross = gid[ci] != gid[cj]
    ci, cj, cs = ci[cross], cj[cross], cs[cross]
    cpos = pids[ci] == pids[cj]
    cand_keys = set((ci * len(pids) + cj).tolist())
    in_cand = np.array([(min(x, y) * len(pids) + max(x, y)) in cand_keys for x, y in zip(mi, mj)])
    out["missed"] = {
        "n_missed_pairs": int(len(mi)),
        "share_cross_camera": float((cams[mi] != cams[mj]).mean()) if len(mi) else float("nan"),
        "share_in_knn_candidates": float(in_cand.mean()) if len(mi) else float("nan"),
        "cos_percentile_among_cross_cluster_candidates": {
            q: float(np.mean(cs <= np.quantile(msim, q / 100))) * 100 for q in (25, 50, 75)} if len(mi) and len(cs) else {},
        "cos_quantiles_missed": [float(x) for x in np.quantile(msim, [0.1, 0.25, 0.5, 0.75, 0.9])] if len(mi) else [],
    }
    order = np.argsort(-cs)
    dec = np.array_split(order, 10)  # decile 0 = most similar
    out["candidates"] = {
        "n_cross_cluster": int(len(cs)), "pos_rate": float(cpos.mean()) if len(cs) else float("nan"),
        "pos_rate_by_decile_high_to_low": [float(cpos[d].mean()) if len(d) else float("nan") for d in dec],
        "pos_rate_top50": float(cpos[order[:50]].mean()) if len(cs) else float("nan"),
        "pos_rate_top250": float(cpos[order[:250]].mean()) if len(cs) else float("nan"),
        "n_pos_cross_cluster_candidates": int(cpos.sum()),
    }
    # shortlist recall at the cluster level
    C = int(gid.max()) + 1
    cent = np.zeros((C, Xn.shape[1]), np.float64)
    np.add.at(cent, gid, Xn)
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12
    maj = np.zeros(C, np.int64)
    for c in range(C):
        v, n = np.unique(pids[gid == c], return_counts=True)
        maj[c] = v[np.argmax(n)]
    camset = [set(cams[gid == c].tolist()) for c in range(C)]
    S = cent @ cent.T
    np.fill_diagonal(S, -np.inf)
    rec = {"all": {1: [], 5: [], 10: []}, "complementary": {1: [], 5: [], 10: []}}
    for c in range(C):
        partners = np.flatnonzero(maj == maj[c])
        partners = partners[partners != c]
        if not len(partners):
            continue
        for mode in rec:
            s = S[c].copy()
            if mode == "complementary":
                s[[k for k in range(C) if camset[k] & camset[c]]] = -np.inf
                if not np.isfinite(s[partners]).any():
                    continue
            top = np.argsort(-s)[:10]
            for K in (1, 5, 10):
                rec[mode][K].append(bool(np.isin(top[:K], partners).any()))
    out["shortlist_recall"] = {m: {"n": len(v[1]), **{"@{}".format(K): float(np.mean(v[K])) if v[K] else float("nan")
                                                     for K in v}} for m, v in rec.items()}
    # cross-camera retrieval in the pool
    Sx = X @ X.T
    Sx.fill_diagonal_(-2)
    ct = torch.as_tensor(cams, device=X.device)
    if has_cameras:
        Sx.masked_fill_(ct[:, None] == ct[None, :], -2)
    pt = torch.as_tensor(pids, device=X.device)
    same = (pt[:, None] == pt[None, :]) & (Sx > -2)
    has = same.any(1)
    top = Sx.topk(10, dim=1).indices
    hit = torch.gather(same, 1, top)
    out["retrieval"] = {"hit@1": float(hit[:, 0][has].float().mean()), "hit@10": float(hit.any(1)[has].float().mean())}
    return out


def main():
    parser = transformers.HfArgumentParser((ReIDTrainingArguments, DiagArgs))
    args, a = parser.parse_args_into_dataclasses()
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    torch.set_num_threads(int(os.environ.get("FERREID_CPU_THREADS", "8")))
    model = load_checkpoint_model(args, device, a.checkpoint)
    os.makedirs(args.output_dir, exist_ok=True)
    result = {}
    for name in a.domains.split(","):
        split = load_split(name, device, a.cache_max, args.eval_num_workers)
        prompts = [("base", default_prompt(model))]
        for p in [x for x in a.prompts.split(",") if x]:
            prompts.append((os.path.basename(p).replace(".pt", ""), torch.load(p, map_location=device)["prompt"].to(device)))
        for pname, prompt in prompts:
            f = features(model, split.pool, prompt)
            for cn in (False, True):
                if cn and not split.has_cameras:
                    continue
                ff = camera_normalize(f, split.pool_cams) if cn else f
                key = "{}|{}|{}".format(name, pname, "camnorm" if cn else "raw")
                result[key] = analyse(ff, split.pool_pids, split.pool_cams, split.has_cameras, a)
                r = result[key]
                print("{:<40} F1 {:.3f} split_ids {:>4} | missed {:>6} (in kNN {:.2f}, median pct {:.0f}) | cand pos {:.3f} "
                      "top50 {:.2f} top250 {:.2f} | shortlist all @1/5/10 {:.2f}/{:.2f}/{:.2f} compl {:.2f}/{:.2f}/{:.2f} | "
                      "hit@1 {:.3f} hit@10 {:.3f}".format(
                          key, r["clustering"]["pw_f"], r["clustering"]["pseudo_split_ids"], r["missed"]["n_missed_pairs"],
                          r["missed"]["share_in_knn_candidates"],
                          r["missed"]["cos_percentile_among_cross_cluster_candidates"].get(50, float("nan")),
                          r["candidates"]["pos_rate"], r["candidates"]["pos_rate_top50"], r["candidates"]["pos_rate_top250"],
                          *[r["shortlist_recall"]["all"]["@{}".format(K)] for K in (1, 5, 10)],
                          *[r["shortlist_recall"]["complementary"]["@{}".format(K)] for K in (1, 5, 10)],
                          r["retrieval"]["hit@1"], r["retrieval"]["hit@10"]), flush=True)
                json.dump(result, open(os.path.join(args.output_dir, "diag_missed.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
