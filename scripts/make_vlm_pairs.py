"""Question sets for an MLLM annotator (scripts/diag_vlm.py): the pairs the active loop would actually ask, and the
hard cases of the clustering, with their true label. Person ids only label the pairs; nothing is trained.

Features of the base model as deployed (round 1, where the front-loaded budget is spent), pool clustered as in
the active loop. Per domain:
  q_rule   the shortlist questions of the current rule (K=1, fewest cameras first, then the most similar
           complementary cluster): medoid of cluster A vs medoid of its candidate, first `n_questions`
  hardpos  same person, different clusters, different cameras (what the clustering misses), random sample
  hardneg  different persons in different clusters, the most similar cross-camera k-NN pairs
  q_all    (--all_k K) every cluster vs its K nearest complementary clusters, medoid vs medoid: the questions a
           machine annotator answers without a budget (scripts/eval_active.py --strategies file)
Writes <output_dir>/verify_pairs.csv: domain, kind, path_a, path_b, same, sim, tau (tau: the similarity threshold
with the best accuracy on that domain's pairs -- an oracle threshold, favouring the ReID baseline).

  python scripts/make_vlm_pairs.py --output_dir experiments/vlm_pairs --checkpoint ... --domains msmt17 \\
      --fp16 True --report_to none
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
from adapters.active.candidates import candidate_pairs
from adapters.active.image_store import features
from adapters.active.loop import default_prompt
from adapters.active.pseudo import PoolGraph
from adapters.baseline_model import load_checkpoint_model
from scripts.eval_active import load_split


@dataclass
class PairArgs:
    checkpoint: str = field(default="")
    domains: str = field(default="msmt17")
    n_questions: int = field(default=250)
    all_k: int = field(default=0)        # > 0: also "q_all", every cluster vs its all_k nearest complementary clusters
    n_hard: int = field(default=250)
    pseudo_eps: float = field(default=0.6)
    cache_max: int = field(default=40000)


def best_tau(sim, same):
    order = np.sort(sim)
    acc = [((sim > t) == same).mean() for t in order]
    return float(order[int(np.argmax(acc))])


def main():
    parser = transformers.HfArgumentParser((ReIDTrainingArguments, PairArgs))
    args, a = parser.parse_args_into_dataclasses()
    device = "cuda"
    model = load_checkpoint_model(args, device, a.checkpoint)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for name in a.domains.split(","):
        split = load_split(name, device, a.cache_max, args.eval_num_workers)
        pids, cams, paths = np.asarray(split.pool_pids), np.asarray(split.pool_cams), split.pool_paths
        f = features(model, split.pool, default_prompt(model))
        X = f.cpu().numpy()
        labels = PoolGraph(f, 30, 6, a.pseudo_eps, 4).cluster()
        C = int(labels.max()) + 1
        members = [np.flatnonzero(labels == c) for c in range(C)]
        cent = np.stack([X[m].mean(0) for m in members])
        cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12
        medoid = np.array([m[np.argmax(X[m] @ cent[c])] for c, m in enumerate(members)])
        cam_idx = np.unique(cams, return_inverse=True)[1].reshape(-1)
        mask = np.zeros((C, cam_idx.max() + 1), bool)
        for c, m in enumerate(members):
            mask[c, cam_idx[m]] = True
        S = cent @ cent.T
        np.fill_diagonal(S, -np.inf)
        S = np.where((mask.astype(np.int32) @ mask.T.astype(np.int32)) > 0, -np.inf, S)
        top1 = S.max(1)
        order = np.lexsort((-top1, mask.sum(1)))
        pairs = []
        for c in order:
            if len(pairs) >= a.n_questions:
                break
            if np.isfinite(top1[c]):
                pairs.append(("q_rule", int(medoid[c]), int(medoid[int(np.argmax(S[c]))])))
        if a.all_k > 0:  # machine-answered questions: no budget, every cluster (pairs deduplicated)
            done = set()
            for c in range(C):
                for d in np.argsort(-S[c])[:a.all_k]:
                    if np.isfinite(S[c, d]) and (min(c, d), max(c, d)) not in done:
                        done.add((min(c, d), max(c, d)))
                        pairs.append(("q_all", int(medoid[c]), int(medoid[d])))
        gid = labels.copy()
        o = np.flatnonzero(gid < 0)
        gid[o] = gid.max() + 1 + np.arange(len(o))
        missed = []
        for p in np.unique(pids):
            idx = np.flatnonzero(pids == p)
            if len(idx) < 2:
                continue
            i, j = np.triu_indices(len(idx), 1)
            i, j = idx[i], idx[j]
            keep = (gid[i] != gid[j]) & (cams[i] != cams[j])
            missed += list(zip(i[keep], j[keep]))
        for k in rng.choice(len(missed), size=min(a.n_hard, len(missed)), replace=False):
            pairs.append(("hardpos", int(missed[k][0]), int(missed[k][1])))
        cand = candidate_pairs(f, cams, True, k=10)
        ci, cj, cs = cand["i"], cand["j"], cand["sim"]
        neg = (gid[ci] != gid[cj]) & (pids[ci] != pids[cj])
        for t in np.flatnonzero(neg)[np.argsort(-cs[neg])][:a.n_hard]:
            pairs.append(("hardneg", int(ci[t]), int(cj[t])))
        sim = np.array([float(X[i] @ X[j]) for _, i, j in pairs])
        same = np.array([pids[i] == pids[j] for _, i, j in pairs])
        tau = best_tau(sim, same)
        for (kind, i, j), s, y in zip(pairs, sim, same):
            rows.append({"domain": name, "kind": kind, "i": i, "j": j, "path_a": paths[i], "path_b": paths[j], "same": int(y),
                         "sim": s, "tau": tau})
        for kind in ("q_rule", "q_all", "hardpos", "hardneg"):
            if not any(p[0] == kind for p in pairs):
                continue
            m = np.array([p[0] == kind for p in pairs])
            print("{} {:<8} n {:>4} pos rate {:.3f} reid acc@tau {:.3f}".format(
                name, kind, int(m.sum()), same[m].mean(), ((sim[m] > tau) == same[m]).mean()), flush=True)
        del split, f
        torch.cuda.empty_cache()
    with open(os.path.join(args.output_dir, "verify_pairs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
