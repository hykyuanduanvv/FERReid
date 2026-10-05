"""Query-by-committee for cluster repair: does the disagreement of a committee of clusterings predict which pseudo
clusters are split (their person also owns another cluster)? Person ids are used only to score; nothing is trained.

Main clustering: the pool under the main prompt (base: shared prompt + source-mean tokens; or a saved target prompt
of eval_active), PoolGraph as in the active loop. Committees, each member clusters the same pool the same way:
  src      the base model's prompt with one source domain's tokens per member (shared prompt | domain_prompts[d])
  src_mix  (saved target prompts only) the target tokens mixed half-half with one source domain's tokens
  eps      the main features, DBSCAN eps of 0.5 / 0.55 / 0.65 / 0.7 (no prompt; the control committee)
Scores per main cluster c (higher: more likely split):
  coassoc   max over other main clusters c' of the committee's co-association of c and c' (share of image pairs
            (i in c, j in c') clustered together, averaged over members); partner = the argmax c'
  frag      share of c not in its largest piece, averaged over members (the committee cuts c)
  vote      share of members that put >= 30 % of c together with >= 30 % of some other main cluster
  rule      the current shortlist rule: fewest cameras first, then the highest centroid cosine of a complementary
            cluster (no camera in common); partner = that cluster
  sim       highest centroid cosine of a complementary cluster; random
Reported: AUC of every score against "split", hit rate (share of split clusters) of the top 50 / 100 / 250 / 500,
and the K=1 partner hit rate of those top clusters (is the proposed partner the same person?) with the score's own
partner and with the complementary nearest cluster. Writes diag_committee.json.

  python scripts/diag_committee.py --output_dir experiments/diag_committee_cuhk03 \\
      --checkpoint experiments/base_md_cuhk03/checkpoint-12000 --domains cuhk03 \\
      --prompts experiments/pilot_s0a/prompts/cuhk03_none_seed0.pt --fp16 True --report_to none
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import json
from dataclasses import dataclass, field

import numpy as np
import torch
import transformers
from sklearn.metrics import roc_auc_score

from adapters.args_reid import ReIDTrainingArguments
from adapters.active.image_store import features
from adapters.active.loop import default_prompt
from adapters.active.oracle import PairOracle
from adapters.active.pseudo import PoolGraph
from adapters.baseline_model import load_checkpoint_model
from scripts.eval_active import load_split

TOPS = (50, 100, 250, 500)


@dataclass
class CommitteeArgs:
    checkpoint: str = field(default="")
    domains: str = field(default="cuhk03")
    prompts: str = field(default="")          # comma-separated saved target prompts (eval_active prompts/*.pt)
    pseudo_k1: int = field(default=30)
    pseudo_k2: int = field(default=6)
    pseudo_eps: float = field(default=0.6)
    pseudo_min_samples: int = field(default=4)
    member_eps: str = field(default="0.5,0.55,0.65,0.7")
    mix: float = field(default=0.5)
    cache_max: int = field(default=40000)


def cluster(feats, a, eps=None):
    g = PoolGraph(feats.float(), a.pseudo_k1, a.pseudo_k2, a.pseudo_eps if eps is None else eps, a.pseudo_min_samples)
    labels = g.cluster()
    del g
    torch.cuda.empty_cache()
    return labels


def share_matrix(main, labels, C):
    """(C, K) share of each main cluster's images in each member cluster (member outliers dropped)."""
    ok = (main >= 0) & (labels >= 0)
    K = int(labels.max()) + 1 if (labels >= 0).any() else 1
    M = np.zeros((C, K))
    np.add.at(M, (main[ok], labels[ok]), 1)
    size = np.bincount(main[main >= 0], minlength=C).astype(float)
    return M / size[:, None]


def committee_scores(main, members, C, comp):
    """comp: (C, C) bool, cluster pairs with no camera in common (all True without cameras)."""
    A = np.zeros((C, C))
    frag = np.zeros(C)
    vote = np.zeros(C)
    for lab in members:
        P = share_matrix(main, lab, C)
        Am = P @ P.T
        np.fill_diagonal(Am, 0)
        A += Am
        frag += 1 - P.max(1)
        big = P >= 0.3
        both = (big.astype(np.int32) @ big.T.astype(np.int32))
        np.fill_diagonal(both, 0)
        vote += both.max(1) > 0
    A /= len(members)
    Ac = np.where(comp, A, -1.0)
    return {"coassoc": (A.max(1), A.argmax(1)), "frag": (frag / len(members), A.argmax(1)),
            "vote": (vote / len(members) + 1e-3 * A.max(1), A.argmax(1)),
            "coassoc_c": (Ac.max(1), Ac.argmax(1))}


def evaluate(main, X, pids, cams, has_cameras, committees, seed):
    C = int(main.max()) + 1
    members = [np.flatnonzero(main == c) for c in range(C)]
    upid, pinv = np.unique(pids, return_inverse=True)
    maj = upid[np.array([np.bincount(pinv.reshape(-1)[m]).argmax() for m in members])]
    split = np.array([(maj == maj[c]).sum() > 1 for c in range(C)])
    split_any = np.array([(pids == maj[c]).sum() > (pids[m] == maj[c]).sum() for c, m in enumerate(members)])
    # similarity baselines
    cent = np.stack([X[m].mean(0) for m in members])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12
    cam_idx = np.unique(cams, return_inverse=True)[1].reshape(-1)
    mask = np.zeros((C, cam_idx.max() + 1), bool)
    for c, m in enumerate(members):
        mask[c, cam_idx[m]] = True
    S = cent @ cent.T
    np.fill_diagonal(S, -np.inf)
    if has_cameras:
        S = np.where((mask.astype(np.int32) @ mask.T.astype(np.int32)) > 0, -np.inf, S)
    top1 = S.max(1)
    sim_partner = S.argmax(1)
    fin = np.isfinite(top1)
    t1 = np.where(fin, top1, -9.0)
    ncam = mask.sum(1)
    rng = np.random.default_rng(seed)
    scores = {"random": (rng.random(C), sim_partner), "sim": (t1, sim_partner),
              "rule": (-ncam * 10.0 + t1, sim_partner)}
    comp = ~((mask.astype(np.int32) @ mask.T.astype(np.int32)) > 0) if has_cameras else np.ones((C, C), bool)
    np.fill_diagonal(comp, False)
    for cname, labs in committees.items():
        for sname, (s, p) in committee_scores(main, labs, C, comp).items():
            scores["{}:{}".format(cname, sname)] = (s, p)
            if sname.startswith("coassoc"):  # current rule, committee instead of similarity (ties: fewest cameras)
                scores["rule+{}:{}".format(cname, sname)] = (-ncam * 10.0 + s, p)
    out = {"n_clusters": C, "split_rate": float(split.mean()), "split_any_rate": float(split_any.mean()),
           "scores": {}}
    for name, (s, partner) in scores.items():
        order = np.argsort(-s, kind="stable")
        r = {"auc": float(roc_auc_score(split, s)) if 0 < split.sum() < C else float("nan"),
             "auc_any": float(roc_auc_score(split_any, s)) if 0 < split_any.sum() < C else float("nan")}
        for n in TOPS:
            top = order[:n]
            r["hit@{}".format(n)] = float(split[top].mean())
            r["partner_own@{}".format(n)] = float((maj[partner[top]] == maj[top]).mean())
            r["partner_sim@{}".format(n)] = float(((maj[sim_partner[top]] == maj[top]) & fin[top]).mean())
        out["scores"][name] = r
    return out


def main():
    parser = transformers.HfArgumentParser((ReIDTrainingArguments, CommitteeArgs))
    args, a = parser.parse_args_into_dataclasses()
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    torch.set_num_threads(int(os.environ.get("FERREID_CPU_THREADS", "8")))
    model = load_checkpoint_model(args, device, a.checkpoint)
    if getattr(model, "domain_prompts", None) is None:
        raise ValueError("the checkpoint has no source-domain tokens (train it with --source_domain_tokens)")
    shared = model.prompt.detach().float()
    src = model.domain_prompts.detach().float()  # (n_dom, L, m, D)
    m_tok = src.size(2)
    os.makedirs(args.output_dir, exist_ok=True)
    result = {}
    for name in a.domains.split(","):
        split = load_split(name, device, a.cache_max, args.eval_num_workers)
        pids, cams = np.asarray(split.pool_pids), np.asarray(split.pool_cams)
        oracle = PairOracle(pids, cams, split.has_cameras)
        src_feats = [features(model, split.pool, torch.cat([shared, src[d:d + 1]], dim=2)) for d in range(len(src))]
        src_labels = [cluster(f, a) for f in src_feats]
        del src_feats
        settings = [("base", default_prompt(model))]
        for p in [x for x in a.prompts.split(",") if x]:
            settings.append((os.path.basename(p).replace(".pt", ""),
                             torch.load(p, map_location=device)["prompt"].to(device).float()))
        for sname, prompt in settings:
            f = features(model, split.pool, prompt)
            main_labels = cluster(f, a)
            committees = {"src": src_labels,
                          "eps": [cluster(f, a, float(e)) for e in a.member_eps.split(",")]}
            if sname != "base" and prompt.size(2) == shared.size(2) + m_tok:
                tgt = prompt[:, :, -m_tok:]
                mixed = []
                for d in range(len(src)):
                    pm = torch.cat([prompt[:, :, :-m_tok], (1 - a.mix) * tgt + a.mix * src[d:d + 1]], dim=2)
                    mixed.append(cluster(features(model, split.pool, pm), a))
                committees["src_mix"] = mixed
                committees["src_mix+eps"] = mixed + committees["eps"]
            committees["src+eps"] = committees["src"] + committees["eps"]
            key = "{}|{}".format(name, sname)
            r = evaluate(main_labels, f.cpu().numpy(), pids, cams, split.has_cameras, committees, args.seed)
            r["main_clustering"] = oracle.pseudo_report(main_labels)
            r["member_f1"] = {c: [oracle.pseudo_report(l)["pw_f"] for l in labs] for c, labs in committees.items()}
            result[key] = r
            print("== {}  clusters {}  split {:.3f}  main F1 {:.3f}  member F1 {}".format(
                key, r["n_clusters"], r["split_rate"], r["main_clustering"]["pw_f"],
                {c: [round(x, 3) for x in v] for c, v in r["member_f1"].items()}), flush=True)
            print("{:<18} {:>6} {:>6} | {} | {} | {}".format(
                "score", "AUC", "AUCany", " ".join("hit@{:<4}".format(n) for n in TOPS),
                " ".join("own@{:<4}".format(n) for n in TOPS), " ".join("sim@{:<4}".format(n) for n in TOPS)))
            for s, v in r["scores"].items():
                print("{:<18} {:>6.3f} {:>6.3f} | {} | {} | {}".format(
                    s, v["auc"], v["auc_any"], " ".join("{:>8.3f}".format(v["hit@{}".format(n)]) for n in TOPS),
                    " ".join("{:>8.3f}".format(v["partner_own@{}".format(n)]) for n in TOPS),
                    " ".join("{:>8.3f}".format(v["partner_sim@{}".format(n)]) for n in TOPS)), flush=True)
            json.dump(result, open(os.path.join(args.output_dir, "diag_committee.json"), "w"), indent=1)
            del f
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
