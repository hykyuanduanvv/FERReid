"""Rounds of active querying + domain-prompt tuning on one target split (see adapters/active/__init__.py).

A strategy is either a pair strategy (pair_selection.STRATEGIES: random, confident, uncertain, balanced,
cover) or "anchor:<selector>": the ID-level protocol of the in-context setting -- a label-free image
selector (adapters/context_selection.IMAGE_SELECTORS) picks anchor images and the annotator finds each
anchor's person in another camera. Pair strategies spend the budget on yes/no answers, anchor strategies
on "find this person" annotations; both are reported.

--pseudo True (cluster repair): every round the pool is clustered into pseudo identities (pseudo.PoolGraph,
the answers enforced as constraints) and the domain prompt is trained on all of them (+ cluster-memory loss).
Strategies "repair", "repair_unc", "repair_random" then ask merge / split questions about those clusters
(repair.py); "none" asks nothing (the unsupervised baseline: 0 answers). Pair strategies work as before, their
answers entering the clustering as constraints.
"""
import time
from dataclasses import dataclass

import numpy as np
import torch

from adapters.active.candidates import candidate_pairs, random_pair_quantile, fit_threshold
from adapters.active.constraints import ConstraintStore
from adapters.active.image_store import features
from adapters.active.oracle import PairOracle
from adapters.active.pair_selection import NEEDS_GRAPH, STRATEGIES, SelectionContext
from adapters.active.prompt_tuning import tune_domain_prompt
from adapters.active.pseudo import PoolGraph, camera_normalize, label_clusters
from adapters.active.repair import REPAIR_STRATEGIES, Calibrator, rank_repair, repair_candidates, repair_candidates_knn


@dataclass
class ActiveConfig:
    rounds: int = 4
    budget: int = 50              # queries per round (pairs, or anchors for anchor:<selector>)
    candidate_k: int = 10         # cross-camera neighbours per image proposed as pairs
    expand_ratio: float = 0.5     # cover: share of the budget for pairs touching annotated images
    prompt_mode: str = "append"   # append | replace
    domain_tokens: int = 0        # append: new tokens per layer (0: as many as the source-domain tokens, else 8)
    token_init: str = "source_mean"  # append: "source_mean" of the base model's domain tokens (if any) | "random"
    init_std: float = 0.02
    steps: int = 300
    lr: float = 3e-4
    ids_per_batch: int = 32
    hn_prob: float = 0.5
    warm_start: bool = False      # tune each round from the previous round's prompt (default: from the base)
    eval_rounds: str = "all"      # "all", "last", or comma-separated round numbers
    # cluster repair (pseudo.py, repair.py)
    pseudo: bool = False          # train on the constrained pseudo identities of the whole pool
    pseudo_k1: int = 30           # k-reciprocal neighbours (Jaccard distance)
    pseudo_k2: int = 6            # local query expansion
    pseudo_eps: float = 0.6       # DBSCAN radius on the Jaccard distance
    pseudo_min_samples: int = 4
    contrast_weight: float = 1.0  # cluster-memory contrastive loss (pseudo only)
    contrast_temp: float = 0.05
    cross_cam: bool = True        # pseudo only: the two images of a cluster from two cameras
    repair_k: int = 5             # merge questions: nearest clusters per cluster
    repair_per_cluster: int = 1   # questions per cluster and round
    budget_schedule: str = ""     # questions per round, comma-separated (e.g. "250,0,0,0,0"); overrides budget
    cam_norm: bool = False        # clustering + question selection on camera-normalised features (training: raw)


def base_prompt(model):
    """The shared prompt (1, L, V, D) of a VPT model (without its source-domain tokens)."""
    if getattr(model, "prompt", None) is None or not hasattr(model, "default_prompt"):
        raise ValueError("the active module tunes a deep visual prompt: use a --model_type vpt checkpoint")
    return model.prompt.detach().float()


def default_prompt(model):
    """What the base model uses on an unseen domain: shared prompt (+ source-mean tokens)."""
    base_prompt(model)  # type check
    return model.default_prompt().detach().float()


def _eval_round(cfg, r):
    if cfg.eval_rounds == "all":
        return True
    if cfg.eval_rounds == "last":
        return r == cfg.rounds
    return r in {int(x) for x in cfg.eval_rounds.split(",") if x}


class ActiveRun:
    """One (split, strategy, seed): rounds of query -> constraints -> prompt; returns one row per round."""

    def __init__(self, model, split, strategy, cfg, seed, log=print):
        self.model, self.split, self.strategy, self.cfg, self.seed, self.log = model, split, strategy, cfg, seed, log
        if not (strategy in STRATEGIES or strategy in REPAIR_STRATEGIES or strategy in ("none", "oracle")
                or strategy.startswith("anchor:")):
            raise ValueError("unknown strategy {}; pair strategies: {}, repair strategies: {}, none, or "
                             "anchor:<image selector>".format(strategy, sorted(STRATEGIES), REPAIR_STRATEGIES))
        if strategy == "none" and not cfg.pseudo:
            raise ValueError("strategy none (no questions) only makes sense with --pseudo True")
        if cfg.budget_schedule:
            sched = [int(x) for x in cfg.budget_schedule.split(",") if x != ""]
            if len(sched) != cfg.rounds:
                raise ValueError("--budget_schedule needs one entry per round ({} rounds)".format(cfg.rounds))
        self.graph = None
        self.budget = cfg.budget  # questions of the current round (budget_schedule)
        self.oracle = PairOracle(split.pool_pids, split.pool_cams, split.has_cameras)
        self.store = ConstraintStore()
        self.rng = np.random.RandomState(seed)
        self.base = base_prompt(model)
        self.prompt = default_prompt(model)  # round 1 proposes pairs with the base model as deployed
        self._prompt0 = self.prompt
        self.anchors_used = 0
        src = model.domain_token_init()  # (1, L, m, D) or None
        self.init_tokens = src if (cfg.token_init == "source_mean" and src is not None) else None
        if cfg.token_init not in ("source_mean", "random"):
            raise ValueError("--token_init must be source_mean or random")
        self.n_tokens = cfg.domain_tokens or (src.size(2) if src is not None else 8)
        if self.init_tokens is not None and self.n_tokens != src.size(2):
            raise ValueError("--domain_tokens {} differs from the base model's {} source-domain tokens "
                             "(use 0, or --token_init random)".format(cfg.domain_tokens, src.size(2)))

    # ---------------------------------------------------------------- one round of questions

    def _round_budget(self, r):
        if self.cfg.budget_schedule:
            return [int(x) for x in self.cfg.budget_schedule.split(",") if x != ""][r - 1]
        return self.cfg.budget

    def _needs_graph(self):
        if self.strategy == "oracle":
            return False
        return self.cfg.pseudo or self.strategy in REPAIR_STRATEGIES or self.strategy in NEEDS_GRAPH

    def _ask_pairs(self, feats):
        cfg, store = self.cfg, self.store
        X = feats.cpu().numpy()
        dup_tau = random_pair_quantile(X, 0.99, seed=self.seed)
        ans_sims, ans_lab = np.zeros(0), np.zeros(0, bool)
        if store.answers:  # threshold from the answers, on the current features
            a = np.array([(i, j) for i, j, _ in store.answers])
            ans_sims, ans_lab = (X[a[:, 0]] * X[a[:, 1]]).sum(1), np.array([s for *_, s in store.answers])
            tau = fit_threshold(ans_sims, ans_lab, dup_tau)
        else:
            tau = dup_tau
        extra = {}
        budget = self.budget
        if self.strategy in REPAIR_STRATEGIES:
            labels = self.graph.cluster(store)
            if self.strategy == "repair2":
                knn = candidate_pairs(feats, self.split.pool_cams, self.split.has_cameras, k=cfg.candidate_k)
                cand = repair_candidates_knn(X, labels, knn)
            else:
                cand = repair_candidates(X, labels, k_merge=cfg.repair_k,
                                         cams=self.split.pool_cams if self.split.has_cameras else None)
            p = Calibrator(ans_sims, ans_lab, tau)(cand["sim"])
            mode = "repair" if self.strategy == "repair2" else self.strategy
            order = rank_repair(cand, p, self.rng, mode, cfg.repair_per_cluster)
            top = order[:budget]
            extra = {"n_merge_q": int((cand["kind"][top] == 0).sum()), "n_split_q": int((cand["kind"][top] == 1).sum()),
                     "exp_change": float((np.where(cand["kind"] == 0, p, 1 - p) * cand["impact"])[top].sum())}
        else:
            cand = candidate_pairs(feats, self.split.pool_cams, self.split.has_cameras, k=cfg.candidate_k)
            ctx = SelectionContext(store, budget, self.rng, X, self.split.pool_cams, tau, dup_tau,
                                   self.split.has_cameras, cfg.expand_ratio, graph=self.graph)
            order = STRATEGIES[self.strategy](cand, ctx)
        asked = pos = 0
        for p in order:
            if asked >= budget:
                break
            i, j = int(cand["i"][p]), int(cand["j"][p])
            if store.infer(i, j) is not None:
                store.n_inferred += 1
                continue
            same = self.oracle.same(i, j)
            store.add(i, j, same)
            asked += 1
            pos += same
        return {"tau": tau, "dup_tau": dup_tau, "n_candidates": len(cand["sim"]),
                "cand_mutual": float(cand["mutual"].mean()) if len(cand["sim"]) else float("nan"),
                "round_pos_rate": pos / max(asked, 1), **extra}

    def _ask_anchors(self, feats):
        from adapters.context_selection import IMAGE_SELECTORS, ImagePool
        method = self.strategy.split(":", 1)[1]
        pool = ImagePool([(p, -1, c) for p, c in zip(self.split.pool_paths, self.split.pool_cams)],
                         has_cameras=self.split.has_cameras)  # no person ids: the selector cannot see them
        pool.attach_features(feats.cpu().numpy())
        anchors = IMAGE_SELECTORS[method](pool, min(self.budget, len(pool)), self.rng)
        new = fail = 0
        for a in anchors:
            b = self.oracle.partner(int(a), self.rng)
            self.anchors_used += 1
            if b is None:
                fail += 1
                continue
            if self.store.infer(int(a), b) is None:
                new += 1
            self.store.add(int(a), b, True)
        return {"n_anchor_fail": fail, "round_new_pairs": new}

    # ---------------------------------------------------------------- rounds

    def run(self, evaluate=True, oracle_all=False, tune=True):
        """tune=False: the prompt stays the base model's (features and graph computed once) -- the offline
        simulation of the questions (scripts/sim_selection.py), no training, no evaluation."""
        cfg, rows = self.cfg, []
        if oracle_all:  # upper bound: every pool identity annotated
            feats = features(self.model, self.split.pool, self.prompt) if cfg.pseudo else None
            return [self._tune_and_eval(0, {}, clusters=self.oracle.all_identities(), cannot=[], evaluate=evaluate,
                                        feats=feats)]
        feats = None
        for r in range(1, cfg.rounds + 1):
            t0 = time.time()
            if tune or feats is None:
                feats = self._features()
                sel = camera_normalize(feats, self.split.pool_cams) if (cfg.cam_norm and self.split.has_cameras) else feats
                self.graph = PoolGraph(sel, cfg.pseudo_k1, cfg.pseudo_k2, cfg.pseudo_eps,
                                       cfg.pseudo_min_samples) if self._needs_graph() else None
            self.budget = self._round_budget(r)
            if self.strategy in ("none", "oracle") or self.budget <= 0:
                info = {}
            elif self.strategy.startswith("anchor:"):
                info = self._ask_anchors(sel)
            else:
                info = self._ask_pairs(sel)
            if self.strategy == "oracle":  # upper bound, every round: the true identities of the pool
                clusters, cannot = self.oracle.all_identities(), []
            elif cfg.pseudo:
                labels = self.graph.cluster(self.store)
                clusters, cannot = self._pseudo_clusters(labels)
                info.update(self.oracle.pseudo_report(labels))
            else:
                clusters = self.store.clusters()
                cannot = self.store.cannot_links(clusters)
            info["t_query"] = time.time() - t0
            if not tune:
                rows.append({"round": r, **self.store.stats(), "n_anchors": self.anchors_used,
                             **self.oracle.report(self.store.clusters()), **info})
                self.log("  [{} r{}] queries {} | pos {} | {}".format(
                    self.strategy, r, rows[-1]["n_queries"], rows[-1]["n_pos"],
                    "pairwise F {:.3f}".format(info["pw_f"]) if "pw_f" in info else
                    "true ids {}".format(rows[-1]["true_ids"])))
                continue
            row = self._tune_and_eval(r, info, clusters, cannot, evaluate=evaluate and _eval_round(cfg, r),
                                      feats=feats)
            rows.append(row)
            self.log("  [{} r{}] queries {} anchors {} | pos {} neg {} inferred {} | clusters {} ({} true ids) | "
                     "mAP {} R1 {}".format(self.strategy, r, row["n_queries"], row["n_anchors"], row["n_pos"],
                                          row["n_neg"], row["n_inferred"], row["n_clusters"], row["true_ids"],
                                          _fmt(row["mAP"]), _fmt(row["rank1"])))
        return rows

    def _features(self):
        """Pool features under the current prompt; features_cache (a (N, D) tensor of the base model's
        features, set by scripts/sim_selection.py) skips the extraction while the prompt is the base one."""
        cache = getattr(self, "features_cache", None)
        if cache is not None and self.prompt is self._prompt0:
            return cache
        return features(self.model, self.split.pool, self.prompt)

    def _pseudo_clusters(self, labels):
        """Training clusters from pseudo labels: the pseudo identities, plus answered images the clustering
        left as outliers (singletons, negatives only); answered cannot-links as positions into that list."""
        clusters = label_clusters(labels)
        pos = {i: n for n, c in enumerate(clusters) for i in c}
        for i in self.store.images():
            if labels[i] < 0:
                pos[i] = len(clusters)
                clusters.append([i])
        hc = self.store.clusters()
        cannot = set()
        for x, y in self.store.cannot_links(hc):
            a, b = pos[hc[x][0]], pos[hc[y][0]]
            if a != b:
                cannot.add((min(a, b), max(a, b)))
        return clusters, sorted(cannot)

    def _tune_and_eval(self, r, info, clusters, cannot, evaluate, feats=None):
        cfg = self.cfg
        t0 = time.time()
        init_tokens = self.init_tokens
        if cfg.prompt_mode == "replace":
            start = self.prompt if (cfg.warm_start and r > 1) else default_prompt(self.model)
        else:
            start = self.base
            if cfg.warm_start and r > 1:  # continue from the previous round's domain tokens
                init_tokens = self.prompt[:, :, self.base.size(2):]
        pseudo = cfg.pseudo
        self.prompt, tinfo = tune_domain_prompt(
            self.model, self.split.pool, clusters, cannot, start, mode=cfg.prompt_mode,
            domain_tokens=self.n_tokens, init_std=cfg.init_std, steps=cfg.steps, lr=cfg.lr,
            ids_per_batch=cfg.ids_per_batch, hn_prob=cfg.hn_prob, seed=self.seed + 7919 * r,
            init_tokens=init_tokens, cams=self.split.pool_cams if self.split.has_cameras else None,
            cross_cam=pseudo and cfg.cross_cam, feats=feats if pseudo else None,
            contrast_weight=cfg.contrast_weight if pseudo else 0.0, temp=cfg.contrast_temp)
        t_tune = time.time() - t0
        rank1 = mAP = float("nan")
        if evaluate:
            rank1, mAP = self.split.evaluate(self.model, self.prompt)
        row = {"round": r, **self.store.stats(), "n_anchors": self.anchors_used,
               **self.oracle.report(clusters), "rank1": rank1, "mAP": mAP, **tinfo,
               "t_tune": t_tune, "prompt_tokens": int(self.prompt.size(2)), **info}
        if self.cfg.pseudo and r > 0:  # pseudo: the row's cluster columns describe the answers alone
            row.update({"ans_" + k: v for k, v in self.oracle.report(self.store.clusters()).items()})
        if r == 0:  # oracle_all: every identity, no queries
            row.update(n_clusters=sum(len(c) >= 2 for c in clusters), n_cluster_imgs=sum(len(c) for c in clusters))
        return row


def _fmt(v):
    return "-" if v != v else "{:.2f}".format(v)


@torch.no_grad()
def evaluate_base(model, split):
    """Round 0: the base model as deployed (shared prompt + source-mean tokens), no target labels."""
    return split.evaluate(model, default_prompt(model))
