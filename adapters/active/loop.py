"""Rounds of active querying + domain-prompt tuning on one target split (see adapters/active/__init__.py).

A strategy is either a pair strategy (pair_selection.STRATEGIES: random, confident, uncertain, balanced,
cover) or "anchor:<selector>": the ID-level protocol of the in-context setting -- a label-free image
selector (adapters/context_selection.IMAGE_SELECTORS) picks anchor images and the annotator finds each
anchor's person in another camera. Pair strategies spend the budget on yes/no answers, anchor strategies
on "find this person" annotations; both are reported.
"""
import time
from dataclasses import dataclass

import numpy as np
import torch

from adapters.active.candidates import candidate_pairs, random_pair_quantile, fit_threshold
from adapters.active.constraints import ConstraintStore
from adapters.active.image_store import features
from adapters.active.oracle import PairOracle
from adapters.active.pair_selection import STRATEGIES, SelectionContext
from adapters.active.prompt_tuning import tune_domain_prompt


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
        if not (strategy in STRATEGIES or strategy.startswith("anchor:")):
            raise ValueError("unknown strategy {}; pair strategies: {}, or anchor:<image selector>".format(
                strategy, sorted(STRATEGIES)))
        self.oracle = PairOracle(split.pool_pids, split.pool_cams, split.has_cameras)
        self.store = ConstraintStore()
        self.rng = np.random.RandomState(seed)
        self.base = base_prompt(model)
        self.prompt = default_prompt(model)  # round 1 proposes pairs with the base model as deployed
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

    def _ask_pairs(self, feats):
        cfg, store = self.cfg, self.store
        cand = candidate_pairs(feats, self.split.pool_cams, self.split.has_cameras, k=cfg.candidate_k)
        X = feats.cpu().numpy()
        dup_tau = random_pair_quantile(X, 0.99, seed=self.seed)
        if store.answers:  # threshold from the answers, on the current features
            a = np.array([(i, j) for i, j, _ in store.answers])
            tau = fit_threshold((X[a[:, 0]] * X[a[:, 1]]).sum(1), [s for *_, s in store.answers], dup_tau)
        else:
            tau = dup_tau
        ctx = SelectionContext(store, cfg.budget, self.rng, X, self.split.pool_cams, tau, dup_tau,
                               self.split.has_cameras, cfg.expand_ratio)
        order = STRATEGIES[self.strategy](cand, ctx)
        asked = pos = 0
        for p in order:
            if asked >= cfg.budget:
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
                "round_pos_rate": pos / max(asked, 1)}

    def _ask_anchors(self, feats):
        from adapters.context_selection import IMAGE_SELECTORS, ImagePool
        method = self.strategy.split(":", 1)[1]
        pool = ImagePool([(p, -1, c) for p, c in zip(self.split.pool_paths, self.split.pool_cams)],
                         has_cameras=self.split.has_cameras)  # no person ids: the selector cannot see them
        pool.attach_features(feats.cpu().numpy())
        anchors = IMAGE_SELECTORS[method](pool, min(self.cfg.budget, len(pool)), self.rng)
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

    def run(self, evaluate=True, oracle_all=False):
        cfg, rows = self.cfg, []
        if oracle_all:  # upper bound: every pool identity annotated
            return [self._tune_and_eval(0, {}, clusters=self.oracle.all_identities(), cannot=[], evaluate=evaluate)]
        for r in range(1, cfg.rounds + 1):
            t0 = time.time()
            feats = features(self.model, self.split.pool, self.prompt)
            info = self._ask_anchors(feats) if self.strategy.startswith("anchor:") else self._ask_pairs(feats)
            info["t_query"] = time.time() - t0
            clusters = self.store.clusters()
            row = self._tune_and_eval(r, info, clusters, self.store.cannot_links(clusters),
                                      evaluate=evaluate and _eval_round(cfg, r))
            rows.append(row)
            self.log("  [{} r{}] queries {} anchors {} | pos {} neg {} inferred {} | clusters {} ({} true ids) | "
                     "mAP {} R1 {}".format(self.strategy, r, row["n_queries"], row["n_anchors"], row["n_pos"],
                                          row["n_neg"], row["n_inferred"], row["n_clusters"], row["true_ids"],
                                          _fmt(row["mAP"]), _fmt(row["rank1"])))
        return rows

    def _tune_and_eval(self, r, info, clusters, cannot, evaluate):
        cfg = self.cfg
        t0 = time.time()
        if cfg.prompt_mode == "replace":
            start = self.prompt if (cfg.warm_start and r > 1) else default_prompt(self.model)
        else:
            start = self.base
        self.prompt, tinfo = tune_domain_prompt(
            self.model, self.split.pool, clusters, cannot, start, mode=cfg.prompt_mode,
            domain_tokens=self.n_tokens, init_std=cfg.init_std, steps=cfg.steps, lr=cfg.lr,
            ids_per_batch=cfg.ids_per_batch, hn_prob=cfg.hn_prob, seed=self.seed + 7919 * r,
            init_tokens=self.init_tokens)
        t_tune = time.time() - t0
        rank1 = mAP = float("nan")
        if evaluate:
            rank1, mAP = self.split.evaluate(self.model, self.prompt)
        row = {"round": r, **self.store.stats(), "n_anchors": self.anchors_used,
               **self.oracle.report(clusters), "rank1": rank1, "mAP": mAP, **tinfo,
               "t_tune": t_tune, "prompt_tokens": int(self.prompt.size(2)), **info}
        if r == 0:  # oracle_all: every identity, no queries
            row.update(n_clusters=sum(len(c) >= 2 for c in clusters), n_cluster_imgs=sum(len(c) for c in clusters))
        return row


def _fmt(v):
    return "-" if v != v else "{:.2f}".format(v)


@torch.no_grad()
def evaluate_base(model, split):
    """Round 0: the base model as deployed (shared prompt + source-mean tokens), no target labels."""
    return split.evaluate(model, default_prompt(model))
