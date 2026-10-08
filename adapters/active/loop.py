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


OFFLINE_STRATEGIES = ("off:rule", "off:cos", "off:ac")
# Decomposition of the full-label upper bound (no questions; true ids fix one error type of every round's clustering):
# oracle_merge fixes splits only (clusters with the same majority id are merged, outliers join the cluster of their
# id), oracle_purify fixes impurity only (every cluster is split by true id; groups of one image become outliers).
ORACLE_FIX = ("oracle_merge", "oracle_purify")
_OFFLINE = []


def mix_eps(strategy):
    """Share of a round's budget for member questions: mix:<eps> -> eps, off:member -> 1, else None.
    mix:<eps> asks round(eps * B) member questions (_ask_member) and B - round(eps * B) merge questions (off:rule)."""
    if strategy == "off:member":
        return 1.0
    if strategy.startswith("mix:"):
        eps = float(strategy.split(":", 1)[1])
        if not 0 <= eps <= 1:
            raise ValueError("mix:<eps> needs 0 <= eps <= 1")
        return eps
    return None


def oracle_fix(labels, pids, mode):
    labels, pids = np.asarray(labels), np.asarray(pids)
    out = np.full(len(labels), -1, np.int64)
    clusters = np.unique(labels[labels >= 0])
    if mode == "oracle_merge":
        new = {}
        for c in clusters:
            m = labels == c
            u, n = np.unique(pids[m], return_counts=True)
            out[m] = new.setdefault(u[np.argmax(n)].item(), len(new))
        for i in np.flatnonzero(labels < 0):
            out[i] = new.get(pids[i].item(), -1)
        return out
    nxt = 0
    for c in clusters:
        idx = np.flatnonzero(labels == c)
        u, n = np.unique(pids[idx], return_counts=True)
        for p, k in zip(u, n):
            if k >= 2:
                out[idx[pids[idx] == p]] = nxt
                nxt += 1
    return out


def _offline_modules():
    """(offline_recall_1006_2, offline_AC_1007), loaded once by path (they are scripts, not package modules)."""
    if not _OFFLINE:
        import importlib.util
        import os

        def load(name, path):
            s = importlib.util.spec_from_file_location(name, path)
            m = importlib.util.module_from_spec(s)
            s.loader.exec_module(m)
            return m
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        base_path = os.path.join(root, "scripts", "offline_recall_1006_2.py")
        ac_path = os.environ.get("FERREID_OFFLINE_AC", "/data1/yangbin/dz/code/exp_1007_AC/offline_AC_1007.py")
        _OFFLINE.extend([load("offline_recall_1006_2", base_path), load("offline_AC_1007", ac_path)])
    return _OFFLINE


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
    shortlist_k: int = 5          # shortlist: candidate clusters shown per question (cost: one comparison each)
    shortlist_rank: str = "rule"  # shortlist: "rule" (fewest cameras, then centroid cosine) | "committee" (co-association
    #                               of the clusterings under each source domain's tokens, complementary clusters only)
    answers_file: str = ""        # strategy "file": machine answers (csv with i, j, vlm_score; scripts/diag_vlm.py)
    answer_yes: float = 2.2       # file: accept "same" when vlm_score > answer_yes (log-odds; 2.2 ~ P(yes) 0.9)
    answer_no: float = -2.2       # file: accept "different" when vlm_score < answer_no; in between: no answer
    human_verify: int = 0         # file: a human checks the top-N "same" candidates of the machine (by score); 0: none
    cannot_use: str = "all"       # "different" answers: "all" (split clusters + hard negatives in training), "cluster"
    #                               (split only), "train" (hard negatives only), "none" (ignored)
    unverified_yes: str = "trust" # file + human_verify: "trust" the machine's remaining "same" answers or "drop" them
    # 10-08 night (task list v3)
    select_respect_cannot_use: bool = False  # BUG switch: the clustering used for question selection also follows
    #                                          --cannot_use (default False = the 10-07 / 10-08 behaviour, where it
    #                                          splits on every "different" answer)
    subset: str = "random"        # oracle_merge_subset: random | persist | all
    persist_file: str = ""        # oracle_merge_subset persist: json of round-1 split pairs labelled persist (E1a)


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
        if not (strategy in STRATEGIES or strategy in REPAIR_STRATEGIES or strategy in ("none", "oracle", "shortlist", "file")
                or strategy.startswith("anchor:") or strategy in OFFLINE_STRATEGIES or strategy in ORACLE_FIX
                or mix_eps(strategy) is not None or strategy == "oracle_merge_subset"):
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
        return (self.cfg.pseudo or self.strategy in REPAIR_STRATEGIES or self.strategy in NEEDS_GRAPH
                or self.strategy == "shortlist" or self.strategy in OFFLINE_STRATEGIES
                or mix_eps(self.strategy) is not None or self.strategy == "oracle_merge_subset")

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

    def _ask_shortlist(self, feats):
        """Shortlist questions: "which of these K clusters (no camera in common with cluster A) is A's person,
        if any?" -- several may be chosen. Query clusters A, most likely split first: fewest cameras, then the
        highest centroid similarity of the best complementary candidate. The simulated annotator compares A's
        medoid with each candidate's medoid; the cost is one comparison per candidate shown (candidates whose
        answer follows from earlier answers are not shown)."""
        K, store, X = self.cfg.shortlist_k, self.store, feats.cpu().numpy()
        labels = self.graph.cluster(store)
        C = int(labels.max()) + 1 if (labels >= 0).any() else 0
        if C < 2:
            return {"n_shortlist_q": 0}
        members = [np.flatnonzero(labels == c) for c in range(C)]
        cent = np.stack([X[m].mean(0) for m in members])
        cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12
        medoid = np.array([m[np.argmax(X[m] @ cent[c])] for c, m in enumerate(members)])
        cams = np.asarray(self.split.pool_cams)
        cam_idx = np.unique(cams, return_inverse=True)[1].reshape(-1)
        mask = np.zeros((C, cam_idx.max() + 1), bool)
        for c, m in enumerate(members):
            mask[c, cam_idx[m]] = True
        S = cent @ cent.T
        np.fill_diagonal(S, -np.inf)
        if self.split.has_cameras:
            S = np.where((mask.astype(np.int32) @ mask.T.astype(np.int32)) > 0, -np.inf, S)  # complementary only
        top1 = S.max(1)
        ncam = mask.sum(1)
        order = np.lexsort((-top1, ncam))  # fewest cameras first, then the most similar complementary candidate
        if self.cfg.shortlist_rank == "committee":
            # query-by-committee: co-association of clusters A, B (share of their image pairs clustered together)
            # under each source domain's tokens, averaged; complementary pairs only. Clusters with the strongest
            # co-association first, candidates by co-association (ties / zeros: centroid cosine)
            A = np.zeros((C, C))
            size = np.array([len(m) for m in members], float)
            for lab in self._committee_labels():
                ok = lab >= 0
                P = np.zeros((C, int(lab.max()) + 1 if ok.any() else 1))
                np.add.at(P, (labels[ok & (labels >= 0)], lab[ok & (labels >= 0)]), 1)
                P /= size[:, None]
                A += P @ P.T
            A /= len(self._committee_labels())
            S = np.where(np.isfinite(S), A + 1e-3 * (S + 1), -np.inf)
            top1 = S.max(1)
            order = np.argsort(-top1, kind="stable")
        elif self.cfg.shortlist_rank != "rule":
            raise ValueError("--shortlist_rank must be rule or committee")
        asked = pos = nq = hits = 0
        for a in order:
            if asked >= self.budget or not np.isfinite(top1[a]):
                continue
            cand = [int(c) for c in np.argsort(-S[a])[:K] if np.isfinite(S[a, c])]
            todo = [c for c in cand if store.infer(int(medoid[a]), int(medoid[c])) is None]
            if not todo:
                continue
            todo = todo[:self.budget - asked]
            hit = 0
            for c in todo:
                same = self.oracle.same(int(medoid[a]), int(medoid[c]))
                store.add(int(medoid[a]), int(medoid[c]), same)
                hit += same
            pos += hit
            hits += hit > 0
            asked += len(todo)
            nq += 1
        return {"n_shortlist_q": nq, "round_pos_rate": pos / max(asked, 1), "shortlist_hit_rate": hits / max(nq, 1)}

    def _ask_file(self):
        """Machine-answered questions (e.g. a fine-tuned MLLM, scripts/diag_vlm.py): only confident answers enter
        the constraints, at most `budget` of them; the person ids only report how many accepted answers are right."""
        import csv
        cfg, store = self.cfg, self.store
        rows = [r for r in csv.DictReader(open(cfg.answers_file)) if r.get("domain", self.split.name) == self.split.name]
        if any(r.get("kind") == "q_all" for r in rows):
            rows = [r for r in rows if r.get("kind") == "q_all"]
        rows.sort(key=lambda r: -abs(float(r["vlm_score"])))  # most confident first
        # VLM-assisted active learning: a human checks the machine's most confident "same" candidates
        verify = {(int(r["i"]), int(r["j"])) for r in sorted(rows, key=lambda r: -float(r["vlm_score"]))
                  [:cfg.human_verify] if float(r["vlm_score"]) > cfg.answer_yes}
        n_yes = n_no = right_yes = right_no = n_human = human_pos = 0
        for r in rows:
            if n_yes + n_no >= self.budget:
                break
            i, j, s = int(r["i"]), int(r["j"]), float(r["vlm_score"])
            if cfg.answer_no <= s <= cfg.answer_yes or store.infer(i, j) is not None:
                continue
            if (i, j) in verify:  # human answer (counted as a human query)
                truth = self.oracle.same(i, j)
                store.add(i, j, truth)
                n_human += 1; human_pos += truth
                continue
            same = s > cfg.answer_yes
            if same and cfg.human_verify and cfg.unverified_yes == "drop":
                continue
            store.add(i, j, same)
            truth = bool(self.oracle._pids[i] == self.oracle._pids[j])  # reporting only (not a human query)
            n_yes += same; n_no += not same
            right_yes += same and truth; right_no += (not same) and (not truth)
        return {"n_human_verified": n_human, "n_human_yes": human_pos,
                "n_machine_yes": n_yes, "n_machine_no": n_no, "machine_yes_prec": right_yes / max(n_yes, 1),
                "machine_no_prec": right_no / max(n_no, 1), "round_pos_rate": n_yes / max(n_yes + n_no, 1)}

    def _ask_member(self, feats):
        """Member questions ("is this member the same person as its cluster's medoid?") on this round's constrained
        clustering. Suspicion of a member = 1 - cos(member, normalised cluster centroid); clusters of >= 2 images;
        the medoid is the member closest to the centroid. Most suspicious first over the whole pool, at most 2
        questions per cluster and round; pairs whose answer follows from earlier answers are skipped (not charged).
        "Different" -> cannot-link member / medoid that splits clusters (split=True, kept across rounds);
        "same" -> must-link. Label-free: features, cameras-free, answers only."""
        store, X = self.store, feats.float().cpu().numpy()
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        labels = self.graph.cluster(store)
        cand = []  # (suspicion, member, medoid, cluster)
        for c, m in enumerate(label_clusters(labels)):
            if len(m) < 2:
                continue
            m = np.asarray(m)
            cent = X[m].mean(0)
            cent /= np.linalg.norm(cent) + 1e-12
            cos = X[m] @ cent
            med = int(m[np.argmax(cos)])
            for i, s in zip(m.tolist(), (1 - cos).tolist()):
                if i != med:
                    cand.append((s, i, med, c))
        cand.sort(key=lambda t: -t[0])
        asked = no = 0
        per = {}
        susp = []
        for s, i, med, c in cand:
            if asked >= self.budget:
                break
            if per.get(c, 0) >= 2:
                continue
            if store.infer(i, med) is not None:
                store.n_inferred += 1
                continue
            same = self.oracle.same(i, med)
            store.add(i, med, same, split=True)
            per[c] = per.get(c, 0) + 1
            asked += 1
            no += not same
            susp.append(s)
        return {"n_member_q": asked, "n_member_no": no, "member_cands": len(cand),
                "member_susp_mean": float(np.mean(susp)) if susp else float("nan")}

    def _ask_mix(self, feats):
        """mix:<eps> / off:member: the round's budget B split into B - round(eps B) merge questions (off:rule, asked
        first, exactly as the off:rule strategy; their "different" answers do not split clusters) and round(eps B)
        member questions (_ask_member, on the clustering after the merge answers)."""
        B = self.budget
        n_member = int(round(mix_eps(self.strategy) * B))
        info = {}
        before = self.store.stats()
        if B - n_member > 0:
            self.budget = B - n_member
            info = self._ask_offline(feats, method="rule", split_no=False)
        mid = self.store.stats()
        info.update(n_merge_q=mid["n_queries"] - before["n_queries"], n_merge_yes=mid["n_pos"] - before["n_pos"])
        self.budget = n_member
        if n_member > 0:
            info.update(self._ask_member(feats))
        else:
            info.update(n_member_q=0, n_member_no=0)
        self.budget = B
        after = self.store.stats()
        q = after["n_queries"] - before["n_queries"]
        info["round_pos_rate"] = (after["n_pos"] - before["n_pos"]) / max(q, 1)
        return info

    def _ask_oracle_subset(self, feats):
        """E2 (task list v3), oracle: merge B split pairs of this round's clustering with the true ids, each as one
        "same person" answer between the two units' representatives (split_pairs.py), kept as must-links.
          random   B pairs drawn uniformly from this round's split pairs;
          persist  B pairs drawn uniformly from the round-1 split pairs labelled persist in --persist_file (E1a, the
                   same seed's no-question run: its round-1 clustering is this run's round-1 clustering);
          random_file  B pairs drawn uniformly from all round-1 split pairs of --persist_file (task list v7: the
                   round-2 asking groups draw from the same round-1 list as persist, not from round 2's clustering);
          all      every split pair of this round (budget ignored; only in rounds with a budget).
        random / random_file / persist are stratified by pair kind with the same quotas (task list v4). Pairs
        whose representatives this round's clustering already put together are still asked and charged
        (e2_already_together counts them).
        Pairs already merged by earlier answers are skipped (not charged)."""
        import json
        from adapters.active.split_pairs import split_pairs
        X = feats.float().cpu().numpy()
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        labels = self.graph.cluster(self.store)
        cur = split_pairs(labels, self.oracle._pids, X)
        mode = self.cfg.subset
        if mode not in ("random", "random_file", "persist", "all"):
            raise ValueError("--subset must be random, random_file, persist or all")
        if mode in ("random", "random_file", "persist") and getattr(self, "_persist", None) is None:
            # task list v4: random and persist are stratified by pair kind with the same quotas, taken from the
            # same seed's persist list (random reads it only for the kind counts)
            d = json.load(open(self.cfg.persist_file))
            if int(d["seed"]) != int(self.seed):
                raise ValueError("persist file of seed {} used by a seed-{} run".format(d["seed"], self.seed))
            self._file_pairs = list(d["pairs"])  # every round-1 split pair (random_file, task list v7)
            self._persist = [p for p in d["pairs"] if p["persist"]]
            self._kind_share = {k: sum(p["kind"] == k for p in self._persist) for k in ("cluster", "outlier")}
            self._persist_left = dict(self._kind_share)  # persist pairs of each kind not used yet
        if mode == "all":
            pool, quota = cur, {"cluster": None, "outlier": None}
        else:
            tot = sum(self._kind_share.values())
            qc = int(round(self.budget * self._kind_share["cluster"] / tot)) if tot else 0
            quota = {"cluster": qc, "outlier": self.budget - qc}
            # fewer persist pairs of a kind than its quota: both groups use what persist has left of it
            quota = {k: min(q, self._persist_left[k]) for k, q in quota.items()}
            pool = {"persist": self._persist, "random_file": self._file_pairs, "random": cur}[mode]
            pool = list(pool)
        order = self.rng.permutation(len(pool)) if mode != "all" else np.arange(len(pool))
        used = {"cluster": 0, "outlier": 0}
        skipped = together = 0
        for k in order:
            p = pool[k]
            q = quota[p["kind"]]
            if q is not None and used[p["kind"]] >= q:
                continue
            i, j = int(p["rep_a"]), int(p["rep_b"])
            if self.store.infer(i, j) is not None:
                self.store.n_inferred += 1
                skipped += 1
                continue
            same = self.oracle.same(i, j)
            assert same, "split pair {} {} is not one person".format(i, j)
            together += bool(labels[i] >= 0 and labels[i] == labels[j])  # already merged by this round's clustering
            self.store.add(i, j, True)
            used[p["kind"]] += 1
        if mode != "all":
            for kd in used:
                self._persist_left[kd] -= used[kd]
        asked = used["cluster"] + used["outlier"]
        return {"e2_subset": mode, "e2_pool": len(pool), "e2_merged": asked, "e2_merged_cluster": used["cluster"],
                "e2_merged_outlier": used["outlier"], "e2_quota_cluster": quota["cluster"],
                "e2_quota_outlier": quota["outlier"], "e2_skipped": skipped, "e2_split_pairs_now": len(cur),
                "e2_already_together": together,
                "round_pos_rate": 1.0 if asked else float("nan")}

    def _ask_offline(self, feats, method=None, split_no=True):
        """Selectors of the offline recall study (scripts/offline_recall_1006_2.py, exp_1007_AC/offline_AC_1007.py)
        on this round's constrained clustering: off:rule (shortlist rule, K=1), off:cos (camera-subcluster cosine,
        K=10), off:ac (camera-pair KISSME d128 + cross-camera Sinkhorn, q90, eps 0.05, fix; all clusters as KISSME
        positives -- no eps-stability filter in the loop). Questions are the selector's representative image pairs in
        its order; pairs whose answer follows from earlier answers are skipped (not charged). After the questions,
        the round's split-pair recall and cosine rank diagnostic are logged (true ids: reporting only)."""
        base, ac = _offline_modules()
        method = method or self.strategy.split(":", 1)[1]
        labels = self.graph.cluster(self.store)
        pool = base.Pool(feats.float().cpu().numpy(), labels, np.asarray(self.split.pool_cams))
        sub = ac.Sub(pool)
        if method == "rule":
            order = base.old_queries(pool, 1, rule=True)
        elif method == "cos":
            order = base.camera_queries(pool)
        else:
            km = ac.KissmeScore(pool, sub, 128)
            vals, med, spread = ac.score_stats(km, sub)
            zq = (float(np.quantile(vals, .9)) - med) / spread
            order = base.dedup(ac.sinkhorn_queries(km, sub, med, spread, zq, .05, "fix")[0])
        asked = pos = 0
        picked = []
        for i, j in order:
            if asked >= self.budget:
                break
            i, j = int(i), int(j)
            if self.store.infer(i, j) is not None:
                self.store.n_inferred += 1
                continue
            same = self.oracle.same(i, j)
            self.store.add(i, j, same, split=split_no)
            asked += 1
            pos += same
            picked.append((i, j))
        pids = self.oracle._pids  # reporting only, after every question of the round
        M, _ = base.gold_pairs(pool, pids)
        row, _, _ = base.metrics(pool, pids, M, np.array(picked, np.int64).reshape(-1, 2))
        ranks = {r["rank_bucket"]: r["weight_share"] for r in ac.rank_diagnostic(ac.CosineScore(sub), sub, pool, M)}
        return {"round_pos_rate": pos / max(asked, 1), "off_candidates": len(order), "off_asked": asked,
                "off_clusters": pool.c, "off_outliers": len(pool.outliers), "off_M_pairs": len(M),
                "off_M_weight": sum(M.values()), "off_wrecall": row["weighted_recall"],
                "off_eff_wrecall": row["effective_weighted_recall"],
                "off_rank1": ranks["1-1"], "off_rank2_10": ranks["2-3"] + ranks["4-10"],
                "off_rank11_200": ranks["11-50"] + ranks["51-200"], "off_rank_gt200": ranks["201-inf"],
                "off_rank_none": ranks["no_cross_camera_subcluster_pair"]}

    def _committee_labels(self):
        """Pool clusterings under each source domain's tokens (shared prompt | domain_prompts[d]); the base model
        is frozen, so they are computed once per run."""
        if getattr(self, "_committee", None) is None:
            dom = getattr(self.model, "domain_prompts", None)
            if dom is None:
                raise ValueError("--shortlist_rank committee needs a base model with source-domain tokens")
            cfg, out = self.cfg, []
            for d in range(dom.size(0)):
                p = torch.cat([self.base, dom[d:d + 1].detach().float()], dim=2)
                f = features(self.model, self.split.pool, p)
                if cfg.cam_norm and self.split.has_cameras:
                    f = camera_normalize(f, self.split.pool_cams)
                g = PoolGraph(f, cfg.pseudo_k1, cfg.pseudo_k2, cfg.pseudo_eps, cfg.pseudo_min_samples)
                out.append(g.cluster())
                del g, f
                torch.cuda.empty_cache()
            self._committee = out
        return self._committee

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
                if cfg.select_respect_cannot_use and self.graph is not None:  # BUG fix switch (task list v3)
                    self.graph.split_cannot = cfg.cannot_use in ("all", "cluster")
                    if mix_eps(self.strategy) is not None:
                        self.graph.split_cannot, self.graph.member_split_only = True, True
            self.budget = self._round_budget(r)
            if self.strategy in ("none", "oracle") + ORACLE_FIX or self.budget <= 0:
                info = {}
            elif self.strategy.startswith("anchor:"):
                info = self._ask_anchors(sel)
            elif self.strategy == "shortlist":
                info = self._ask_shortlist(sel)
            elif self.strategy == "file":
                info = self._ask_file()
            elif self.strategy in OFFLINE_STRATEGIES:
                info = self._ask_offline(sel)
            elif mix_eps(self.strategy) is not None:
                info = self._ask_mix(sel)
            elif self.strategy == "oracle_merge_subset":
                info = self._ask_oracle_subset(sel)
            else:
                info = self._ask_pairs(sel)
            if self.strategy == "oracle":  # upper bound, every round: the true identities of the pool
                clusters, cannot = self.oracle.all_identities(), []
            elif cfg.pseudo:
                self.graph.split_cannot = cfg.cannot_use in ("all", "cluster")
                if mix_eps(self.strategy) is not None:  # member "different" answers split, merge ones do not
                    self.graph.split_cannot, self.graph.member_split_only = True, True
                labels = self.graph.cluster(self.store)
                if self.strategy in ORACLE_FIX:  # report the clustering before the fix, then train on the fixed one
                    info.update({"raw_" + k: v for k, v in self.oracle.pseudo_report(labels).items()})
                    labels = oracle_fix(labels, self.oracle._pids, self.strategy)
                clusters, cannot = self._pseudo_clusters(labels)
                if cfg.cannot_use in ("cluster", "none"):
                    cannot = []
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
            self._save_round(r)
            self.log("  [{} r{}] queries {} anchors {} | pos {} neg {} inferred {} | clusters {} ({} true ids) | "
                     "mAP {} R1 {}".format(self.strategy, r, row["n_queries"], row["n_anchors"], row["n_pos"],
                                          row["n_neg"], row["n_inferred"], row["n_clusters"], row["true_ids"],
                                          _fmt(row["mAP"]), _fmt(row["rank1"])))
        return rows

    def _save_round(self, r):
        """Per-round snapshot when round_dir is set (scripts/eval_active.py): the domain prompt after round r and
        every answer so far, so that features / clusterings of any round can be recomputed later. Saving only."""
        d = getattr(self, "round_dir", None)
        if not d:
            return
        import os
        os.makedirs(d, exist_ok=True)
        torch.save({"prompt": self.prompt.detach().cpu(), "round": r, "strategy": self.strategy, "seed": self.seed,
                    "answers": list(self.store.answers),
                    "member_split": {int(k): sorted(int(x) for x in v) for k, v in self.store._cannot_split.items() if v}},
                   os.path.join(d, "round{:02d}.pt".format(r)))

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
