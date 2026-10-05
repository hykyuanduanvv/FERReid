"""CPU checks of cluster repair (adapters/active/pseudo.py, repair.py). No data or weights needed.

  python tests/test_repair.py

1. PoolGraph: Jaccard similarity symmetric and in [0, 1]; on well-separated synthetic identities DBSCAN recovers
   them (pairwise F1 high, oracle.pseudo_report);
2. constraints enforced on the pseudo labels: a "yes" merges two pseudo clusters, a "no" splits one, answered
   clusters are never cut;
3. repair questions: merge / split candidates with their impact; rankings are reproducible permutations, the
   per-cluster quota holds, "repair" orders by expected change; Calibrator falls back before both answers;
4. the point of the method on a camera-split toy domain (every person = one cluster per camera): with the same
   budget, "repair" repairs more of the clustering (pairwise F1) than "repair_random" and the kNN "random";
5. end to end with a random tiny ViT: --pseudo True runs (none / repair / cover / disagree), warm start, the
   cluster-memory loss, offline simulation (tune=False) with a feature cache.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from adapters.active.constraints import ConstraintStore
from adapters.active.oracle import PairOracle
from adapters.active.pseudo import PoolGraph, label_clusters
from adapters.active.repair import Calibrator, expected_change, rank_repair, repair_candidates


def blobs(n_people=30, n_cams=2, per_cam=4, cam_shift=0.0, noise=0.15, dim=32, seed=0):
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_people, dim)
    shifts = rng.randn(n_cams, n_people, dim) * cam_shift  # person-specific camera shift (pose, light)
    X, pids, cams = [], [], []
    for p in range(n_people):
        for c in range(n_cams):
            for _ in range(per_cam):
                X.append(centers[p] + shifts[c, p] + noise * rng.randn(dim))
                pids.append(p)
                cams.append(c)
    X = np.array(X, np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return torch.from_numpy(X), np.array(pids), np.array(cams)


def test_graph():
    X, pids, cams = blobs()
    g = PoolGraph(X, k1=10, k2=3, eps=0.6, min_samples=3)
    a = np.random.RandomState(0).randint(len(X), size=200)
    b = np.random.RandomState(1).randint(len(X), size=200)
    s1, s2 = g.jaccard_sim(a, b), g.jaccard_sim(b, a)
    assert np.allclose(s1, s2) and (s1 >= -1e-9).all() and (s1 <= 1 + 1e-9).all()
    assert np.allclose(g.jaccard_sim(a, a), 1)
    labels = g.cluster()
    rep = PairOracle(pids, cams).pseudo_report(labels)
    assert rep["pw_f"] > 0.9, rep
    print("graph: {} clusters, pairwise F1 {:.3f}, NMI {:.3f}, outliers {:.2f}: ok".format(
        rep["pseudo_clusters"], rep["pw_f"], rep["nmi"], rep["pseudo_outliers"]))


def test_constraints_on_clusters():
    X, pids, cams = blobs(n_people=6, per_cam=4)
    g = PoolGraph(X, k1=10, k2=3, eps=0.6, min_samples=3)
    base = g.cluster()
    cl = label_clusters(base)
    assert len(cl) >= 2
    s = ConstraintStore()
    s.add(cl[0][0], cl[1][0], True)          # a (false) "yes": the two pseudo clusters merge
    lab = g.cluster(s)
    assert lab[cl[0][0]] == lab[cl[1][0]] and len(set(lab[cl[0] + cl[1]].tolist())) == 1
    s = ConstraintStore()
    c = cl[0]
    s.add(c[0], c[1], True)
    s.add(c[1], c[-1], False)                # a "no" inside one pseudo cluster: it is split
    s.add(c[-1], c[-2], True)
    lab = g.cluster(s)
    assert lab[c[0]] == lab[c[1]] and lab[c[-1]] == lab[c[-2]] and lab[c[0]] != lab[c[-1]]
    assert (lab[c] >= 0).all()
    print("constraints on pseudo labels: merge / split enforced: ok")


def test_repair_candidates():
    X, pids, cams = blobs(n_people=20, cam_shift=0.9)
    g = PoolGraph(X, k1=8, k2=2, eps=0.5, min_samples=3)
    labels = g.cluster()
    Xn = X.numpy()
    cand = repair_candidates(Xn, labels, k_merge=3)
    n = len(cand["i"])
    assert n > 0 and (cand["kind"] == 0).any()
    assert (cand["i"] < cand["j"]).all() and (cand["impact"] >= 1).all()
    assert np.allclose(cand["sim"], (Xn[cand["i"]] * Xn[cand["j"]]).sum(1), atol=1e-5)
    merge = np.flatnonzero((cand["kind"] == 0) & (labels[cand["i"]] >= 0) & (labels[cand["j"]] >= 0))
    sizes = np.bincount(labels[labels >= 0])
    for p in merge[:20]:
        assert labels[cand["i"][p]] != labels[cand["j"][p]]
        assert cand["impact"][p] == min(sizes[labels[cand["i"][p]]], sizes[labels[cand["j"][p]]])
    cal = Calibrator([], [], tau=0.5)
    assert abs(cal(0.5) - 0.5) < 1e-9 and cal(0.9) > cal(0.1)
    cal = Calibrator(np.r_[np.linspace(0.6, 1, 10), np.linspace(0, 0.4, 10)], [True] * 10 + [False] * 10, tau=0.35)
    assert cal(0.95) > 0.5 > cal(0.05)
    p = cal(cand["sim"])
    for mode in ("repair", "repair_unc", "repair_random"):
        o1 = rank_repair(cand, p, np.random.RandomState(4), mode)
        o2 = rank_repair(cand, p, np.random.RandomState(4), mode)
        assert np.array_equal(o1, o2) and sorted(o1.tolist()) == list(range(n)), mode
    o = rank_repair(cand, p, np.random.RandomState(4), "repair", per_cluster=1)
    used = {}
    head = []
    for q in o:
        ga, gb = int(cand["ga"][q]), int(cand["gb"][q])
        if used.get(ga, 0) or used.get(gb, 0):
            break
        used[ga] = used[gb] = 1
        head.append(q)
    ec = expected_change(cand, p)[head]
    assert len(head) > 0 and np.all(np.diff(ec) <= 1e-9), "repair head not sorted by expected change"
    print("repair candidates: {} merge, {} split questions; rankings / quota / calibration: ok".format(
        int((cand["kind"] == 0).sum()), int((cand["kind"] == 1).sum())))


def _simulate(strategy, X, pids, cams, budget, rounds, seed):
    """The loop's question phase on fixed features (as ActiveRun.run(tune=False)), without a model."""
    from adapters.active.candidates import candidate_pairs, random_pair_quantile, fit_threshold
    from adapters.active.pair_selection import STRATEGIES, SelectionContext
    g = PoolGraph(X, k1=8, k2=2, eps=0.5, min_samples=3)
    oracle, store, rng = PairOracle(pids, cams), ConstraintStore(), np.random.RandomState(seed)
    Xn = X.numpy()
    dup_tau = random_pair_quantile(Xn, 0.99)
    for _ in range(rounds):
        sims = np.array([Xn[i] @ Xn[j] for i, j, _ in store.answers]) if store.answers else np.zeros(0)
        lab = np.array([s for *_, s in store.answers], bool)
        tau = fit_threshold(sims, lab, dup_tau) if len(sims) else dup_tau
        if strategy.startswith("repair"):
            cand = repair_candidates(Xn, g.cluster(store), k_merge=3, cams=cams)
            order = rank_repair(cand, Calibrator(sims, lab, tau)(cand["sim"]), rng, strategy)
        else:
            cand = candidate_pairs(X, cams, True, k=5)
            order = STRATEGIES[strategy](cand, SelectionContext(store, budget, rng, Xn, cams, tau, dup_tau))
        asked = 0
        for p in order:
            if asked >= budget:
                break
            i, j = int(cand["i"][p]), int(cand["j"][p])
            if store.infer(i, j) is None:
                store.add(i, j, oracle.same(i, j))
                asked += 1
    return oracle.pseudo_report(g.cluster(store))["pw_f"]


def test_camera_split_toy():
    X, pids, cams = blobs(n_people=40, n_cams=3, per_cam=4, cam_shift=0.9, noise=0.12)
    f0 = PairOracle(pids, cams).pseudo_report(PoolGraph(X, k1=8, k2=2, eps=0.5, min_samples=3).cluster())["pw_f"]
    res = {s: np.mean([_simulate(s, X, pids, cams, budget=15, rounds=2, seed=k) for k in range(3)])
           for s in ("repair", "repair_random", "random")}
    print("camera-split toy, pairwise F1: no answers {:.3f} | {}".format(
        f0, " | ".join("{} {:.3f}".format(k, v) for k, v in res.items())))
    assert res["repair"] > max(res["repair_random"], res["random"]) and res["repair"] > f0


def test_end_to_end():
    import transformers
    from adapters.args_reid import ReIDTrainingArguments
    from adapters.baseline_model import VPTReIDModel
    from adapters.active.image_store import TargetSplit, features
    from adapters.active.loop import ActiveConfig, ActiveRun
    from tests.test_active import make_images, _DS

    torch.manual_seed(0)
    args = transformers.HfArgumentParser(ReIDTrainingArguments).parse_args_into_dataclasses(
        ["--output_dir", "/tmp/ferreid_test", "--report_to", "none", "--model_type", "vpt", "--backbone", "tiny_test",
         "--num_vpt_tokens", "4", "--lora_layers", "2", "--lora_rank", "8", "--source_domain_tokens", "3",
         "--num_source_domains", "3"])[0]
    model = VPTReIDModel(args).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    with tempfile.TemporaryDirectory() as tmp:
        train = make_images(tmp, 12, 3, 2, seed=1)
        test = make_images(tmp, 6, 3, 1, seed=2, start_pid=100)
        split = TargetSplit(_DS(train, [x for x in test if x[2] == 0], [x for x in test if x[2] != 0]),
                            "synthetic", "cpu", has_cameras=True)
        cfg = ActiveConfig(rounds=2, budget=6, candidate_k=4, steps=3, ids_per_batch=4, pseudo=True, warm_start=True,
                           pseudo_k1=6, pseudo_k2=2, pseudo_min_samples=2, pseudo_eps=0.7)
        for strategy in ("none", "repair", "repair_random", "cover", "disagree"):
            run = ActiveRun(model, split, strategy, cfg, seed=0)
            rows = run.run()
            assert [r["round"] for r in rows] == [1, 2] and all(np.isfinite(r["mAP"]) for r in rows), strategy
            assert "pw_f" in rows[-1] and "ans_true_ids" in rows[-1]
            assert rows[-1]["n_queries"] == (0 if strategy == "none" else rows[-1]["n_queries"])
            assert run.prompt.shape[2] == 4 + 3
            if rows[-1]["n_train_ids"] > 0:
                assert np.isfinite(rows[-1]["loss_start"]), strategy
        assert rows[-1]["n_queries"] <= 12
        try:
            ActiveRun(model, split, "none", ActiveConfig(), seed=0)
            raise AssertionError("none without pseudo accepted")
        except ValueError:
            pass
        cfg_np = ActiveConfig(rounds=1, budget=6, candidate_k=4, steps=2, ids_per_batch=4, pseudo_k1=6, pseudo_k2=2)
        row = ActiveRun(model, split, "disagree", cfg_np, seed=0).run()[0]  # graph without pseudo training
        assert row["n_queries"] <= 6
        feats = features(model, split.pool, model.default_prompt())
        run = ActiveRun(model, split, "repair", cfg, seed=1)
        run.features_cache = feats
        rows = run.run(tune=False)
        assert len(rows) == 2 and rows[-1]["n_queries"] <= 12 and "pw_f" in rows[-1]
        assert run.prompt is run._prompt0
        row = ActiveRun(model, split, "random", cfg, seed=0).run(oracle_all=True)[0]
        assert row["true_ids"] == 12 and np.isfinite(row["mAP"])
    print("end to end (--pseudo True): none / repair / repair_random / cover / disagree, offline simulation, "
          "oracle bound: ok")


def main():
    test_graph()
    test_constraints_on_clusters()
    test_repair_candidates()
    test_camera_split_toy()
    test_end_to_end()
    print("PASS")


if __name__ == "__main__":
    main()
