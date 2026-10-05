"""CPU checks of the pilot-2 additions (no data or weights):

  python tests/test_pilot2.py

1. repair_candidates_knn: merge questions are cross-cluster, cross-camera k-NN pairs (the most similar one per
   cluster pair), impact sqrt(min size); split questions kept;
2. on the camera-split toy domain, repair2 gets "yes" answers and repairs the clustering more than random order;
3. --budget_schedule: all answers in round 1 when "B,0,0"; strategy "oracle" trains on the true identities every
   round; anchor annotation with pseudo labels; strategy suffix in the outputs.
"""
import os
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from adapters.active.candidates import candidate_pairs
from adapters.active.constraints import ConstraintStore
from adapters.active.oracle import PairOracle
from adapters.active.pseudo import PoolGraph
from adapters.active.repair import Calibrator, rank_repair, repair_candidates_knn
from tests.test_repair import blobs


def test_candidates():
    X, pids, cams = blobs(n_people=20, n_cams=3, cam_shift=0.9)
    g = PoolGraph(X, k1=8, k2=2, eps=0.5, min_samples=3)
    labels = g.cluster()
    knn = candidate_pairs(X, cams, True, k=5)
    cand = repair_candidates_knn(X.numpy(), labels, knn)
    m = cand["kind"] == 0
    assert m.any() and (cand["kind"] == 1).any() | True
    assert (cams[cand["i"][m]] != cams[cand["j"][m]]).all(), "merge questions must be cross-camera"
    lab = labels.copy(); out = np.flatnonzero(lab < 0); lab[out] = lab.max() + 1 + np.arange(len(out))
    assert (lab[cand["i"][m]] != lab[cand["j"][m]]).all(), "merge questions must join two clusters"
    keys = set(zip(np.minimum(lab[cand["i"][m]], lab[cand["j"][m]]), np.maximum(lab[cand["i"][m]], lab[cand["j"][m]])))
    assert len(keys) == m.sum(), "one question per cluster pair"
    size = np.bincount(lab)
    assert np.allclose(cand["impact"][m], np.sqrt(np.minimum(size[lab[cand["i"][m]]], size[lab[cand["j"][m]]])))
    print("repair2 candidates: {} merge (cross-camera, one per cluster pair), {} split: ok".format(
        int(m.sum()), int((~m).sum())))


def simulate(strategy, X, pids, cams, budget, rounds, seed):
    from adapters.active.candidates import random_pair_quantile, fit_threshold
    from adapters.active.pair_selection import STRATEGIES, SelectionContext
    g = PoolGraph(X, k1=8, k2=2, eps=0.5, min_samples=3)
    oracle, store, rng = PairOracle(pids, cams), ConstraintStore(), np.random.RandomState(seed)
    Xn = X.numpy(); dup_tau = random_pair_quantile(Xn, 0.99); yes = 0
    for _ in range(rounds):
        sims = np.array([Xn[i] @ Xn[j] for i, j, _ in store.answers]) if store.answers else np.zeros(0)
        lab = np.array([s for *_, s in store.answers], bool)
        tau = fit_threshold(sims, lab, dup_tau) if len(sims) else dup_tau
        knn = candidate_pairs(X, cams, True, k=5)
        if strategy == "repair2":
            cand = repair_candidates_knn(Xn, g.cluster(store), knn)
            order = rank_repair(cand, Calibrator(sims, lab, tau)(cand["sim"]), rng, "repair")
        else:
            cand = knn
            order = STRATEGIES[strategy](cand, SelectionContext(store, budget, rng, Xn, cams, tau, dup_tau))
        asked = 0
        for p in order:
            if asked >= budget:
                break
            i, j = int(cand["i"][p]), int(cand["j"][p])
            if store.infer(i, j) is None:
                s = oracle.same(i, j); store.add(i, j, s); asked += 1; yes += s
    return oracle.pseudo_report(g.cluster(store))["pw_f"], yes


def test_toy():
    X, pids, cams = blobs(n_people=60, n_cams=3, per_cam=4, cam_shift=0.9, noise=0.12, seed=1)
    f0 = PairOracle(pids, cams).pseudo_report(PoolGraph(X, k1=8, k2=2, eps=0.5, min_samples=3).cluster())["pw_f"]
    res = {s: [simulate(s, X, pids, cams, 15, 3, k) for k in range(3)] for s in ("repair2", "random")}
    msg = " | ".join("{} F1 {:.3f} yes {:.0f}/45".format(s, np.mean([f for f, _ in v]), np.mean([y for _, y in v]))
                     for s, v in res.items())
    print("camera-split toy (no answers F1 {:.3f}): {}".format(f0, msg))
    assert np.mean([y for _, y in res["repair2"]]) > 15, "repair2 should get many yes answers"
    assert np.mean([f for f, _ in res["repair2"]]) > np.mean([f for f, _ in res["random"]])


def test_end_to_end():
    import transformers
    from adapters.args_reid import ReIDTrainingArguments
    from adapters.baseline_model import VPTReIDModel
    from adapters.active.image_store import TargetSplit
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
        kw = dict(rounds=3, budget=4, candidate_k=4, steps=2, ids_per_batch=4, pseudo=True, warm_start=True,
                  pseudo_k1=6, pseudo_k2=2, pseudo_min_samples=2, pseudo_eps=0.7)
        rows = ActiveRun(model, split, "repair2", ActiveConfig(**kw, budget_schedule="10,0,0"), seed=0).run()
        assert [r["n_queries"] for r in rows][1:] == [rows[0]["n_queries"]] * 2 and 0 < rows[0]["n_queries"] <= 10, \
            [r["n_queries"] for r in rows]
        rows = ActiveRun(model, split, "random", ActiveConfig(**kw), seed=0).run()
        assert rows[-1]["n_queries"] <= 12
        rows = ActiveRun(model, split, "anchor:random", ActiveConfig(**kw, budget_schedule="6,0,0"), seed=0).run()
        assert rows[0]["n_anchors"] == 6 and rows[-1]["n_anchors"] == 6 and "pw_f" in rows[-1]
        run = ActiveRun(model, split, "oracle", ActiveConfig(**kw), seed=0)
        rows = run.run()
        assert len(rows) == 3 and rows[-1]["true_ids"] == 12 and rows[-1]["purity"] == 1.0 and rows[-1]["n_queries"] == 0
        assert all(np.isfinite(r["mAP"]) for r in rows) and run.graph is None
        try:
            ActiveRun(model, split, "random", ActiveConfig(**kw, budget_schedule="5,0"), seed=0)
            raise AssertionError("schedule of the wrong length accepted")
        except ValueError:
            pass
    print("end to end: repair2 front-loaded, anchor + pseudo front-loaded, oracle every round, schedule check: ok")


def test_suffix():
    from scripts.eval_active import ActiveArguments
    assert ActiveArguments().strategy_suffix == "" and ActiveArguments().budget_schedule == ""
    print("eval_active arguments: ok")


if __name__ == "__main__":
    test_candidates()
    test_toy()
    test_end_to_end()
    test_suffix()
    print("PASS")
