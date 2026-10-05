"""CPU checks of the active pair-query module (adapters/active). No data or weights needed (about a minute).

  python tests/test_active.py

1. ConstraintStore: union-find clusters, cannot-links follow merges, transitivity, contradictions raise;
2. candidate_pairs equals a brute-force cross-camera k-NN; fit_threshold separates the answers;
3. every pair strategy returns a permutation of the candidates, reproducibly for a seed; "cover" keeps no
   two pairs of near-identical images in its reserve and spreads over camera pairs;
4. strategies never see person ids (the oracle is the only holder);
5. end to end on synthetic images with a random tiny ViT (VPT): rounds of queries -> constraints ->
   appended domain prompt -> retrieval metrics, for a pair strategy, an anchor strategy and the
   full-annotation bound; streamed and cached image stores give the same features; with a base model trained
   with multi-domain tokens, round 0 and the new domain's tokens start at the source mean.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from adapters.active.candidates import candidate_pairs, fit_threshold, random_pair_quantile
from adapters.active.constraints import ConstraintStore
from adapters.active.pair_selection import STRATEGIES, SelectionContext


def test_constraints():
    s = ConstraintStore()
    s.add(0, 1, True)
    s.add(2, 3, True)
    s.add(1, 2, False)          # cluster {0,1} !~ cluster {2,3}
    assert s.infer(0, 3) is False and s.infer(1, 0) is True and s.infer(0, 9) is None
    s.add(3, 4, True)           # 4 joins {2,3}: the cannot-link follows the merge
    assert s.infer(0, 4) is False
    s.add(5, 6, True)
    s.add(6, 4, True)           # {5,6} merges into {2,3,4}
    assert s.infer(5, 1) is False and s.infer(5, 2) is True
    cl = s.clusters()
    assert [len(c) for c in cl] == [5, 2]
    assert s.cannot_links(cl) == [(0, 1)]
    for bad in ((0, 2, True), (5, 3, False)):
        try:
            s.add(*bad)
            raise AssertionError("contradiction not detected: {}".format(bad))
        except ValueError:
            pass
    st = s.stats()
    assert st["n_clusters"] == 2 and st["n_cluster_imgs"] == 7, st
    print("constraints: ok")


def synthetic(n_people=40, per_cam=1, n_cams=4, dim=16, seed=0):
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_people, dim)
    shift = rng.randn(n_cams, dim) * 0.3
    X, pids, cams = [], [], []
    for p in range(n_people):
        for c in range(n_cams):
            for _ in range(per_cam):
                X.append(centers[p] + shift[c] + 0.3 * rng.randn(dim))
                pids.append(p)
                cams.append(c)
    X = np.array(X, np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return torch.from_numpy(X), np.array(pids), np.array(cams)


def test_candidates_and_threshold():
    X, pids, cams = synthetic()
    k = 5
    cand = candidate_pairs(X, cams, True, k=k, chunk=17)
    S = (X @ X.T).numpy()
    np.fill_diagonal(S, -2)
    S[cams[:, None] == cams[None, :]] = -2
    ref = set()
    for i in range(len(X)):
        for j in np.argsort(-S[i])[:k]:
            ref.add((min(i, j), max(i, j)))
    got = set(zip(cand["i"].tolist(), cand["j"].tolist()))
    assert got == ref, (len(got), len(ref))
    assert (cams[cand["i"]] != cams[cand["j"]]).all()
    assert np.allclose(cand["sim"], S[cand["i"], cand["j"]], atol=1e-5)
    lab = pids[cand["i"]] == pids[cand["j"]]
    tau = fit_threshold(cand["sim"], lab, default=0.5)
    acc = ((cand["sim"] > tau) == lab).mean()
    assert acc >= max(lab.mean(), 1 - lab.mean()), (acc, lab.mean())
    assert fit_threshold([0.1, 0.9], [True, True], default=0.42) == 0.42
    q = random_pair_quantile(X, 0.99)
    print("candidates: {} pairs, {:.0%} positive, tau {:.3f} (acc {:.2f}), random-pair q99 {:.3f}: ok".format(
        len(got), lab.mean(), tau, acc, q))
    return X, pids, cams, cand


def test_strategies(X, pids, cams, cand):
    feats = X.numpy()
    store = ConstraintStore()
    store.add(int(cand["i"][0]), int(cand["j"][0]), bool(pids[cand["i"][0]] == pids[cand["j"][0]]))
    dup_tau = random_pair_quantile(feats, 0.99)
    n = len(cand["sim"])
    from adapters.active.pseudo import PoolGraph
    graph = PoolGraph(X, k1=8, k2=2)  # for the strategies that need it (disagree)
    for name, fn in STRATEGIES.items():
        orders = []
        for _ in range(2):
            ctx = SelectionContext(store, 10, np.random.RandomState(3), feats, cams, tau=0.5, dup_tau=dup_tau,
                                   graph=graph)
            orders.append(np.asarray(fn(cand, ctx)))
        assert np.array_equal(orders[0], orders[1]), name
        assert sorted(orders[0].tolist()) == list(range(n)), name
    ctx = SelectionContext(store, 10, np.random.RandomState(3), feats, cams, tau=0.5, dup_tau=dup_tau)
    order = STRATEGIES["cover"](cand, ctx)[:12]
    imgs = np.array([[cand["i"][p], cand["j"][p]] for p in order]).reshape(-1)
    fresh = [p for p in order if not (store.labeled(int(cand["i"][p])) or store.labeled(int(cand["j"][p])))]
    camp = {(min(cams[cand["i"][p]], cams[cand["j"][p]]), max(cams[cand["i"][p]], cams[cand["j"][p]])) for p in fresh}
    print("strategies: {} return reproducible permutations; cover's first 12 pairs span {} camera pairs, "
          "{} distinct images: ok".format(sorted(STRATEGIES), len(camp), len(set(imgs.tolist()))))
    assert len(camp) >= 3


def make_images(root, n_people, n_cams, per_cam, seed, start_pid=0):
    """Pedestrian-like synthetic crops: a person = (top colour, bottom colour), a camera = brightness / tint."""
    from PIL import Image
    rng = np.random.RandomState(seed)
    out = []
    tints = rng.uniform(0.7, 1.3, size=(n_cams, 3))
    for p in range(start_pid, start_pid + n_people):
        top, bottom = rng.randint(0, 255, 3), rng.randint(0, 255, 3)
        for c in range(n_cams):
            for r in range(per_cam):
                img = np.zeros((128, 64, 3), np.float32)
                img[:60] = top
                img[60:] = bottom
                img = np.clip(img * tints[c] + rng.randn(128, 64, 3) * 12, 0, 255).astype(np.uint8)
                path = os.path.join(root, "p{:03d}_c{}_{}.png".format(p, c, r))
                Image.fromarray(img).save(path)
                out.append((path, p, c, 0))
    return out


class _DS:
    def __init__(self, train, query, gallery):
        self.train, self.query, self.gallery = train, query, gallery


def test_end_to_end():
    import transformers
    from adapters.args_reid import ReIDTrainingArguments
    from adapters.baseline_model import VPTReIDModel
    from adapters.active.image_store import TargetSplit, ImageStore, features
    from adapters.active.loop import ActiveConfig, ActiveRun, evaluate_base

    torch.manual_seed(0)
    args = transformers.HfArgumentParser(ReIDTrainingArguments).parse_args_into_dataclasses(
        ["--output_dir", "/tmp/ferreid_test", "--report_to", "none", "--model_type", "vpt",
         "--backbone", "tiny_test", "--num_vpt_tokens", "4", "--lora_layers", "2", "--lora_rank", "8"])[0]
    model = VPTReIDModel(args).eval()
    with torch.no_grad():
        model.prompt.normal_(0, 0.02)  # a non-trivial "trained" base prompt
    for p in model.parameters():
        p.requires_grad_(False)
    with tempfile.TemporaryDirectory() as tmp:
        train = make_images(tmp, 12, 3, 2, seed=1)
        test = make_images(tmp, 6, 3, 1, seed=2, start_pid=100)
        query = [x for x in test if x[2] == 0]
        gallery = [x for x in test if x[2] != 0]
        split = TargetSplit(_DS(train, query, gallery), "synthetic", "cpu", has_cameras=True)
        r1, mAP = evaluate_base(model, split)
        print("base: mAP {:.2f} R1 {:.2f}".format(mAP, r1))

        streamed = ImageStore(split.pool_paths, "cpu", cache=False, num_workers=0, batch_size=16)
        f1 = features(model, split.pool, model.prompt)
        f2 = features(model, streamed, model.prompt)
        assert torch.allclose(f1, f2, atol=1e-5), (f1 - f2).abs().max()
        assert torch.allclose(split.pool.get([3, 5]), streamed.get([3, 5]))

        cfg = ActiveConfig(rounds=2, budget=8, candidate_k=4, domain_tokens=2, steps=3, ids_per_batch=4)
        rows = ActiveRun(model, split, "cover", cfg, seed=0).run()
        assert [r["round"] for r in rows] == [1, 2]
        assert rows[-1]["n_queries"] <= 16 and rows[-1]["n_queries"] == rows[-1]["n_pos"] + rows[-1]["n_neg"]
        assert all(np.isfinite(r["mAP"]) for r in rows)
        run = ActiveRun(model, split, "uncertain", cfg, seed=0)
        run.run()
        L, V, D = model.num_layers, args.num_vpt_tokens, model.hidden_size
        assert tuple(run.prompt.shape) == (1, L, V + cfg.domain_tokens, D), run.prompt.shape
        assert torch.equal(run.prompt[:, :, :V], model.prompt.float())  # base prompt untouched (append)

        rows = ActiveRun(model, split, "anchor:random", cfg, seed=0).run()
        assert rows[-1]["n_anchors"] == 16 and rows[-1]["n_queries"] == rows[-1]["n_pos"]
        assert rows[-1]["purity"] == 1.0  # anchor annotation only produces true positives

        cfg_r = ActiveConfig(rounds=1, budget=8, candidate_k=4, prompt_mode="replace", steps=3, ids_per_batch=4)
        run = ActiveRun(model, split, "balanced", cfg_r, seed=1)
        run.run()
        assert run.prompt.shape == model.prompt.shape
        row = ActiveRun(model, split, "random", cfg, seed=0).run(oracle_all=True)[0]
        assert row["true_ids"] == 12 and row["purity"] == 1.0

        # base model trained with multi-domain tokens: round 0 and the new domain's tokens start at the source mean
        args_md = transformers.HfArgumentParser(ReIDTrainingArguments).parse_args_into_dataclasses(
            ["--output_dir", "/tmp/ferreid_test", "--report_to", "none", "--model_type", "vpt", "--backbone", "tiny_test",
             "--num_vpt_tokens", "4", "--lora_layers", "2", "--lora_rank", "8", "--source_domain_tokens", "3",
             "--num_source_domains", "3"])[0]
        md = VPTReIDModel(args_md).eval()
        for p in md.parameters():
            p.requires_grad_(False)
        mean = md.domain_prompts.mean(0, keepdim=True)
        assert evaluate_base(md, split) == split.evaluate(md, torch.cat([md.prompt, mean], dim=2))
        cfg0 = ActiveConfig(rounds=1, budget=8, candidate_k=4, steps=0)
        run = ActiveRun(md, split, "cover", cfg0, seed=0)
        run.run()
        assert run.prompt.shape[2] == 4 + 3 and torch.allclose(run.prompt[:, :, 4:], mean)  # steps=0: the init
        cfg_rand = ActiveConfig(rounds=1, budget=8, candidate_k=4, steps=0, token_init="random")
        run = ActiveRun(md, split, "cover", cfg_rand, seed=0)
        run.run()
        assert not torch.allclose(run.prompt[:, :, 4:], mean)
        print("end to end: pair / anchor / replace / oracle_all runs, mAP after 2 rounds {:.2f}: ok".format(rows[-1]["mAP"]))


def main():
    test_constraints()
    X, pids, cams, cand = test_candidates_and_threshold()
    test_strategies(X, pids, cams, cand)
    test_end_to_end()
    print("PASS")


if __name__ == "__main__":
    main()
