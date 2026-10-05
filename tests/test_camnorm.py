"""CPU checks of camera normalisation and the missed-identity diagnostics (no data or weights).

  python tests/test_camnorm.py
"""
import os
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from adapters.active.oracle import PairOracle
from adapters.active.pseudo import PoolGraph, camera_normalize


def camera_shift_toy(n_people=40, n_cams=3, per_cam=4, shift=1.5, noise=0.15, dim=32, seed=0):
    """Every camera adds the same strong offset to all its images (a global camera bias)."""
    rng = np.random.RandomState(seed)
    centers, offs = rng.randn(n_people, dim), rng.randn(n_cams, dim) * shift
    X, pids, cams = [], [], []
    for p in range(n_people):
        for c in range(n_cams):
            for _ in range(per_cam):
                X.append(centers[p] + offs[c] + noise * rng.randn(dim)); pids.append(p); cams.append(c)
    X = torch.nn.functional.normalize(torch.tensor(np.array(X), dtype=torch.float32), dim=1)
    return X, np.array(pids), np.array(cams)


def test_camera_normalize():
    X, pids, cams = camera_shift_toy()
    Y = camera_normalize(X, cams)
    assert torch.allclose(Y.norm(dim=1), torch.ones(len(Y)), atol=1e-5)
    f = {}
    for name, F in (("raw", X), ("camnorm", Y)):
        f[name] = PairOracle(pids, cams).pseudo_report(PoolGraph(F, k1=10, k2=3, eps=0.6, min_samples=3).cluster())["pw_f"]
    print("global camera bias toy, pairwise F1: raw {:.3f} -> camera-normalised {:.3f}".format(f["raw"], f["camnorm"]))
    assert f["camnorm"] > f["raw"] + 0.2


def test_diag():
    from dataclasses import dataclass
    from scripts.diag_missed import analyse

    @dataclass
    class A:
        candidate_k: int = 5; pseudo_k1: int = 10; pseudo_k2: int = 3; pseudo_eps: float = 0.6; pseudo_min_samples: int = 3
    X, pids, cams = camera_shift_toy()
    r_raw = analyse(X, pids, cams, True, A())
    r_cn = analyse(camera_normalize(X, cams), pids, cams, True, A())
    for k in ("clustering", "missed", "candidates", "shortlist_recall", "retrieval"):
        assert k in r_raw
    assert r_raw["missed"]["n_missed_pairs"] > r_cn["missed"]["n_missed_pairs"]
    assert 0 <= r_raw["shortlist_recall"]["all"]["@10"] <= 1
    print("diag_missed: raw missed {} pairs (shortlist @1/@10 {:.2f}/{:.2f}, complementary {:.2f}/{:.2f}) -> camnorm missed {}"
          .format(r_raw["missed"]["n_missed_pairs"], r_raw["shortlist_recall"]["all"]["@1"],
                  r_raw["shortlist_recall"]["all"]["@10"], r_raw["shortlist_recall"]["complementary"]["@1"],
                  r_raw["shortlist_recall"]["complementary"]["@10"], r_cn["missed"]["n_missed_pairs"]))


def test_loop_cam_norm():
    import transformers
    from adapters.args_reid import ReIDTrainingArguments
    from adapters.baseline_model import VPTReIDModel
    from adapters.active.image_store import TargetSplit
    from adapters.active.loop import ActiveConfig, ActiveRun
    from tests.test_active import make_images, _DS
    torch.manual_seed(0)
    args = transformers.HfArgumentParser(ReIDTrainingArguments).parse_args_into_dataclasses(
        ["--output_dir", "/tmp/ferreid_test", "--report_to", "none", "--model_type", "vpt", "--backbone", "tiny_test",
         "--num_vpt_tokens", "4", "--lora_layers", "2", "--lora_rank", "8"])[0]
    model = VPTReIDModel(args).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    with tempfile.TemporaryDirectory() as tmp:
        train = make_images(tmp, 12, 3, 2, seed=1)
        test = make_images(tmp, 6, 3, 1, seed=2, start_pid=100)
        split = TargetSplit(_DS(train, [x for x in test if x[2] == 0], [x for x in test if x[2] != 0]),
                            "synthetic", "cpu", has_cameras=True)
        kw = dict(rounds=2, budget=4, candidate_k=4, steps=2, ids_per_batch=4, pseudo=True, warm_start=True,
                  pseudo_k1=6, pseudo_k2=2, pseudo_min_samples=2, pseudo_eps=0.7, cam_norm=True, domain_tokens=2)
        for s in ("none", "repair2", "anchor:random", "random"):
            rows = ActiveRun(model, split, s, ActiveConfig(**kw), seed=0).run()
            assert len(rows) == 2 and np.isfinite(rows[-1]["mAP"]) and "pw_f" in rows[-1], s
    print("active loop with --cam_norm True (none / repair2 / anchor / random): ok")


if __name__ == "__main__":
    test_camera_normalize()
    test_diag()
    test_loop_cam_norm()
    print("PASS")
