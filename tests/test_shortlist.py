"""CPU checks of shortlist questions (no data or weights):

  python tests/test_shortlist.py

1. on the camera-split toy domain (every person one cluster per camera): costs are counted per candidate shown;
   shortlist K=1 (most similar complementary cluster) gets mostly "yes" and repairs the clustering more than
   random k-NN pairs with the same number of comparisons (on this toy the top candidate is almost always right,
   so K=5 spends 5 comparisons per hit: the K trade-off is measured on real data);
2. end to end with a tiny ViT: shortlist front-loaded and uniform, with and without camera normalisation.
"""
import os
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from adapters.active.loop import ActiveConfig, ActiveRun
from tests.test_repair import blobs


class _Model:  # what ActiveRun needs when tune=False with a feature cache
    def __init__(self):
        self.prompt = torch.zeros(1, 2, 4, 8)
    def default_prompt(self):
        return self.prompt
    def domain_token_init(self):
        return None


class _Split:
    def __init__(self, pids, cams):
        self.pool_pids, self.pool_cams, self.has_cameras = pids, cams, True
        self.pool, self.pool_paths = None, ["x"] * len(pids)


def sim(strategy, X, pids, cams, k, budget, rounds, seed):
    cfg = ActiveConfig(rounds=rounds, budget=budget, candidate_k=5, pseudo=True, pseudo_k1=8, pseudo_k2=2,
                       pseudo_eps=0.5, pseudo_min_samples=3, shortlist_k=k)
    run = ActiveRun(_Model(), _Split(pids, cams), strategy, cfg, seed=seed, log=lambda *a: None)
    run.features_cache = X
    rows = run.run(tune=False)
    return rows[-1]


def test_toy():
    X, pids, cams = blobs(n_people=60, n_cams=3, per_cam=4, cam_shift=0.9, noise=0.12, seed=1)
    res = {}
    for name, strat, k in (("shortlist K=5", "shortlist", 5), ("shortlist K=1", "shortlist", 1), ("random", "random", 5)):
        rows = [sim(strat, X, pids, cams, k, 40, 3, s) for s in range(2)]
        res[name] = (np.mean([r["pw_f"] for r in rows]), np.mean([r["n_queries"] for r in rows]),
                     np.mean([r["n_pos"] for r in rows]))
        assert all(r["n_queries"] <= 120 for r in rows), name
    print("camera-split toy, 3 x 40 comparisons: " + " | ".join(
        "{}: F1 {:.3f} ({:.0f} comparisons, {:.0f} yes)".format(k, *v) for k, v in res.items()))
    assert res["shortlist K=1"][0] > res["random"][0] and res["shortlist K=1"][2] > 0.8 * res["shortlist K=1"][1]
    assert res["shortlist K=5"][1] <= 120 and res["shortlist K=5"][2] > 0


def test_end_to_end():
    import transformers
    from adapters.args_reid import ReIDTrainingArguments
    from adapters.baseline_model import VPTReIDModel
    from adapters.active.image_store import TargetSplit
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
        kw = dict(rounds=3, budget=6, candidate_k=4, steps=2, ids_per_batch=4, pseudo=True, warm_start=True,
                  pseudo_k1=6, pseudo_k2=2, pseudo_min_samples=2, pseudo_eps=0.7, domain_tokens=2, shortlist_k=3)
        for extra in (dict(budget_schedule="12,0,0"), dict(), dict(cam_norm=True, budget_schedule="12,0,0")):
            rows = ActiveRun(model, split, "shortlist", ActiveConfig(**kw, **extra), seed=0).run()
            assert len(rows) == 3 and np.isfinite(rows[-1]["mAP"]) and rows[-1]["n_queries"] <= 18, (extra, rows[-1]["n_queries"])
            if "budget_schedule" in extra:
                assert rows[0]["n_queries"] == rows[-1]["n_queries"] <= 12
    print("end to end: shortlist front-loaded / uniform / + camera normalisation: ok")


if __name__ == "__main__":
    test_toy()
    test_end_to_end()
    print("PASS")
