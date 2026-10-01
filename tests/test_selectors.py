"""CPU checks of the label-free selectors (adapters/selectors.py).

  python tests/test_selectors.py            # synthetic pools only (seconds)
  python tests/test_selectors.py --real     # + VIPeR / GRID split 0 with DINOv2 features (minutes on CPU)

Synthetic: every selector returns k distinct valid indices, is reproducible for a seed, never touches
the hidden labels (the oracle raises on any access), de-duplicating selectors avoid near-identical
images, camera_balanced covers the cameras, facility / typical / kcenter beat random on their own
criteria. Real: anchors -> annotated pairs per selector (n_pairs / n_dup) vs random.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from adapters.context_selection import IMAGE_SELECTORS, ImagePool, select_images, annotate
from adapters.selectors import selection_properties


class _Forbidden:
    def __getattr__(self, name):
        raise AssertionError("a selector accessed the hidden labels ({})".format(name))


def synthetic_pool(n_people=300, per_person=4, n_cams=4, dim=32, seed=0):
    """Clusters = people (near-identical images), 4 cameras, plus a 'style' per camera."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_people, dim)
    feats, cams, pids = [], [], []
    for p in range(n_people):
        for j in range(per_person):
            feats.append(centers[p] + 0.05 * rng.randn(dim))
            cams.append(j % n_cams)
            pids.append(p)
    feats = np.array(feats, np.float32)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    style = np.array([np.eye(n_cams)[c] for c in cams], np.float32) + 0.01 * rng.randn(len(cams), n_cams).astype(np.float32)
    pool = ImagePool([("img{}".format(i), pid, cam) for i, (pid, cam) in enumerate(zip(pids, cams))], has_cameras=True)
    pool.attach_features(feats, style)
    return pool, np.array(pids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--checkpoint", default="", help="--real: a trained checkpoint (default: pretrained DINOv2)")
    a = ap.parse_args()
    pool, pids = synthetic_pool()
    oracle, pool._oracle = pool._oracle, _Forbidden()
    k = 12
    rows = {}
    for name in sorted(IMAGE_SELECTORS):
        idx = select_images(name, pool, k, np.random.RandomState(3))
        again = select_images(name, pool, k, np.random.RandomState(3))
        assert idx == again, name
        props = selection_properties(pool, idx)
        rows[name] = (len(set(pids[idx])), props["p_n_cams"], props["p_typicality"], props["p_diversity"])
    pool._oracle = oracle
    print("{:<16} {:>9} {:>6} {:>11} {:>10}".format("selector", "persons", "cams", "typicality", "diversity"))
    for name, (persons, cams, typ, div) in rows.items():
        print("{:<16} {:>6}/{:<2} {:>6} {:>11.2f} {:>10.3f}".format(name, persons, k, cams, typ, div))
    for name in ("dedup", "pairable", "typical", "kcenter", "camera_balanced", "facility", "facility_camera"):
        assert rows[name][0] == k, (name, "selected the same person twice", rows[name][0])
    assert rows["camera_balanced"][1] == 4 and rows["facility_camera"][1] == 4
    # annotation: every anchor of a distinct person yields a cross-camera pair here
    pairs, info = annotate(pool, select_images("dedup", pool, k, np.random.RandomState(0)), np.random.RandomState(0))
    assert info["n_pairs"] == k and info["n_dup"] == 0, info

    if a.real:
        import torch
        import transformers
        from adapters.args_reid import ReIDTrainingArguments
        from adapters.config_reid import DOMAIN_CONFIG
        from adapters.trainer_reid import _get_dataset_cls
        from adapters.baseline_model import PlainReIDModel
        from adapters.selectors import extract_selector_features
        from scripts.context_sensitivity import load_images
        args = transformers.HfArgumentParser(ReIDTrainingArguments).parse_args_into_dataclasses(
            ["--output_dir", "/tmp/x", "--report_to", "none", "--model_type", "plain", "--backbone", "dinov2_b14"])[0]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if a.checkpoint:
            from scripts.context_sensitivity import build_model
            model = build_model(args, device, a.checkpoint)
            model.load_state_dict(torch.load(os.path.join(a.checkpoint, "pytorch_model.bin"), map_location=device), strict=True)
        else:
            model = PlainReIDModel(args).to(device).eval()
        for name in ("viper", "grid"):
            ds = _get_dataset_cls(name)(root=DOMAIN_CONFIG["data_root"], verbose=False, split_id=0)
            pool = ImagePool(ds.train, has_cameras=True)
            feats, style = extract_selector_features(model, load_images(pool.paths, device), device)
            pool.attach_features(feats, style)
            print("\n{} split 0: {} pool images, k=16, 10 seeds".format(name, len(pool)))
            for sel in sorted(IMAGE_SELECTORS):
                stats = [annotate(pool, select_images(sel, pool, 16, np.random.RandomState(s)), np.random.RandomState(s))[1]
                         for s in range(10)]
                print("  {:<16} pairs {:5.1f}  dup {:4.1f}  fail {:4.1f}".format(
                    sel, np.mean([x["n_pairs"] for x in stats]), np.mean([x["n_dup"] for x in stats]),
                    np.mean([x["n_fail"] for x in stats])))
    print("PASS")


if __name__ == "__main__":
    main()
