"""CPU checks of context selection (needs the datasets under $FERREID_DATA_ROOT; no GPU).

  python tests/test_selection.py [--legacy /path/to/old/adapters/context_selection.py]

1. unit "identity" reproduces the historical select()+make_pairs() random stream exactly
   (compared with an older copy of the module when --legacy is given);
2. unit "image": the public ImagePool has no person ids; every returned pair is two different images
   of one person from different cameras (any two images in NO_CAMERA_DOMAINS); each person at most once;
   k = n_pairs + n_fail + n_dup; same seed -> same result.
"""
import argparse
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from adapters.config_reid import DOMAIN_CONFIG, NO_CAMERA_DOMAINS
from adapters.context_selection import ContextSampler, ImagePool
from adapters.trainer_reid import _get_dataset_cls


def load(name, split_id=0):
    kwargs = {"split_id": split_id} if name in ("viper", "grid", "ilids") else {}
    return _get_dataset_cls(name)(root=DOMAIN_CONFIG["data_root"], verbose=False, **kwargs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--legacy", default="")
    p.add_argument("--domains", default="viper,grid,ilids,market1501,cuhk03")
    a = p.parse_args()
    legacy = None
    if a.legacy:
        spec = importlib.util.spec_from_file_location("legacy_selection", a.legacy)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)

    public = {k for k in vars(ImagePool([("x", 0, 0)])) if not k.startswith("_")}
    assert public == {"paths", "camids", "has_cameras", "feats", "style"}, public  # all label-free
    for name in a.domains.split(","):
        ds = load(name)
        cams = name not in NO_CAMERA_DOMAINS
        sampler = ContextSampler(ds.train, has_cameras=cams)
        info_of = {p: (pid, cam) for p, pid, cam, *_ in ds.train}
        for k in (2, 4, 16):
            for seed in range(5):
                # identity unit == historical stream
                pairs, info = sampler.draw("identity", "random", k, np.random.RandomState(seed))
                if legacy is not None:
                    rng = np.random.RandomState(seed)
                    pool = legacy.CandidatePool(ds.train)
                    old = legacy.make_pairs(pool, legacy.select("random", pool, k, rng), rng)
                    assert pairs == old, (name, k, seed)
                # image unit
                pairs, info = sampler.draw("image", "random", k, np.random.RandomState(seed))
                again, _ = sampler.draw("image", "random", k, np.random.RandomState(seed))
                assert pairs == again
                assert info["k"] == k == info["n_pairs"] + info["n_fail"] + info["n_dup"], info
                persons = []
                for x, y in pairs:
                    (px, cx), (py, cy) = info_of[x], info_of[y]
                    assert x != y and px == py, (x, y)
                    assert (cx != cy) if cams else True, (x, y, cx, cy)
                    persons.append(px)
                assert len(persons) == len(set(persons))
        stats = [sampler.draw("image", "random", 16, np.random.RandomState(s))[1] for s in range(50)]
        print("{:<11} cameras={}  k=16 over 50 seeds: pairs {:.1f}, fail {:.2f}, dup {:.2f}{}".format(
            name, cams, np.mean([s["n_pairs"] for s in stats]), np.mean([s["n_fail"] for s in stats]),
            np.mean([s["n_dup"] for s in stats]), "  identity stream == legacy" if legacy else ""))
    print("PASS")


if __name__ == "__main__":
    main()
