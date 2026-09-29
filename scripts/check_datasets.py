"""Load every configured domain the same way training/evaluation does and check it.

For each domain (and every split of the small target sets): #IDs / #images / #cams
of train (context pool for val/target) / query / gallery vs reference counts, number
of identities eligible as context (seen by >= 2 cameras), and a path-based leak check
between the context pool and query/gallery.
"""
import os
import sys
import warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
warnings.filterwarnings("ignore")

from adapters.config_reid import DOMAIN_CONFIG, NUM_SPLITS
from adapters.context_selection import CandidatePool
from adapters.trainer_reid import _get_dataset_cls

# (ids, images) per subset; None = not checked (varies by split)
REFERENCE = {
    "market1501": {"train": (751, 12936), "query": (750, 3368), "gallery": (751, 15913)},
    "msmt17":     {"train": (1041, 30248), "query": (3060, 11659), "gallery": (3060, 82161)},
    "cuhk03":     {"train": (767, 7368), "query": (700, 1400), "gallery": (700, 5328)},  # CUHK03-NP labeled
    "viper":      {"train": (316, 632), "query": (316, 316), "gallery": (316, 316)},
    "grid":       {"train": (125, 250), "query": (125, 125), "gallery": (126, 900)},
    "ilids":      {"train": (59, None), "query": (60, 60), "gallery": (60, 60)},
    "prid2011":   {"train": (100, 200), "query": (100, 100), "gallery": (649, 649)},
}


def stats(data):
    return len({x[1] for x in data}), len(data), len({x[2] for x in data})


def check(name, split_id=None):
    kwargs = {} if split_id is None else {"split_id": split_id}
    ds = _get_dataset_cls(name)(root=DOMAIN_CONFIG["data_root"], verbose=False, **kwargs)
    ok = True
    parts = []
    for subset in ("train", "query", "gallery"):
        ids, imgs, cams = stats(getattr(ds, subset))
        ref_ids, ref_imgs = REFERENCE.get(name, {}).get(subset, (None, None))
        match = (ref_ids is None or ids == ref_ids) and (ref_imgs is None or imgs == ref_imgs)
        ok &= match
        parts.append("{} {}/{}/{}{}".format(subset, ids, imgs, cams, "" if match else "(!)"))
    pool_paths = {x[0] for x in ds.train}
    leak = len(pool_paths & {x[0] for x in ds.query + ds.gallery})
    eligible = len(CandidatePool(ds.train))
    ok &= leak == 0
    tag = name if split_id is None else "{}[{}]".format(name, split_id)
    print("{:<14} {}  eligible={}  leak={}  {}".format(tag, "  ".join(parts), eligible, leak, "OK" if ok else "MISMATCH"))
    return ok


def main():
    print("format: subset ids/images/cams\n")
    summary = {}
    for role in ("source_domains", "val_domains", "target_domains"):
        print("== {}".format(role))
        for name in DOMAIN_CONFIG[role]:
            try:
                splits = range(NUM_SPLITS[name]) if name in NUM_SPLITS else [None]
                summary[name] = "OK" if all([check(name, s) for s in splits]) else "MISMATCH"
            except Exception as e:
                print("{:<14} FAILED: {}".format(name, e))
                summary[name] = "MISSING"
        print()
    print("Summary:", summary)


if __name__ == "__main__":
    main()
