"""Load every dataset of the leave-one-out protocol the way training / evaluation does and check it.

Per dataset: #IDs / #images / #cams of train (source data; the unlabeled pool when it is the target) /
query / gallery vs reference counts, identities seen by >= 2 cameras, the share of pool images a
cross-camera pair can be formed for (any other image of the person in NO_CAMERA_DOMAINS), and a path-based
leak check between the train split and query / gallery.

  python scripts/check_datasets.py                       # Market1501, MSMT17, CUHK03, CUHK-SYSU
  python scripts/check_datasets.py --domains cuhksysu
"""
import argparse
import os
import sys
import warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
warnings.filterwarnings("ignore")

from adapters.config_reid import DOMAIN_CONFIG, DATASETS, NO_CAMERA_DOMAINS
from adapters.context_selection import CandidatePool
from adapters.trainer_reid import _get_dataset_cls

# (ids, images) per subset; None = not checked (varies by split)
REFERENCE = {
    "market1501": {"train": (751, 12936), "query": (750, 3368), "gallery": (751, 15913)},
    "msmt17":     {"train": (1041, 30248), "query": (3060, 11659), "gallery": (3060, 82161)},
    "cuhk03":     {"train": (767, 7368), "query": (700, 1400), "gallery": (700, 5328)},  # CUHK03-NP labeled
    # CUHK-SYSU (ReID crops): sizes reported by DG-ReID papers, to be confirmed on the data
    "cuhksysu":   {"train": (5532, 15088), "query": (2900, 2900), "gallery": (2900, 5447)},
}


def stats(data):
    return len({x[1] for x in data}), len(data), len({x[2] for x in data})


def pairable_share(train, cross_camera):
    """Share of pool images whose person has another image (from another camera if cross_camera)."""
    by_pid = {}
    for _, pid, cam, *_ in train:
        by_pid.setdefault(pid, []).append(cam)
    ok = 0
    for _, pid, cam, *_ in train:
        cams = by_pid[pid]
        ok += (any(c != cam for c in cams) if cross_camera else len(cams) > 1)
    return ok / max(len(train), 1)


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
    share = pairable_share(ds.train, cross_camera=name not in NO_CAMERA_DOMAINS)
    ok &= leak == 0
    tag = name if split_id is None else "{}[{}]".format(name, split_id)
    print("{:<14} {}  eligible_ids={}  pairable_images={:.1%}  leak={}  {}".format(
        tag, "  ".join(parts), eligible, share, leak, "OK" if ok else "MISMATCH"))
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domains", default="", help="comma-separated domain names (overrides the roles)")
    a = parser.parse_args()
    print("format: subset ids/images/cams  (!) = differs from the reference counts\n")
    if a.domains:
        roles = {"domains": a.domains.split(",")}
    else:
        roles = {"leave-one-out datasets (Market / MSMT17 / CUHK03 are each the target of one fold)": DATASETS}
    summary = {}
    for role, names in roles.items():
        print("== {}".format(role))
        for name in names:
            try:
                summary[name] = "OK" if check(name) else "MISMATCH"
            except Exception as e:
                print("{:<14} FAILED: {}".format(name, e))
                summary[name] = "MISSING"
        print()
    print("Summary:", summary)


if __name__ == "__main__":
    main()
