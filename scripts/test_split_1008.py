"""E0 (task list v2): split each target's test set by identity into a development half and a final half.

Every identity of query + gallery (gallery-only distractor identities included, so both halves keep a comparable
gallery) is assigned to one half with a fixed seed; query and gallery use the same assignment. Evaluation on a
half = its queries against its gallery (scripts/eval_rounds_1008.py). Labels of the pool (train split) are not used.

  python scripts/test_split_1008.py data_manifests/test_split_dev_final.json        (CPU)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hashlib
import json

import numpy as np

from adapters.config_reid import DOMAIN_CONFIG
from adapters.trainer_reid import _get_dataset_cls

SEED = 20261008
DOMAINS = ("cuhk03", "msmt17", "market1501")


def main(out):
    res = {"seed": SEED, "method": "identity-level 50/50 split of query+gallery identities (distractors included); "
           "query and gallery share the assignment", "domains": {}}
    for name in DOMAINS:
        ds = _get_dataset_cls(name)(root=DOMAIN_CONFIG["data_root"], verbose=False)
        qp = np.array([x[1] for x in ds.query])
        gp = np.array([x[1] for x in ds.gallery])
        ids = np.unique(np.concatenate([qp, gp]))
        rng = np.random.RandomState(SEED)
        perm = rng.permutation(ids)
        dev = np.sort(perm[: len(ids) // 2])
        final = np.sort(perm[len(ids) // 2:])
        dq, dg = np.isin(qp, dev), np.isin(gp, dev)
        qids = np.unique(qp)
        stats = {"ids_total": int(len(ids)), "query_ids": int(len(qids)),
                 "dev": {"ids": int(len(dev)), "query_ids": int(np.isin(qids, dev).sum()),
                         "query_imgs": int(dq.sum()), "gallery_imgs": int(dg.sum())},
                 "final": {"ids": int(len(final)), "query_ids": int((~np.isin(qids, dev)).sum()),
                           "query_imgs": int((~dq).sum()), "gallery_imgs": int((~dg).sum())}}
        res["domains"][name] = {"dev_pids": dev.tolist(), "final_pids": final.tolist(), "stats": stats}
        print(name, json.dumps(stats), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    blob = json.dumps(res, sort_keys=True).encode()
    open(out, "wb").write(blob)
    sha = hashlib.sha256(blob).hexdigest()
    open(out + ".sha256", "w").write(sha + "  " + os.path.basename(out) + "\n")
    print("written", out, "sha256", sha)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data_manifests/test_split_dev_final.json")
