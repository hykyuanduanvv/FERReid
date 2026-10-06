"""Questions for the VLM from a Cluster Contrast / PASS epoch-0 clustering (examples/vlm_constraints.py dump):
every pseudo cluster vs its K nearest complementary clusters (no camera in common), medoid vs medoid -- the same
rule as scripts/make_vlm_pairs.py --all_k. Person ids only label the pairs for reporting.

  python scripts/pass_questions.py --dump epoch0.npz --out verify_pairs.csv --domain msmt17 --k 3
"""
import argparse
import csv

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--domain", default="msmt17")
    p.add_argument("--k", type=int, default=3)
    a = p.parse_args()
    z = np.load(a.dump, allow_pickle=True)
    X, labels, paths, pids, cams = z["features"], z["labels"], z["paths"], z["pids"], z["cams"]
    C = int(labels.max()) + 1
    members = [np.flatnonzero(labels == c) for c in range(C)]
    cent = np.stack([X[m].mean(0) for m in members])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12
    medoid = np.array([m[np.argmax(X[m] @ cent[c])] for c, m in enumerate(members)])
    cam_idx = np.unique(cams, return_inverse=True)[1].reshape(-1)
    mask = np.zeros((C, cam_idx.max() + 1), bool)
    for c, m in enumerate(members):
        mask[c, cam_idx[m]] = True
    S = cent @ cent.T
    np.fill_diagonal(S, -np.inf)
    S = np.where((mask.astype(np.int32) @ mask.T.astype(np.int32)) > 0, -np.inf, S)
    done, rows = set(), []
    for c in range(C):
        for d in np.argsort(-S[c])[:a.k]:
            if np.isfinite(S[c, d]) and (min(c, d), max(c, d)) not in done:
                done.add((min(c, d), max(c, d)))
                i, j = int(medoid[c]), int(medoid[d])
                rows.append({"domain": a.domain, "kind": "q_all", "i": i, "j": j, "path_a": paths[i], "path_b": paths[j],
                             "same": int(pids[i] == pids[j]), "sim": float(X[i] @ X[j]), "tau": 0.5})
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("{} clusters, {} questions, positive rate {:.3f}".format(C, len(rows), np.mean([r["same"] for r in rows])))


if __name__ == "__main__":
    main()
