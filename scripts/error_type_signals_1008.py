"""T5 (task order 10-08): label-free signals of the dominant pseudo-label error (impure / merged vs split) on the
round-0 exports. The signals use features, camera ids and the round-0 clustering only; person ids are read
afterwards to compute the truth (pairwise precision / recall, per-cluster purity) and the per-cluster AUCs.
Evaluation only: nothing here selects a parameter.

  python scripts/error_type_signals_1008.py <round0 dir> <out dir>      (CPU, torch threads from OMP_NUM_THREADS)

Signals (dataset level unless noted):
  S1 cluster size: median / p90; images per identity estimated label-free as (same-camera near-duplicate group
     size) x (median cameras per cluster); ratio = median cluster size / that estimate. Near-duplicate group:
     connected components of same-camera pairs with cosine >= the 0.999 quantile of random cross-camera pairs.
  S2 cameras per cluster: mean / median; per cluster: number of cameras (score for "impure").
  S3 bimodality of the within-cluster pairwise cosine (Sarle's coefficient, clusters >= 4 images): mean; per cluster
     (score). Also per cluster: 1 - min cos(member, centroid) (the off:member suspicion of the worst member).
  S4 cross-cluster mutual neighbours: share of an image's mutual 10-NN (clustered images) in another cluster
     (signal of splits): mean over clustered images.
"""
import json
import os
import sys

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import kurtosis, skew
from sklearn.metrics import roc_auc_score

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))


def knn(X, k, chunk=4096):
    Xt = torch.from_numpy(X)
    out = []
    for s in range(0, len(X), chunk):
        S = Xt[s:s + chunk] @ Xt.T
        S[torch.arange(S.size(0)), torch.arange(s, s + S.size(0))] = -2
        out.append(S.topk(k, dim=1).indices.numpy())
    return np.concatenate(out)


def dup_groups(X, cams, tau, chunk=4096):
    """Mean size of same-camera near-duplicate groups (connected components over cos >= tau)."""
    rows, cols = [], []
    Xt = torch.from_numpy(X)
    ct = torch.from_numpy(cams)
    for s in range(0, len(X), chunk):
        S = Xt[s:s + chunk] @ Xt.T
        m = (S >= tau) & (ct[s:s + chunk, None] == ct[None, :])
        r, c = torch.nonzero(m, as_tuple=True)
        rows.append(r.numpy() + s)
        cols.append(c.numpy())
    r, c = np.concatenate(rows), np.concatenate(cols)
    A = coo_matrix((np.ones(len(r)), (r, c)), shape=(len(X), len(X)))
    n, lab = connected_components(A, directed=False)
    return float(np.bincount(lab).mean()), float(np.median(np.bincount(lab)))


def sarle(v):
    n = len(v)
    if n < 6 or np.std(v) < 1e-9:
        return np.nan
    g, k = skew(v), kurtosis(v)  # excess kurtosis
    return (g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def run(npz):
    z = np.load(npz)
    X = z["features"].astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    lab, cams = z["labels"], z["cams"]
    rng = np.random.RandomState(0)
    a, b = rng.randint(len(X), size=200000), rng.randint(len(X), size=200000)
    keep = cams[a] != cams[b]
    rnd = (X[a[keep]] * X[b[keep]]).sum(1)
    tau = float(np.quantile(rnd, 0.999))
    dup_mean, dup_med = dup_groups(X, cams, tau)
    clusters = [np.flatnonzero(lab == c) for c in np.unique(lab[lab >= 0])]
    size = np.array([len(m) for m in clusters])
    ncam = np.array([len(np.unique(cams[m])) for m in clusters])
    bim, worst = [], []
    for m in clusters:
        S = X[m] @ X[m].T
        bim.append(sarle(S[np.triu_indices(len(m), 1)]) if len(m) >= 4 else np.nan)
        cent = X[m].mean(0)
        cent /= np.linalg.norm(cent)
        worst.append(float(1 - (X[m] @ cent).min()))
    bim, worst = np.array(bim), np.array(worst)
    nn = knn(X, 10)
    mutual = np.zeros(len(X))
    ok = lab >= 0
    for i in np.flatnonzero(ok):
        mu = [j for j in nn[i] if i in nn[j] and lab[j] >= 0]
        mutual[i] = np.mean([lab[j] != lab[i] for j in mu]) if mu else np.nan
    est_ids = dup_mean * np.median(ncam)
    sig = {"n_images": len(X), "n_clusters": len(clusters), "outlier_share": float(1 - ok.mean()),
           "S1_size_median": float(np.median(size)), "S1_size_p90": float(np.quantile(size, .9)),
           "S1_dup_group_mean": dup_mean, "S1_imgs_per_id_est": float(est_ids),
           "S1_ratio": float(np.median(size) / est_ids), "S2_cams_mean": float(ncam.mean()),
           "S2_cams_median": float(np.median(ncam)), "S2_cams_total": int(len(np.unique(cams))),
           "S3_bimodality_mean": float(np.nanmean(bim)), "S3_worst_member_susp_mean": float(worst.mean()),
           "S4_cross_cluster_mutual": float(np.nanmean(mutual[ok])), "tau_dup": tau}
    # ---------------------------------------------------------------- truth (person ids, evaluation only)
    pids = z["pids"]
    y, c = pids[ok], lab[ok]
    pair = lambda n: (n * (n - 1) // 2).sum()
    _, joint = np.unique(np.stack([y, c]), axis=1, return_counts=True)
    tp, pred, true = pair(joint), pair(np.unique(c, return_counts=True)[1]), pair(np.unique(pids, return_counts=True)[1])
    pur = np.array([np.unique(pids[m], return_counts=True)[1].max() / len(m) for m in clusters])
    mixed = (pur < 1).astype(int)
    per_id = np.unique(pids, return_counts=True)[1]
    cams_per_id = np.array([len(np.unique(cams[pids == p])) for p in np.unique(pids)])
    truth = {"T_pw_prec": float(tp / max(pred, 1)), "T_pw_rec": float(tp / max(true, 1)),
             "T_purity_img": float((pur * size).sum() / size.sum()), "T_mixed_cluster_share": float(mixed.mean()),
             "T_imgs_per_id": float(per_id.mean()), "T_cams_per_id": float(cams_per_id.mean()),
             "T_error_type": "impure (merge too much)" if tp / max(pred, 1) < tp / max(true, 1) else "split"}
    auc = {}
    for name, s in (("S2_ncam", ncam), ("S3_bimodality", bim), ("S3_worst_member", worst), ("S1_size", size)):
        m = np.isfinite(s)
        if 0 < mixed[m].sum() < m.sum():
            auc["AUC_mixed_" + name] = float(roc_auc_score(mixed[m], s[m]))
    return {**sig, **truth, **auc}


if __name__ == "__main__":
    root, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    res = {}
    for d in ("cuhk03", "msmt17", "market1501"):
        p = os.path.join(root, d, d + "_epoch0.npz")
        if os.path.exists(p):
            res[d] = run(p)
            print(d, json.dumps(res[d], indent=1), flush=True)
    json.dump(res, open(os.path.join(out, "error_type_signals.json"), "w"), indent=1)
    keys = list(next(iter(res.values())).keys())
    with open(os.path.join(out, "error_type_signals.md"), "w") as f:
        f.write("| signal | " + " | ".join(res) + " |\n|---|" + "---:|" * len(res) + "\n")
        for k in keys:
            f.write("| {} | ".format(k) + " | ".join(
                ("{:.3f}".format(v[k]) if isinstance(v[k], float) else str(v[k])) for v in res.values()) + " |\n")
