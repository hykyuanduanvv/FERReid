"""Checks of the torch (GPU) implementations against the CPU references they replace. CPU-only, no data.

  python tests/test_gpu_ops.py

1. image_store.retrieval_metrics equals torchreid's evaluate_rank (market protocol);
2. pseudo.dbscan_graph equals sklearn DBSCAN on the same sparse graph (border ties aside);
3. prompt_tuning.augment: shapes, flips / shifts / erasing rates, reproducible for a generator;
4. ClusterMemory.update: one momentum step per cluster with the batch mean.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch


def test_metrics():
    from torchreid.metrics import evaluate_rank
    from adapters.active.image_store import retrieval_metrics
    rng = np.random.RandomState(0)
    for trial in range(3):
        P, nq, ng, D = 30, 80, 400, 16
        c = rng.randn(P, D)
        qp, gp = rng.randint(P, size=nq), rng.randint(P, size=ng)
        qc, gc = rng.randint(4, size=nq), rng.randint(4, size=ng)
        qf = torch.nn.functional.normalize(torch.tensor(c[qp] + 0.9 * rng.randn(nq, D), dtype=torch.float32), dim=1)
        gf = torch.nn.functional.normalize(torch.tensor(c[gp] + 0.9 * rng.randn(ng, D), dtype=torch.float32), dim=1)
        r1, mAP = retrieval_metrics(qf, gf, qp, gp, qc, gc, chunk=17)
        dist = (1 - qf @ gf.T).numpy()
        cmc, ref = evaluate_rank(dist, qp, gp, qc, gc, max_rank=10, use_cython=False)
        assert abs(mAP - ref * 100) < 1e-3 and abs(r1 - cmc[0] * 100) < 1e-3, (mAP, ref * 100, r1, cmc[0] * 100)
    print("retrieval metrics == torchreid evaluate_rank: ok (mAP {:.2f})".format(mAP))


def test_dbscan():
    import scipy.sparse as sp
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import adjusted_rand_score
    from adapters.active.pseudo import PoolGraph, dbscan_graph
    rng = np.random.RandomState(1)
    c = rng.randn(60, 32)
    pid = rng.randint(60, size=900)
    X = torch.nn.functional.normalize(torch.tensor(c[pid] + 0.5 * rng.randn(900, 32), dtype=torch.float32), dim=1)
    g = PoolGraph(X, k1=15, k2=4, eps=0.6, min_samples=4)
    a, b, d = g._dist
    for eps, ms in ((0.6, 4), (0.5, 3), (0.7, 6)):
        lab = dbscan_graph(a, b, d, g.n, eps, ms).numpy()
        A, B, Dd = a.numpy(), b.numpy(), d.numpy()
        M = sp.coo_matrix((np.r_[Dd, Dd], (np.r_[A, B], np.r_[B, A])), shape=(g.n, g.n))
        rows, cols, vals = M.row, M.col, M.data
        key = rows.astype(np.int64) * g.n + cols
        o = np.lexsort((vals, key)); first = np.r_[True, key[o][1:] != key[o][:-1]]
        M = sp.csr_matrix((vals[o][first], (rows[o][first], cols[o][first])), shape=(g.n, g.n))
        M = M + sp.diags(np.full(g.n, 1e-9))  # sklearn counts stored entries only: store the point itself
        ref = DBSCAN(eps=eps, min_samples=ms, metric="precomputed").fit_predict(M.tocsr())
        ari = adjusted_rand_score(ref, lab)
        assert ((lab >= 0) == (ref >= 0)).all() and ari > 0.99, (eps, ms, ari)
    print("dbscan_graph == sklearn DBSCAN: ok (ARI {:.4f}, {} clusters)".format(ari, len(set(lab[lab >= 0]))))
    s1 = g.jaccard_sim([0, 5, 9], [7, 5, 2]); s2 = g.jaccard_sim([7, 5, 2], [0, 5, 9])
    assert np.allclose(s1, s2) and abs(s1[1] - 1) < 1e-5


def test_augment_and_memory():
    from adapters.active.prompt_tuning import augment, ClusterMemory
    x = torch.randn(64, 3, 256, 128) + 5  # real crop size, no zeros in the input
    o1 = augment(x, torch.Generator().manual_seed(3))
    o2 = augment(x, torch.Generator().manual_seed(3))
    assert o1.shape == x.shape and torch.equal(o1, o2)
    zero_frac = (o1 == 0).float().mean().item()  # padding of the shift + erasing
    erased = ((o1 == 0).flatten(1).float().mean(1) > 0.12).float().mean().item()  # shift alone pads < 12%
    assert 0.05 < zero_frac < 0.3 and 0.15 < erased < 0.7, (zero_frac, erased)
    xs = torch.arange(128.0).repeat(64, 3, 256, 1) + 1
    flipped = (augment(xs, torch.Generator().manual_seed(0))[:, 0, 128, 40:60].diff(dim=1) < 0).any(1).float().mean()
    m = ClusterMemory(torch.eye(4, 8), momentum=0.5)
    f = torch.nn.functional.normalize(torch.tensor([[0., 1, 0, 0, 0, 0, 0, 0], [0., 1, 0, 0, 0, 0, 0, 0]]), dim=1)
    m.update(f, torch.tensor([0, 0]))
    assert torch.allclose(m.M[0], torch.nn.functional.normalize(torch.tensor([.5, .5, 0, 0, 0, 0, 0, 0]), dim=0))
    assert torch.equal(m.M[1:], torch.eye(4, 8)[1:])
    assert 0.3 < float(flipped) < 0.7, float(flipped)
    print("augment (zeros {:.3f}, erased images {:.2f}, flipped {:.2f}) and memory update: ok".format(
        zero_frac, erased, float(flipped)))


if __name__ == "__main__":
    test_metrics()
    test_dbscan()
    test_augment_and_memory()
    print("PASS")
