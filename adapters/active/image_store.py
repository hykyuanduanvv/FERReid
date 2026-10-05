"""Images of one subset (pool / query / gallery) for small and large target domains.

Small sets (VIPeR, GRID, i-LIDS: hundreds of images) are decoded once and kept on the device. Large sets
(Market-1501, MSMT17 as targets: 10^4 - 10^5 images) are streamed from disk for feature extraction; only
the images that prompt tuning needs are decoded on demand and kept in host memory (FP16), up to `host_max`
images -- with pseudo labels prompt tuning draws from the whole pool, so beyond the cap images are decoded
for every batch instead of being kept.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from adapters.reid_dataset import EVAL_TRANSFORM, PathDataset


def _decode(path):
    return EVAL_TRANSFORM(Image.open(path).convert("RGB"))


def load_images(paths, device=None, dtype=None, threads=16):
    with ThreadPoolExecutor(threads) as ex:
        imgs = torch.stack(list(ex.map(_decode, paths)))
    if dtype is not None:
        imgs = imgs.to(dtype)
    return imgs.to(device) if device is not None else imgs


class ImageStore:
    """cache=None: decode everything onto the device when len(paths) <= cache_max, else stream."""

    def __init__(self, paths, device, cache=None, cache_max=6000, batch_size=256, num_workers=8, host_max=12000):
        self.paths = list(paths)
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cached = len(self.paths) <= cache_max if cache is None else bool(cache)
        # half precision on the GPU halves the cache; the models run in fp16 / fp32 anyway
        dtype = torch.float16 if torch.device(device).type == "cuda" else None
        self._imgs = load_images(self.paths, device, dtype) if self.cached else None
        self._host = {}  # streamed sets: index -> decoded image (CPU, FP16), filled by get()
        self.host_max = host_max

    def __len__(self):
        return len(self.paths)

    def get(self, idx):
        """Images idx (list / array of positions) as one float32 tensor on the device."""
        idx = [int(i) for i in idx]
        if self.cached:
            return self._imgs[torch.tensor(idx, device=self._imgs.device)].float()
        missing = sorted({i for i in idx if i not in self._host})
        new = {}
        if missing:
            for i, img in zip(missing, load_images([self.paths[i] for i in missing])):
                new[i] = img
                if len(self._host) < self.host_max:
                    self._host[i] = img.half()
        return torch.stack([new[i] if i in new else self._host[i].float() for i in idx]).to(self.device)

    def batches(self):
        """Every image in order, in batches (on the device)."""
        if self.cached:
            for i in range(0, len(self), self.batch_size):
                yield self._imgs[i:i + self.batch_size].float()
            return
        loader = DataLoader(PathDataset(self.paths), batch_size=self.batch_size, shuffle=False,
                            num_workers=self.num_workers, pin_memory=torch.device(self.device).type == "cuda")
        for x, _ in loader:
            yield x.to(self.device, non_blocking=True)


@torch.no_grad()
def features(model, store, prompts):
    """L2-normalised retrieval features (flip-averaged, float32, on the device) of every image of a store
    under the given prompt (None: the model's own prompt / no prompt)."""
    model.eval()
    out = []
    for x in store.batches():
        f = model(x, prompts=prompts)["features"] + model(torch.flip(x, dims=(3,)), prompts=prompts)["features"]
        out.append(F.normalize(f.float(), dim=1))
    return torch.cat(out)


@torch.no_grad()
def retrieval_metrics(qf, gf, q_pids, g_pids, q_cams, g_cams, chunk=1024):
    """Rank-1 / mAP (%) of query vs gallery with cosine distance, on the features' device (the GPU): the
    torchreid market protocol (gallery images of the query's person seen by the query's camera are ignored;
    queries without a valid match are skipped)."""
    dev = qf.device
    qp, gp = torch.as_tensor(np.asarray(q_pids), device=dev), torch.as_tensor(np.asarray(g_pids), device=dev)
    qc, gc = torch.as_tensor(np.asarray(q_cams), device=dev), torch.as_tensor(np.asarray(g_cams), device=dev)
    ap_sum, r1_sum, n_valid = 0.0, 0.0, 0
    for i in range(0, qf.size(0), chunk):
        order = torch.argsort(1 - qf[i:i + chunk].float() @ gf.float().T, dim=1, stable=True)
        same = gp[order] == qp[i:i + chunk, None]
        keep = ~(same & (gc[order] == qc[i:i + chunk, None]))
        match = same & keep
        n_rel = match.sum(1)
        valid = n_rel > 0
        rank = keep.long().cumsum(1)                        # 1-based rank among kept gallery images
        prec = match.long().cumsum(1).float() / rank.clamp_min(1).float()
        ap = (prec * match).sum(1) / n_rel.clamp_min(1)
        first_kept = torch.argmax(keep.int(), dim=1)
        r1 = match.gather(1, first_kept[:, None])[:, 0].float()
        ap_sum += float(ap[valid].sum())
        r1_sum += float(r1[valid].sum())
        n_valid += int(valid.sum())
    assert n_valid > 0, "no query has a valid gallery match"
    return r1_sum / n_valid * 100, ap_sum / n_valid * 100


class TargetSplit:
    """One target split: the unlabeled pool (train) and the query / gallery used for evaluation.

    Person ids of the pool are kept here only to build the annotation oracle and for reporting; the
    selection code receives features and camera ids."""

    def __init__(self, ds, name, device, has_cameras, cache_max=40000, num_workers=8):
        self.name = name
        self.has_cameras = has_cameras
        self.pool_paths = [x[0] for x in ds.train]
        self.pool_pids = np.array([x[1] for x in ds.train])
        self.pool_cams = np.array([x[2] for x in ds.train])
        self.pool = ImageStore(self.pool_paths, device, cache_max=cache_max, num_workers=num_workers)
        self.query = ImageStore([x[0] for x in ds.query], device, cache_max=cache_max, num_workers=num_workers)
        self.gallery = ImageStore([x[0] for x in ds.gallery], device, cache_max=cache_max, num_workers=num_workers)
        self.q_pids = np.array([x[1] for x in ds.query]); self.q_cams = np.array([x[2] for x in ds.query])
        self.g_pids = np.array([x[1] for x in ds.gallery]); self.g_cams = np.array([x[2] for x in ds.gallery])

    @torch.no_grad()
    def evaluate(self, model, prompts):
        qf = features(model, self.query, prompts)
        gf = features(model, self.gallery, prompts)
        return retrieval_metrics(qf, gf, self.q_pids, self.g_pids, self.q_cams, self.g_cams)
