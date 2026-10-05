"""Images of one subset (pool / query / gallery) for small and large target domains.

Every image is decoded once, on the CPU with torchvision (libjpeg-turbo / libpng, many threads: ~8k MSMT17 images
/ s; nvjpeg on the GPU was 15x slower for these small images); the 256x128 resize (bicubic, antialiased like PIL) runs on the GPU and
the result is kept as uint8 -- on the device up to `cache_max` images per set, in host memory beyond (MSMT17's
82k gallery images: ~8 GB). Batches are normalised (ImageNet mean / std, as EVAL_TRANSFORM) on the device.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import ImageReadMode, decode_image

SIZE = (256, 128)
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _read(path):
    with open(path, "rb") as f:
        return torch.frombuffer(bytearray(f.read()), dtype=torch.uint8)


def _resize(img, device):
    """(3, H, W) uint8 -> (3, 256, 128) uint8, resized on the device."""
    x = img.to(device, non_blocking=True).float()[None]
    if tuple(x.shape[-2:]) != SIZE:
        x = F.interpolate(x, size=SIZE, mode="bicubic", align_corners=False, antialias=True)
    return x.round_().clamp_(0, 255).to(torch.uint8)[0]


def decode_resized(paths, device, threads=32, chunk=2048, out_device=None):
    """uint8 (N, 3, 256, 128) on `out_device` (default: device); CPU decoding, resizing on `device`."""
    dev = torch.device(device)
    odev = dev if out_device is None else torch.device(out_device)
    dec = lambda p: decode_image(_read(p), mode=ImageReadMode.RGB)
    out = torch.empty((len(paths), 3) + SIZE, dtype=torch.uint8, device=odev,
                      pin_memory=odev.type == "cpu" and dev.type == "cuda")
    with ThreadPoolExecutor(threads) as ex:
        for s in range(0, len(paths), chunk):
            imgs = list(ex.map(dec, paths[s:s + chunk]))
            out[s:s + len(imgs)] = torch.stack([_resize(im, dev) for im in imgs]).to(odev)
    return out


def normalize(x_uint8, device):
    """uint8 batch -> float32 normalised batch on the device."""
    x = x_uint8.to(device, non_blocking=True).float().div_(255)
    m = torch.tensor(_MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(_STD, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


def load_images(paths, device=None, dtype=None, threads=32):
    dev = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    imgs = normalize(decode_resized(list(paths), dev, threads), dev)
    if dtype is not None:
        imgs = imgs.to(dtype)
    return imgs if device is not None else imgs.cpu()


class ImageStore:
    """Decoded once (uint8, 256x128): on the device when len(paths) <= cache_max (or cache=True), else in host
    memory (pinned), moved to the device per batch."""

    def __init__(self, paths, device, cache=None, cache_max=6000, batch_size=256, num_workers=8, host_max=None):
        self.paths = list(paths)
        self.device = device
        self.batch_size = batch_size
        self.cached = len(self.paths) <= cache_max if cache is None else bool(cache)
        self._imgs = decode_resized(self.paths, device, out_device=None if self.cached else "cpu")

    def __len__(self):
        return len(self.paths)

    def get(self, idx):
        """Images idx (list / array of positions) as one float32 tensor on the device."""
        idx = torch.as_tensor(np.asarray([int(i) for i in idx], np.int64))
        return normalize(self._imgs[idx.to(self._imgs.device)], self.device)

    def batches(self):
        """Every image in order, in batches (on the device)."""
        for i in range(0, len(self), self.batch_size):
            yield normalize(self._imgs[i:i + self.batch_size], self.device)


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
