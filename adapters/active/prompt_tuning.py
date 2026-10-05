"""Target-domain prompt learned from the annotated clusters; the base model stays frozen.

  append  (default): the shared base prompt (1, L, V, D) is frozen and m new tokens per layer are appended
          and trained -> prompt (1, L, V + m, D). The new tokens start from the mean of the base model's
          source-domain tokens when it was trained with --source_domain_tokens (init_tokens), else from
          N(0, init_std). The base prompt is untouched: the domain tokens are a separate, swappable parameter.
  replace: a copy of the base model's default prompt is trained.

Each step draws `ids_per_batch` identity clusters with >= 2 images (two images each); with probability
`hn_prob` a cluster brings one of its cannot-link clusters along (an annotated hard negative). Loss: the
model's batch-hard triplet loss on the retrieval features, in FP32.

With pseudo labels (loop: --pseudo True) the clusters are the constrained pseudo identities of the whole pool,
and two additions of the cluster-then-train UDA recipe apply:
  contrast_weight > 0  cluster-memory contrastive loss (Cluster Contrast): a memory of cluster centroids,
                       initialised from the round's pool features, InfoNCE of every batch feature against
                       all centroids (temperature `temp`), momentum update of the batch's centroids;
  cross_cam            the two images drawn per cluster come from two cameras when the cluster has them
                       (camera-invariant positives; the camera ids are part of the unlabeled pool).
"""
import numpy as np
import torch
import torch.nn.functional as F


def augment(x, gen):
    """Flip, +-10 px translation (pad + crop), random erasing -- the tensor analogue of the training
    transform without colour jitter. x: (B, 3, H, W), normalised."""
    B, _, H, W = x.shape
    flip = torch.rand(B, generator=gen, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], torch.flip(x, dims=(3,)), x)
    xp = F.pad(x, (10, 10, 10, 10))
    dy = torch.randint(0, 21, (B,), generator=gen, device=x.device).tolist()
    dx = torch.randint(0, 21, (B,), generator=gen, device=x.device).tolist()
    out = torch.stack([xp[i, :, dy[i]:dy[i] + H, dx[i]:dx[i] + W] for i in range(B)])
    for i in range(B):  # RandomErasing(p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3)), fill 0
        if torch.rand(1, generator=gen, device=x.device).item() >= 0.5:
            continue
        for _ in range(10):
            area = H * W * (0.02 + 0.38 * torch.rand(1, generator=gen, device=x.device).item())
            logr = np.log(0.3) + (np.log(3.3) - np.log(0.3)) * torch.rand(1, generator=gen, device=x.device).item()
            h, w = int(round((area * np.exp(logr)) ** 0.5)), int(round((area / np.exp(logr)) ** 0.5))
            if h < H and w < W:
                y0 = torch.randint(0, H - h + 1, (1,), generator=gen, device=x.device).item()
                x0 = torch.randint(0, W - w + 1, (1,), generator=gen, device=x.device).item()
                out[i, :, y0:y0 + h, x0:x0 + w] = 0
                break
    return out


class DomainPrompt(torch.nn.Module):

    def __init__(self, base, mode="append", domain_tokens=8, init_std=0.02, seed=0, init_tokens=None):
        super().__init__()
        self.mode = mode
        base = base.detach().float().reshape(1, base.size(1), -1, base.size(-1))
        self.register_buffer("base", base.clone())
        if mode == "append":
            if init_tokens is not None:
                tok = init_tokens.detach().float().reshape(1, base.size(1), -1, base.size(-1)).clone()
            else:
                g = torch.Generator(device="cpu").manual_seed(seed)
                tok = torch.randn(1, base.size(1), domain_tokens, base.size(-1), generator=g) * init_std
            self.tokens = torch.nn.Parameter(tok.to(base.device))
        elif mode == "replace":
            self.tokens = torch.nn.Parameter(base.clone())
        else:
            raise ValueError("--prompt_mode must be append or replace, got {}".format(mode))

    def forward(self):
        return torch.cat([self.base, self.tokens], dim=2) if self.mode == "append" else self.tokens


def _two(rng, members, cams=None):
    if cams is None or len(members) < 2:
        return list(rng.choice(members, size=2, replace=len(members) < 2))
    a = members[rng.randint(len(members))]
    other = [m for m in members if cams[m] != cams[a]]
    if not other:
        other = [m for m in members if m != a]
    return [a, other[rng.randint(len(other))]]


class ClusterMemory:
    """Cluster-centroid memory (C, D) for the contrastive loss; updated with momentum, kept L2-normalised."""

    def __init__(self, centroids, temp=0.05, momentum=0.2):
        self.M = F.normalize(centroids.float(), dim=1)
        self.temp, self.momentum = temp, momentum

    def loss(self, f, y):
        return F.cross_entropy(f @ self.M.T / self.temp, y)

    @torch.no_grad()
    def update(self, f, y):
        for v, c in zip(f.detach(), y.tolist()):
            self.M[c] = F.normalize(self.momentum * self.M[c] + (1 - self.momentum) * v, dim=0)


def tune_domain_prompt(model, store, clusters, cannot, base, mode="append", domain_tokens=8, init_std=0.02,
                       steps=300, lr=3e-4, ids_per_batch=32, hn_prob=0.5, seed=0, init_tokens=None,
                       cams=None, cross_cam=False, feats=None, contrast_weight=0.0, temp=0.05, momentum=0.2):
    """clusters: lists of pool indices (one per annotated identity, singletons allowed: they only serve as
    negatives); cannot: pairs of positions into clusters; base: the frozen shared prompt (append) or the prompt
    to start from (replace). cams: pool camera ids (cross_cam sampling); feats: (N, D) pool features under the
    current prompt (memory initialisation, contrast_weight > 0). Returns (prompt (1, L, V', D) detached, info)."""
    device = base.device
    prompt = DomainPrompt(base, mode, domain_tokens, init_std, seed, init_tokens).to(device)
    positives = [n for n, c in enumerate(clusters) if len(c) >= 2]
    info = {"loss_start": float("nan"), "loss_end": float("nan"), "n_train_ids": len(positives),
            "n_train_imgs": sum(len(clusters[n]) for n in positives)}
    if not positives or steps <= 0:
        return prompt().detach(), info
    partners = {}
    for a, b in cannot:
        partners.setdefault(a, []).append(b)
        partners.setdefault(b, []).append(a)
    rng = np.random.RandomState(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    opt = torch.optim.Adam(prompt.parameters(), lr=lr)
    P = min(ids_per_batch, len(positives))
    memory = None
    if contrast_weight > 0:
        if feats is None:
            raise ValueError("the contrastive loss needs the pool features (feats)")
        f = feats.to(device).float()
        memory = ClusterMemory(torch.stack([f[c].mean(0) for c in clusters]), temp, momentum)
    cam_arr = np.asarray(cams) if (cross_cam and cams is not None) else None
    losses = []
    model.eval()  # frozen statistics (BNNeck) and no dropout; gradients flow to the prompt only
    with torch.enable_grad():
        for _ in range(steps):
            chosen = [int(c) for c in rng.choice(positives, size=P, replace=False)]
            ids = list(chosen)
            for c in chosen:
                if c in partners and rng.rand() < hn_prob:
                    n = int(partners[c][rng.randint(len(partners[c]))])
                    if n not in ids:
                        ids.append(n)
            idx, labels, cl_ids = [], [], []
            for lab, c in enumerate(ids):
                members = clusters[c]
                pick = _two(rng, members, cam_arr) if len(members) >= 2 else [members[0]]
                idx += pick
                labels += [lab] * len(pick)
                cl_ids += [c] * len(pick)
            x = augment(store.get(idx), gen)
            out = F.normalize(model(x, prompts=prompt())["features"].float(), dim=1)
            loss = model.loss(out, torch.tensor(labels, device=device))
            if memory is not None:
                y = torch.tensor(cl_ids, device=device)
                loss = loss + contrast_weight * memory.loss(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if memory is not None:
                memory.update(out, y)
            losses.append(loss.item())
    info.update(loss_start=float(np.mean(losses[:20])), loss_end=float(np.mean(losses[-20:])))
    return prompt().detach(), info
