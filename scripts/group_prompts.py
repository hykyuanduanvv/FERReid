"""Camera-group pseudo-domains and per-group "teacher" prompts (docs/DIRECTION_A_DISTILL.md).

Three stages, run in order:

  cluster   style of every camera unit of the source datasets (a unit = one camera; for MSMT17 one camera x
            time of day, read from the file name, e.g. 0303morning -> morning), measured with the *pretrained*
            DINOv2 CLS feature (no ReID training, so it keeps the style). Units are clustered within each
            dataset (average linkage on the cosine distance of the unit means) into the largest number of
            groups (<= --max_groups) where every group has >= 2 real cameras and >= --min_ids identities seen by
            two of its cameras, so each group still holds cross-camera positives.
            -> <out>/groups.json, <out>/style.json
  tune      for every group, start from the VPT prompt (--checkpoint, a VPT run) and tune only the prompt on the
            group's tuning identities (triplet loss, cross-camera pairs) -> one teacher prompt per group.
            --shard i/n tunes groups i, i+n, ... so several GPUs can share the work.
            -> <out>/teacher_<group>.pt
  matrix    mAP of every prompt (VPT base + every teacher) on the held-out identities of every group.
            Feasibility check: is a group's own teacher clearly better on that group than the other teachers
            and than the VPT prompt? If not, there is nothing domain-specific to distill.
            -> <out>/matrix.csv, <out>/matrix_summary.json, <out>/teachers.pt (all prompts, for --prompt_teacher)

  python scripts/group_prompts.py --stage cluster --out_dir experiments/groups --report_to none --output_dir /tmp/x
  python scripts/group_prompts.py --stage tune --shard 0/2 --checkpoint $VPT_WARM --out_dir experiments/groups ...
  python scripts/group_prompts.py --stage matrix --checkpoint $VPT_WARM --out_dir experiments/groups ...
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import csv
import json
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
import transformers

from adapters.args_reid import ReIDTrainingArguments
from adapters.config_reid import DOMAIN_CONFIG, NO_CAMERA_DOMAINS
from adapters.reid_dataset import camera_unit
from adapters.trainer_reid import _get_dataset_cls, _seed_all
from scripts.context_sensitivity import build_model, load_images
from scripts.oracle_prompt import augment
from torchreid.metrics import evaluate_rank


@dataclass
class GroupArguments:
    stage: str = field(default="cluster")          # cluster | tune | matrix
    out_dir: str = field(default="experiments/groups")
    datasets: str = field(default="market1501,msmt17")
    checkpoint: str = field(default="")            # VPT checkpoint (tune / matrix)
    # cluster
    imgs_per_unit: int = field(default=200)
    max_groups: int = field(default=6)             # per dataset
    min_ids: int = field(default=64)
    # tune / matrix
    holdout_frac: float = field(default=0.2)
    max_eval_ids: int = field(default=150)
    eval_imgs_per_id: int = field(default=6)
    tune_imgs_per_id: int = field(default=0)       # 0 = every image of every tuning identity
    # cluster: "style" = clustered camera units (above); "pair" = the n_pairs camera pairs (dataset, cam a, cam b)
    # with the most identities seen by both cameras, i.e. the camera-pair pseudo-domains of --pseudo_domains camera_pair
    grouping: str = field(default="style")
    n_pairs: int = field(default=10)
    steps: int = field(default=300)
    lr: float = field(default=1e-3)
    ids_per_batch: int = field(default=32)
    shard: str = field(default="0/1")
    base_seed: int = field(default=0)             # identity split, image picks, evaluation
    tune_seed: int = field(default=0)             # teacher tuning only (batches, augmentation): noise estimate


# ----------------------------------------------------------------------------- units and groups

unit_of = camera_unit  # the same units as the training dataset (--pseudo_domains camera_group)


def load_units(dataset, with_time=True):
    """{unit: [(path, pid, camid), ...]} over the train split of a source dataset."""
    ds = _get_dataset_cls(dataset)(root=DOMAIN_CONFIG["data_root"], combineall=False, verbose=False)
    units = {}
    for path, pid, camid, *_ in ds.train:
        units.setdefault(unit_of(dataset, path, camid, with_time), []).append((path, pid, camid))
    return units


def dataset_units(cache, dataset, us):
    """Units of a dataset as named in groups.json (camera-only names "cNN" for the pair grouping)."""
    with_time = not all(len(u) == 3 and u.startswith("c") for u in us)
    key = (dataset, with_time)
    if key not in cache:
        cache[key] = load_units(dataset, with_time)
    return cache[key]


def pair_groups(gargs):
    """The n_pairs camera pairs with the most identities seen by both cameras (over all datasets)."""
    cands = []
    for dataset in gargs.datasets.split(","):
        if dataset in NO_CAMERA_DOMAINS:
            continue
        units = load_units(dataset, with_time=False)
        names = sorted(units)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                n = len(group_items(units, [names[i], names[j]]))
                if n >= gargs.min_ids:
                    cands.append((n, dataset, names[i], names[j]))
    cands.sort(reverse=True)
    groups = {}
    for n, dataset, a, b in cands[:gargs.n_pairs]:
        groups.setdefault(dataset, {})["{}-{}".format(a, b)] = [a, b]
        print("   {} {}-{}: {} ids seen by both cameras".format(dataset, a, b, n))
    return groups


SYNTH = ("orig", "lowres", "dark", "washed", "blue", "yellow", "blur", "gray")
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def synth_groups(gargs):
    """--grouping synth: one synthetic domain per fixed style transform, all over the same identities (every
    camera of the first dataset in --datasets), so the matrix compares styles on identical people."""
    dataset = gargs.datasets.split(",")[0]
    units = sorted(load_units(dataset, with_time=False))
    return {dataset: {"synth-" + t: units for t in SYNTH}}


@torch.no_grad()
def synth_apply(name, x):
    """Fixed style transform of a group named "...|synth-<t>" on normalized images (B, 3, H, W); other
    groups are returned unchanged. Low resolution, darkness, low contrast, colour casts, blur and grayscale
    are the ways the small target domains (VIPeR, GRID, i-LIDS) differ from the sources."""
    t = name.split("|")[-1]
    if not t.startswith("synth-") or t == "synth-orig":
        return x
    t = t[len("synth-"):]
    mean, std = _MEAN.to(x.device, x.dtype), _STD.to(x.device, x.dtype)
    y = (x.float() * std + mean).clamp(0, 1)
    H, W = y.shape[-2:]
    if t == "lowres":
        y = F.interpolate(F.interpolate(y, size=(H // 4, W // 4), mode="bilinear", antialias=True),
                          size=(H, W), mode="bilinear", align_corners=False)
    elif t == "dark":
        y = 0.8 * y ** 2.2
    elif t == "washed":
        y = 0.5 * y + 0.35
    elif t == "blue":
        y = y * torch.tensor([0.75, 0.9, 1.25], device=y.device).view(1, 3, 1, 1)
    elif t == "yellow":
        y = y * torch.tensor([1.2, 1.05, 0.7], device=y.device).view(1, 3, 1, 1)
    elif t == "blur":
        from torchvision.transforms.functional import gaussian_blur
        y = gaussian_blur(y, kernel_size=9, sigma=2.5)
    elif t == "gray":
        y = (0.299 * y[:, :1] + 0.587 * y[:, 1:2] + 0.114 * y[:, 2:]).expand(-1, 3, -1, -1)
    else:
        raise ValueError("unknown synthetic domain " + t)
    return ((y.clamp(0, 1) - mean) / std).to(x.dtype)


def split_key(dataset, units):
    """Seed key of a group's identity split / image picks: groups over the same units (synthetic domains)
    get the same held-out identities and images."""
    return sum(map(ord, dataset + "|" + ",".join(sorted(units))))


def seed_key(name, dataset, units):
    """Synthetic domains share the split of their units; real groups keep the name-based seed (as first run)."""
    return split_key(dataset, units) if "|synth-" in name else sum(map(ord, name))


def group_items(units, unit_names, no_camera=False):
    """{pid: {camid: [paths]}} for the identities of a group that have a cross-camera positive inside it
    (any two images for a dataset without cameras)."""
    by_pid = {}
    for u in unit_names:
        for path, pid, camid in units[u]:
            by_pid.setdefault(pid, {}).setdefault(camid, []).append(path)
    if no_camera:
        return {p: c for p, c in by_pid.items() if sum(len(v) for v in c.values()) >= 2}
    return {p: c for p, c in by_pid.items() if len(c) >= 2}


def read_groups(out_dir):
    """[(group name, dataset, [units])] in a fixed order. Group name = "<dataset>|<g>"."""
    with open(os.path.join(out_dir, "groups.json")) as f:
        groups = json.load(f)
    return [("{}|{}".format(d, g), d, us) for d in sorted(groups) for g, us in sorted(groups[d].items())]


@torch.no_grad()
def stage_cluster(args, gargs, device):
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    from adapters.reid_model import load_backbone, backbone_name
    os.makedirs(gargs.out_dir, exist_ok=True)
    if gargs.grouping in ("pair", "synth"):
        with open(os.path.join(gargs.out_dir, "groups.json"), "w") as f:
            json.dump(pair_groups(gargs) if gargs.grouping == "pair" else synth_groups(gargs), f, indent=1)
        return
    enc = load_backbone(backbone_name(args)).to(device).eval().half()
    rng = np.random.RandomState(gargs.base_seed)
    groups, style = {}, {}
    for dataset in gargs.datasets.split(","):
        units = load_units(dataset)
        names = sorted(units)
        means, info = [], {}
        for u in names:
            items = units[u]
            pick = rng.choice(len(items), size=min(gargs.imgs_per_unit, len(items)), replace=False)
            imgs = load_images([items[i][0] for i in pick], device).half()
            f = torch.cat([F.normalize(enc.forward_features(imgs[i:i + 256])["x_norm_clstoken"].float(), dim=-1)
                           for i in range(0, len(imgs), 256)])
            means.append(F.normalize(f.mean(0), dim=0))
            info[u] = {"images": len(items), "ids": len({p for _, p, _ in items}), "spread": float(1 - (f @ means[-1]).mean())}
        M = torch.stack(means)
        D = (1 - M @ M.T).clamp_min(0).cpu().numpy()
        np.fill_diagonal(D, 0)
        Z = linkage(squareform(D, checks=False), method="average")
        def valid(us):
            cams = {c for u in us for _, _, c in units[u]}
            ids = group_items(units, us, dataset in NO_CAMERA_DOMAINS)
            return (len(cams) >= 2 or dataset in NO_CAMERA_DOMAINS) and len(ids) >= gargs.min_ids

        idx = {u: i for i, u in enumerate(names)}

        def dist(a, b):  # average linkage between two groups of units
            return float(np.mean([D[idx[x], idx[y]] for x in a for y in b]))

        chosen = {"g0": names}
        for n in range(2, len(names) + 1):  # cut into n clusters, fold invalid clusters into their nearest
            lab = fcluster(Z, t=n, criterion="maxclust")
            cl = {}
            for u, l in zip(names, lab):
                cl.setdefault(l, []).append(u)
            cl = list(cl.values())
            while len(cl) > 1 and not all(valid(c) for c in cl):
                bad = min((c for c in cl if not valid(c)), key=len)
                cl.remove(bad)
                near = min(cl, key=lambda c: dist(bad, c))
                near.extend(bad)
            if all(valid(c) for c in cl) and len(cl) > len(chosen):
                chosen = {"g{}".format(i): c for i, c in enumerate(cl)}
            if len(chosen) >= gargs.max_groups:
                break
        # renumber groups by their first unit so names are stable
        chosen = {"g{}".format(i): us for i, us in enumerate(sorted(chosen.values()))}
        groups[dataset] = chosen
        style[dataset] = {"units": names, "distance": D.round(4).tolist(), "info": info,
                          "groups": {g: {"units": us, "ids_cross_camera": len(group_items(units, us, dataset in NO_CAMERA_DOMAINS)),
                                         "cameras": len({c for u in us for _, _, c in units[u]})}
                                     for g, us in chosen.items()}}
        print("== {}: {} units -> {} groups".format(dataset, len(names), len(chosen)))
        for g, s in style[dataset]["groups"].items():
            print("   {} cams {} ids {} : {}".format(g, s["cameras"], s["ids_cross_camera"], " ".join(s["units"])))
        within = [D[i, j] for i in range(len(names)) for j in range(i + 1, len(names))]
        print("   unit distance: min {:.4f} median {:.4f} max {:.4f}".format(min(within), float(np.median(within)), max(within)))
    os.makedirs(gargs.out_dir, exist_ok=True)
    with open(os.path.join(gargs.out_dir, "groups.json"), "w") as f:
        json.dump(groups, f, indent=1)
    with open(os.path.join(gargs.out_dir, "style.json"), "w") as f:
        json.dump(style, f)


# ----------------------------------------------------------------------------- per-group data

def held_out(gargs, pid):
    """Held-out identities are chosen per dataset, not per group: an identity seen by several groups is
    held out in all of them, so no teacher is ever tuned on another group's evaluation identities."""
    return np.random.RandomState(gargs.base_seed * 100003 + int(pid) % 100003).rand() < gargs.holdout_frac


def split_group(gargs, name, items, key):
    """Tuning identities and (at most max_eval_ids) held-out identities of a group."""
    pids = sorted(items)
    tune = [p for p in pids if not held_out(gargs, p)]
    ev = [p for p in pids if held_out(gargs, p)]
    rng = np.random.RandomState(gargs.base_seed + key)
    rng.shuffle(ev)
    return tune, sorted(ev[:gargs.max_eval_ids])


def pick_images(cams, n, rng):
    """Up to n images (n <= 0: all) of one identity, cycling over its cameras so several cameras are kept."""
    pools = [list(rng.permutation(v)) for _, v in sorted(cams.items())]
    n = n if n > 0 else sum(len(p) for p in pools)
    out = []
    while len(out) < n and any(pools):
        for p in pools:
            if p and len(out) < n:
                out.append(p.pop())
    return out


def eval_set(gargs, name, items, eval_pids, device, key):
    rng = np.random.RandomState(gargs.base_seed + 1 + key)
    paths, pids, cams = [], [], []
    for pid in eval_pids:
        cam_of = {p: c for c, ps in items[pid].items() for p in ps}
        for p in pick_images(items[pid], gargs.eval_imgs_per_id, rng):
            paths.append(p)
            pids.append(pid)
            cams.append(cam_of[p])
    imgs = synth_apply(name, load_images(paths, device)).half()
    pids, cams = np.array(pids), np.array(cams)
    # query: the first image of every identity; gallery: the rest (same-camera matches are ignored by the metric)
    first = np.zeros(len(pids), dtype=bool)
    seen = set()
    for i, p in enumerate(pids):
        if p not in seen:
            first[i] = True
            seen.add(p)
    return {"imgs": imgs, "pids": pids, "cams": cams, "q": first}


@torch.no_grad()
def features(model, imgs, prompt):
    out = []
    for i in range(0, len(imgs), 256):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(imgs[i:i + 256], prompts=prompt)["features"].float())
    return F.normalize(torch.cat(out), dim=-1)


def group_map(model, ev, prompt):
    f = features(model, ev["imgs"], prompt)
    q, g = ev["q"], ~ev["q"]
    distmat = (1 - f[torch.from_numpy(q).to(f.device)] @ f[torch.from_numpy(g).to(f.device)].T).cpu().numpy()
    cmc, mAP = evaluate_rank(distmat, ev["pids"][q], ev["pids"][g], ev["cams"][q], ev["cams"][g], max_rank=10)
    return float(mAP) * 100, float(cmc[0]) * 100


def load_vpt(args, gargs, device):
    model = build_model(args, device, gargs.checkpoint)
    if args.model_type != "vpt":
        raise ValueError("--checkpoint must be a VPT run (model_type vpt), got {}".format(args.model_type))
    state = torch.load(os.path.join(gargs.checkpoint, "pytorch_model.bin"), map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("checkpoint:", gargs.checkpoint, "missing:", len(missing), "unexpected:", len(unexpected))
    if missing or unexpected:
        raise ValueError("VPT checkpoint does not match the model: missing {} unexpected {}".format(missing[:5], unexpected[:5]))
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def stage_tune(args, gargs, device):
    i, n = map(int, gargs.shard.split("/"))
    model = load_vpt(args, gargs, device)
    base = model.prompt.detach().float()
    unit_cache = {}
    for gi, (name, dataset, us) in enumerate(read_groups(gargs.out_dir)):
        if gi % n != i:
            continue
        out = os.path.join(gargs.out_dir, "teacher_{}.pt".format(name.replace("|", "_")))
        if os.path.exists(out):
            print("skip", name, "(done)")
            continue
        t0 = time.time()
        items = group_items(dataset_units(unit_cache, dataset, us), us, dataset in NO_CAMERA_DOMAINS)
        tune_pids, eval_pids = split_group(gargs, name, items, seed_key(name, dataset, us))
        rng = np.random.RandomState(gargs.base_seed + 2 + 7919 * gargs.tune_seed + seed_key(name, dataset, us))
        paths, id2imgs = [], []
        for pid in tune_pids:
            cam_of = {p: c for c, ps in items[pid].items() for p in ps}
            ps = pick_images(items[pid], gargs.tune_imgs_per_id, rng)
            by_cam = {}
            for p in ps:
                by_cam.setdefault(cam_of[p], []).append(len(paths))
                paths.append(p)
            id2imgs.append(list(by_cam.values()))
        imgs = synth_apply(name, load_images(paths, device)).half()
        gen = torch.Generator(device=device).manual_seed(gargs.base_seed + 7919 * gargs.tune_seed)
        prompt = torch.nn.Parameter(base.clone())
        opt = torch.optim.Adam([prompt], lr=gargs.lr)
        P = min(gargs.ids_per_batch, len(id2imgs))
        losses = []
        with torch.enable_grad():
            for step in range(gargs.steps):
                ids = rng.choice(len(id2imgs), size=P, replace=False)
                idx, labels = [], []
                for j, k in enumerate(ids):
                    cams = id2imgs[k]
                    if len(cams) >= 2:  # one image from each of two different cameras
                        a, b = rng.choice(len(cams), size=2, replace=False)
                        idx += [rng.choice(cams[a]), rng.choice(cams[b])]
                    else:
                        idx += list(rng.choice(cams[0], size=2, replace=len(cams[0]) < 2))
                    labels += [j, j]
                x = augment(imgs[torch.tensor(idx, device=device)].float(), gen)
                with torch.autocast("cuda", dtype=torch.float16):
                    feats = model(x, prompts=prompt)["features"]
                loss = model.loss(feats.float(), torch.tensor(labels, device=device))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(loss.item())
        rel = ((prompt.detach() - base).norm() / base.norm()).item()
        torch.save({"group": name, "prompt": prompt.detach().cpu(), "tune_ids": len(tune_pids),
                    "eval_ids": len(eval_pids), "loss_start": float(np.mean(losses[:20])),
                    "loss_end": float(np.mean(losses[-20:])), "rel_change": rel}, out)
        print("{}: {} tune ids / {} images, loss {:.4f} -> {:.4f}, prompt change {:.3f}, {:.0f}s".format(
            name, len(tune_pids), len(paths), np.mean(losses[:20]), np.mean(losses[-20:]), rel, time.time() - t0))
        del imgs
        torch.cuda.empty_cache()


@torch.no_grad()
def stage_matrix(args, gargs, device):
    model = load_vpt(args, gargs, device)
    groups = read_groups(gargs.out_dir)
    prompts = {"VPT": model.prompt.detach().float()}
    meta = {}
    for name, _, _ in groups:
        t = torch.load(os.path.join(gargs.out_dir, "teacher_{}.pt".format(name.replace("|", "_"))), weights_only=False)
        prompts[name] = t["prompt"].to(device)
        meta[name] = {k: v for k, v in t.items() if k != "prompt"}
    unit_cache, evals = {}, {}
    for name, dataset, us in groups:
        items = group_items(dataset_units(unit_cache, dataset, us), us, dataset in NO_CAMERA_DOMAINS)
        _, eval_pids = split_group(gargs, name, items, seed_key(name, dataset, us))
        evals[name] = eval_set(gargs, name, items, eval_pids, device, seed_key(name, dataset, us))
    rows, M = [], {}
    for pname, prompt in prompts.items():
        for gname in evals:
            _seed_all(gargs.base_seed)
            mAP, r1 = group_map(model, evals[gname], prompt)
            M[(pname, gname)] = mAP
            rows.append({"prompt": pname, "eval_group": gname, "mAP": mAP, "rank1": r1})
        print("{:<22} ".format(pname) + " ".join("{:6.2f}".format(M[(pname, g)]) for g in evals))
    with open(os.path.join(gargs.out_dir, "matrix.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for g, dataset, _ in groups:
        others = [M[(t, g)] for t, _, _ in groups if t != g]
        same_ds = [M[(t, g)] for t, d, _ in groups if t != g and d == dataset]
        summary[g] = {"vpt": M[("VPT", g)], "own_teacher": M[(g, g)], "other_teachers_mean": float(np.mean(others)),
                      "other_teachers_same_dataset_mean": float(np.mean(same_ds)) if same_ds else None,
                      "own_minus_vpt": M[(g, g)] - M[("VPT", g)],
                      "own_minus_others": M[(g, g)] - float(np.mean(others)),
                      **meta[g]}
    adv_vpt = np.mean([s["own_minus_vpt"] for s in summary.values()])
    adv_oth = np.mean([s["own_minus_others"] for s in summary.values()])
    print("mean over groups: own teacher - VPT {:+.2f} | own teacher - other teachers {:+.2f}".format(adv_vpt, adv_oth))
    with open(os.path.join(gargs.out_dir, "matrix_summary.json"), "w") as f:
        json.dump({"groups": summary, "mean_own_minus_vpt": adv_vpt, "mean_own_minus_others": adv_oth}, f, indent=1)
    torch.save({"base": prompts["VPT"].cpu(), "prompts": {g: prompts[g].cpu() for g, _, _ in groups},
                "groups": json.load(open(os.path.join(gargs.out_dir, "groups.json"))), "checkpoint": gargs.checkpoint},
               os.path.join(gargs.out_dir, "teachers.pt"))


def main():
    parser = transformers.HfArgumentParser((ReIDTrainingArguments, GroupArguments))
    args, gargs = parser.parse_args_into_dataclasses()
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not args.backbone:
        args.backbone = "dinov2_b14"
    {"cluster": stage_cluster, "tune": stage_tune, "matrix": stage_matrix}[gargs.stage](args, gargs, device)


if __name__ == "__main__":
    main()
