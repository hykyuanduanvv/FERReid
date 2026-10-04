import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


# TransReID-style pedestrian augmentations (256×128, ViT-B/16 compatible)
PEDESTRIAN_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 128), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.Pad(10),
    transforms.RandomCrop((256, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3)),
])

PEDESTRIAN_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 128), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# DINOv2-compatible transforms (224×112, multiples of patch_size=14) — kept as reference
DINO_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 112), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.Pad(10),
    transforms.RandomCrop((224, 112)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3)),
])

DINO_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 112), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Default to TransReID pedestrian transforms
TRAIN_TRANSFORM = PEDESTRIAN_TRAIN_TRANSFORM
EVAL_TRANSFORM = PEDESTRIAN_EVAL_TRANSFORM


def _set_domain_index(ds, sample_domain_names):
    """ds.domain_names (sorted training domains, i.e. what the batch sampler groups by) and
    ds.sample_domain[i] = index of sample i's domain; returned per sample as "domains" (only the
    contrastive context loss reads it; the other models ignore it)."""
    ds.domain_names = sorted(set(sample_domain_names))
    index = {d: i for i, d in enumerate(ds.domain_names)}
    ds.sample_domain = [index[d] for d in sample_domain_names]


class DomainPersonTrainDataset(Dataset):
    """One sample = one source identity -> K augmented images of that person.

    instances_per_id=2, cross_camera=False is the historical sampler (random.choices, with
    replacement, cameras ignored). cross_camera=True draws the K images so they span at least two
    cameras whenever the person has them (first image random, second from another camera, the rest
    without replacement while possible).
    """

    def __init__(self, domain_datasets, transform=None, instances_per_id=2, cross_camera=False):
        self.transform = transform or TRAIN_TRANSFORM
        if instances_per_id < 2 or instances_per_id % 2:
            raise ValueError("instances_per_id must be an even number >= 2 (the ICL questions pair "
                             "consecutive images of one identity)")
        self.instances_per_id = instances_per_id
        self.cross_camera = cross_camera
        self.label2images = {}
        self.label2cams = {}
        self.pid2label = {}
        pid_offset = 0
        for domain_name, ds_obj in domain_datasets:
            pid_set = set()
            for img_path, pid, camid, dsetid in ds_obj.train:
                pid_set.add(pid)
            pid_map = {pid: pid + pid_offset for pid in sorted(pid_set)}
            pid_offset += max(pid_set) + 1 if pid_set else 0
            for img_path, pid, camid, dsetid in ds_obj.train:
                global_pid = pid_map[pid]
                key = "{}/{}".format(domain_name, global_pid)
                if key not in self.label2images:
                    self.label2images[key] = []
                    self.label2cams[key] = []
                    self.pid2label[key] = domain_name
                self.label2images[key].append(img_path)
                self.label2cams[key].append(camid)
        self.label2images = {k: v for k, v in self.label2images.items() if len(v) >= 2}
        self.pid2label = {k: v for k, v in self.pid2label.items() if k in self.label2images}
        self.keys = list(self.label2images.keys())
        self.num_ids = len(self.keys)  # one sample per identity: label = sample index
        _set_domain_index(self, [self.pid2label[k] for k in self.keys])

    def __len__(self):
        return len(self.keys)

    def _paths(self, key):
        paths = self.label2images[key]
        K = self.instances_per_id
        if not self.cross_camera:
            return random.choices(paths, k=K)  # the historical draw when K == 2
        cams = self.label2cams[key]
        first = random.randrange(len(paths))
        other = [j for j in range(len(paths)) if cams[j] != cams[first]]
        chosen = [first] + ([random.choice(other)] if other else [])
        rest = [j for j in range(len(paths)) if j not in chosen]
        random.shuffle(rest)
        while len(chosen) < K:
            chosen.append(rest.pop() if rest else random.randrange(len(paths)))
        # the cross-camera pair sits in the first two slots (the ICL questions use consecutive pairs)
        return [paths[j] for j in chosen]

    def __getitem__(self, idx):
        key = self.keys[idx]
        images = [self.transform(Image.open(p).convert("RGB")) for p in self._paths(key)]
        return {"image_crops": torch.stack(images), "labels": torch.tensor(idx, dtype=torch.long),
                "domains": torch.tensor(self.sample_domain[idx], dtype=torch.long)}


class CameraPairTrainDataset(Dataset):
    """Direction A (--pseudo_domains camera_pair): many training "domains" instead of two or three.

    A pseudo-domain is (source dataset, camera pair (a, b)); its samples are the identities seen by both
    cameras, and a sample gives K images alternating a, b, a, b, ... (random with replacement), so every
    consecutive pair used by the ICL questions is a cross-camera positive of that pseudo-domain. With
    --batch_domain_mode single every batch comes from one pseudo-domain, i.e. the context and the queries
    share the same camera pair, as a deployment shares one camera network. Pseudo-domains with fewer than
    min_ids identities are dropped. Labels are global identity indices (shared by an identity's samples
    in different camera pairs), so the ID-classification head has num_ids classes.
    """

    def __init__(self, domain_datasets, transform=None, instances_per_id=2, min_ids=8):
        self.transform = transform or TRAIN_TRANSFORM
        if instances_per_id < 2 or instances_per_id % 2:
            raise ValueError("instances_per_id must be an even number >= 2")
        self.instances_per_id = instances_per_id
        self.label2images, self.pid2label = {}, {}
        self.items = []  # (paths in camera a, paths in camera b, identity index)
        id_index = {}
        for domain_name, ds_obj in domain_datasets:
            by_pid = {}
            for img_path, pid, camid, *_ in ds_obj.train:
                by_pid.setdefault(pid, {}).setdefault(camid, []).append(img_path)
            groups = {}
            for pid, cams in by_pid.items():
                cs = sorted(cams)
                for i in range(len(cs)):
                    for j in range(i + 1, len(cs)):
                        groups.setdefault((cs[i], cs[j]), []).append(pid)
            for (a, b), pids in sorted(groups.items()):
                if len(pids) < min_ids:
                    continue
                group = "{}|c{}-c{}".format(domain_name, a, b)
                for pid in pids:
                    gid = id_index.setdefault((domain_name, pid), len(id_index))
                    key = "{}/{}".format(group, gid)  # the sampler groups batches by the part before "/"
                    self.label2images[key] = by_pid[pid][a] + by_pid[pid][b]
                    self.pid2label[key] = group
                    self.items.append((by_pid[pid][a], by_pid[pid][b], gid))
        self.keys = list(self.label2images.keys())
        self.num_ids = len(id_index)
        _set_domain_index(self, [self.pid2label[k] for k in self.keys])  # items and keys share the order
        print("camera-pair pseudo-domains: {} groups, {} samples, {} identities".format(
            len(set(self.pid2label.values())), len(self.items), self.num_ids))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        paths_a, paths_b, gid = self.items[idx]
        paths = [random.choice(paths_a if i % 2 == 0 else paths_b) for i in range(self.instances_per_id)]
        images = [self.transform(Image.open(p).convert("RGB")) for p in paths]
        return {"image_crops": torch.stack(images), "labels": torch.tensor(gid, dtype=torch.long),
                "domains": torch.tensor(self.sample_domain[idx], dtype=torch.long)}


def camera_unit(dataset, path, camid, with_time=True):
    """Camera unit of an image for --pseudo_domains camera_group: the camera ("c03"), and for MSMT17 also the
    time of day of the file name ("c03-morning" from 0303morning). "all" for datasets without cameras."""
    from adapters.config_reid import NO_CAMERA_DOMAINS
    if dataset in NO_CAMERA_DOMAINS:
        return "all"
    if dataset == "msmt17" and with_time:
        return "c{:02d}-{}".format(camid, os.path.basename(path).split("_")[3].lstrip("0123456789"))
    return "c{:02d}".format(camid)


class CameraGroupTrainDataset(Dataset):
    """Direction A (--pseudo_domains camera_group --camera_groups groups.json): a training "domain" is a group
    of camera units of one source dataset (scripts/group_prompts.py --stage cluster: units clustered by style,
    or camera pairs). A sample is an identity seen by >= 2 cameras of the group; its K images come in
    consecutive pairs from two different cameras of the group, so every ICL question pair is a cross-camera
    positive of that pseudo-domain. A source dataset absent from groups.json is one group "<dataset>|all"
    (for a dataset without cameras, any two images of a person form a pair). Labels are global identity
    indices; group names are "<dataset>|<group>", the keys of the teacher prompts (--prompt_teacher)."""

    def __init__(self, domain_datasets, groups_path, transform=None, instances_per_id=2, min_ids=8):
        import json
        from adapters.config_reid import NO_CAMERA_DOMAINS
        self.transform = transform or TRAIN_TRANSFORM
        if instances_per_id < 2 or instances_per_id % 2:
            raise ValueError("instances_per_id must be an even number >= 2")
        self.instances_per_id = instances_per_id
        with open(groups_path) as f:
            spec = json.load(f)
        self.label2images, self.pid2label = {}, {}
        self.items = []  # ([paths of camera 1], [paths of camera 2], ...), identity index
        id_index = {}
        for domain_name, ds_obj in domain_datasets:
            groups = spec.get(domain_name) or {"all": None}
            names = [u for us in groups.values() if us for u in us]
            with_time = not names or not all(len(u) == 3 for u in names)
            no_cam = domain_name in NO_CAMERA_DOMAINS
            by_unit = {}
            for img_path, pid, camid, *_ in ds_obj.train:
                u = camera_unit(domain_name, img_path, camid, with_time)
                by_unit.setdefault(u, {}).setdefault(pid, {}).setdefault(camid, []).append(img_path)
            for gname, units in sorted(groups.items()):
                by_pid = {}
                for u in (units if units else sorted(by_unit)):
                    for pid, cams in by_unit.get(u, {}).items():
                        for c, ps in cams.items():
                            by_pid.setdefault(pid, {}).setdefault(c, []).extend(ps)
                if no_cam:  # no camera labels: split a person's images into two halves as the "two cameras"
                    by_pid = {p: {0: sum(c.values(), [])} for p, c in by_pid.items()}
                    by_pid = {p: {0: c[0][::2], 1: c[0][1::2]} for p, c in by_pid.items() if len(c[0]) >= 2}
                pids = sorted(p for p, c in by_pid.items() if len(c) >= 2)
                if len(pids) < min_ids:
                    print("camera group {}|{}: {} identities < {}, dropped".format(domain_name, gname, len(pids), min_ids))
                    continue
                group = "{}|{}".format(domain_name, gname)
                for pid in pids:
                    gid = id_index.setdefault((domain_name, pid), len(id_index))
                    key = "{}/{}".format(group, gid)  # the sampler groups batches by the part before "/"
                    self.label2images[key] = sum(by_pid[pid].values(), [])
                    self.pid2label[key] = group
                    self.items.append(([v for _, v in sorted(by_pid[pid].items())], gid))
        self.keys = list(self.label2images.keys())
        self.num_ids = len(id_index)
        _set_domain_index(self, [self.pid2label[k] for k in self.keys])
        print("camera-group pseudo-domains: {} groups, {} samples, {} identities".format(
            len(self.domain_names), len(self.items), self.num_ids))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        cams, gid = self.items[idx]
        paths = []
        for _ in range(self.instances_per_id // 2):
            a, b = random.sample(range(len(cams)), 2)
            paths += [random.choice(cams[a]), random.choice(cams[b])]
        images = [self.transform(Image.open(p).convert("RGB")) for p in paths]
        return {"image_crops": torch.stack(images), "labels": torch.tensor(gid, dtype=torch.long),
                "domains": torch.tensor(self.sample_domain[idx], dtype=torch.long)}


class ContextPairDataset(Dataset):
    """Annotated context at test time: one cross-camera positive pair per identity.

    Uses the eval transform so the context is deterministic.
    """

    def __init__(self, pairs, transform=None):
        self.transform = transform or EVAL_TRANSFORM
        self.pairs = list(pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path1, img_path2 = self.pairs[idx]
        image1 = self.transform(Image.open(img_path1).convert("RGB"))
        image2 = self.transform(Image.open(img_path2).convert("RGB"))
        return {"image_crops": torch.stack([image1, image2]), "labels": torch.tensor(idx, dtype=torch.long)}


class DomainReIDEvalDataset(Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform or EVAL_TRANSFORM
        self.data = list(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, pid, camid, dsetid = self.data[idx]
        image = self.transform(Image.open(img_path).convert("RGB"))
        return image, torch.tensor(pid, dtype=torch.long), torch.tensor(camid, dtype=torch.long)
