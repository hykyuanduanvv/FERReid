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

# Default to TransReID pedestrian transforms
TRAIN_TRANSFORM = PEDESTRIAN_TRAIN_TRANSFORM
EVAL_TRANSFORM = PEDESTRIAN_EVAL_TRANSFORM


def _set_domain_index(ds, sample_domain_names):
    """ds.domain_names (sorted training domains, i.e. what the batch sampler groups by) and
    ds.sample_domain[i] = index of sample i's domain (returned per sample as "domains"; the models ignore it)."""
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


class PathDataset(Dataset):
    """Images by path with the eval transform; returns (image, position) so a DataLoader keeps the order."""

    def __init__(self, paths, transform=None):
        self.transform = transform or EVAL_TRANSFORM
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert("RGB")), idx
