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


class DomainPersonTrainDataset(Dataset):
    def __init__(self, domain_datasets, transform=None):
        self.transform = transform or TRAIN_TRANSFORM
        self.label2images = {}
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
                    self.pid2label[key] = domain_name
                self.label2images[key].append(img_path)
        self.label2images = {k: v for k, v in self.label2images.items() if len(v) >= 2}
        self.pid2label = {k: v for k, v in self.pid2label.items() if k in self.label2images}
        self.keys = list(self.label2images.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        img_path1, img_path2 = random.choices(self.label2images[key], k=2)
        image1 = self.transform(Image.open(img_path1).convert("RGB"))
        image2 = self.transform(Image.open(img_path2).convert("RGB"))
        return {"image_crops": torch.stack([image1, image2]), "labels": torch.tensor(idx, dtype=torch.long)}


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
