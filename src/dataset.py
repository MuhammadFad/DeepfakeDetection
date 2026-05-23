"""
DeepfakeDataset — loads images from the split layout:

    images/train/real/  images/train/fake/
    images/val/real/    images/val/fake/
    images/test/real/   images/test/fake/

Label mapping: Real -> 0, Fake -> 1
"""

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from .config import (
    IMAGE_SIZE, VIT_IMAGE_SIZE,
    IMAGENET_MEAN, IMAGENET_STD,
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    MODEL_VIT,
)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

_SPLIT_DIRS = {'train': TRAIN_DIR, 'val': VAL_DIR, 'test': TEST_DIR}


def _build_transform(split: str, target_size: int) -> transforms.Compose:
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.05),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.10), value=0),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def _build_cache_transform(target_size: int) -> transforms.Compose:
    """Non-augmentation transform used for the one-time GPU cache fill."""
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _scan_dir(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


class DeepfakeDataset(Dataset):
    def __init__(self, split: str = 'test', model_name: str = None):
        target_size = VIT_IMAGE_SIZE if model_name == MODEL_VIT else IMAGE_SIZE
        self.transform = _build_transform(split, target_size)
        self.target_size = target_size

        split_dir = Path(_SPLIT_DIRS.get(split, TEST_DIR))
        real_paths = _scan_dir(split_dir / 'real')
        fake_paths = _scan_dir(split_dir / 'fake')
        self.samples: list[tuple[Path, int]] = (
            [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img), label
        except Exception:
            return torch.zeros(3, self.target_size, self.target_size), label

    def num_real(self) -> int:
        return sum(1 for _, lb in self.samples if lb == 0)

    def num_fake(self) -> int:
        return sum(1 for _, lb in self.samples if lb == 1)


class GPUCachedDataset(Dataset):
    """
    Preloads the entire split onto the GPU as FP16 tensors.
    GPU-native augmentation eliminates CPU data-loading overhead.
    Only use for the train split on CUDA devices with sufficient VRAM.
    """

    def __init__(self, split: str, model_name: str, device: torch.device):
        self.split = split
        self.device = device
        target_size = VIT_IMAGE_SIZE if model_name == MODEL_VIT else IMAGE_SIZE
        self.target_size = target_size

        transform = _build_cache_transform(target_size)

        split_dir = Path(_SPLIT_DIRS.get(split, TEST_DIR))
        real_paths = _scan_dir(split_dir / 'real')
        fake_paths = _scan_dir(split_dir / 'fake')
        all_samples = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]

        n = len(all_samples)
        labels = []
        # Pre-allocate one contiguous CPU buffer — avoids the double-allocation
        # that torch.stack(list_of_tensors) causes (list ~5 GB + stacked ~5 GB = OOM).
        cpu_buf = torch.empty(n, 3, target_size, target_size, dtype=torch.float16)
        for i, (path, label) in enumerate(tqdm(all_samples, desc='Caching to GPU', ncols=80)):
            try:
                img = Image.open(path).convert('RGB')
                cpu_buf[i] = transform(img).half()
            except Exception:
                cpu_buf[i] = 0.0
            labels.append(label)

        try:
            self.data = cpu_buf.to(device)
            del cpu_buf
            self.labels = torch.tensor(labels, dtype=torch.int64).to(device)
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError(
                'GPU cache OOM — reduce batch size or use use_cache=False'
            )
        self.n = len(self.data)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GPU-native augmentation to a single FP16 (C, H, W) tensor."""
        if self.split != 'train':
            return x

        # RandomHorizontalFlip
        if torch.rand(1, device=self.device).item() < 0.5:
            x = torch.flip(x, dims=[2])

        # ColorJitter brightness ±0.3
        b_factor = torch.empty(1, device=self.device).uniform_(0.7, 1.3).item()
        x = (x * b_factor).clamp(-3.0, 3.0)

        # ColorJitter contrast ±0.3
        c_factor = torch.empty(1, device=self.device).uniform_(0.7, 1.3).item()
        mean_val = x.mean()
        x = ((x - mean_val) * c_factor + mean_val).clamp(-3.0, 3.0)

        # RandomRotation ±10° via affine grid (tensor-safe, no PIL dependency)
        angle = torch.empty(1).uniform_(-10.0, 10.0).item()
        if abs(angle) > 0.5:
            angle_rad = math.radians(angle)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            theta = torch.tensor(
                [[cos_a, -sin_a, 0.0],
                 [sin_a,  cos_a, 0.0]],
                dtype=torch.float32, device=self.device,
            ).unsqueeze(0)
            x_f = x.float().unsqueeze(0)
            grid = F.affine_grid(theta, x_f.shape, align_corners=False)
            x_f = F.grid_sample(x_f, grid, align_corners=False,
                                padding_mode='zeros', mode='bilinear')
            x = x_f.squeeze(0).half()

        # RandomErasing p=0.2, scale=(0.02, 0.10)
        if torch.rand(1, device=self.device).item() < 0.2:
            _, h, w = x.shape
            area = h * w
            erase_area = torch.empty(1).uniform_(0.02, 0.10).item() * area
            aspect = torch.empty(1).uniform_(0.3, 3.3).item()
            eh = min(int(math.sqrt(erase_area * aspect)), h)
            ew = min(int(math.sqrt(erase_area / aspect)), w)
            if eh > 0 and ew > 0:
                top  = torch.randint(0, h - eh + 1, (1,), device=self.device).item()
                left = torch.randint(0, w - ew + 1, (1,), device=self.device).item()
                x[:, top:top + eh, left:left + ew] = 0.0

        return x

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        x = self.data[idx].clone()
        return self._augment(x), self.labels[idx]


def get_dataloader(split: str, model_name: str,
                   batch_size: int, num_workers: int,
                   use_cache: bool = False,
                   device=None) -> DataLoader:
    import torch as _torch
    if use_cache and split == 'train':
        assert num_workers == 0, 'use_cache=True requires num_workers=0 (data is already on GPU)'
        _device = device if device is not None else _torch.device('cuda' if _torch.cuda.is_available() else 'cpu')
        dataset = GPUCachedDataset(split=split, model_name=model_name, device=_device)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

    dataset = DeepfakeDataset(split=split, model_name=model_name)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=_torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
