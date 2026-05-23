"""
DeepfakeDataset — loads images from the split layout:

    images/train/real/  images/train/fake/
    images/val/real/    images/val/fake/
    images/test/real/   images/test/fake/

Label mapping: Real -> 0, Fake -> 1
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

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


def get_dataloader(split: str, model_name: str,
                   batch_size: int, num_workers: int) -> DataLoader:
    import torch as _torch
    dataset = DeepfakeDataset(split=split, model_name=model_name)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=_torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )