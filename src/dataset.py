from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class DatasetConfig:
    root: str = "./data/raw"
    image_size: int = 224
    batch_size: int = 32
    train_split: float = 0.8
    num_workers: int = 4


class NoiseInjector:
    """Applies synthetic corruption to image tensors."""

    def add_gaussian_noise(self, image: torch.Tensor, std: float = 0.3) -> torch.Tensor:
        noise = torch.randn_like(image) * std
        return torch.clamp(image + noise, min=-3.0, max=3.0)

    def add_salt_pepper(self, image: torch.Tensor, prob: float = 0.1) -> torch.Tensor:
        out = image.clone()
        salt = torch.rand_like(out[0]) < (prob / 2.0)
        pepper = torch.rand_like(out[0]) < (prob / 2.0)
        ch = out.shape[0]
        min_val = float(out.min())
        max_val = float(out.max())
        for c in range(ch):
            out[c][salt] = max_val
            out[c][pepper] = min_val
        return out

    def add_occlusion(self, image: torch.Tensor, patch_size: int = 32) -> torch.Tensor:
        out = image.clone()
        _, h, w = out.shape
        patch = min(patch_size, h, w)
        top = random.randint(0, h - patch)
        left = random.randint(0, w - patch)
        out[:, top : top + patch, left : left + patch] = 0.0
        return out

    def apply(self, image: torch.Tensor, noise_type: str, **kwargs) -> torch.Tensor:
        if noise_type == "gaussian":
            return self.add_gaussian_noise(image, std=kwargs.get("std", 0.3))
        if noise_type == "salt_pepper":
            return self.add_salt_pepper(image, prob=kwargs.get("prob", 0.1))
        if noise_type == "occlusion":
            return self.add_occlusion(image, patch_size=kwargs.get("patch_size", 32))
        raise ValueError(f"Unsupported noise_type: {noise_type}")


class ImageNet100Dataset(Dataset):
    """ImageNet-100 loader based on torchvision.datasets.ImageFolder."""

    def __init__(self, root: str = "./data/raw", image_size: int = 224):
        self.root = Path(root)
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.dataset = datasets.ImageFolder(self.root, transform=self.transform)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]


class NoisyImageNet100Dataset(Dataset):
    """Wraps ImageNet100Dataset and injects noise into each sample."""

    def __init__(
        self,
        base_dataset: Dataset,
        noise_type: str = "gaussian",
        noise_params: Optional[Dict] = None,
        return_clean: bool = True,
    ):
        self.base_dataset = base_dataset
        self.noise_type = noise_type
        self.noise_params = noise_params or {}
        self.return_clean = return_clean
        self.injector = NoiseInjector()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        clean_image, label = self.base_dataset[idx]
        noisy_image = self.injector.apply(clean_image, self.noise_type, **self.noise_params)
        if self.return_clean:
            return noisy_image, clean_image, label
        return noisy_image, label


def _split_dataset(dataset: Dataset, train_split: float) -> Tuple[Subset, Subset, Subset]:
    n_total = len(dataset)
    n_train = int(n_total * train_split)
    n_remaining = n_total - n_train
    n_val = n_remaining // 2
    n_test = n_remaining - n_val
    return random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )


def get_dataloaders(
    root: str = "./data/raw",
    image_size: int = 224,
    batch_size: int = 32,
    train_split: float = 0.8,
    num_workers: int = 4,
    noisy: bool = False,
    noise_type: str = "gaussian",
    noise_params: Optional[Dict] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Returns train/val/test DataLoaders.

    If noisy=True, each batch returns noisy and clean tensors for reconstruction tasks.
    """

    dataset = ImageNet100Dataset(root=root, image_size=image_size)
    train_set, val_set, test_set = _split_dataset(dataset, train_split)

    if noisy:
        train_set = NoisyImageNet100Dataset(train_set, noise_type, noise_params, return_clean=True)
        val_set = NoisyImageNet100Dataset(val_set, noise_type, noise_params, return_clean=True)
        test_set = NoisyImageNet100Dataset(test_set, noise_type, noise_params, return_clean=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
