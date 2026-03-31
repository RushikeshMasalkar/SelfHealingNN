from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]

CIFAR100_CLASSES = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "computer_keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]


def normalize_tensor(image: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=image.dtype, device=image.device).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=image.dtype, device=image.device).view(3, 1, 1)
    return (image - mean_t) / std_t


def denormalize_tensor(image: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=image.dtype, device=image.device).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=image.dtype, device=image.device).view(3, 1, 1)
    return torch.clamp(image * std_t + mean_t, 0.0, 1.0)


class NoiseInjector:
    """Applies synthetic noise to image tensors in [0, 1] range."""

    def gaussian_noise(self, image: torch.Tensor, std: float = 0.3) -> torch.Tensor:
        noisy = image + torch.randn_like(image) * std
        return torch.clamp(noisy, 0.0, 1.0)

    def salt_pepper(self, image: torch.Tensor, prob: float = 0.05) -> torch.Tensor:
        noisy = image.clone()
        salt_mask = torch.rand_like(noisy) < (prob / 2.0)
        pepper_mask = torch.rand_like(noisy) < (prob / 2.0)
        noisy[salt_mask] = 1.0
        noisy[pepper_mask] = 0.0
        return noisy

    def occlusion(self, image: torch.Tensor, patch_size: int = 8) -> torch.Tensor:
        noisy = image.clone()
        _, height, width = noisy.shape
        patch = min(patch_size, height, width)
        top = random.randint(0, height - patch)
        left = random.randint(0, width - patch)
        noisy[:, top : top + patch, left : left + patch] = 0.0
        return noisy

    def inject(self, image: torch.Tensor, noise_type: str = "gaussian", **kwargs) -> torch.Tensor:
        if noise_type == "gaussian":
            return self.gaussian_noise(image, std=kwargs.get("std", 0.3))
        if noise_type == "salt_pepper":
            return self.salt_pepper(image, prob=kwargs.get("prob", 0.05))
        if noise_type == "occlusion":
            return self.occlusion(image, patch_size=kwargs.get("patch_size", 8))
        raise ValueError(f"Unsupported noise_type: {noise_type}")


class NoisyCIFAR100(Dataset):
    """Wraps CIFAR-100 and returns (noisy_image, clean_image, label)."""

    def __init__(
        self,
        base_dataset: Dataset,
        noise_types: Optional[List[str]] = None,
        noise_params: Optional[Dict[str, float]] = None,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        fixed_noise_type: Optional[str] = None,
    ):
        self.base_dataset = base_dataset
        self.noise_types = noise_types or ["gaussian"]
        self.noise_params = noise_params or {}
        self.mean = mean or CIFAR100_MEAN
        self.std = std or CIFAR100_STD
        self.fixed_noise_type = fixed_noise_type
        self.injector = NoiseInjector()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        clean_image, label = self.base_dataset[idx]
        clean_image = clean_image.float()

        # Base CIFAR100 is normalized by transform; convert to [0,1] before corruption.
        clean_pixel = denormalize_tensor(clean_image, self.mean, self.std)

        noise_type = self.fixed_noise_type or random.choice(self.noise_types)
        noisy_pixel = self.injector.inject(
            clean_pixel,
            noise_type=noise_type,
            std=self.noise_params.get("gaussian_std", 0.3),
            prob=self.noise_params.get("salt_pepper_prob", 0.05),
            patch_size=int(self.noise_params.get("occlusion_size", 8)),
        )

        noisy_image = normalize_tensor(noisy_pixel, self.mean, self.std)
        clean_image = normalize_tensor(clean_pixel, self.mean, self.std)
        return noisy_image, clean_image, int(label)


def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset_cfg = config.get("dataset", {})
    noise_cfg = config.get("noise", {})
    training_cfg = config.get("training", {})

    root = dataset_cfg.get("root", "./data/raw/")
    train_split = float(dataset_cfg.get("train_split", 0.9))
    mean = dataset_cfg.get("mean", CIFAR100_MEAN)
    std = dataset_cfg.get("std", CIFAR100_STD)
    seed = int(training_cfg.get("seed", 42))
    requested_device = str(training_cfg.get("device", "auto")).strip().lower()
    use_cuda = torch.cuda.is_available() and requested_device in {"auto", "cuda"}

    num_workers = int(training_cfg.get("num_workers", 4 if use_cuda else 0))
    pin_memory_cfg = training_cfg.get("pin_memory", "auto")
    if isinstance(pin_memory_cfg, str):
        if pin_memory_cfg.strip().lower() == "auto":
            pin_memory = use_cuda
        else:
            pin_memory = pin_memory_cfg.strip().lower() in {"1", "true", "yes", "on"}
    else:
        pin_memory = bool(pin_memory_cfg)

    vae_batch = int(config.get("vae", {}).get("batch_size", 64))
    cls_batch = int(config.get("classifier", {}).get("batch_size", vae_batch))
    batch_size = max(vae_batch, cls_batch)

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_base = datasets.CIFAR100(root=root, train=True, transform=train_transform, download=True)
    val_base = datasets.CIFAR100(root=root, train=True, transform=eval_transform, download=True)
    test_base = datasets.CIFAR100(root=root, train=False, transform=eval_transform, download=True)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_base), generator=generator).tolist()
    split_idx = int(len(indices) * train_split)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    if len(val_indices) == 0:
        val_indices = train_indices[-1:]
        train_indices = train_indices[:-1]

    train_subset = Subset(train_base, train_indices)
    val_subset = Subset(val_base, val_indices)

    noise_params = {
        "gaussian_std": float(noise_cfg.get("gaussian_std", 0.3)),
        "salt_pepper_prob": float(noise_cfg.get("salt_pepper_prob", 0.05)),
        "occlusion_size": int(noise_cfg.get("occlusion_size", 8)),
    }
    noise_types = list(noise_cfg.get("types", ["gaussian", "salt_pepper", "occlusion"]))

    train_dataset = NoisyCIFAR100(train_subset, noise_types=noise_types, noise_params=noise_params, mean=mean, std=std)
    val_dataset = NoisyCIFAR100(val_subset, noise_types=noise_types, noise_params=noise_params, mean=mean, std=std)
    test_dataset = NoisyCIFAR100(test_base, noise_types=noise_types, noise_params=noise_params, mean=mean, std=std)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


# Backward compatibility exports for existing imports in package init.
ImageNet100Dataset = datasets.CIFAR100
NoisyImageNet100Dataset = NoisyCIFAR100
