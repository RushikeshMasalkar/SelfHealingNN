from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision.utils import make_grid

from .dataset import NoiseInjector


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_imagenet(batch: torch.Tensor) -> torch.Tensor:
    """Converts [0, 1] image tensors into ImageNet normalized space."""
    mean = torch.tensor(IMAGENET_MEAN, device=batch.device, dtype=batch.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=batch.device, dtype=batch.dtype).view(1, 3, 1, 1)
    return (batch - mean) / std


def compute_psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    orig = original.detach().cpu().permute(1, 2, 0).numpy()
    rec = reconstructed.detach().cpu().permute(1, 2, 0).numpy()
    data_range = float(orig.max() - orig.min()) or 1.0
    return float(peak_signal_noise_ratio(orig, rec, data_range=data_range))


def compute_ssim(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    orig = original.detach().cpu().permute(1, 2, 0).numpy()
    rec = reconstructed.detach().cpu().permute(1, 2, 0).numpy()
    data_range = float(orig.max() - orig.min()) or 1.0
    return float(structural_similarity(orig, rec, channel_axis=2, data_range=data_range))


def _accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())


def _topk_accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    _, pred = logits.topk(k, dim=1)
    correct = pred.eq(labels.view(-1, 1)).any(dim=1).float().mean().item()
    return float(correct)


def _mean_confidence_from_logits(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=1)
    return float(probs.max(dim=1).values.mean().item())


def evaluate_pipeline(
    vae,
    classifier,
    test_loader,
    device: str = "cuda",
    noise_levels: Iterable[float] = (0.1, 0.2, 0.3, 0.5, 0.7),
    vae_outputs_denormalized: bool = True,
    debug_ranges: bool = False,
) -> List[Dict[str, float]]:
    """Evaluates Clean->Classifier, Noisy->Classifier, and Noisy->VAE->Classifier."""

    vae = vae.to(device).eval()
    classifier = classifier.to(device).eval()
    injector = NoiseInjector()

    results = []
    with torch.no_grad():
        for level in noise_levels:
            clean_acc = []
            noisy_acc = []
            healed_acc = []
            clean_top5 = []
            noisy_top5 = []
            healed_top5 = []
            clean_conf = []
            noisy_conf = []
            healed_conf = []

            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                clean_logits = classifier(images)
                clean_acc.append(_accuracy_from_logits(clean_logits, labels))
                clean_top5.append(_topk_accuracy_from_logits(clean_logits, labels, k=5))
                clean_conf.append(_mean_confidence_from_logits(clean_logits))

                noisy_images = torch.stack([
                    injector.add_gaussian_noise(img, std=level) for img in images
                ])
                noisy_logits = classifier(noisy_images)
                noisy_acc.append(_accuracy_from_logits(noisy_logits, labels))
                noisy_top5.append(_topk_accuracy_from_logits(noisy_logits, labels, k=5))
                noisy_conf.append(_mean_confidence_from_logits(noisy_logits))

                recon, _, _ = vae(noisy_images)
                healed_input = normalize_imagenet(recon) if vae_outputs_denormalized else recon

                if debug_ranges:
                    print(
                        "[EvalRanges] "
                        f"clean=[{images.min().item():.3f},{images.max().item():.3f}] "
                        f"noisy=[{noisy_images.min().item():.3f},{noisy_images.max().item():.3f}] "
                        f"recon=[{recon.min().item():.3f},{recon.max().item():.3f}] "
                        f"healed_input=[{healed_input.min().item():.3f},{healed_input.max().item():.3f}]"
                    )
                    debug_ranges = False

                healed_logits = classifier(healed_input)
                healed_acc.append(_accuracy_from_logits(healed_logits, labels))
                healed_top5.append(_topk_accuracy_from_logits(healed_logits, labels, k=5))
                healed_conf.append(_mean_confidence_from_logits(healed_logits))

            results.append(
                {
                    "noise": float(level),
                    "clean": float(np.mean(clean_acc) * 100.0),
                    "noHealing": float(np.mean(noisy_acc) * 100.0),
                    "withHealing": float(np.mean(healed_acc) * 100.0),
                    "clean_top5": float(np.mean(clean_top5) * 100.0),
                    "noHealing_top5": float(np.mean(noisy_top5) * 100.0),
                    "withHealing_top5": float(np.mean(healed_top5) * 100.0),
                    "clean_conf": float(np.mean(clean_conf)),
                    "noHealing_conf": float(np.mean(noisy_conf)),
                    "withHealing_conf": float(np.mean(healed_conf)),
                }
            )

    return results


def save_comparison_grid(
    originals: torch.Tensor,
    noisy: torch.Tensor,
    reconstructed: torch.Tensor,
    save_path: str = "outputs/plots/comparison_grid.png",
    max_samples: int = 8,
) -> None:
    """Saves grid as [Original | Noisy | Reconstructed] for up to 8 samples."""

    n = min(max_samples, originals.size(0))
    triplets = []
    for i in range(n):
        triplets.extend([originals[i].cpu(), noisy[i].cpu(), reconstructed[i].cpu()])

    grid = make_grid(triplets, nrow=3, normalize=True)
    np_grid = grid.permute(1, 2, 0).numpy()

    target = Path(save_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 2 * n))
    plt.imshow(np_grid)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(target, dpi=200)
    plt.close()
