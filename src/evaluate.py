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


def denormalize_imagenet(batch: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=batch.device, dtype=batch.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=batch.device, dtype=batch.dtype).view(1, 3, 1, 1)
    return torch.clamp(batch * std + mean, min=0.0, max=1.0)


def _to_unit_range_image(image: torch.Tensor) -> np.ndarray:
    if image.dim() != 3:
        raise ValueError("Expected CHW image tensor")
    img = image.detach()
    if img.min().item() < -0.25 or img.max().item() > 1.25:
        img = denormalize_imagenet(img.unsqueeze(0)).squeeze(0)
    else:
        img = torch.clamp(img, min=0.0, max=1.0)
    return img.cpu().permute(1, 2, 0).numpy()


def compute_psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    orig = _to_unit_range_image(original)
    rec = _to_unit_range_image(reconstructed)
    return float(peak_signal_noise_ratio(orig, rec, data_range=1.0))


def compute_ssim(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    orig = _to_unit_range_image(original)
    rec = _to_unit_range_image(reconstructed)
    return float(structural_similarity(orig, rec, channel_axis=2, data_range=1.0))


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
                    noisy_min, noisy_max = noisy_images.min().item(), noisy_images.max().item()
                    recon_min, recon_max = recon.min().item(), recon.max().item()
                    healed_min, healed_max = healed_input.min().item(), healed_input.max().item()
                    print(
                        "[EvalRanges] "
                        f"clean=[{images.min().item():.3f},{images.max().item():.3f}] "
                        f"noisy=[{noisy_min:.3f},{noisy_max:.3f}] "
                        f"recon=[{recon_min:.3f},{recon_max:.3f}] "
                        f"healed_input=[{healed_min:.3f},{healed_max:.3f}]"
                    )
                    # Check if recon looks like Sigmoid output
                    if recon_max <= 1.2 and recon_min >= -0.2:
                        print("[EvalWarn] Reconstruction appears to be in [0,1] range (Sigmoid pattern). May indicate training issue.")

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

    # Validation: warn if healing is making things worse
    if results:
        last_result = results[-1]
        if last_result["withHealing"] <= last_result["noHealing"]:
            print(
                f"[EvalWarn] At high noise ({last_result['noise']}): "
                f"Healing accuracy ({last_result['withHealing']:.2f}%) <= "
                f"No-healing accuracy ({last_result['noHealing']:.2f}%). "
                f"VAE is not improving classifier performance. Check tensor ranges and VAE training config."
            )
        if last_result["withHealing_conf"] < last_result["noHealing_conf"] * 0.95:
            print(
                f"[EvalWarn] At high noise ({last_result['noise']}): "
                f"Healing confidence ({last_result['withHealing_conf']:.4f}) much lower than "
                f"no-healing confidence ({last_result['noHealing_conf']:.4f}). "
                f"VAE outputs may be in wrong tensor space."
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
