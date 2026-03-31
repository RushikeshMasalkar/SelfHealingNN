from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm


CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]


def normalize_batch(batch: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=batch.dtype, device=batch.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=batch.dtype, device=batch.device).view(1, 3, 1, 1)
    return (batch - mean_t) / std_t


def denormalize_batch(batch: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=batch.dtype, device=batch.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=batch.dtype, device=batch.device).view(1, 3, 1, 1)
    return torch.clamp(batch * std_t + mean_t, 0.0, 1.0)


def add_gaussian_noise(batch: torch.Tensor, std: float) -> torch.Tensor:
    return torch.clamp(batch + torch.randn_like(batch) * std, 0.0, 1.0)


def compute_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
    original_np = original.detach().cpu().permute(1, 2, 0).numpy()
    recon_np = reconstructed.detach().cpu().permute(1, 2, 0).numpy()

    psnr = peak_signal_noise_ratio(original_np, recon_np, data_range=1.0)
    try:
        ssim = structural_similarity(original_np, recon_np, data_range=1.0, multichannel=True)
    except TypeError:
        ssim = structural_similarity(original_np, recon_np, data_range=1.0, channel_axis=2)

    return {"psnr": float(psnr), "ssim": float(ssim)}


def save_comparison_grid(
    originals: torch.Tensor,
    noisy: torch.Tensor,
    reconstructed: torch.Tensor,
    path: str,
) -> None:
    n = min(8, originals.size(0))

    fig, axes = plt.subplots(4, n, figsize=(2 * n, 8))
    row_titles = ["Original", "Noisy", "Reconstructed", "Difference"]

    for i in range(n):
        original = originals[i].detach().cpu().permute(1, 2, 0).numpy()
        noisy_img = noisy[i].detach().cpu().permute(1, 2, 0).numpy()
        recon = reconstructed[i].detach().cpu().permute(1, 2, 0).numpy()
        diff = np.abs(original - recon)

        images = [original, noisy_img, recon, diff]
        for row in range(4):
            axes[row, i].imshow(images[row])
            axes[row, i].axis("off")
            if i == 0:
                axes[row, i].set_ylabel(row_titles[row], fontsize=10)

    plt.tight_layout()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def evaluate_pipeline(
    vae,
    classifier,
    test_loader,
    noise_levels: Iterable[float] = (0.1, 0.2, 0.3, 0.5, 0.7),
    device: str = "cpu",
) -> List[Dict[str, float]]:
    vae = vae.to(device).eval()
    classifier = classifier.to(device).eval()

    results: List[Dict[str, float]] = []

    with torch.no_grad():
        for noise_level in tqdm(list(noise_levels), desc="Evaluating noise levels"):
            stats = {
                "clean": {"correct": 0, "total": 0, "psnr": [], "ssim": []},
                "noisy": {"correct": 0, "total": 0, "psnr": [], "ssim": []},
                "healed": {"correct": 0, "total": 0, "psnr": [], "ssim": []},
            }

            preview_originals = None
            preview_noisy = None
            preview_recon = None

            for _, (noisy_norm, clean_norm, labels) in enumerate(test_loader):
                labels = labels.to(device)
                clean_norm = clean_norm.to(device)

                clean_pixel = denormalize_batch(clean_norm, CIFAR100_MEAN, CIFAR100_STD)
                noisy_pixel = add_gaussian_noise(clean_pixel, std=float(noise_level))
                noisy_norm = normalize_batch(noisy_pixel, CIFAR100_MEAN, CIFAR100_STD)

                clean_logits = classifier(clean_norm)
                noisy_logits = classifier(noisy_norm)

                recon_pixel, _, _ = vae(noisy_pixel)
                recon_norm = normalize_batch(recon_pixel, CIFAR100_MEAN, CIFAR100_STD)
                healed_logits = classifier(recon_norm)

                stats["clean"]["correct"] += int((clean_logits.argmax(dim=1) == labels).sum().item())
                stats["noisy"]["correct"] += int((noisy_logits.argmax(dim=1) == labels).sum().item())
                stats["healed"]["correct"] += int((healed_logits.argmax(dim=1) == labels).sum().item())

                batch_size = labels.size(0)
                stats["clean"]["total"] += batch_size
                stats["noisy"]["total"] += batch_size
                stats["healed"]["total"] += batch_size

                for idx in range(batch_size):
                    noisy_metrics = compute_metrics(clean_pixel[idx], noisy_pixel[idx])
                    healed_metrics = compute_metrics(clean_pixel[idx], recon_pixel[idx])

                    stats["noisy"]["psnr"].append(noisy_metrics["psnr"])
                    stats["noisy"]["ssim"].append(noisy_metrics["ssim"])
                    stats["healed"]["psnr"].append(healed_metrics["psnr"])
                    stats["healed"]["ssim"].append(healed_metrics["ssim"])

                if preview_originals is None:
                    preview_originals = clean_pixel[:8].detach().cpu()
                    preview_noisy = noisy_pixel[:8].detach().cpu()
                    preview_recon = recon_pixel[:8].detach().cpu()

            clean_acc = 100.0 * stats["clean"]["correct"] / max(stats["clean"]["total"], 1)
            noisy_acc = 100.0 * stats["noisy"]["correct"] / max(stats["noisy"]["total"], 1)
            healed_acc = 100.0 * stats["healed"]["correct"] / max(stats["healed"]["total"], 1)

            results.append(
                {
                    "noise_level": float(noise_level),
                    "condition": "Clean -> Classifier",
                    "accuracy": float(clean_acc),
                    "psnr": np.nan,
                    "ssim": np.nan,
                }
            )
            results.append(
                {
                    "noise_level": float(noise_level),
                    "condition": "Noisy -> Classifier",
                    "accuracy": float(noisy_acc),
                    "psnr": float(np.mean(stats["noisy"]["psnr"])) if stats["noisy"]["psnr"] else np.nan,
                    "ssim": float(np.mean(stats["noisy"]["ssim"])) if stats["noisy"]["ssim"] else np.nan,
                }
            )
            results.append(
                {
                    "noise_level": float(noise_level),
                    "condition": "Noisy -> VAE -> Classifier",
                    "accuracy": float(healed_acc),
                    "psnr": float(np.mean(stats["healed"]["psnr"])) if stats["healed"]["psnr"] else np.nan,
                    "ssim": float(np.mean(stats["healed"]["ssim"])) if stats["healed"]["ssim"] else np.nan,
                }
            )

            if preview_originals is not None:
                save_comparison_grid(
                    preview_originals,
                    preview_noisy,
                    preview_recon,
                    path="outputs/plots/comparison_grid.png",
                )

    return results


def save_results_csv(results_dict: List[Dict[str, float]]) -> None:
    output_path = Path("outputs/results/final_metrics.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["noise_level", "condition", "accuracy", "psnr", "ssim"]
    pd.DataFrame(results_dict, columns=columns).to_csv(output_path, index=False)
