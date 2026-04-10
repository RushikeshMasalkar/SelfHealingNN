from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import confusion_matrix, f1_score
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

from .dataset import NoiseInjector


def load_config(config_path: str = "configs/config.yaml") -> Dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


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


def corrupt_batch_pixels(
    clean_pixel: torch.Tensor,
    noise_type: str,
    injector: NoiseInjector,
    gaussian_std: float,
    salt_pepper_prob: float,
) -> torch.Tensor:
    kwargs = {
        "gaussian": {"std": gaussian_std},
        "salt_pepper": {"prob": salt_pepper_prob},
    }[noise_type]
    out = []
    for i in range(clean_pixel.size(0)):
        out.append(injector.inject(clean_pixel[i], noise_type=noise_type, **kwargs))
    return torch.stack(out, dim=0)


def top5_correct_count(logits: torch.Tensor, labels: torch.Tensor) -> int:
    top5 = logits.topk(5, dim=1).indices
    correct = top5.eq(labels.view(-1, 1)).any(dim=1)
    return int(correct.sum().item())


def compute_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
    original_np = original.detach().cpu().permute(1, 2, 0).numpy()
    recon_np = reconstructed.detach().cpu().permute(1, 2, 0).numpy()

    mse = float(np.mean((original_np - recon_np) ** 2))
    if mse <= 0.0:
        psnr = float("inf")
    else:
        psnr = float(10.0 * np.log10(1.0 / mse))
    min_side = min(original_np.shape[0], original_np.shape[1])
    win_size = min(7, min_side)
    if win_size % 2 == 0:
        win_size = max(3, win_size - 1)

    try:
        ssim = structural_similarity(
            original_np,
            recon_np,
            data_range=1.0,
            channel_axis=-1,
            win_size=win_size,
        )
    except TypeError:
        ssim = structural_similarity(
            original_np,
            recon_np,
            data_range=1.0,
            multichannel=True,
            win_size=win_size,
        )

    return {"psnr": float(psnr), "ssim": float(ssim)}


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        ece += (cnt / n) * float(np.abs(accuracies[mask].mean() - confidences[mask].mean()))
    return float(ece)


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


def save_confusion_matrix_plot(cm: np.ndarray, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


class _AggStats:
    def __init__(self) -> None:
        self.correct = 0
        self.top5_correct = 0
        self.total = 0
        self.psnr: List[float] = []
        self.ssim: List[float] = []
        self.all_preds: List[int] = []
        self.all_labels: List[int] = []
        self.all_probs: List[np.ndarray] = []

    def add_batch(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        psnr_ssim_pairs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        collect_probs: bool = False,
    ) -> None:
        bs = labels.size(0)
        self.correct += int((logits.argmax(dim=1) == labels).sum().item())
        self.top5_correct += top5_correct_count(logits, labels)
        self.total += bs
        if psnr_ssim_pairs:
            for a, b in psnr_ssim_pairs:
                m = compute_metrics(a, b)
                self.psnr.append(m["psnr"])
                self.ssim.append(m["ssim"])
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        labs = labels.detach().cpu().numpy()
        self.all_preds.extend(preds.tolist())
        self.all_labels.extend(labs.tolist())
        if collect_probs:
            self.all_probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())

    def finalize(self) -> Dict[str, float]:
        t = max(self.total, 1)
        out: Dict[str, float] = {
            "accuracy": 100.0 * self.correct / t,
            "top5_accuracy": 100.0 * self.top5_correct / t,
            "psnr": float(np.mean(self.psnr)) if self.psnr else float("nan"),
            "ssim": float(np.mean(self.ssim)) if self.ssim else float("nan"),
        }
        if self.all_preds:
            out["macro_f1"] = float(
                f1_score(self.all_labels, self.all_preds, average="macro", zero_division=0)
            )
        else:
            out["macro_f1"] = float("nan")
        if self.all_probs:
            probs = np.concatenate(self.all_probs, axis=0)
            labs = np.array(self.all_labels, dtype=np.int64)
            out["ece"] = expected_calibration_error(probs, labs)
        else:
            out["ece"] = float("nan")
        return out


def _evaluate_corruption_setting(
    vae: torch.nn.Module,
    classifier: torch.nn.Module,
    test_loader,
    mean: List[float],
    std: List[float],
    device: torch.device,
    noisy_pixel_fn: Callable[[torch.Tensor], torch.Tensor],
    protocol: str,
    noise_type_label: str,
    severity_label: str,
    collect_matrix_key: Optional[str] = None,
    save_preview: bool = False,
) -> Tuple[List[Dict[str, float]], Optional[Tuple[np.ndarray, str]]]:
    vae.eval()
    classifier.eval()
    stats_clean = _AggStats()
    stats_noisy = _AggStats()
    stats_healed = _AggStats()
    preview = None

    collect_probs_clean = collect_matrix_key == "clean"
    collect_probs_healed = collect_matrix_key == "healed"

    with torch.no_grad():
        for _, (_, clean_norm, labels) in enumerate(test_loader):
            labels = labels.to(device)
            clean_norm = clean_norm.to(device)
            clean_pixel = denormalize_batch(clean_norm, mean, std)
            noisy_pixel = noisy_pixel_fn(clean_pixel)
            noisy_norm = normalize_batch(noisy_pixel, mean, std)

            clean_logits = classifier(clean_norm)
            noisy_logits = classifier(noisy_norm)
            recon_pixel, _, _ = vae(noisy_pixel)
            recon_norm = normalize_batch(recon_pixel, mean, std)
            healed_logits = classifier(recon_norm)

            batch_psnr_noisy = [(clean_pixel[i], noisy_pixel[i]) for i in range(labels.size(0))]
            batch_psnr_healed = [(clean_pixel[i], recon_pixel[i]) for i in range(labels.size(0))]

            stats_clean.add_batch(clean_logits, labels, collect_probs=collect_probs_clean)
            stats_noisy.add_batch(noisy_logits, labels, batch_psnr_noisy)
            stats_healed.add_batch(
                healed_logits,
                labels,
                batch_psnr_healed,
                collect_probs=collect_probs_healed,
            )

            if preview is None:
                preview = (
                    clean_pixel[:8].detach().cpu(),
                    noisy_pixel[:8].detach().cpu(),
                    recon_pixel[:8].detach().cpu(),
                )

    rows: List[Dict[str, float]] = []
    condition_map = [
        ("clean", "Clean -> Classifier", stats_clean),
        ("noisy", "Noisy -> Classifier", stats_noisy),
        ("healed", "Noisy -> VAE -> Classifier", stats_healed),
    ]

    cm_out: Optional[Tuple[np.ndarray, str]] = None
    for key, condition, st in condition_map:
        fin = st.finalize()
        if collect_matrix_key == key and st.all_preds:
            cm = confusion_matrix(st.all_labels, st.all_preds, labels=np.arange(100))
            safe_sev = severity_label.replace("=", "_").replace(".", "p")
            cm_out = (cm, f"{protocol}_{noise_type_label}_{safe_sev}_{key}")
        noise_level_val = float("nan")
        if protocol == "gaussian_stress" and severity_label.startswith("stress_std="):
            try:
                noise_level_val = float(severity_label.split("=", 1)[1])
            except ValueError:
                pass

        row = {
            "protocol": protocol,
            "noise_type": noise_type_label,
            "severity": severity_label,
            "noise_level": noise_level_val,
            "condition": condition,
            "accuracy": fin["accuracy"],
            "top5_accuracy": fin["top5_accuracy"],
            "macro_f1": fin["macro_f1"],
            "ece": fin["ece"],
            "psnr": fin["psnr"],
            "ssim": fin["ssim"],
        }
        rows.append(row)

    if save_preview and preview is not None:
        o, n, r = preview
        save_comparison_grid(o, n, r, path="outputs/plots/comparison_grid.png")

    return rows, cm_out


def evaluate_pipeline(
    vae,
    classifier,
    test_loader,
    config_path: str = "configs/config.yaml",
    device: str = "cpu",
    noise_levels: Optional[Iterable[float]] = None,
) -> List[Dict[str, float]]:
    """
    Run evaluation:
    - train_matched: each noise type from config at training severities
    - gaussian_stress: extra Gaussian-only sweep (stress test; severities from config or noise_levels)
    """
    config = load_config(config_path)
    dataset_cfg = config.get("dataset", {})
    noise_cfg = config.get("noise", {})
    eval_cfg = config.get("evaluation", {})

    mean = list(dataset_cfg.get("mean", [0.5071, 0.4865, 0.4409]))
    std = list(dataset_cfg.get("std", [0.2673, 0.2564, 0.2761]))
    dev = torch.device(device)

    gaussian_std = float(noise_cfg.get("gaussian_std", 0.15))
    salt_pepper_prob = float(noise_cfg.get("salt_pepper_prob", 0.05))
    noise_types = list(noise_cfg.get("types", ["gaussian", "salt_pepper"]))

    stress_levels = list(noise_levels) if noise_levels is not None else list(
        eval_cfg.get("gaussian_stress_levels", [0.1, 0.2, 0.3, 0.5, 0.7])
    )

    injector = NoiseInjector()
    all_rows: List[Dict[str, float]] = []
    confusion_to_save: List[Tuple[np.ndarray, str]] = []

    vae = vae.to(dev)
    classifier = classifier.to(dev)

    rows_baseline, cm_clean = _evaluate_corruption_setting(
        vae,
        classifier,
        test_loader,
        mean,
        std,
        dev,
        lambda cp: cp.clone(),
        protocol="train_matched",
        noise_type_label="none",
        severity_label="identity",
        collect_matrix_key="clean",
        save_preview=True,
    )
    all_rows.extend(rows_baseline)
    if cm_clean:
        confusion_to_save.append(cm_clean)

    train_matched_healed_cm_pending = True
    for nt in noise_types:
        collect_h = "healed" if train_matched_healed_cm_pending else None
        if nt == "gaussian":
            fn_g = lambda cp, gs=gaussian_std: add_gaussian_noise(cp, gs)
            rows, cm = _evaluate_corruption_setting(
                vae,
                classifier,
                test_loader,
                mean,
                std,
                dev,
                fn_g,
                protocol="train_matched",
                noise_type_label="gaussian",
                severity_label=f"std={gaussian_std}",
                collect_matrix_key=collect_h,
            )
            if train_matched_healed_cm_pending and cm:
                train_matched_healed_cm_pending = False
            all_rows.extend(rows)
            if cm:
                confusion_to_save.append(cm)

        elif nt == "salt_pepper":

            def fn_sp(cp: torch.Tensor) -> torch.Tensor:
                return corrupt_batch_pixels(cp, "salt_pepper", injector, gaussian_std, salt_pepper_prob)

            rows, cm = _evaluate_corruption_setting(
                vae,
                classifier,
                test_loader,
                mean,
                std,
                dev,
                fn_sp,
                protocol="train_matched",
                noise_type_label="salt_pepper",
                severity_label=f"prob={salt_pepper_prob}",
                collect_matrix_key=collect_h,
            )
            if train_matched_healed_cm_pending and cm:
                train_matched_healed_cm_pending = False
            all_rows.extend(rows)
            if cm:
                confusion_to_save.append(cm)

    for level in tqdm(stress_levels, desc="Gaussian stress sweep"):
        rows, _ = _evaluate_corruption_setting(
            vae,
            classifier,
            test_loader,
            mean,
            std,
            dev,
            lambda cp, lv=level: add_gaussian_noise(cp, float(lv)),
            protocol="gaussian_stress",
            noise_type_label="gaussian",
            severity_label=f"stress_std={level}",
        )
        all_rows.extend(rows)

    # Display confusion matrices at the end of evaluation instead of saving
    for cm, name in confusion_to_save:
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title(f"Confusion Matrix: {name}")
        plt.colorbar()
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.show()

    return all_rows


def save_results_csv(results_dict: List[Dict[str, float]]) -> None:
    # Print simplified summary table for user (No longer saving to CSV)
    print("\n" + "="*115)
    print(f"{'PROTOCOL':<20} | {'SEVERITY':<15} | {'ACC%':<8} | {'TOP-5%':<8} | {'PSNR':<8} | {'SSIM':<8} | {'MACRO F1':<10}")
    print("-" * 115)
    
    # Filter for interesting conditions: 'Noisy -> VAE -> Classifier' represents the healed data performance
    for res in results_dict:
        if res['condition'] == 'Noisy -> VAE -> Classifier':
            prot = res.get('protocol', 'N/A')
            sev = res.get('severity_label', res.get('severity', 'N/A'))
            acc = res.get('accuracy', 0.0)
            top5 = res.get('top5_accuracy', 0.0)
            psnr = res.get('psnr', 0.0)
            ssim = res.get('ssim', 0.0)
            f1 = res.get('macro_f1', 0.0)
            
            print(f"{prot:<20} | {sev:<15} | {acc:<8.1f} | {top5:<8.1f} | {psnr:<8.2f} | {ssim:<8.3f} | {f1:<10.3f}")
    
    print("="*115)
    print("\n* TOP-5 Accuracy: Measures if the correct class was among the model's top 5 guesses.")
    print("  Essential for CIFAR-100 where many classes (e.g., 'maple' vs 'oak') are very similar.")
    print("="*115 + "\n")


if __name__ == "__main__":
    import sys

    # Support running dynamically from anywhere by making the project root part of PYTHONPATH
    root_dir = Path(__file__).resolve().parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    from src.classifier import get_classifier
    from src.conv_vae import ConvVAE
    from src.dataset import get_dataloaders

    print("Loading config...")
    config_path = str(root_dir / "configs/config.yaml")
    config = load_config(config_path)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")

    print("Loading test dataloader...")
    # 5. DATALOADER RETURNS WRONG SPLIT: Index [2] uses the correct test_loader
    _, _, test_loader = get_dataloaders(config)

    print("Loading models...")
    # 1. MODEL LOADING BUG FIX: Instantiate CIFAR100Classifier correctly
    classifier = get_classifier(config)
    classifier_path = root_dir / "models/resnet_classifier.pth"
    classifier.load_state_dict(torch.load(str(classifier_path), map_location=device))

    vae_path = root_dir / "models/conv_vae_best.pth"
    vae = ConvVAE(latent_dim=int(config.get("vae", {}).get("latent_dim", 256)))
    vae.load_state_dict(torch.load(str(vae_path), map_location=device))

    # 3. MODEL NOT IN EVAL MODE: Must explicitly set eval()
    classifier.eval()
    vae.eval()

    classifier.to(device)
    vae.to(device)

    # Sanity Check implementation
    print("Running Sanity Check on 1 test batch (clean images)...")
    batch_correct = 0
    batch_total = 0

    # Ensure no_grad() is used for evaluation
    with torch.no_grad():
        for _, clean_norm, labels in test_loader:
            clean_norm = clean_norm.to(device)
            labels = labels.to(device)

            logits = classifier(clean_norm)
            preds = logits.argmax(dim=1)

            batch_correct += (preds == labels).sum().item()
            batch_total += labels.size(0)
            break  # Single batch for sanity check

    accuracy = batch_correct / max(batch_total, 1)
    print(f"Sanity Check Batch Accuracy: {accuracy * 100:.2f}%")

    if accuracy < 0.10:
        raise RuntimeError(
            f"Sanity check failed! Accuracy is {accuracy * 100:.2f}% (expected ~75-80%). Please check:\n"
            "1. Model Loading Bug: Ensure CIFAR100Classifier is loaded, not raw resnet18.\n"
            "2. Normalization Mismatch: Ensure CIFAR-100 mean/std are used, not ImageNet.\n"
            "3. Model Not in Eval Mode: Ensure classifier.eval() is called.\n"
            "4. Dataloader Split: Ensure test split (index [2]) is used."
        )
    print("Sanity check passed! Commencing evaluation pipeline...\n")

    # 2. NORMALIZATION / 4. VAE DENORM CONSISTENCY: 
    # eval_pipeline correctly fetches CIFAR100 stats from configs and chains denorm->VAE->norm natively.
    all_results = evaluate_pipeline(
        vae=vae,
        classifier=classifier,
        test_loader=test_loader,
        config_path=config_path,
        device=device_str,
    )

    save_results_csv(all_results)
    print("Evaluation complete. Results saved to outputs/results/final_metrics.csv")
