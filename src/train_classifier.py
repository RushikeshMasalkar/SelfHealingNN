from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .classifier import CIFAR100Classifier, get_classifier
from .conv_vae import ConvVAE
from .dataset import get_dataloaders


def load_config(config_path: str = "configs/config.yaml") -> Dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(training_cfg: Dict) -> torch.device:
    configured = str(training_cfg.get("device", "auto")).strip().lower()
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if configured == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(configured)


def denormalize_batch(batch: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=batch.dtype, device=batch.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=batch.dtype, device=batch.device).view(1, 3, 1, 1)
    return torch.clamp(batch * std_t + mean_t, 0.0, 1.0)


def normalize_batch(batch: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=batch.dtype, device=batch.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=batch.dtype, device=batch.device).view(1, 3, 1, 1)
    return (batch - mean_t) / std_t


def top5_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    top5 = logits.topk(5, dim=1).indices
    correct = top5.eq(labels.view(-1, 1)).any(dim=1)
    return float(correct.float().mean().item())


def run_classifier_epoch(
    classifier: CIFAR100Classifier,
    vae: ConvVAE,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    mean: List[float],
    std: List[float],
    train: bool,
    phase_label: str,
    clean_mix_prob: float,
) -> Tuple[float, float, float, float, float]:
    if train:
        classifier.train()
    else:
        classifier.eval()

    total_loss = 0.0
    total_top1_combined = 0.0
    total_top5_combined = 0.0
    total_top1_clean = 0.0
    total_top1_healed = 0.0
    batch_count = 0

    loop = tqdm(loader, desc=phase_label, leave=False)
    for noisy, clean, labels in loop:
        noisy = noisy.to(device)
        clean = clean.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            noisy_pixel = denormalize_batch(noisy, mean, std)
            cleaned_pixel, _, _ = vae(noisy_pixel)
            cleaned = normalize_batch(cleaned_pixel, mean, std)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            if train:
                batch_size = clean.size(0)
                use_clean = torch.rand(batch_size, device=device) < float(clean_mix_prob)
                mask = use_clean.view(-1, 1, 1, 1).expand_as(clean)
                classifier_input = torch.where(mask, clean, cleaned)
                logits = classifier(classifier_input)
                loss = criterion(logits, labels)
                if train:
                    loss.backward()
                    optimizer.step()
            else:
                logits_clean = classifier(clean)
                logits_healed = classifier(cleaned)
                loss = 0.5 * (criterion(logits_clean, labels) + criterion(logits_healed, labels))
                logits = logits_healed

        if train:
            top1 = float((logits.argmax(dim=1) == labels).float().mean().item())
            top5 = top5_accuracy(logits, labels)
            top1_clean_br = float("nan")
            top1_healed_br = float("nan")
        else:
            top1_clean_br = float((logits_clean.argmax(dim=1) == labels).float().mean().item())
            top1_healed_br = float((logits_healed.argmax(dim=1) == labels).float().mean().item())
            top1 = 0.5 * (top1_clean_br + top1_healed_br)
            top5 = 0.5 * (top5_accuracy(logits_clean, labels) + top5_accuracy(logits_healed, labels))

        total_loss += float(loss.item())
        total_top1_combined += top1
        total_top5_combined += top5
        total_top1_clean += top1_clean_br if not train else 0.0
        total_top1_healed += top1_healed_br if not train else 0.0
        batch_count += 1

    batch_count = max(batch_count, 1)
    if train:
        return (
            total_loss / batch_count,
            total_top1_combined / batch_count,
            total_top5_combined / batch_count,
            float("nan"),
            float("nan"),
        )
    return (
        total_loss / batch_count,
        total_top1_combined / batch_count,
        total_top5_combined / batch_count,
        total_top1_clean / batch_count,
        total_top1_healed / batch_count,
    )


def train_classifier_model(config_path: str = "configs/config.yaml") -> Dict[str, List[float]]:
    config = load_config(config_path)
    train_cfg = config.get("training", {})

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    device = resolve_device(train_cfg)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataset_cfg = config.get("dataset", {})
    classifier_cfg = config.get("classifier", {})

    mean = dataset_cfg.get("mean", [0.5071, 0.4865, 0.4409])
    std = dataset_cfg.get("std", [0.2673, 0.2564, 0.2761])

    clean_mix_prob = float(classifier_cfg.get("clean_input_mix_prob", 0.5))
    phase1_epochs = int(classifier_cfg.get("phase1_epochs", 10))
    phase2_epochs = int(classifier_cfg.get("phase2_epochs", 40))
    head_lr = float(classifier_cfg.get("head_learning_rate", 1e-3))
    finetune_lr = float(classifier_cfg.get("learning_rate", 1e-4))
    patience = int(classifier_cfg.get("patience", 7))
    min_epochs_before_early_stop = int(classifier_cfg.get("min_epochs_before_early_stop", patience))
    target_train_acc = float(classifier_cfg.get("target_train_acc", 0.8))

    train_loader, val_loader, _ = get_dataloaders(config)

    vae = ConvVAE(latent_dim=int(config.get("vae", {}).get("latent_dim", 256))).to(device)
    root_dir = Path(config_path).resolve().parent.parent
    vae.load_state_dict(torch.load(str(root_dir / "models/conv_vae_best.pth"), map_location=device))
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    classifier = get_classifier(config).to(device)
    criterion = nn.CrossEntropyLoss()

    models_dir = root_dir / "models"
    results_dir = root_dir / "outputs/results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    history_rows: List[Dict[str, float]] = []
    best_val_acc = 0.0
    best_path = models_dir / "resnet_classifier.pth"

    patience_counter = 0
    global_epoch = 0

    classifier.freeze_backbone()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, classifier.parameters()),
        lr=head_lr,
        weight_decay=1e-4,
    )

    for epoch in range(1, phase1_epochs + 1):
        global_epoch += 1
        train_loss, train_acc, train_top5, _, _ = run_classifier_epoch(
            classifier,
            vae,
            train_loader,
            criterion,
            optimizer,
            device,
            mean,
            std,
            train=True,
            phase_label=f"Phase 1 Train {epoch}/{phase1_epochs}",
            clean_mix_prob=clean_mix_prob,
        )
        with torch.no_grad():
            val_loss, val_acc, val_top5, val_clean_acc, val_healed_acc = run_classifier_epoch(
                classifier,
                vae,
                val_loader,
                criterion,
                optimizer,
                device,
                mean,
                std,
                train=False,
                phase_label=f"Phase 1 Val {epoch}/{phase1_epochs}",
                clean_mix_prob=clean_mix_prob,
            )

        lr = float(optimizer.param_groups[0]["lr"])
        print(
            f"Phase 1 | Epoch {epoch}/{phase1_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc (avg): {val_acc:.4f} | Val Clean: {val_clean_acc:.4f} | "
            f"Val Healed: {val_healed_acc:.4f} | Val Top5: {val_top5:.4f} | LR: {lr:.6f}"
        )

        history_rows.append(
            {
                "epoch": global_epoch,
                "phase": 1,
                "phase_epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_top5_acc": train_top5,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_acc_clean": val_clean_acc,
                "val_acc_healed": val_healed_acc,
                "val_top5_acc": val_top5,
                "lr": lr,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(classifier.state_dict(), best_path)
        else:
            patience_counter += 1

        if (
            epoch >= min_epochs_before_early_stop
            and patience_counter >= patience
            and train_acc >= target_train_acc
        ):
            print(
                f"Early stopping in Phase 1 at epoch {epoch}: "
                f"patience={patience}, train_acc={train_acc:.4f} (target={target_train_acc:.4f})."
            )
            break

    if patience_counter < patience:
        patience_counter = 0
        classifier.unfreeze_backbone()
        optimizer = AdamW(classifier.parameters(), lr=finetune_lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=phase2_epochs)

        for epoch in range(1, phase2_epochs + 1):
            global_epoch += 1
            train_loss, train_acc, train_top5, _, _ = run_classifier_epoch(
                classifier,
                vae,
                train_loader,
                criterion,
                optimizer,
                device,
                mean,
                std,
                train=True,
                phase_label=f"Phase 2 Train {epoch}/{phase2_epochs}",
                clean_mix_prob=clean_mix_prob,
            )
            with torch.no_grad():
                val_loss, val_acc, val_top5, val_clean_acc, val_healed_acc = run_classifier_epoch(
                    classifier,
                    vae,
                    val_loader,
                    criterion,
                    optimizer,
                    device,
                    mean,
                    std,
                    train=False,
                    phase_label=f"Phase 2 Val {epoch}/{phase2_epochs}",
                    clean_mix_prob=clean_mix_prob,
                )

            lr = float(optimizer.param_groups[0]["lr"])
            scheduler.step()

            print(
                f"Phase 2 | Epoch {epoch}/{phase2_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc (avg): {val_acc:.4f} | Val Clean: {val_clean_acc:.4f} | "
                f"Val Healed: {val_healed_acc:.4f} | Val Top5: {val_top5:.4f} | LR: {lr:.6f}"
            )

            history_rows.append(
                {
                    "epoch": global_epoch,
                    "phase": 2,
                    "phase_epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "train_top5_acc": train_top5,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_acc_clean": val_clean_acc,
                    "val_acc_healed": val_healed_acc,
                    "val_top5_acc": val_top5,
                    "lr": lr,
                }
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(classifier.state_dict(), best_path)
            else:
                patience_counter += 1

            if (
                epoch >= min_epochs_before_early_stop
                and patience_counter >= patience
                and train_acc >= target_train_acc
            ):
                print(
                    f"Early stopping in Phase 2 at epoch {epoch}: "
                    f"patience={patience}, train_acc={train_acc:.4f} (target={target_train_acc:.4f})."
                )
                break

    pd.DataFrame(history_rows).to_csv(results_dir / "classifier_history.csv", index=False)
    print(f"Best validation accuracy (avg clean/healed): {best_val_acc * 100:.2f}%")
    print("Model saved: models/resnet_classifier.pth")

    return {
        "train_loss": [row["train_loss"] for row in history_rows],
        "val_loss": [row["val_loss"] for row in history_rows],
        "train_acc": [row["train_acc"] for row in history_rows],
        "val_acc": [row["val_acc"] for row in history_rows],
        "val_top5_acc": [row["val_top5_acc"] for row in history_rows],
    }


def train_classifier(config_path: str = "configs/config.yaml") -> Dict[str, List[float]]:
    return train_classifier_model(config_path=config_path)


if __name__ == "__main__":
    train_classifier_model()
