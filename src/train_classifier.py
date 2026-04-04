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
) -> Tuple[float, float, float]:
    if train:
        classifier.train()
    else:
        classifier.eval()

    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    batch_count = 0

    loop = tqdm(loader, desc=phase_label, leave=False)
    for noisy, _, labels in loop:
        noisy = noisy.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            noisy_pixel = denormalize_batch(noisy, mean, std)
            cleaned_pixel, _, _ = vae(noisy_pixel)
            cleaned = normalize_batch(cleaned_pixel, mean, std)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = classifier(cleaned)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item())
        total_top1 += float((logits.argmax(dim=1) == labels).float().mean().item())
        total_top5 += top5_accuracy(logits, labels)
        batch_count += 1

    batch_count = max(batch_count, 1)
    return total_loss / batch_count, total_top1 / batch_count, total_top5 / batch_count


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

    train_loader, val_loader, _ = get_dataloaders(config)

    vae = ConvVAE(latent_dim=int(config.get("vae", {}).get("latent_dim", 256))).to(device)
    vae.load_state_dict(torch.load("models/conv_vae_best.pth", map_location=device))
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    classifier = get_classifier(config).to(device)
    criterion = nn.CrossEntropyLoss()

    models_dir = Path("models")
    results_dir = Path("outputs/results")
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    history_rows: List[Dict[str, float]] = []
    best_val_acc = 0.0
    best_path = models_dir / "resnet_classifier.pth"

    patience = int(classifier_cfg.get("patience", 7))
    patience_counter = 0
    global_epoch = 0

    # Phase 1: frozen backbone.
    classifier.freeze_backbone()
    optimizer = AdamW(filter(lambda p: p.requires_grad, classifier.parameters()), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, 11):
        global_epoch += 1
        train_loss, train_acc, train_top5 = run_classifier_epoch(
            classifier,
            vae,
            train_loader,
            criterion,
            optimizer,
            device,
            mean,
            std,
            train=True,
            phase_label=f"Phase 1 Train {epoch}/10",
        )
        with torch.no_grad():
            val_loss, val_acc, val_top5 = run_classifier_epoch(
                classifier,
                vae,
                val_loader,
                criterion,
                optimizer,
                device,
                mean,
                std,
                train=False,
                phase_label=f"Phase 1 Val {epoch}/10",
            )

        lr = float(optimizer.param_groups[0]["lr"])
        print(
            f"Phase 1 | Epoch {epoch}/10 | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Top5: {val_top5:.4f} | LR: {lr:.6f}"
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

        if patience_counter >= patience:
            print(f"Early stopping triggered in Phase 1 at epoch {epoch}.")
            break

    # Phase 2: full fine-tuning.
    if patience_counter < patience:
        classifier.unfreeze_backbone()
        optimizer = AdamW(classifier.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=40)

        for epoch in range(1, 41):
            global_epoch += 1
            train_loss, train_acc, train_top5 = run_classifier_epoch(
                classifier,
                vae,
                train_loader,
                criterion,
                optimizer,
                device,
                mean,
                std,
                train=True,
                phase_label=f"Phase 2 Train {epoch}/40",
            )
            with torch.no_grad():
                val_loss, val_acc, val_top5 = run_classifier_epoch(
                    classifier,
                    vae,
                    val_loader,
                    criterion,
                    optimizer,
                    device,
                    mean,
                    std,
                    train=False,
                    phase_label=f"Phase 2 Val {epoch}/40",
                )

            lr = float(optimizer.param_groups[0]["lr"])
            scheduler.step()

            print(
                f"Phase 2 | Epoch {epoch}/40 | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Top5: {val_top5:.4f} | LR: {lr:.6f}"
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

            if patience_counter >= patience:
                print(f"Early stopping triggered in Phase 2 at epoch {epoch}.")
                break

    pd.DataFrame(history_rows).to_csv(results_dir / "classifier_history.csv", index=False)
    print(f"Best validation accuracy: {best_val_acc * 100:.2f}%")
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
