from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .conv_vae import ConvVAE, print_model_summary, vae_loss
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


def run_epoch(
    vae: ConvVAE,
    loader,
    optimizer,
    device: torch.device,
    mean: List[float],
    std: List[float],
    beta: float,
    train: bool,
) -> Tuple[float, float, float]:
    if train:
        vae.train()
    else:
        vae.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    sample_count = 0

    loop = tqdm(loader, desc="Train" if train else "Val", leave=False)
    for noisy, clean, _ in loop:
        noisy = denormalize_batch(noisy.to(device), mean, std)
        clean = denormalize_batch(clean.to(device), mean, std)
        batch_size = noisy.size(0)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            recon, mu, logvar = vae(noisy)
            loss, recon_l, kl_l = vae_loss(recon, clean, mu, logvar, beta=beta)

            if train:
                loss.backward()
                clip_grad_norm_(vae.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += float(loss.item())
        total_recon += float(recon_l.item())
        total_kl += float(kl_l.item())
        sample_count += batch_size

    sample_count = max(sample_count, 1)
    return total_loss / sample_count, total_recon / sample_count, total_kl / sample_count


def train_vae_model(config_path: str = "configs/config.yaml") -> Dict[str, List[float]]:
    config = load_config(config_path)
    train_cfg = config.get("training", {})

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    device = resolve_device(train_cfg)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataset_cfg = config.get("dataset", {})
    vae_cfg = config.get("vae", {})

    mean = dataset_cfg.get("mean", [0.5071, 0.4867, 0.4408])
    std = dataset_cfg.get("std", [0.2675, 0.2565, 0.2761])

    latent_dim = int(vae_cfg.get("latent_dim", 256))
    beta = float(vae_cfg.get("beta", 0.5))
    learning_rate = float(vae_cfg.get("learning_rate", 1e-3))
    epochs = int(vae_cfg.get("epochs", 100))
    patience = int(vae_cfg.get("patience", 10))
    save_every = int(train_cfg.get("save_every", 5))

    train_loader, val_loader, _ = get_dataloaders(config)

    vae = ConvVAE(latent_dim=latent_dim).to(device)
    print_model_summary(vae)

    optimizer = AdamW(vae.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    models_dir = Path("models")
    results_dir = Path("outputs/results")
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    best_path = models_dir / "conv_vae_best.pth"
    last_path = models_dir / "conv_vae_last.pth"

    history_rows: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_recon, train_kl = run_epoch(
            vae,
            train_loader,
            optimizer,
            device,
            mean,
            std,
            beta,
            train=True,
        )

        with torch.no_grad():
            val_loss, val_recon, val_kl = run_epoch(
                vae,
                val_loader,
                optimizer,
                device,
                mean,
                std,
                beta,
                train=False,
            )

        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_recon": train_recon,
                "train_kl": train_kl,
                "val_recon": val_recon,
                "val_kl": val_kl,
                "lr": current_lr,
            }
        )

        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Recon: {val_recon:.4f} | "
            f"KL: {val_kl:.4f} | LR: {current_lr:.6f}"
        )

        if epoch % save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": vae.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                },
                models_dir / f"conv_vae_epoch_{epoch}.pth",
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(vae.state_dict(), best_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    torch.save(vae.state_dict(), last_path)
    pd.DataFrame(history_rows).to_csv(results_dir / "vae_history.csv", index=False)

    print("Training complete!")
    print(f"Best Val Loss: {best_val_loss:.4f} at Epoch {best_epoch}")
    print("Model saved: models/conv_vae_best.pth")

    return {
        "train_loss": [row["train_loss"] for row in history_rows],
        "val_loss": [row["val_loss"] for row in history_rows],
        "train_recon": [row["train_recon"] for row in history_rows],
        "train_kl": [row["train_kl"] for row in history_rows],
        "val_recon": [row["val_recon"] for row in history_rows],
        "val_kl": [row["val_kl"] for row in history_rows],
        "lr": [row["lr"] for row in history_rows],
    }


def train_vae(config_path: str = "configs/config.yaml") -> Dict[str, List[float]]:
    return train_vae_model(config_path=config_path)


if __name__ == "__main__":
    train_vae_model()
