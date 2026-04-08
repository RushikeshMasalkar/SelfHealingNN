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

        total_loss += float(loss.item()) * float(batch_size)
        total_recon += float(recon_l.item()) * float(batch_size)
        total_kl += float(kl_l.item()) * float(batch_size)
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

    mean = dataset_cfg.get("mean", [0.5071, 0.4865, 0.4409])
    std = dataset_cfg.get("std", [0.2673, 0.2564, 0.2761])

    latent_dim = int(vae_cfg.get("latent_dim", 256))
    beta = float(vae_cfg.get("beta", 0.08))
    beta_warmup_epochs = int(vae_cfg.get("beta_warmup_epochs", 0))
    beta_start_factor = float(vae_cfg.get("beta_start_factor", 0.3))
    beta_start_factor = min(max(beta_start_factor, 0.0), 1.0)
    learning_rate = float(vae_cfg.get("learning_rate", 1e-3))
    epochs = int(vae_cfg.get("epochs", 100))
    patience = int(vae_cfg.get("patience", 10))
    min_epochs_before_early_stop = int(vae_cfg.get("min_epochs_before_early_stop", 0))
    min_improvement = float(vae_cfg.get("min_improvement", 0.0))
    save_every = int(train_cfg.get("save_every", 5))

    vae_augment_train = bool(vae_cfg.get("augment_train", False))
    vae_noise_types = list(vae_cfg.get("noise_types", config.get("noise", {}).get("types", ["gaussian"])))
    vae_noise_params = {
        "gaussian_std": float(vae_cfg.get("gaussian_std", config.get("noise", {}).get("gaussian_std", 0.15))),
        "salt_pepper_prob": float(vae_cfg.get("salt_pepper_prob", config.get("noise", {}).get("salt_pepper_prob", 0.05))),
        "occlusion_size": int(vae_cfg.get("occlusion_size", config.get("noise", {}).get("occlusion_size", 8))),
    }

    train_loader, val_loader, _ = get_dataloaders(
        config,
        augment_train=vae_augment_train,
        noise_types_override=vae_noise_types,
        noise_params_override=vae_noise_params,
        batch_size_override=int(vae_cfg.get("batch_size", 64)),
    )

    vae = ConvVAE(latent_dim=latent_dim).to(device)
    print_model_summary(vae)

    optimizer = AdamW(vae.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    repo_root = Path(__file__).resolve().parents[1]
    models_dir = repo_root / "models"
    results_dir = repo_root / "outputs" / "results"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    best_path = models_dir / "conv_vae_best.pth"
    last_path = models_dir / "conv_vae_last.pth"

    history_rows: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        if beta_warmup_epochs > 0:
            progress = min(1.0, float(epoch) / float(beta_warmup_epochs))
            beta_eff = beta * (beta_start_factor + (1.0 - beta_start_factor) * progress)
        else:
            beta_eff = beta

        train_loss, train_recon, train_kl = run_epoch(
            vae,
            train_loader,
            optimizer,
            device,
            mean,
            std,
            beta_eff,
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
                beta_eff,
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
                "beta_effective": beta_eff,
            }
        )

        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Recon: {val_recon:.4f} | "
            f"KL: {val_kl:.6e} | beta: {beta_eff:.4f} | LR: {current_lr:.6f}"
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

        if val_loss < (best_val_loss - min_improvement):
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(vae.state_dict(), best_path)
        else:
            patience_counter += 1

        print(
            f"No-improvement counter: {patience_counter}/{patience} | "
            f"Best epoch: {best_epoch}"
        )

        if epoch >= min_epochs_before_early_stop and patience_counter >= patience:
            print(
                f"Early stopping triggered at epoch {epoch} "
                f"(min_epochs_before_early_stop={min_epochs_before_early_stop})."
            )
            break

    torch.save(vae.state_dict(), last_path)

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
        "beta_effective": [row["beta_effective"] for row in history_rows],
    }


def train_vae(config_path: str = "configs/config.yaml") -> Dict[str, List[float]]:
    return train_vae_model(config_path=config_path)


if __name__ == "__main__":
    train_vae_model()
