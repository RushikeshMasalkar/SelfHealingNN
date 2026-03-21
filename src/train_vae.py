from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .conv_vae import ConvVAE, vae_loss


class EarlyStopper:
    def __init__(self, patience: int = 7):
        self.patience = patience
        self.best = float("inf")
        self.counter = 0

    def step(self, value: float) -> bool:
        if value < self.best:
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_vae(
    model: ConvVAE,
    train_loader,
    val_loader,
    device: str = "cuda",
    epochs: int = 50,
    learning_rate: float = 3e-4,
    beta: float = 0.5,
    save_path: str = "models/conv_vae_best.pth",
    patience: int = 7,
) -> Dict[str, List[float]]:
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    early_stopper = EarlyStopper(patience=patience)

    history = {
        "train_recon": [],
        "train_kl": [],
        "train_total": [],
        "val_recon": [],
        "val_kl": [],
        "val_total": [],
    }

    best_val = float("inf")
    save_target = Path(save_path)
    save_target.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_recon = 0.0
        train_kl = 0.0
        train_total = 0.0

        train_bar = tqdm(train_loader, desc=f"[VAE][Train] Epoch {epoch}/{epochs}", leave=False)
        for noisy, clean, _ in train_bar:
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(noisy)
            loss, parts = vae_loss(recon, clean, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()

            train_recon += parts["recon_loss"]
            train_kl += parts["kl_loss"]
            train_total += parts["total_loss"]

        n_train = max(len(train_loader), 1)
        train_recon /= n_train
        train_kl /= n_train
        train_total /= n_train

        model.eval()
        val_recon = 0.0
        val_kl = 0.0
        val_total = 0.0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"[VAE][Val]   Epoch {epoch}/{epochs}", leave=False)
            for noisy, clean, _ in val_bar:
                noisy = noisy.to(device)
                clean = clean.to(device)
                recon, mu, logvar = model(noisy)
                _, parts = vae_loss(recon, clean, mu, logvar, beta=beta)

                val_recon += parts["recon_loss"]
                val_kl += parts["kl_loss"]
                val_total += parts["total_loss"]

        n_val = max(len(val_loader), 1)
        val_recon /= n_val
        val_kl /= n_val
        val_total /= n_val

        history["train_recon"].append(train_recon)
        history["train_kl"].append(train_kl)
        history["train_total"].append(train_total)
        history["val_recon"].append(val_recon)
        history["val_kl"].append(val_kl)
        history["val_total"].append(val_total)

        scheduler.step()

        print(
            f"Epoch {epoch:03d} | "
            f"recon_loss={train_recon:.4f}/{val_recon:.4f} | "
            f"kl_loss={train_kl:.4f}/{val_kl:.4f} | "
            f"total_loss={train_total:.4f}/{val_total:.4f}"
        )

        if val_total < best_val:
            best_val = val_total
            torch.save(model.state_dict(), save_target)

        if early_stopper.step(val_total):
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    return history
