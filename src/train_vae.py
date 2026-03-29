from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .conv_vae import ConvVAE, vae_loss


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def _resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[VAE] CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"
    return device


def _save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model: ConvVAE,
    optimizer,
    scheduler,
    scaler: GradScaler,
    history: Dict[str, List[float]],
    best_val: float,
    early_stopper: EarlyStopper,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
        "history": history,
        "best_val": best_val,
        "early_stopper": {
            "best": early_stopper.best,
            "counter": early_stopper.counter,
            "patience": early_stopper.patience,
        },
    }
    torch.save(payload, checkpoint_path)


def _state_dict_is_finite(state_dict: Dict[str, torch.Tensor]) -> bool:
    for value in state_dict.values():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            return False
    return True


def _denormalize_imagenet(batch: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=batch.device, dtype=batch.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=batch.device, dtype=batch.dtype).view(1, 3, 1, 1)
    return torch.clamp(batch * std + mean, min=0.0, max=1.0)


def _epoch_from_checkpoint_path(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return -1


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
    checkpoint_dir: str = "models/checkpoints/vae",
    checkpoint_interval: int = 2,
    auto_resume: bool = True,
    use_amp: bool = True,
    grad_clip_norm: float = 1.0,
    denormalize_targets: bool = True,
) -> Dict[str, List[float]]:
    device = _resolve_device(device)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    early_stopper = EarlyStopper(patience=patience)
    amp_enabled = bool(use_amp and device.startswith("cuda"))
    scaler = GradScaler("cuda", enabled=amp_enabled)

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
    checkpoint_root = Path(checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    latest_checkpoint = checkpoint_root / "latest.pt"

    start_epoch = 1
    resume_checkpoint: Path | None = None
    if auto_resume:
        if latest_checkpoint.exists():
            resume_checkpoint = latest_checkpoint
        else:
            epoch_checkpoints = sorted(
                checkpoint_root.glob("epoch_*.pt"),
                key=_epoch_from_checkpoint_path,
            )
            if epoch_checkpoints:
                resume_checkpoint = epoch_checkpoints[-1]

    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        state = checkpoint.get("model_state")
        if state is not None and _state_dict_is_finite(state):
            model.load_state_dict(state)
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            if checkpoint.get("scaler_state") is not None and scaler.is_enabled():
                scaler.load_state_dict(checkpoint["scaler_state"])

            loaded_history = checkpoint.get("history", {})
            for key in history:
                history[key] = list(loaded_history.get(key, []))

            best_val = float(checkpoint.get("best_val", best_val))
            early = checkpoint.get("early_stopper", {})
            early_stopper.best = float(early.get("best", early_stopper.best))
            early_stopper.counter = int(early.get("counter", early_stopper.counter))
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            print(f"[VAE] Resuming from epoch {start_epoch} ({resume_checkpoint.name}).")
        else:
            print(f"[VAE] Ignoring {resume_checkpoint.name} because it contains non-finite values.")

    if start_epoch > epochs:
        print("[VAE] Training already completed for configured epochs. Returning loaded history.")
        return history

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_recon = 0.0
        train_kl = 0.0
        train_total = 0.0
        train_valid_batches = 0
        saw_non_finite = False

        train_bar = tqdm(train_loader, desc=f"[VAE][Train] Epoch {epoch}/{epochs}", leave=False)
        for noisy, clean, _ in train_bar:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            target_clean = _denormalize_imagenet(clean) if denormalize_targets else clean

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=amp_enabled):
                recon, mu, logvar = model(noisy)
                loss, parts = vae_loss(recon, target_clean, mu, logvar, beta=beta)

            if not torch.isfinite(loss):
                saw_non_finite = True
                continue

            scaler.scale(loss).backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            train_recon += parts["recon_loss"]
            train_kl += parts["kl_loss"]
            train_total += parts["total_loss"]
            train_valid_batches += 1

        if train_valid_batches == 0:
            raise RuntimeError(
                "All training batches became non-finite. Try lower learning_rate/beta and disable AMP temporarily."
            )

        n_train = train_valid_batches
        train_recon /= n_train
        train_kl /= n_train
        train_total /= n_train

        model.eval()
        val_recon = 0.0
        val_kl = 0.0
        val_total = 0.0
        val_valid_batches = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"[VAE][Val]   Epoch {epoch}/{epochs}", leave=False)
            for noisy, clean, _ in val_bar:
                noisy = noisy.to(device, non_blocking=True)
                clean = clean.to(device, non_blocking=True)
                target_clean = _denormalize_imagenet(clean) if denormalize_targets else clean
                with autocast("cuda", enabled=amp_enabled):
                    recon, mu, logvar = model(noisy)
                    val_loss, parts = vae_loss(recon, target_clean, mu, logvar, beta=beta)

                if not torch.isfinite(val_loss):
                    saw_non_finite = True
                    continue

                val_recon += parts["recon_loss"]
                val_kl += parts["kl_loss"]
                val_total += parts["total_loss"]
                val_valid_batches += 1

        if val_valid_batches == 0:
            raise RuntimeError(
                "All validation batches became non-finite. Stop and lower learning_rate/beta before resuming."
            )

        n_val = val_valid_batches
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

        if saw_non_finite:
            print("[VAE] Warning: Non-finite batches were detected and skipped during this epoch.")

        if not all(torch.isfinite(torch.tensor(v)) for v in [train_recon, train_kl, train_total, val_recon, val_kl, val_total]):
            raise RuntimeError("Epoch metrics became non-finite. Aborting to avoid corrupt checkpoints.")

        if val_total < best_val:
            best_val = val_total
            torch.save(model.state_dict(), save_target)

        if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
            epoch_checkpoint = checkpoint_root / f"epoch_{epoch:04d}.pt"
            _save_checkpoint(
                epoch_checkpoint,
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                history,
                best_val,
                early_stopper,
            )
            _save_checkpoint(
                latest_checkpoint,
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                history,
                best_val,
                early_stopper,
            )

        if early_stopper.step(val_total):
            _save_checkpoint(
                latest_checkpoint,
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                history,
                best_val,
                early_stopper,
            )
            print(f"Early stopping triggered at epoch {epoch}.")
            break

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if checkpoint_interval <= 0:
        _save_checkpoint(
            latest_checkpoint,
            epochs,
            model,
            optimizer,
            scheduler,
            scaler,
            history,
            best_val,
            early_stopper,
        )

    return history
