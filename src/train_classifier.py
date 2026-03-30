from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .classifier import SelfHealingClassifier


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
        print("[Classifier] CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"
    return device


def _save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model: SelfHealingClassifier,
    optimizer,
    scheduler,
    scaler: GradScaler,
    history: Dict[str, List[float]],
    best_val_top1: float,
    early_stopper: EarlyStopper,
    backbone_unfrozen: bool,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
        "history": history,
        "best_val_top1": best_val_top1,
        "backbone_unfrozen": backbone_unfrozen,
        "early_stopper": {
            "best": early_stopper.best,
            "counter": early_stopper.counter,
            "patience": early_stopper.patience,
        },
    }
    torch.save(payload, checkpoint_path)


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    _, pred = logits.topk(k, dim=1)
    correct = pred.eq(targets.view(-1, 1)).sum().item()
    return correct / targets.size(0)


def _set_backbone_frozen(model: SelfHealingClassifier, frozen: bool) -> None:
    for name, param in model.backbone.named_parameters():
        if name.startswith("fc."):
            param.requires_grad = True
        else:
            param.requires_grad = not frozen


def _run_epoch(
    model: SelfHealingClassifier,
    loader,
    criterion,
    optimizer,
    device: str,
    train: bool,
    scaler: GradScaler,
    use_amp: bool,
    grad_clip_norm: float,
    mixup_alpha: float,
) -> Tuple[float, float, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_confidence = 0.0

    iterator = tqdm(loader, desc="Train" if train else "Val", leave=False)
    for images, labels in iterator:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        labels_for_metrics = labels
        mixup_active = train and mixup_alpha > 0
        if mixup_active:
            lam = float(torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item())
            index = torch.randperm(images.size(0), device=device)
            images = lam * images + (1.0 - lam) * images[index]
            labels_a = labels
            labels_b = labels[index]

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with autocast("cuda", enabled=use_amp):
                logits = model(images)
                if mixup_active:
                    loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
                else:
                    loss = criterion(logits, labels)
            if train:
                scaler.scale(loss).backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

        top1 = (logits.argmax(dim=1) == labels_for_metrics).float().mean().item()
        top5 = topk_accuracy(logits, labels_for_metrics, k=5)
        confidence = torch.softmax(logits, dim=1).max(dim=1).values.mean().item()

        total_loss += float(loss.detach().cpu())
        total_top1 += top1
        total_top5 += top5
        total_confidence += confidence

    n = max(len(loader), 1)
    return total_loss / n, total_top1 / n, total_top5 / n, total_confidence / n


def train_classifier(
    model: SelfHealingClassifier,
    train_loader,
    val_loader,
    device: str = "cuda",
    epochs: int = 30,
    learning_rate: float = 1e-4,
    freeze_backbone_epochs: int = 5,
    save_path: str = "models/resnet_classifier.pth",
    patience: int = 7,
    checkpoint_dir: str = "models/checkpoints/classifier",
    checkpoint_interval: int = 2,
    auto_resume: bool = True,
    use_amp: bool = True,
    grad_clip_norm: float = 1.0,
    early_stop_metric: str = "val_top1",
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    mixup_alpha: float = 0.2,
) -> Dict[str, List[float]]:
    device = _resolve_device(device)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    amp_enabled = bool(use_amp and device.startswith("cuda"))
    scaler = GradScaler("cuda", enabled=amp_enabled)

    checkpoint_root = Path(checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    latest_checkpoint = checkpoint_root / "latest.pt"
    save_target = Path(save_path)
    save_target.parent.mkdir(parents=True, exist_ok=True)

    early_stopper = EarlyStopper(patience=patience)

    _set_backbone_frozen(model, frozen=True)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    backbone_unfrozen = False

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_top1": [],
        "val_top1": [],
        "train_top5": [],
        "val_top5": [],
        "train_confidence": [],
        "val_confidence": [],
        "lr": [],
    }

    best_val_top1 = 0.0
    start_epoch = 1

    if auto_resume and latest_checkpoint.exists():
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state"])

        backbone_unfrozen = bool(checkpoint.get("backbone_unfrozen", False))
        if backbone_unfrozen:
            _set_backbone_frozen(model, frozen=False)
            optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            _set_backbone_frozen(model, frozen=True)
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=learning_rate,
                weight_decay=weight_decay,
            )

        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint.get("scaler_state") is not None and scaler.is_enabled():
            scaler.load_state_dict(checkpoint["scaler_state"])

        loaded_history = checkpoint.get("history", {})
        for key in history:
            history[key] = list(loaded_history.get(key, []))

        best_val_top1 = float(checkpoint.get("best_val_top1", 0.0))
        early = checkpoint.get("early_stopper", {})
        early_stopper.best = float(early.get("best", early_stopper.best))
        early_stopper.counter = int(early.get("counter", early_stopper.counter))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"[Classifier] Resuming from epoch {start_epoch} (latest checkpoint).")

    if start_epoch > epochs:
        print("[Classifier] Training already completed for configured epochs. Returning loaded history.")
        return history

    for epoch in range(start_epoch, epochs + 1):
        if (not backbone_unfrozen) and epoch == freeze_backbone_epochs + 1:
            _set_backbone_frozen(model, frozen=False)
            optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs - epoch + 1, 1))
            backbone_unfrozen = True

        train_loss, train_top1, train_top5, train_confidence = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
            scaler=scaler,
            use_amp=amp_enabled,
            grad_clip_norm=grad_clip_norm,
            mixup_alpha=mixup_alpha,
        )
        val_loss, val_top1, val_top5, val_confidence = _run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            train=False,
            scaler=scaler,
            use_amp=amp_enabled,
            grad_clip_norm=grad_clip_norm,
            mixup_alpha=0.0,
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_top1"].append(train_top1)
        history["val_top1"].append(val_top1)
        history["train_top5"].append(train_top5)
        history["val_top5"].append(val_top5)
        history["train_confidence"].append(train_confidence)
        history["val_confidence"].append(val_confidence)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        print(
            f"Epoch {epoch:03d} | "
            f"loss={train_loss:.4f}/{val_loss:.4f} | "
            f"top1={train_top1:.4f}/{val_top1:.4f} | "
            f"top5={train_top5:.4f}/{val_top5:.4f} | "
            f"conf={train_confidence:.4f}/{val_confidence:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
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
                best_val_top1,
                early_stopper,
                backbone_unfrozen,
            )
            _save_checkpoint(
                latest_checkpoint,
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                history,
                best_val_top1,
                early_stopper,
                backbone_unfrozen,
            )

        stop_value = val_loss if early_stop_metric == "val_loss" else -val_top1
        if early_stopper.step(stop_value):
            _save_checkpoint(
                latest_checkpoint,
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                history,
                best_val_top1,
                early_stopper,
                backbone_unfrozen,
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
            best_val_top1,
            early_stopper,
            backbone_unfrozen,
        )

    return history
