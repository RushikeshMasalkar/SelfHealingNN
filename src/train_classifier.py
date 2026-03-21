from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .classifier import SelfHealingClassifier


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
) -> Tuple[float, float, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0

    iterator = tqdm(loader, desc="Train" if train else "Val", leave=False)
    for images, labels in iterator:
        images = images.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()

        top1 = (logits.argmax(dim=1) == labels).float().mean().item()
        top5 = topk_accuracy(logits, labels, k=5)

        total_loss += float(loss.detach().cpu())
        total_top1 += top1
        total_top5 += top5

    n = max(len(loader), 1)
    return total_loss / n, total_top1 / n, total_top5 / n


def train_classifier(
    model: SelfHealingClassifier,
    train_loader,
    val_loader,
    device: str = "cuda",
    epochs: int = 30,
    learning_rate: float = 1e-4,
    freeze_backbone_epochs: int = 5,
    save_path: str = "models/resnet_classifier.pth",
) -> Dict[str, List[float]]:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    _set_backbone_frozen(model, frozen=True)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_top1": [],
        "val_top1": [],
        "train_top5": [],
        "val_top5": [],
    }

    best_val_top1 = 0.0
    save_target = Path(save_path)
    save_target.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        if epoch == freeze_backbone_epochs + 1:
            _set_backbone_frozen(model, frozen=False)
            optimizer = AdamW(model.parameters(), lr=learning_rate)
            scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs - epoch + 1, 1))

        train_loss, train_top1, train_top5 = _run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_top1, val_top5 = _run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_top1"].append(train_top1)
        history["val_top1"].append(val_top1)
        history["train_top5"].append(train_top5)
        history["val_top5"].append(val_top5)

        print(
            f"Epoch {epoch:03d} | "
            f"loss={train_loss:.4f}/{val_loss:.4f} | "
            f"top1={train_top1:.4f}/{val_top1:.4f} | "
            f"top5={train_top5:.4f}/{val_top5:.4f}"
        )

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            torch.save(model.state_dict(), save_target)

    return history
