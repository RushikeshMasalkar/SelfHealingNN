from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class CIFAR100Classifier(nn.Module):
    def __init__(self, num_classes: int = 100, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        # CIFAR-100 images are 32x32, so use a gentler stem.
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        for name, param in self.backbone.named_parameters():
            param.requires_grad = name.startswith("fc.")

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_classifier(config: Optional[Dict] = None) -> CIFAR100Classifier:
    if config is None:
        return CIFAR100Classifier()

    dataset_cfg = config.get("dataset", {})
    classifier_cfg = config.get("classifier", {})

    return CIFAR100Classifier(
        num_classes=int(dataset_cfg.get("num_classes", 100)),
        pretrained=bool(classifier_cfg.get("pretrained", True)),
        dropout=float(classifier_cfg.get("dropout", 0.3)),
    )


# Backward compatibility alias.
SelfHealingClassifier = CIFAR100Classifier
