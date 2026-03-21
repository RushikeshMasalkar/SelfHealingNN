from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class SelfHealingClassifier(nn.Module):
    def __init__(self, num_classes: int = 100, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, cleaned_image: torch.Tensor) -> torch.Tensor:
        return self.backbone(cleaned_image)


def get_classifier(num_classes: int = 100, pretrained: bool = True) -> SelfHealingClassifier:
    return SelfHealingClassifier(num_classes=num_classes, pretrained=pretrained)
