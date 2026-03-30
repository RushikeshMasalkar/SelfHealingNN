from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from .classifier import SelfHealingClassifier
from .conv_vae import ConvVAE


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_imagenet(batch: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=batch.device, dtype=batch.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=batch.device, dtype=batch.dtype).view(1, 3, 1, 1)
    return (batch - mean) / std


class SelfHealingPipeline(nn.Module):
    def __init__(
        self,
        vae: ConvVAE,
        classifier: SelfHealingClassifier,
        vae_outputs_denormalized: bool = True,
    ):
        super().__init__()
        self.vae = vae
        self.classifier = classifier
        self.vae_outputs_denormalized = vae_outputs_denormalized

    @torch.no_grad()
    def heal(self, noisy_image: torch.Tensor) -> torch.Tensor:
        cleaned, _, _ = self.vae(noisy_image)
        return cleaned

    @torch.no_grad()
    def classify(self, cleaned_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # VAE now outputs normalized images directly; no re-normalization needed
        # (unless explicitly configured otherwise via vae_outputs_denormalized flag)
        classifier_input = (
            normalize_imagenet(cleaned_image)
            if self.vae_outputs_denormalized
            else cleaned_image
        )
        logits = self.classifier(classifier_input)
        probs = torch.softmax(logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)
        return prediction, confidence

    @torch.no_grad()
    def forward(self, noisy_image: torch.Tensor):
        cleaned = self.heal(noisy_image)
        prediction, confidence = self.classify(cleaned)
        return prediction, confidence, cleaned

    def save_pipeline(self, path: str) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "vae_state_dict": self.vae.state_dict(),
                "classifier_state_dict": self.classifier.state_dict(),
            },
            save_path,
        )

    @classmethod
    def load_pipeline(
        cls,
        path: str,
        vae: ConvVAE,
        classifier: SelfHealingClassifier,
        device: str = "cpu",
    ) -> "SelfHealingPipeline":
        ckpt = torch.load(path, map_location=device)
        vae.load_state_dict(ckpt["vae_state_dict"])
        classifier.load_state_dict(ckpt["classifier_state_dict"])
        pipeline = cls(vae=vae, classifier=classifier)
        return pipeline.to(device)
