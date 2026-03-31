from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .classifier import CIFAR100Classifier
from .conv_vae import ConvVAE
from .dataset import CIFAR100_CLASSES


CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]


def normalize_batch(batch: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR100_MEAN, dtype=batch.dtype, device=batch.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR100_STD, dtype=batch.dtype, device=batch.device).view(1, 3, 1, 1)
    return (batch - mean) / std


class SelfHealingPipeline(nn.Module):
    def __init__(self, vae: ConvVAE, classifier: CIFAR100Classifier, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.vae = vae.to(self.device)
        self.classifier = classifier.to(self.device)

    def _ensure_batch(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        if tensor.dim() == 3:
            return tensor.unsqueeze(0), True
        return tensor, False

    def heal(self, noisy_image: torch.Tensor) -> torch.Tensor:
        noisy_batch, is_single = self._ensure_batch(noisy_image.to(self.device))
        self.vae.eval()
        with torch.no_grad():
            cleaned, _, _ = self.vae(noisy_batch)
        return cleaned.squeeze(0) if is_single else cleaned

    def classify(self, cleaned_image: torch.Tensor) -> Tuple[int, float, List[Dict[str, float]]]:
        cleaned_batch, _ = self._ensure_batch(cleaned_image.to(self.device))
        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(normalize_batch(cleaned_batch))
            probs = torch.softmax(logits, dim=1)
            top5_probs, top5_indices = probs.topk(5, dim=1)

        prediction = int(top5_indices[0, 0].item())
        confidence = float(top5_probs[0, 0].item())
        top5 = [
            {
                "class_name": CIFAR100_CLASSES[int(top5_indices[0, i].item())],
                "confidence": float(top5_probs[0, i].item()),
            }
            for i in range(5)
        ]
        return prediction, confidence, top5

    def forward(self, noisy_image: torch.Tensor) -> Dict:
        cleaned = self.heal(noisy_image)
        prediction, confidence, top5 = self.classify(cleaned)
        return {
            "prediction": int(prediction),
            "class_name": CIFAR100_CLASSES[int(prediction)],
            "confidence": float(confidence),
            "top5": top5,
            "cleaned_image": cleaned,
        }

    def save(self, path: str) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vae_state_dict": self.vae.state_dict(),
            "classifier_state_dict": self.classifier.state_dict(),
            "latent_dim": int(self.vae.encoder.fc_mu.out_features),
            "num_classes": len(CIFAR100_CLASSES),
        }
        torch.save(payload, save_path)

    def save_pipeline(self, path: str) -> None:
        self.save(path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "SelfHealingPipeline":
        checkpoint = torch.load(path, map_location=device)
        vae = ConvVAE(latent_dim=int(checkpoint.get("latent_dim", 256)))
        classifier = CIFAR100Classifier(
            num_classes=int(checkpoint.get("num_classes", 100)),
            pretrained=False,
        )
        vae.load_state_dict(checkpoint["vae_state_dict"])
        classifier.load_state_dict(checkpoint["classifier_state_dict"])
        return cls(vae=vae, classifier=classifier, device=device)

    @classmethod
    def load_pipeline(cls, path: str, device: str = "cpu") -> "SelfHealingPipeline":
        return cls.load(path=path, device=device)
