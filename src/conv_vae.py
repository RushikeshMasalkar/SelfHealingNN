from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        # 4 downsampling blocks: 32->16->8->4->2 spatial dims
        self.features = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # Block 2: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Block 3: 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Block 4: 4x4 -> 2x2
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.flatten = nn.Flatten()
        # 512 channels * 2 * 2 spatial = 2048
        self.fc = nn.Linear(512 * 2 * 2, 1024)
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        x = self.flatten(x)
        x = F.relu(self.fc(x), inplace=True)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 1024)
        # Project back to 512 * 2 * 2 = 2048
        self.fc2 = nn.Linear(1024, 512 * 2 * 2)
        # 4 upsampling blocks: 2->4->8->16->32 spatial dims
        self.deconv = nn.Sequential(
            # Block 1: 2x2 -> 4x4
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Block 2: 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Block 3: 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Block 4: 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.fc1(z), inplace=True)
        z = F.relu(self.fc2(z), inplace=True)
        z = z.view(z.size(0), 512, 2, 2)
        return self.deconv(z)


class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim=latent_dim)
        self.decoder = ConvDecoder(latent_dim=latent_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = F.mse_loss(recon, target, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


def print_model_summary(vae: ConvVAE) -> None:
    encoder_params = sum(p.numel() for p in vae.encoder.parameters())
    decoder_params = sum(p.numel() for p in vae.decoder.parameters())
    total_params = encoder_params + decoder_params

    print("=" * 60)
    print("ConvVAE Model Summary")
    print("=" * 60)
    print(f"Encoder parameters : {encoder_params:,}")
    print(f"Decoder parameters : {decoder_params:,}")
    print(f"Total parameters   : {total_params:,}")
    print("=" * 60)
