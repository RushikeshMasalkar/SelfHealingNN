from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        # 3 downsampling blocks: 32->16->8->4 spatial dims
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
        )
        self.flatten = nn.Flatten()
        # 256 channels * 4 * 4 spatial = 4096
        self.fc = nn.Linear(256 * 4 * 4, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        x = self.flatten(x)
        x = F.relu(self.fc(x), inplace=True)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        # Project back to 256 * 4 * 4 = 4096
        self.fc2 = nn.Linear(512, 256 * 4 * 4)
        # 3 upsampling blocks: 4->8->16->32 spatial dims
        self.deconv = nn.Sequential(
            # Block 1: 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Block 2: 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Block 3: 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Output: 32x32
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.fc1(z), inplace=True)
        z = F.relu(self.fc2(z), inplace=True)
        z = z.view(z.size(0), 256, 4, 4)
        return self.deconv(z)


class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim=latent_dim)
        self.decoder = ConvDecoder(latent_dim=latent_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.to(next(self.parameters()).device)
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
    # Blend L1+MSE to reduce blur. Sum over pixels, mean over batch.
    recon_l1 = F.l1_loss(recon, target, reduction="none").view(recon.size(0), -1).sum(dim=1)
    recon_mse = F.mse_loss(recon, target, reduction="none").view(recon.size(0), -1).sum(dim=1)
    
    recon_loss_per_sample = 0.7 * recon_l1 + 0.3 * recon_mse
    recon_loss = torch.mean(recon_loss_per_sample)

    # Compute KL per-sample and average across the batch.
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = torch.mean(kl_per_sample)

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
