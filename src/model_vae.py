"""MLP-VAE for 42-dim BTC vol surfaces with masked-input encoder.

Encoder input: concat(value * mask, mask) → 84 dims. Hidden cells contribute
zero to the value half, and the mask channel tells the encoder which inputs
were observed vs masked. Encoder outputs (mu, logvar) for an 8-dim Gaussian
latent. Decoder maps z → full 42-dim surface reconstruction.
"""

from __future__ import annotations

import torch
from torch import nn

from .data_btc import N_CELLS

LATENT_DIM = 8
HIDDEN = 128


class MaskedVAE(nn.Module):
    def __init__(self, z_dim: int = LATENT_DIM, hidden: int = HIDDEN):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Linear(2 * N_CELLS, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.fc_mu = nn.Linear(hidden, z_dim)
        self.fc_logvar = nn.Linear(hidden, z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, N_CELLS),
        )

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([x * mask, mask], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        mu, logvar = self.encode(x, mask)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
