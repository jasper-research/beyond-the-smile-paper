"""2D convolutional VAE for the (tenor x delta) volatility-surface grid.

Treats each surface as a 2-channel image: the masked value and the mask
indicator. The encoder uses 3x3 convolutions over the 6x7 grid to
exploit local spatial structure across both tenor and delta axes, then
flattens to a small MLP head producing (mu, logvar). The decoder
mirrors this with a linear-to-spatial projection followed by ConvTranspose
layers.

Interface matches MaskedVAE: forward(x, mask) -> (recon, mu, logvar),
with x and mask as flat (N, 42) tensors so the existing training loop
can be reused unchanged.
"""

from __future__ import annotations

import torch
from torch import nn

from .data_btc import N_CELLS, N_DELTAS, N_TENORS  # 6, 7, 42


class ConvVAE(nn.Module):
    """2D convolutional VAE.

    `hidden` here is the number of feature channels inside the conv
    stack. The latent-projection MLP head sits on top of the flattened
    final feature map (`hidden * N_TENORS * N_DELTAS` features).
    """

    def __init__(self, z_dim: int = 16, hidden: int = 32):
        super().__init__()
        self.z_dim = z_dim
        self.hidden = hidden

        # Encoder: (B, 2, 6, 7) -> (B, h, 6, 7) -> (B, h, 6, 7)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(2, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        feat_dim = hidden * N_TENORS * N_DELTAS
        self.fc_mu = nn.Linear(feat_dim, z_dim)
        self.fc_logvar = nn.Linear(feat_dim, z_dim)

        # Decoder: z -> (B, h, 6, 7) via linear, then 3 conv layers, then 1x1 to 1 channel
        self.decoder_fc = nn.Linear(z_dim, feat_dim)
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def _to_grid(self, flat: torch.Tensor) -> torch.Tensor:
        return flat.view(-1, N_TENORS, N_DELTAS)

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x, mask: (B, 42) -> stack into (B, 2, 6, 7)
        x2 = self._to_grid(x * mask).unsqueeze(1)
        m2 = self._to_grid(mask).unsqueeze(1)
        inp = torch.cat([x2, m2], dim=1)  # (B, 2, 6, 7)
        h = self.encoder_conv(inp)
        h_flat = h.flatten(1)
        return self.fc_mu(h_flat), self.fc_logvar(h_flat)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z).view(-1, self.hidden, N_TENORS, N_DELTAS)
        recon = self.decoder_conv(h)  # (B, 1, 6, 7)
        return recon.flatten(1)  # (B, 42)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        mu, logvar = self.encode(x, mask)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
