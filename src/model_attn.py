"""Set-attention VAE for the (tenor x delta) volatility-surface grid.

Tokens are the 42 cells. Each cell carries a learned (tenor, delta)
embedding and, in the encoder, its (masked value, mask flag). Multi-head
self-attention layers let every cell attend to every other; the
encoder pools the cell tokens to (mu, logvar). The decoder broadcasts
the latent to every cell position together with the cell embedding,
applies a symmetric attention stack, and projects each token to a
scalar reconstructed value.

Interface matches MaskedVAE: forward(x, mask) -> (recon, mu, logvar)
on flat (N, 42) tensors.
"""

from __future__ import annotations

import torch
from torch import nn

from .data_btc import N_CELLS, N_DELTAS, N_TENORS  # 6, 7, 42


class _AttnBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + a)
        x = self.ln2(x + self.ff(x))
        return x


class AttnVAE(nn.Module):
    """Per-cell set-attention VAE.

    `hidden` (here `d_model`) is the per-token embedding dimension used
    inside the attention stack; `n_layers` is the depth of the encoder
    and the decoder stacks (both have the same depth).
    """

    def __init__(self, z_dim: int = 16, hidden: int = 64,
                 n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.z_dim = z_dim
        self.d_model = hidden

        # Learned (tenor, delta) positional embeddings; identical between
        # encoder and decoder.
        self.tenor_embed = nn.Embedding(N_TENORS, hidden)
        self.delta_embed = nn.Embedding(N_DELTAS, hidden)

        cell_idx = torch.arange(N_CELLS)
        self.register_buffer("tenor_idx", cell_idx // N_DELTAS, persistent=False)
        self.register_buffer("delta_idx", cell_idx %  N_DELTAS, persistent=False)

        # Encoder: project (value, mask, tenor_emb, delta_emb) -> d_model
        self.encoder_in = nn.Linear(2 + 2 * hidden, hidden)
        self.encoder_blocks = nn.ModuleList(
            [_AttnBlock(hidden, n_heads) for _ in range(n_layers)]
        )
        self.fc_mu = nn.Linear(hidden, z_dim)
        self.fc_logvar = nn.Linear(hidden, z_dim)

        # Decoder: z (broadcast) + (tenor_emb, delta_emb) -> d_model
        self.decoder_in = nn.Linear(z_dim + 2 * hidden, hidden)
        self.decoder_blocks = nn.ModuleList(
            [_AttnBlock(hidden, n_heads) for _ in range(n_layers)]
        )
        self.decoder_out = nn.Linear(hidden, 1)

    def _pos_embed(self, batch: int, device) -> torch.Tensor:
        te = self.tenor_embed(self.tenor_idx.to(device))          # (42, h)
        de = self.delta_embed(self.delta_idx.to(device))          # (42, h)
        pe = torch.cat([te, de], dim=-1).unsqueeze(0).expand(batch, -1, -1)
        return pe                                                  # (B, 42, 2h)

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)
        pe = self._pos_embed(B, x.device)
        per_cell = torch.stack([x * mask, mask], dim=-1)           # (B, 42, 2)
        tokens = torch.cat([per_cell, pe], dim=-1)                 # (B, 42, 2 + 2h)
        h = self.encoder_in(tokens)                                # (B, 42, h)
        for blk in self.encoder_blocks:
            h = blk(h)
        pooled = h.mean(dim=1)                                     # (B, h)
        return self.fc_mu(pooled), self.fc_logvar(pooled)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        pe = self._pos_embed(B, z.device)
        z_broadcast = z.unsqueeze(1).expand(-1, N_CELLS, -1)        # (B, 42, z)
        tokens = torch.cat([z_broadcast, pe], dim=-1)               # (B, 42, z + 2h)
        h = self.decoder_in(tokens)                                  # (B, 42, h)
        for blk in self.decoder_blocks:
            h = blk(h)
        return self.decoder_out(h).squeeze(-1)                       # (B, 42)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        mu, logvar = self.encode(x, mask)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
