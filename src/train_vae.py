"""Train the MaskedVAE for a given TrainConfig, into a per-run directory.

Loss = w_hidden * MSE(recon, target)[hidden cells]
     + w_obs    * MSE(recon, target)[observed cells]
     + beta     * KL(q(z|x) || N(0, I))

Hidden-cell reconstruction is the headline objective. Observed-cell loss keeps
the decoder consistent with the encoder. KL is the standard VAE prior.

`train(cfg)` returns the run_dir path. Each run writes:
  config.json, history.json, best.pt
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import TrainConfig, make_run_dir, save_config
from .data_btc import BTCSurfaceDataset
from .model_attn import AttnVAE
from .model_conv import ConvVAE
from .model_smooth import DeepSmoother
from .model_vae import MaskedVAE


def build_model(cfg: TrainConfig):
    """Dispatch to the requested architecture from a TrainConfig."""
    return _build(cfg.arch, cfg.z_dim, cfg.hidden)


def build_model_from_ckpt(ckpt: dict, cfg: TrainConfig):
    """Reconstruct the model that produced a checkpoint, falling back to
    the run's config for any field the checkpoint predates (e.g. `arch`
    was added after the MLP-only baselines)."""
    arch = ckpt.get("arch", cfg.arch)
    z_dim = ckpt.get("z_dim", cfg.z_dim)
    hidden = ckpt.get("hidden", cfg.hidden)
    return _build(arch, z_dim, hidden)


def _build(arch: str, z_dim: int, hidden: int):
    if arch == "mlp":
        return MaskedVAE(z_dim=z_dim, hidden=hidden)
    if arch == "conv2d":
        return ConvVAE(z_dim=z_dim, hidden=hidden)
    if arch == "attention":
        return AttnVAE(z_dim=z_dim, hidden=hidden)
    if arch == "smoother":
        return DeepSmoother(z_dim=z_dim, hidden=hidden)
    raise ValueError(f"unknown arch {arch!r}")


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def step_loss(model: MaskedVAE, surf: torch.Tensor, mask: torch.Tensor, cfg: TrainConfig
              ) -> tuple[torch.Tensor, dict[str, float]]:
    recon, mu, logvar = model(surf, mask)
    sq_err = (recon - surf) ** 2
    hidden = (1.0 - mask)
    obs = mask
    hidden_loss = (sq_err * hidden).sum(-1) / hidden.sum(-1).clamp(min=1.0)
    obs_loss = (sq_err * obs).sum(-1) / obs.sum(-1).clamp(min=1.0)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)
    loss = (cfg.w_hidden * hidden_loss + cfg.w_obs * obs_loss + cfg.beta * kl).mean()
    # Models that attach an extra penalty (e.g. DeepSmoother's
    # calendar-arbitrage term) expose it via _last_arb_penalty.
    if hasattr(model, "_last_arb_penalty"):
        loss = loss + model._last_arb_penalty
    return loss, {
        "loss": float(loss.detach()),
        "hidden": float(hidden_loss.mean().detach()),
        "obs": float(obs_loss.mean().detach()),
        "kl": float(kl.mean().detach()),
    }


def run_epoch(model, loader, optimizer, device, train: bool, cfg: TrainConfig
              ) -> dict[str, float]:
    model.train(train)
    totals = {"loss": 0.0, "hidden": 0.0, "obs": 0.0, "kl": 0.0, "n": 0}
    for surf, mask in loader:
        surf = surf.to(device)
        mask = mask.to(device)
        if train:
            optimizer.zero_grad()
        loss, parts = step_loss(model, surf, mask, cfg)
        if train:
            loss.backward()
            optimizer.step()
        bs = surf.size(0)
        for k in ("loss", "hidden", "obs", "kl"):
            totals[k] += parts[k] * bs
        totals["n"] += bs
    n = totals["n"]
    return {k: totals[k] / n for k in ("loss", "hidden", "obs", "kl")}


def train(cfg: TrainConfig, verbose: bool = True) -> Path:
    run_dir = make_run_dir(cfg)
    save_config(cfg, run_dir)
    device = pick_device()
    if verbose:
        print(f"=== {run_dir.name} ===")
        print(f"device: {device}  symbols={cfg.symbols}  z_dim={cfg.z_dim}  "
              f"hidden={cfg.hidden}  beta={cfg.beta}  epochs={cfg.epochs}")

    torch.manual_seed(cfg.seed)

    train_ds = BTCSurfaceDataset("train", rate_min=cfg.mask_rate_min,
                                 rate_max=cfg.mask_rate_max, seed=cfg.seed,
                                 mask_scheme=cfg.mask_scheme,
                                 symbols=cfg.symbols)
    val_ds = BTCSurfaceDataset("val", rate_min=cfg.mask_rate_min,
                               rate_max=cfg.mask_rate_max, seed=cfg.seed + 1,
                               mask_scheme=cfg.mask_scheme,
                               symbols=cfg.symbols)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history: list[dict] = []
    best_val = float("inf")
    patience_left = cfg.patience
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, True, cfg)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, optimizer, device, False, cfg)
        row = {"epoch": epoch,
               **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        improved = val_metrics["loss"] < best_val - 1e-6
        if improved:
            best_val = val_metrics["loss"]
            patience_left = cfg.patience
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_loss": best_val,
                        "arch": cfg.arch,
                        "z_dim": cfg.z_dim, "hidden": cfg.hidden},
                       run_dir / "best.pt")
        else:
            patience_left -= 1
        if verbose and (epoch == 1 or epoch % 25 == 0 or improved and epoch % 5 == 0):
            print(f"  ep {epoch:3d}  train_h {train_metrics['hidden']:.4f}  "
                  f"val_h {val_metrics['hidden']:.4f}  "
                  f"val_loss {val_metrics['loss']:.4f}  "
                  f"{'(best)' if improved else f'(pat {patience_left})'}")
        if patience_left <= 0:
            if verbose:
                print(f"  early stop at epoch {epoch}")
            break

    if verbose:
        print(f"  done in {time.time()-t0:.1f}s, best val_loss={best_val:.4f}")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    return run_dir


def main() -> None:
    train(TrainConfig(name="default",
                      notes="MVP baseline: z=8 h=128 beta=1e-3, 200 epochs"))


if __name__ == "__main__":
    main()
