"""Eval a trained run vs smile-refit baseline + write metrics + plots to run_dir.

Usage:
  python -m src.eval_vae <run_dir>           # eval one run
  python -m src.eval_vae --latest             # eval the most recent run

Writes to run_dir/:
  metrics.json                    machine-readable results
  rmse_vs_mask.png                headline plot
  recon_loss_timeseries.png       anomaly-detection demo

The smile-refit baseline doesn't depend on the run; it's recomputed once and
cached at runs/vae_btc/_baseline.json. Subsequent evals reuse it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .baseline_smile import _load_k_grid, predict_one
from .config import RUNS_ROOT, TrainConfig, list_runs, load_config
from .data_btc import N_CELLS, N_DELTAS, N_TENORS, load_split
from .model_vae import MaskedVAE  # noqa: F401  (kept for back-compat imports)
from .train_vae import build_model_from_ckpt

def baseline_cache_path(symbol: str = "BTCUSDT") -> Path:
    return RUNS_ROOT / f"_baseline_{symbol.lower()}.json"


# Back-compat alias (BTC default).
BASELINE_CACHE = baseline_cache_path()


def _build_masks(n: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    n_hidden = max(1, min(N_CELLS - 1, int(round(rate * N_CELLS))))
    masks = np.ones((n, N_CELLS), dtype=np.float32)
    for i in range(n):
        ix = rng.choice(N_CELLS, size=n_hidden, replace=False)
        masks[i, ix] = 0.0
    return masks


def vae_predict(model: MaskedVAE, surfaces_z: np.ndarray, masks: np.ndarray,
                device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(surfaces_z).to(device)
        m = torch.from_numpy(masks).to(device)
        recon, _, _ = model(x, m)
    return recon.cpu().numpy()


def _rmse(pred: np.ndarray, truth: np.ndarray, hidden: np.ndarray) -> float:
    err = (pred - truth)[hidden]
    err = err[np.isfinite(err)]
    return float(np.sqrt(np.mean(err ** 2)))


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_baseline(mask_rates: tuple[float, ...], seed: int = 0,
                     symbol: str = "BTCUSDT") -> dict[float, float]:
    """Smile-refit RMSE per mask rate (depends only on data + seed). Cached per symbol."""
    cache = baseline_cache_path(symbol)
    if cache.exists():
        cached = json.loads(cache.read_text())
        if (cached.get("mask_rates") == list(mask_rates)
                and cached.get("seed") == seed
                and cached.get("symbol", "BTCUSDT") == symbol):
            return {float(k): v for k, v in cached["rmse"].items()}

    bundle = load_split("test", symbol=symbol)
    train_bundle = load_split("train", symbol=symbol)
    fallback = train_bundle.surfaces.mean(axis=0)
    k_all, ts_all = _load_k_grid(symbol)
    ts_to_k = {t: k_all[i] for i, t in enumerate(ts_all)}
    k_split = np.stack([ts_to_k[t] for t in bundle.ts], axis=0)
    surfaces = bundle.surfaces
    surfaces_flat = surfaces.reshape(-1, N_CELLS)

    rng = np.random.default_rng(seed)
    rmse_by_rate: dict[float, float] = {}
    for rate in mask_rates:
        masks_flat = _build_masks(len(surfaces), rate, rng)
        preds = np.full_like(surfaces, np.nan)
        for i in range(len(surfaces)):
            m2d = masks_flat[i].reshape(N_TENORS, N_DELTAS)
            preds[i] = predict_one(surfaces[i], k_split[i], m2d, fallback)
        rmse_by_rate[rate] = _rmse(preds.reshape(-1, N_CELLS),
                                   surfaces_flat, masks_flat < 0.5)

    cache.write_text(json.dumps({
        "symbol": symbol,
        "mask_rates": list(mask_rates), "seed": seed,
        "rmse": {str(r): v for r, v in rmse_by_rate.items()},
    }, indent=2))
    return rmse_by_rate


def eval_vae(run_dir: Path, target_symbol: str | None = None) -> dict:
    cfg = load_config(run_dir)
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    device = _device()
    model = build_model_from_ckpt(ckpt, cfg).to(device)
    model.load_state_dict(ckpt["model"])

    # Default target: the run's first training symbol (the natural "in-dist"
    # for that model). Override to evaluate any model on any symbol.
    primary = cfg.symbols[0] if cfg.symbols else "BTCUSDT"
    symbol = target_symbol or primary

    bundle = load_split("test", symbol=symbol)
    train_bundle = load_split("train", symbol=symbol)
    mean, std = train_bundle.mean.reshape(-1), train_bundle.std.reshape(-1)
    surfaces_flat = bundle.surfaces.reshape(-1, N_CELLS)
    surfaces_z = ((surfaces_flat - mean) / std).astype(np.float32)

    rng = np.random.default_rng(cfg.eval_seed)
    vae_rmse: dict[float, float] = {}
    for rate in cfg.eval_mask_rates:
        masks_flat = _build_masks(len(bundle.surfaces), rate, rng)
        recon_z = vae_predict(model, surfaces_z, masks_flat, device)
        recon = recon_z * std + mean
        vae_rmse[rate] = _rmse(recon, surfaces_flat, masks_flat < 0.5)

    base_rmse = compute_baseline(cfg.eval_mask_rates, cfg.eval_seed, symbol=symbol)

    metrics = {
        "target_symbol": symbol,
        "training_symbols": list(cfg.symbols),
        "ckpt_epoch": int(ckpt["epoch"]),
        "ckpt_val_loss": float(ckpt["val_loss"]),
        "mask_rates": list(cfg.eval_mask_rates),
        "vae_rmse_vol": {str(r): vae_rmse[r] for r in cfg.eval_mask_rates},
        "baseline_rmse_vol": {str(r): base_rmse[r] for r in cfg.eval_mask_rates},
    }
    # Default eval (primary symbol) goes to metrics.json for compare_runs;
    # cross-symbol evals get a per-symbol suffix so nothing overwrites.
    out_name = "metrics.json" if symbol == primary else f"metrics_{symbol.lower()}.json"
    (run_dir / out_name).write_text(json.dumps(metrics, indent=2))

    if symbol == primary:
        _plot_rmse_vs_mask(cfg, vae_rmse, base_rmse, run_dir)
        _plot_anomaly(model, device, run_dir, mean, std, symbol)
    print(f"\n[{run_dir.name}]  target={symbol}")
    for r in cfg.eval_mask_rates:
        print(f"  rate={r:.2f}  VAE={vae_rmse[r]:.4f}  base={base_rmse[r]:.4f}")
    return metrics


def _plot_rmse_vs_mask(cfg: TrainConfig, vae_rmse, base_rmse, run_dir: Path) -> None:
    rates = sorted(vae_rmse)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([r * 100 for r in rates], [vae_rmse[r] * 100 for r in rates],
            "o-", label=f"VAE ({cfg.name}, z={cfg.z_dim} h={cfg.hidden})", linewidth=2)
    ax.plot([r * 100 for r in rates], [base_rmse[r] * 100 for r in rates],
            "s-", label="Smile refit (baseline)", linewidth=2)
    ax.set_xlabel("Mask rate (% cells hidden)")
    ax.set_ylabel("RMSE on hidden cells (vol pts)")
    ax.set_title(f"BTC surface completion — {cfg.name}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "rmse_vs_mask.png", dpi=140)
    plt.close(fig)


def _plot_anomaly(model: MaskedVAE, device: torch.device, run_dir: Path,
                  mean: np.ndarray, std: np.ndarray, symbol: str = "BTCUSDT") -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for split, color in [("train", "tab:blue"), ("val", "tab:orange"), ("test", "tab:green")]:
        b = load_split(split, symbol=symbol)
        flat = b.surfaces.reshape(-1, N_CELLS)
        z = ((flat - mean) / std).astype(np.float32)
        masks = np.ones_like(z, dtype=np.float32)
        recon_z = vae_predict(model, z, masks, device)
        recon = recon_z * std + mean
        err = np.sqrt(((recon - flat) ** 2).mean(axis=1))
        ax.plot(b.ts, err * 100, label=split, color=color, linewidth=0.7, alpha=0.85)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Per-snapshot reconstruction RMSE (vol pts)")
    ax.set_title(f"BTC: reconstruction error over time — {run_dir.name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "recon_loss_timeseries.png", dpi=140)
    plt.close(fig)


def _parse_run_arg(arg: str | None) -> Path:
    if arg is None or arg == "--latest":
        runs = list_runs()
        if not runs:
            raise SystemExit("no runs found in runs/vae_btc/")
        return runs[-1]
    p = Path(arg)
    if not p.is_absolute():
        p = RUNS_ROOT / arg
    if not p.exists():
        raise SystemExit(f"run dir not found: {p}")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default="--latest")
    ap.add_argument("--target-symbol", default=None,
                    help="symbol to eval on (default = run's first training symbol)")
    args = ap.parse_args()
    eval_vae(_parse_run_arg(args.run_dir), target_symbol=args.target_symbol)


if __name__ == "__main__":
    main()
