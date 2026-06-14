"""TrainConfig + per-run directory helpers.

Every training run produces runs/vae_btc/<timestamp>_<name>/ containing:
  config.json   — TrainConfig that produced the run
  history.json  — per-epoch train/val metrics
  best.pt       — best-val-loss checkpoint
  metrics.json  — eval results (RMSE vs mask rate, etc), written by eval_vae
  *.png         — plots, written by eval_vae

Runs are immutable once written. Compare across runs via src/compare_runs.py.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = ROOT / "runs" / "vae_btc"


@dataclass
class TrainConfig:
    name: str = "default"

    # Data: tuple of Binance symbol(s) to train on. Multi-symbol = per-symbol
    # z-score then concat (the VAE sees a shared *shape* manifold).
    symbols: tuple[str, ...] = ("BTCUSDT",)

    # Model
    arch: str = "mlp"           # 'mlp' (default) | 'conv2d' | 'attention'
    z_dim: int = 8
    hidden: int = 128

    # Loss
    w_hidden: float = 1.0
    w_obs: float = 0.1
    beta: float = 1e-3

    # Optim / training
    lr: float = 1e-3
    epochs: int = 200
    batch_size: int = 128
    patience: int = 30
    seed: int = 42

    # Masking (training-time augmentation)
    # mask_scheme: 'random' = iid cell mask at rate U(min, max).
    #              'mixed'  = 50% random / 50% structured rotation
    #                        (row_random, col_random, wing_put, wing_call, long_tenor).
    mask_scheme: str = "random"
    mask_rate_min: float = 0.10
    mask_rate_max: float = 0.50

    # Eval (also stored so eval is reproducible from config alone)
    eval_mask_rates: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50)
    eval_seed: int = 0

    # Free-text notes (what this ablation is testing)
    notes: str = ""


def make_run_dir(cfg: TrainConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = RUNS_ROOT / f"{stamp}_{cfg.name}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def save_config(cfg: TrainConfig, run_dir: Path) -> None:
    data = asdict(cfg)
    # JSON doesn't have tuples — normalize for round-trip equality.
    data["eval_mask_rates"] = list(cfg.eval_mask_rates)
    data["symbols"] = list(cfg.symbols)
    (run_dir / "config.json").write_text(json.dumps(data, indent=2))


def load_config(run_dir: Path) -> TrainConfig:
    data = json.loads((run_dir / "config.json").read_text())
    data["eval_mask_rates"] = tuple(data["eval_mask_rates"])
    if "symbols" in data:
        data["symbols"] = tuple(data["symbols"])
    # Drop any unknown fields so old runs still load after schema changes.
    valid = {f.name for f in dataclasses.fields(TrainConfig)}
    data = {k: v for k, v in data.items() if k in valid}
    return TrainConfig(**data)


def list_runs() -> list[Path]:
    if not RUNS_ROOT.exists():
        return []
    return sorted(p for p in RUNS_ROOT.iterdir()
                  if p.is_dir() and (p / "config.json").exists())
