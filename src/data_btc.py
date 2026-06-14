"""BTC dataset: load gridded surfaces, split time-ordered, mask cells during training.

Pipeline:
  1. Read data/grid/binance_eoh/BTCUSDT_grid.parquet.
  2. Pivot to (n_snapshots, 6 tenors, 7 deltas).
  3. Keep only fully-filled snapshots (no NaN across all 42 cells).
  4. Time-ordered 70/15/15 split.
  5. Per-cell z-score normalizer fit on train.
  6. PyTorch Dataset returns (surface_z, mask) where mask=1 means observed.

A `prepare()` step materializes data/processed/btc_splits.npz, which the
baseline and eval scripts also read.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SYMBOL = "BTCUSDT"


def grid_path(symbol: str = DEFAULT_SYMBOL) -> Path:
    return ROOT / "data" / "grid" / "binance_eoh" / f"{symbol}_grid.parquet"


def splits_path(symbol: str = DEFAULT_SYMBOL) -> Path:
    return ROOT / "data" / "processed" / f"{symbol.lower()}_splits.npz"



TENORS = np.array([14, 30, 60, 90, 120, 180], dtype=float)
DELTAS = np.array([0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90], dtype=float)
N_TENORS = len(TENORS)
N_DELTAS = len(DELTAS)
N_CELLS = N_TENORS * N_DELTAS  # 42

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

MASK_RATE_MIN = 0.10
MASK_RATE_MAX = 0.50

STRUCTURED_SCHEMES = ("row_random", "col_random", "wing_put", "wing_call", "long_tenor")


def build_structured_mask_single(scheme: str, rng: np.random.Generator) -> np.ndarray:
    """One (N_TENORS, N_DELTAS) mask for the given structured scheme.

    Keep in sync with eval_structured.SCHEMES so train- and eval-time masks
    sample from the same distribution.
    """
    mask = np.ones((N_TENORS, N_DELTAS), dtype=np.float32)
    if scheme == "row_random":
        mask[int(rng.integers(0, N_TENORS)), :] = 0.0
    elif scheme == "col_random":
        mask[:, int(rng.integers(0, N_DELTAS))] = 0.0
    elif scheme == "wing_put":
        mask[:, 0:2] = 0.0
    elif scheme == "wing_call":
        mask[:, 5:7] = 0.0
    elif scheme == "long_tenor":
        mask[-1, :] = 0.0
    else:
        raise ValueError(f"unknown structured scheme {scheme!r}")
    return mask


@dataclass
class SplitBundle:
    """Numpy arrays for one split. Surfaces are unnormalized vol (iv units)."""
    surfaces: np.ndarray   # (N, 6, 7)
    ts: np.ndarray         # (N,) datetime64
    mean: np.ndarray       # (6, 7) per-cell train mean
    std: np.ndarray        # (6, 7) per-cell train std


def _pivot_to_surfaces(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Pivot long grid → (n_snaps, 6, 7) iv tensor and ts array, sorted by ts."""
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    # Index by canonical order to guarantee axis alignment.
    tenor_to_i = {float(t): i for i, t in enumerate(TENORS)}
    delta_to_j = {float(d): j for j, d in enumerate(DELTAS)}
    df["i"] = df["tenor_days"].astype(float).map(tenor_to_i)
    df["j"] = df["delta"].astype(float).map(delta_to_j)
    assert df["i"].notna().all() and df["j"].notna().all(), \
        "unexpected (tenor, delta) outside canonical grid"

    ts_sorted = np.array(sorted(df["ts"].unique()))
    ts_to_n = {t: n for n, t in enumerate(ts_sorted)}
    n_snaps = len(ts_sorted)

    iv = np.full((n_snaps, N_TENORS, N_DELTAS), np.nan, dtype=np.float32)
    n_idx = df["ts"].map(ts_to_n).to_numpy()
    iv[n_idx, df["i"].to_numpy(int), df["j"].to_numpy(int)] = df["iv"].to_numpy(np.float32)
    return iv, ts_sorted


def prepare(force: bool = False, symbol: str = DEFAULT_SYMBOL) -> None:
    """Build and cache the train/val/test split bundle for `symbol`."""
    sp = splits_path(symbol)
    gp = grid_path(symbol)
    if sp.exists() and not force:
        print(f"splits already at {sp} (use force=True to rebuild)")
        return

    df = pd.read_parquet(gp)
    surfaces_all, ts_all = _pivot_to_surfaces(df)
    print(f"loaded {len(surfaces_all):,} snapshots from {gp.name}")

    fully_filled = np.isfinite(surfaces_all).all(axis=(1, 2))
    surfaces = surfaces_all[fully_filled]
    ts = ts_all[fully_filled]
    print(f"fully-filled: {len(surfaces):,} / {len(surfaces_all):,} "
          f"({fully_filled.mean():.1%})")

    n = len(surfaces)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train_idx = slice(0, n_train)
    val_idx = slice(n_train, n_train + n_val)
    test_idx = slice(n_train + n_val, n)

    train_surf = surfaces[train_idx]
    mean = train_surf.mean(axis=0)
    std = train_surf.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)  # guard div-by-zero on degenerate cells

    sp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        sp,
        train_surfaces=train_surf,
        val_surfaces=surfaces[val_idx],
        test_surfaces=surfaces[test_idx],
        train_ts=ts[train_idx],
        val_ts=ts[val_idx],
        test_ts=ts[test_idx],
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        tenors=TENORS,
        deltas=DELTAS,
    )
    print(f"wrote {sp}")
    print(f"  train: {len(train_surf):,}  {ts[train_idx][0]} → {ts[train_idx][-1]}")
    print(f"  val:   {len(surfaces[val_idx]):,}  {ts[val_idx][0]} → {ts[val_idx][-1]}")
    print(f"  test:  {len(surfaces[test_idx]):,}  {ts[test_idx][0]} → {ts[test_idx][-1]}")


def load_split(name: str, symbol: str = DEFAULT_SYMBOL) -> SplitBundle:
    sp = splits_path(symbol)
    if not sp.exists():
        prepare(symbol=symbol)
    z = np.load(sp, allow_pickle=True)
    return SplitBundle(
        surfaces=z[f"{name}_surfaces"],
        ts=z[f"{name}_ts"],
        mean=z["mean"],
        std=z["std"],
    )


class BTCSurfaceDataset(Dataset):
    """Returns (surface_z, mask) tensors of shape (42,).

    mask = 1 means observed (encoder sees it), 0 means hidden (loss target).
    Mask rate drawn from U(rate_min, rate_max) per sample.
    If `fixed_mask_rate` is set, that rate is used for every sample (for eval).
    """

    def __init__(
        self,
        split: str,
        rate_min: float = MASK_RATE_MIN,
        rate_max: float = MASK_RATE_MAX,
        fixed_mask_rate: float | None = None,
        seed: int | None = None,
        mask_scheme: str = "random",
        mixed_structured_prob: float = 0.5,
        symbols: tuple[str, ...] = (DEFAULT_SYMBOL,),
    ):
        # Per-symbol z-score then concatenate: each market's surfaces enter
        # the model as level-removed shapes, which is what we want the VAE to
        # learn (a shared manifold). At inference time the same logic flips
        # in reverse: normalize the target with its own train stats.
        zs, raws = [], []
        for sym in symbols:
            b = load_split(split, symbol=sym)
            flat = b.surfaces.reshape(-1, N_CELLS)
            mean, std = b.mean.reshape(N_CELLS), b.std.reshape(N_CELLS)
            zs.append((flat - mean) / std)
            raws.append(flat)
        self.surfaces = np.concatenate(raws, axis=0)
        self.surfaces_z = np.concatenate(zs, axis=0).astype(np.float32)
        self.symbols = symbols
        # mean/std are per-symbol; we don't carry a single one (eval uses target's).
        self.rate_min = rate_min
        self.rate_max = rate_max
        self.fixed_mask_rate = fixed_mask_rate
        self.mask_scheme = mask_scheme
        self.mixed_structured_prob = mixed_structured_prob
        self._rng = np.random.default_rng(seed)
        if mask_scheme not in {"random", "mixed"}:
            raise ValueError(f"unknown mask_scheme {mask_scheme!r}")

    def __len__(self) -> int:
        return len(self.surfaces_z)

    def _random_mask(self) -> np.ndarray:
        if self.fixed_mask_rate is not None:
            rate = self.fixed_mask_rate
        else:
            rate = float(self._rng.uniform(self.rate_min, self.rate_max))
        n_hidden = max(1, min(N_CELLS - 1, int(round(rate * N_CELLS))))
        hidden_ix = self._rng.choice(N_CELLS, size=n_hidden, replace=False)
        mask = np.ones(N_CELLS, dtype=np.float32)
        mask[hidden_ix] = 0.0
        return mask

    def _structured_mask(self) -> np.ndarray:
        scheme = STRUCTURED_SCHEMES[int(self._rng.integers(0, len(STRUCTURED_SCHEMES)))]
        return build_structured_mask_single(scheme, self._rng).reshape(N_CELLS)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        surf = self.surfaces_z[idx]
        if self.mask_scheme == "mixed" and self._rng.random() < self.mixed_structured_prob:
            mask = self._structured_mask()
        else:
            mask = self._random_mask()
        return (
            torch.from_numpy(surf.astype(np.float32)),
            torch.from_numpy(mask),
        )


if __name__ == "__main__":
    prepare(force=True)
