"""Smile-refit baseline for surface completion.

For each (snapshot, mask):
  1. Take the OBSERVED (k, iv) cells. Per tenor row with ≥3 observations, fit
     w(k) = a + b·k + c·k² in total-variance space (same parameterization as
     build_grid.py).
  2. For each hidden (tenor, delta) cell:
     - Get (a,b,c) at that tenor by linear-in-T interpolation across the tenors
       that have a smile fit; flat extrapolation at the boundaries.
     - Fixed-point solve for k such that BS call-delta = target delta.
     - Predicted iv is σ at that (k, T).
  3. Return predicted iv for every hidden cell. If a tenor has no smile and no
     bracketing neighbors, fall back to the global train-set mean iv at that
     (tenor, delta) cell.

This is the practitioner's "what would you do without a VAE" baseline — refit
the smile family the chain already implies. The VAE has to beat this to be
worth deploying.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .build_grid import fit_smile, interp_params_in_T, solve_k_for_delta
from .data_btc import (
    DEFAULT_SYMBOL,
    DELTAS,
    N_CELLS,
    N_DELTAS,
    N_TENORS,
    TENORS,
    grid_path,
    load_split,
)

ROOT = Path(__file__).resolve().parent.parent

TENORS_YR = TENORS / 365.25
MIN_OBS_PER_TENOR = 3  # need ≥3 cells in a tenor row to fit a 3-param quadratic


def _load_k_grid(symbol: str = DEFAULT_SYMBOL) -> np.ndarray:
    """Load the canonical k matrix for the fully-filled test snapshots.

    Per-cell k varies snapshot-to-snapshot (k = log(K/F) depends on forward),
    so we cache k aligned to the same (ts, tenor, delta) ordering as the
    surfaces in the split bundle.

    Returns (N_test, 6, 7) array of log-moneyness at each cell.
    """
    import pandas as pd

    df = pd.read_parquet(grid_path(symbol))
    df["ts"] = pd.to_datetime(df["ts"])
    tenor_to_i = {float(t): i for i, t in enumerate(TENORS)}
    delta_to_j = {float(d): j for j, d in enumerate(DELTAS)}
    df["i"] = df["tenor_days"].astype(float).map(tenor_to_i)
    df["j"] = df["delta"].astype(float).map(delta_to_j)
    ts_sorted = np.array(sorted(df["ts"].unique()))
    ts_to_n = {t: n for n, t in enumerate(ts_sorted)}
    n_all = len(ts_sorted)
    k_all = np.full((n_all, N_TENORS, N_DELTAS), np.nan, dtype=np.float32)
    iv_all = np.full((n_all, N_TENORS, N_DELTAS), np.nan, dtype=np.float32)
    n_idx = df["ts"].map(ts_to_n).to_numpy()
    k_all[n_idx, df["i"].to_numpy(int), df["j"].to_numpy(int)] = df["k"].to_numpy(np.float32)
    iv_all[n_idx, df["i"].to_numpy(int), df["j"].to_numpy(int)] = df["iv"].to_numpy(np.float32)
    fully_filled = np.isfinite(iv_all).all(axis=(1, 2))
    return k_all[fully_filled], ts_sorted[fully_filled]


def predict_one(
    surface: np.ndarray,          # (6, 7) iv ground truth (use only observed)
    k_grid: np.ndarray,           # (6, 7) log-moneyness coords
    mask: np.ndarray,             # (6, 7) 1=observed, 0=hidden
    fallback: np.ndarray,         # (6, 7) global mean iv per cell
) -> np.ndarray:
    """Return (6,7) iv predictions for the HIDDEN cells (observed cells = NaN)."""
    pred = np.full((N_TENORS, N_DELTAS), np.nan, dtype=np.float32)

    # Per tenor: fit smile in (k, w) using observed cells in that row.
    expiry_params: dict[float, np.ndarray] = {}
    for i in range(N_TENORS):
        obs_j = np.where(mask[i] > 0.5)[0]
        if len(obs_j) < MIN_OBS_PER_TENOR:
            continue
        k_i = k_grid[i, obs_j].astype(float)
        iv_i = surface[i, obs_j].astype(float)
        T = float(TENORS_YR[i])
        w_i = (iv_i ** 2) * T
        finite = np.isfinite(k_i) & np.isfinite(w_i)
        if finite.sum() < MIN_OBS_PER_TENOR:
            continue
        expiry_params[T] = fit_smile(k_i[finite], w_i[finite])

    if not expiry_params:
        # No usable smiles anywhere — fall back to mean for all hidden cells.
        hidden = mask < 0.5
        pred[hidden] = fallback[hidden]
        return pred

    for i in range(N_TENORS):
        T_target = float(TENORS_YR[i])
        for j in range(N_DELTAS):
            if mask[i, j] > 0.5:
                continue
            params, _ = interp_params_in_T(T_target, expiry_params)
            _, sigma = solve_k_for_delta(params, T_target, float(DELTAS[j]))
            if np.isfinite(sigma):
                pred[i, j] = sigma
            else:
                pred[i, j] = fallback[i, j]
    return pred


def rmse_vol_pts(pred: np.ndarray, truth: np.ndarray, mask_hidden: np.ndarray) -> float:
    """RMSE in volatility points (i.e., 0.01 = 1 vol pt). Computed over hidden cells."""
    err = (pred - truth)[mask_hidden]
    err = err[np.isfinite(err)]
    if len(err) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(err ** 2)))


def evaluate(split: str = "test", mask_rates: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5),
             seed: int = 0) -> dict[float, float]:
    """Evaluate smile-refit RMSE across mask rates. Returns {rate: rmse_vol_pts}."""
    bundle = load_split(split)
    k_all, ts_all = _load_k_grid()
    # Re-align k to the split's ts order (load_split already orders by ts within split).
    ts_to_k = {t: k_all[i] for i, t in enumerate(ts_all)}
    k_split = np.stack([ts_to_k[t] for t in bundle.ts], axis=0)

    train_bundle = load_split("train")
    fallback = train_bundle.surfaces.mean(axis=0)  # (6, 7) train-mean iv

    rng = np.random.default_rng(seed)
    out: dict[float, float] = {}
    for rate in mask_rates:
        n_hidden = max(1, min(N_CELLS - 1, int(round(rate * N_CELLS))))
        all_err = []
        for n in range(len(bundle.surfaces)):
            mask_flat = np.ones(N_CELLS, dtype=np.float32)
            hidden_ix = rng.choice(N_CELLS, size=n_hidden, replace=False)
            mask_flat[hidden_ix] = 0.0
            mask = mask_flat.reshape(N_TENORS, N_DELTAS)
            pred = predict_one(bundle.surfaces[n], k_split[n], mask, fallback)
            hidden_mask = mask < 0.5
            err = (pred - bundle.surfaces[n])[hidden_mask]
            err = err[np.isfinite(err)]
            all_err.append(err)
        all_err = np.concatenate(all_err)
        rmse = float(np.sqrt(np.mean(all_err ** 2)))
        out[rate] = rmse
        print(f"  mask_rate={rate:.2f}  RMSE={rmse:.4f} vol  (n_err={len(all_err):,})")
    return out


if __name__ == "__main__":
    print("=== smile-refit baseline on TEST split ===")
    evaluate("test")
