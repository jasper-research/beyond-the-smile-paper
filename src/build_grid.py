"""Build (tenor × delta) gridded IV surfaces from cleaned chains.

Per snapshot:
  1. Use train-grade rows only.
  2. Per available expiry: fit total-variance w(k) = a + b·k + c·k² in
     log-moneyness k = ln(K/F).
  3. For each target tenor T*: linearly interpolate (a, b, c) across the two
     bracketing available expiries — equivalent to linear-in-T interpolation of
     total variance at fixed k.
  4. For each target call-delta d: fixed-point solve for k_d such that the
     Black-Scholes call-delta at (k_d, σ(k_d, T*)) equals d.
  5. Emit one row per (snapshot, tenor, delta) with iv, log-moneyness, and the
     smile parameters used (a, b, c) for diagnostics.

Grid (6 × 7 = 42 points; trimmed for Binance Options coverage):
  Tenors (days):  14, 30, 60, 90, 120, 180
  Deltas (call):  0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "data" / "clean" / "binance_eoh"
OUT_DIR = ROOT / "data" / "grid" / "binance_eoh"

TENOR_DAYS = np.array([14, 30, 60, 90, 120, 180], dtype=float)
DELTAS = np.array([0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90])

MIN_POINTS_PER_EXPIRY = 5  # need ≥ 5 strikes to fit a quadratic robustly
TARGET_TENOR_YEARS = TENOR_DAYS / 365.25


def fit_smile(k: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Least-squares fit of w(k) = a + b·k + c·k². Returns array([a, b, c])."""
    X = np.column_stack([np.ones_like(k), k, k * k])
    coef, *_ = np.linalg.lstsq(X, w, rcond=None)
    return coef


def w_of_k(k: float | np.ndarray, params: np.ndarray) -> float | np.ndarray:
    a, b, c = params
    return a + b * k + c * k * k


def solve_k_for_delta(params: np.ndarray, T: float, target_delta: float,
                       max_iter: int = 50, tol: float = 1e-7,
                       delta_tol: float = 1e-4) -> tuple[float, float]:
    """Fixed-point: find k such that BS call-delta at (k, σ(k,T)) = target_delta.

    Iterates k_{n+1} = 0.5 σ_n² T − Φ⁻¹(d) · σ_n √T, with σ_n = √(w(k_n)/T).
    Returns NaN if iteration didn't converge OR if the converged (k, σ) doesn't
    actually reproduce target_delta within delta_tol.
    """
    phi_inv = norm.ppf(target_delta)
    sqrtT = np.sqrt(T)
    w0 = max(float(w_of_k(0.0, params)), 1e-6)
    sigma = np.sqrt(w0 / T)
    k = 0.0
    converged = False
    for _ in range(max_iter):
        k = 0.5 * sigma * sigma * T - phi_inv * sigma * sqrtT
        w = max(float(w_of_k(k, params)), 1e-6)
        sigma_new = np.sqrt(w / T)
        if abs(sigma_new - sigma) < tol:
            sigma = sigma_new
            converged = True
            break
        sigma = sigma_new
    if not converged:
        return np.nan, np.nan
    # Validate: the converged (k, σ) must actually produce target_delta.
    d1 = (-k + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    if not np.isfinite(d1) or abs(float(norm.cdf(d1)) - target_delta) > delta_tol:
        return np.nan, np.nan
    return float(k), float(sigma)


def interp_params_in_T(T_target: float, expiry_params: dict[float, np.ndarray]
                       ) -> tuple[np.ndarray, str]:
    """Return (a,b,c) at T_target. Linear-interpolate inside the available
    range; clamp flat-in-time to the nearest expiry's smile at the boundaries."""
    Ts = sorted(expiry_params.keys())
    if T_target <= Ts[0]:
        return expiry_params[Ts[0]], "extrap_short" if T_target < Ts[0] else "exact"
    if T_target >= Ts[-1]:
        return expiry_params[Ts[-1]], "extrap_long" if T_target > Ts[-1] else "exact"
    for T in Ts:
        if abs(T - T_target) < 1e-9:
            return expiry_params[T], "exact"
    lo = max(t for t in Ts if t < T_target)
    hi = min(t for t in Ts if t > T_target)
    alpha = (T_target - lo) / (hi - lo)
    return (1 - alpha) * expiry_params[lo] + alpha * expiry_params[hi], "interp"


def grid_one_snapshot(snap: pd.DataFrame) -> list[dict]:
    """Grid one (ts) snapshot to 63 (tenor, delta, iv) rows."""
    train = snap[snap["is_train_grade"]]
    if len(train) < 3 * MIN_POINTS_PER_EXPIRY:
        return []
    # Build per-expiry smile
    expiry_params: dict[float, np.ndarray] = {}
    expiry_npts: dict[float, int] = {}
    for T_yr, grp in train.groupby("tte_years"):
        if len(grp) < MIN_POINTS_PER_EXPIRY:
            continue
        k = np.log(grp["strike_p"].values / grp["forward"].values)
        w = (grp["mark_iv"].values ** 2) * T_yr
        finite = np.isfinite(k) & np.isfinite(w)
        if finite.sum() < MIN_POINTS_PER_EXPIRY:
            continue
        expiry_params[float(T_yr)] = fit_smile(k[finite], w[finite])
        expiry_npts[float(T_yr)] = int(finite.sum())
    if len(expiry_params) < 2:
        return []

    out: list[dict] = []
    ts = snap["ts"].iloc[0]
    for T_target_days, T_target in zip(TENOR_DAYS, TARGET_TENOR_YEARS):
        params, status = interp_params_in_T(T_target, expiry_params)
        for d in DELTAS:
            k_d, sigma_d = solve_k_for_delta(params, T_target, float(d))
            out.append({
                "ts": ts, "tenor_days": float(T_target_days),
                "delta": float(d), "k": k_d, "iv": sigma_d,
                "status": status, "a": params[0], "b": params[1], "c": params[2],
                "n_expiries": len(expiry_params),
            })
    return out


def grid_symbol(sym: str) -> Path:
    src = SRC_DIR / f"{sym}_eoh_clean.parquet"
    print(f"\n=== {sym} ===")
    df = pd.read_parquet(src)
    df["ts"] = pd.to_datetime(df["ts"])
    print(f"  loaded {len(df):,} rows, {df['ts'].nunique():,} snapshots")
    print(f"  train-grade rows: {df['is_train_grade'].sum():,}")

    rows: list[dict] = []
    snapshots = list(df.groupby("ts", sort=True))
    for i, (ts, snap) in enumerate(snapshots):
        rows.extend(grid_one_snapshot(snap))
        if (i + 1) % 500 == 0:
            print(f"  {sym}: {i+1}/{len(snapshots)} snapshots gridded")

    out_df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{sym}_grid.parquet"
    out_df.to_parquet(out_path, index=False)

    snaps_total = df["ts"].nunique()
    snaps_with_grid = out_df["ts"].nunique()
    point_cov = out_df["iv"].notna().mean()
    print(f"  → {out_path}")
    print(f"  gridded snapshots: {snaps_with_grid:,}/{snaps_total:,} "
          f"({snaps_with_grid/snaps_total:.1%})")
    print(f"  per-point IV coverage: {point_cov:.1%}  (NaN = extrapolation)")
    print(f"  rows written: {len(out_df):,}")
    return out_path


def main() -> None:
    for sym in ["BTCUSDT", "ETHUSDT"]:
        grid_symbol(sym)


if __name__ == "__main__":
    main()
