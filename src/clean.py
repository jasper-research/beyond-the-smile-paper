"""Clean and enrich Binance EOH option-chain data.

Pipeline per symbol:
  1. Parse expiry / strike / right (C|P) from instrument name.
  2. Filter unusable rows (expired, malformed strike, missing price).
  3. Back out forward F per (ts, expiry) via put-call parity, median across strikes.
  4. Invert bid-IV, ask-IV, mid-IV via vectorized Newton on Black-76,
     initialized from the published mark_iv (fast and stable).
  5. Recompute mark_iv from mark_price as a sanity check.
  6. Write enriched parquet to data/clean/binance_eoh/.

Assumes risk-free rate r ≈ 0 (USDT short-dated yield is negligible). Pricing is
in Black-76 forward form so r only enters via the discount factor on the
parity-implied forward, which cancels in the inversion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "data" / "external" / "binance_eoh"
OUT_DIR = ROOT / "data" / "clean" / "binance_eoh"

SYMBOL_RE = r"(\w+)-(\d{6})-(\d+)-([CP])"


def parse_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    parts = df["symbol"].str.extract(SYMBOL_RE)
    df = df.assign(
        date=pd.to_datetime(df["date"]),
        hour=df["hour"].astype(int),
        expiry=pd.to_datetime(parts[1], format="%y%m%d"),
        strike_p=pd.to_numeric(parts[2], errors="coerce"),
        right=parts[3],
    )
    df["ts"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")
    df["dte_days"] = (df["expiry"] - df["date"]).dt.days
    df["tte_years"] = df["dte_days"] / 365.25
    df["mark_iv"] = pd.to_numeric(df["mark_iv"], errors="coerce")
    df["mark_price"] = pd.to_numeric(df["mark_price"], errors="coerce")
    df["best_bid_price"] = pd.to_numeric(df["best_bid_price"], errors="coerce")
    df["best_ask_price"] = pd.to_numeric(df["best_ask_price"], errors="coerce")

    n0 = len(df)
    df = df[
        (df["dte_days"] > 0)
        & (df["strike_p"] > 0)
        & df["mark_price"].notna()
        & (df["mark_price"] > 0)
        & df["right"].isin(["C", "P"])
    ].reset_index(drop=True)
    print(f"  parse+filter: {n0:,} → {len(df):,} rows ({n0 - len(df):,} dropped)")
    return df


def back_out_forward(df: pd.DataFrame) -> pd.DataFrame:
    """Forward per (ts, expiry) via put-call parity: F = K + (C_mid - P_mid)."""
    mids = df.assign(mid=(df["best_bid_price"] + df["best_ask_price"]) / 2)
    mids = mids[(mids["best_bid_price"] > 0) & (mids["best_ask_price"] > 0)]
    calls = (
        mids[mids["right"] == "C"]
        .rename(columns={"mid": "C_mid"})[["ts", "expiry", "strike_p", "C_mid"]]
    )
    puts = (
        mids[mids["right"] == "P"]
        .rename(columns={"mid": "P_mid"})[["ts", "expiry", "strike_p", "P_mid"]]
    )
    pairs = calls.merge(puts, on=["ts", "expiry", "strike_p"], how="inner")
    pairs["fwd_est"] = pairs["strike_p"] + (pairs["C_mid"] - pairs["P_mid"])
    fwd = (
        pairs.groupby(["ts", "expiry"])
        .agg(forward=("fwd_est", "median"), parity_pairs=("fwd_est", "size"))
        .reset_index()
    )
    out = df.merge(fwd, on=["ts", "expiry"], how="left")
    cov = out["forward"].notna().mean()
    print(f"  forward back-out: {cov:.1%} of rows have a parity-implied forward "
          f"(mean pairs/expiry={fwd['parity_pairs'].mean():.1f})")
    return out


def black76_price(sigma, F, K, T, right_sign):
    """Vectorized Black-76 price. right_sign: +1 call, -1 put. Assumes r=0."""
    sqrtT = np.sqrt(T)
    sig_sqrtT = sigma * sqrtT
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / sig_sqrtT
        d2 = d1 - sig_sqrtT
    return right_sign * (F * norm.cdf(right_sign * d1) - K * norm.cdf(right_sign * d2))


def black76_vega(sigma, F, K, T):
    sqrtT = np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrtT)
    return F * norm.pdf(d1) * sqrtT


def invert_iv(price, F, K, T, right_sign, sigma0, max_iter=30, tol=1e-6) -> np.ndarray:
    """Vectorized Newton-Raphson on Black-76. NaN where it can't converge."""
    price = np.asarray(price, dtype=float)
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    right_sign = np.asarray(right_sign, dtype=float)
    sigma0 = np.asarray(sigma0, dtype=float)

    valid = (
        np.isfinite(price) & (price > 0)
        & np.isfinite(F) & (F > 0)
        & np.isfinite(K) & (K > 0)
        & np.isfinite(T) & (T > 0)
    )
    # Intrinsic-value floor: option price >= max(0, right*(F - K)). Below that, no valid IV.
    intrinsic = np.maximum(0.0, right_sign * (F - K))
    valid &= price > intrinsic

    # Init from mark_iv where available; fall back to 0.6 (typical crypto vol).
    sigma = np.where(np.isfinite(sigma0) & (sigma0 > 0.01), sigma0, 0.6).astype(float)
    sigma = np.clip(sigma, 0.01, 5.0)

    for _ in range(max_iter):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            p_est = black76_price(sigma, F, K, T, right_sign)
            vega = black76_vega(sigma, F, K, T)
            step = np.where(np.abs(vega) > 1e-10, (price - p_est) / vega, 0.0)
        sigma = np.clip(sigma + step, 0.001, 5.0)
        if np.nanmax(np.abs(step[valid])) < tol:
            break

    # Final residual check; reject rows that didn't converge tightly.
    p_final = black76_price(sigma, F, K, T, right_sign)
    rel_err = np.abs(p_final - price) / np.where(price > 0, price, 1.0)
    bad = (rel_err > 5e-3) | ~valid
    sigma[bad] = np.nan
    return sigma


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    right_sign = np.where(df["right"].values == "C", 1.0, -1.0)
    F = df["forward"].values
    K = df["strike_p"].values
    T = df["tte_years"].values
    seed = df["mark_iv"].values  # warm-start Newton from the published mark IV

    bid = df["best_bid_price"].values
    ask = df["best_ask_price"].values
    has_bid = (bid > 0) & np.isfinite(bid)
    has_ask = (ask > 0) & np.isfinite(ask)

    df["bid_iv"] = invert_iv(np.where(has_bid, bid, np.nan), F, K, T, right_sign, seed)
    df["ask_iv"] = invert_iv(np.where(has_ask, ask, np.nan), F, K, T, right_sign, seed)

    mid = np.where(has_bid & has_ask, 0.5 * (bid + ask), np.nan)
    df["mid_price"] = mid
    df["mid_iv"] = invert_iv(mid, F, K, T, right_sign, seed)
    df["mark_iv_recomp"] = invert_iv(df["mark_price"].values, F, K, T, right_sign, seed)

    print(
        "  IV fill: "
        f"bid_iv={df['bid_iv'].notna().mean():.1%} "
        f"ask_iv={df['ask_iv'].notna().mean():.1%} "
        f"mid_iv={df['mid_iv'].notna().mean():.1%} "
        f"mark_recomp={df['mark_iv_recomp'].notna().mean():.1%}"
    )
    diff = (df["mark_iv_recomp"] - df["mark_iv"]).abs()
    print(f"  |mark_iv_recomp - mark_iv|: p50={diff.median():.4f}  p95={diff.quantile(0.95):.4f}")
    return df


def tag_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean quality flags. No rows are dropped — filtering happens downstream."""
    with np.errstate(divide="ignore", invalid="ignore"):
        mid = (df["best_bid_price"] + df["best_ask_price"]) / 2
        spread_pct = (df["best_ask_price"] - df["best_bid_price"]) / mid
        moneyness = df["strike_p"] / df["forward"]

    df["spread_pct"] = spread_pct
    df["moneyness"] = moneyness

    df["is_quoted"] = (df["best_bid_price"] > 0) & (df["best_ask_price"] > 0)
    df["is_oi_positive"] = df["openinterest_contracts"] > 0
    df["is_tight_quote"] = df["is_quoted"] & (spread_pct < 0.5)
    df["is_train_tenor"] = df["dte_days"].between(7, 365)
    df["is_train_moneyness"] = moneyness.between(0.5, 2.0)
    df["is_iv_sane"] = df["mark_iv"].between(0.1, 3.0)
    df["is_train_grade"] = (
        df["is_quoted"]
        & df["is_oi_positive"]
        & df["is_tight_quote"]
        & df["is_train_tenor"]
        & df["is_train_moneyness"]
        & df["is_iv_sane"]
    )
    tg = df["is_train_grade"]
    print(
        f"  quality: quoted={df['is_quoted'].mean():.1%} "
        f"oi>0={df['is_oi_positive'].mean():.1%} "
        f"tight={df['is_tight_quote'].mean():.1%} "
        f"tenor={df['is_train_tenor'].mean():.1%} "
        f"mny={df['is_train_moneyness'].mean():.1%}"
    )
    print(f"  train-grade rows: {tg.sum():,} ({tg.mean():.1%})  "
          f"snapshots: {df.loc[tg, 'ts'].nunique():,}")
    return df


def clean_symbol(sym: str) -> Path:
    src = SRC_DIR / f"{sym}_eoh.parquet"
    print(f"\n=== {sym} ===")
    df = pd.read_parquet(src)
    df = parse_and_filter(df)
    df = back_out_forward(df)
    df = enrich(df)
    df = tag_quality(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{sym}_eoh_clean.parquet"
    df.to_parquet(out, index=False)
    print(f"  wrote {out}  ({len(df):,} rows)")
    return out


def main() -> None:
    for sym in ["BTCUSDT", "ETHUSDT"]:
        clean_symbol(sym)


if __name__ == "__main__":
    main()
