# Data Plan — Crypto Vol Surface VAE Paper

End-to-end record of every dataset evaluated, downloaded, cleaned, gridded, and planned for this project. Sources of truth: the scripts under `src/`; this document explains *what* and *why*.

---

## 1. Goal

Train a Variational Autoencoder on crypto options **implied-volatility surfaces** and study its ability to:

- Compress full smiles + term structures into a low-dim latent.
- Reconstruct surfaces from sparse inputs (a few quoted strikes).
- Generate fresh surfaces conditional on spot dynamics.
- Transfer across assets (train BTC → zero-shot test ETH).

The paper sits inside the Bergeron / Lund 2024 family of vol-surface VAEs but extends to **crypto + intraday cadence + cross-exchange transfer**, with $0–$50 total data spend.

---

## 2. Source evaluation summary

Hunt log — what we checked and why we landed on Binance EOH.

| Source | Verdict | Why |
|---|---|---|
| **Binance Options EOHSummary** | ✅ **Primary training set** | Free public archive, hourly chain w/ IV + greeks, BTC+ETH, 147 days |
| OptionsDX BTC EOD (free tier) | ✅ Reserved for daily Lund-style baseline | Free, ~1,200 days of EOD chains, BTC only |
| Tardis.dev first-of-month free | ✅ Reserved as sparse regime cross-check | ~48 days across 4 years, no auth needed |
| Deribit live snapshots | ✅ Reserved for future OOD test set | Free, real-time, both currencies |
| Tardis subscription | ❌ Out of budget | $300/mo minimum |
| CoinAPI Flat Files | ❌ Same tier as Tardis | Paid subscription |
| Amberdata Deribit | ❌ Enterprise pricing | |
| CryptoDataDownload | ❌ Plus+ paywall | Only DVOL is free |
| Polymarket BTC Up/Down (Kaggle) | ❌ Wrong shape | Binary prediction markets, not vol surfaces |
| HuggingFace `KEDevO/crypto-market-datasets` | ⚠️ Limited use | BTC spot useful as auxiliary; options part too thin |
| GitHub collectors (schepal, bottama, etc.) | ❌ Code only, no shared data | |
| Sussex thesis (Figshare) | ❌ PDF only, data not published | |

---

## 3. Primary training dataset — Binance Options EOH

### Source

- **URL**: `https://data.binance.vision/data/option/daily/EOHSummary/<SYMBOL>/`
- **Listing**: `https://s3-ap-northeast-1.amazonaws.com/data.binance.vision/?prefix=data/option/daily/EOHSummary/`
- **Symbols available**: BNBUSDT, BTCUSDT, DOGEUSDT, ETHUSDT, XRPUSDT
- **Auth**: none — public archive
- **License**: public market-data archive, attribution-friendly
- **Window**: 2023-05-18 → 2023-10-23 (147 days). Binance stopped publishing after this date.
- **Format**: daily zip, one CSV inside per (symbol, date)
- **Cadence**: **hourly** — 24 rows per option per day

### Schema (verified from a downloaded sample)

```
date, hour, symbol, underlying, type, strike,
open, high, low, close, volume_contracts, volume_usdt,
best_bid_price, best_ask_price, best_bid_qty, best_ask_qty,
best_buy_iv, best_sell_iv, mark_price, mark_iv,
delta, gamma, vega, theta,
openinterest_contracts, openinterest_usdt
```

Notable: 100% `mark_iv` coverage, 100% `best_sell_iv` (ask-side IV), **0% `best_buy_iv` (bid-side IV always empty)** — Binance simply doesn't publish bid-side IV. We invert it ourselves in `clean.py`.

### Download — `src/fetch_binance_eoh.py`

- Lists all daily zips via S3 listing endpoint, parallel-fetches with 8 workers, unzips in memory, concatenates to one parquet per symbol.
- Output: `data/external/binance_eoh/{BTCUSDT,ETHUSDT}_eoh.parquet`
- Size on disk: ~120 MB total compressed

### Raw row counts

| Symbol | Rows | Days | Avg options/snapshot |
|---|---|---|---|
| BTCUSDT | 796,972 | 147 | 226 |
| ETHUSDT | 979,331 | 147 | 278 |

---

## 4. Cleaning pipeline — `src/clean.py`

Five stages, all logged at runtime. No rows are deleted after stage 1 — every downstream filter becomes a boolean column.

### Stage 1 — Parse + hard filter

- Extract `expiry`, `strike`, `right ∈ {C, P}` from instrument name (`BTC-231027-33000-C` → 2023-10-27, 33000, Call).
- Coerce numeric columns from string (some IV cells are `""` for missing).
- Compute `dte_days` and `tte_years`.
- Drop rows with `dte ≤ 0`, malformed strike, missing or zero `mark_price`, or invalid right.
- **Drop rate: ~2.2%** (17k / 23k rows for BTC / ETH).

### Stage 2 — Forward back-out via put-call parity

For each (snapshot, expiry) group, pair every (strike, call_mid, put_mid) and compute `F = K + (C_mid − P_mid)`, taking the **median** across strikes for robustness. Assumes `r ≈ 0` (USDT short-dated yield is negligible at these tenors).

- Coverage: 99.5–99.7% of rows get a forward.
- Mean call/put pairs per (snapshot, expiry): 11.7 BTC, 14.3 ETH.

### Stage 3 — IV inversion (Black-76, vectorized Newton)

- Black-76 in forward form, r = 0 assumption baked in.
- **Warm-start each Newton iteration from the published `mark_iv`** — this is the key trick. Bid/ask IVs are typically within ±20% of mark, so Newton converges in 2–3 iterations from there.
- Computes `bid_iv`, `ask_iv`, `mid_iv`, and `mark_iv_recomp` for each row.

Sanity checks (BTC):

| Metric | Value |
|---|---|
| `bid_iv ≤ ask_iv` (where both exist) | 100% |
| `ask_iv` vs published `best_sell_iv`, p50 diff | 2.7 vol pts |
| `mark_iv_recomp` vs published `mark_iv`, p50 diff | 2.9 vol pts |
| `mark_iv_recomp` vs published, p95 diff | 22 vol pts (OTM tail) |

### Stage 4 — Decision: trust the published mark_iv

The p95 disagreement is almost certainly a **forward-source mismatch**. Binance most likely uses their perp-index price as the forward, not put-call parity. Two paths considered:

- **Option A (chosen)**: Use Binance's published `mark_iv` as the canonical IV. Our inverted `bid_iv` / `ask_iv` become **auxiliary features** (used for spread information, not for the primary surface).
- Option B: Pull Binance perp-index hourly, re-derive F, re-invert everything. ~1 day of extra work. Deferred unless we hit a need.

### Stage 5 — Quality flags (no rows dropped)

| Flag | Definition |
|---|---|
| `is_quoted` | both bid and ask present |
| `is_oi_positive` | `openinterest_contracts > 0` |
| `is_tight_quote` | `(ask − bid) / mid < 0.5` |
| `is_train_tenor` | `dte ∈ [7, 365]` |
| `is_train_moneyness` | `K / F ∈ [0.5, 2.0]` |
| `is_iv_sane` | `mark_iv ∈ [0.1, 3.0]` |
| **`is_train_grade`** | AND of all above |

| | BTC | ETH |
|---|---|---|
| `is_train_grade` rows | 406,990 (52.2%) | 463,106 (48.4%) |
| Snapshots with train-grade rows | 3,491 | 3,490 |

Why tag instead of delete: lets us widen / narrow filters per experiment (e.g., spread study needs the wider rows, surface training uses the strict band) without re-running the pipeline.

### Output

`data/clean/binance_eoh/{BTCUSDT,ETHUSDT}_eoh_clean.parquet`

---

## 5. Gridding pipeline — `src/build_grid.py`

Convert raw, irregularly-spaced chains into a fixed-shape (tenor × delta) surface ready for the VAE.

### Grid choice — 6 tenors × 7 deltas = **42 points**

```
Tenors (days):  14, 30, 60, 90, 120, 180
Deltas (call):  0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90
```

**Why these specific values:**

- We started with the canonical 7 × 9 = 63 grid (`[7, 14, 30, 60, 90, 180, 365]` × `[0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95]`) → only **9% fully-filled** for BTC.
- Diagnosed root causes:
  - Binance has **no 365-day listings** in the window (per-snapshot max DTE ~ 310d).
  - 7-day tenor isn't reliably bracketed (min DTE typically 10d).
  - The Δ = 0.05 / 0.95 extreme wings depend on quadratic-smile extrapolation that's unreliable at long tenor.
- Trimmed to the corners the data actually supports → **80.9% BTC, 92.1% ETH fully filled**.

### Algorithm (per snapshot)

1. Filter to `is_train_grade` rows.
2. **Per available expiry T_i**, fit total-variance `w(k) = a + b·k + c·k²` (least-squares) over the observed log-moneyness range. Require ≥ 5 strikes per expiry to fit.
3. **Per target tenor T\***, linearly interpolate `(a, b, c)` between the two bracketing expiries — this is equivalent to linear-in-T interpolation of total variance at fixed k. Flat-in-time extrapolation when T\* is outside the available range.
4. **Per target call-delta d**, fixed-point solve for k:
   - `d1 = (−k + 0.5 σ² T) / (σ √T) = Φ⁻¹(d)`
   - Iterate `k ← 0.5 σ² T − Φ⁻¹(d) · σ √T` with `σ = √(w(k)/T)`.
   - Validate convergence: reject (return NaN) if (a) didn't converge in 50 iter, or (b) the final (k, σ) doesn't reproduce d within 10⁻⁴.
5. Emit one row per (ts, tenor, delta) with `iv`, `k`, smile params `(a, b, c)`, and `status ∈ {interp, extrap_short, extrap_long, exact}`.

Why **quadratic in log-moneyness** rather than SVI: simpler, fits in 5 lines of code, captures the dominant smile shape (level + skew + curvature) without the SVI calibration overhead. Upgrading to SVI is a one-day swap if needed for arbitrage tightness.

### Output

`data/grid/binance_eoh/{BTCUSDT,ETHUSDT}_grid.parquet`

| Symbol | Rows | Snapshots | Fully-filled (all 42) | Per-point IV cov |
|---|---|---|---|---|
| BTCUSDT | 146,538 | 3,489 | 2,821 (80.9%) | 99.8% |
| ETHUSDT | 146,454 | 3,487 | 3,213 (92.1%) | 99.8% |

**Total: ~6,000 fully-filled hourly surfaces** for VAE training. For reference, Lund (2024) trained on ~2,500 daily SPX surfaces.

---

## 6. Future data sources

### 6.1 OptionsDX BTC EOD (free tier) — Lund-comparable daily baseline

- Free for EOD frequency, June 2021 – Sept 2024, ~1,200 daily BTC surfaces with full chain + greeks + IV.
- Schema matches what Lund uses for SPX.
- **Plan:** download manually (browser, the free tier requires selecting EOD in their dropdown), drop in `data/external/optionsdx_btc_eod/`. Use as a **daily-cadence baseline** to make our results comparable to the SPX literature.

### 6.2 Tardis free monthly samples — sparse regime cross-check

- First day of every month from 2019 onwards, downloadable without an API key from Tardis.
- ~48 historical days spanning COVID, FTX collapse, 2024 ETF launch — regimes we want to stress-test against.
- **Plan:** opportunistic; pull if/when we need stress-test data the live cron won't reach for years.

### 6.3 Binance spot 1m klines (BTC + ETH) — Δspot conditioning variable

- Already partially pulled via the KEDevO HuggingFace dataset (BTC).
- ETH klines available free at `data.binance.vision/data/spot/daily/klines/ETHUSDT/1m/`.
- Needed only for the **spot-conditional VAE** extension (Lund's spot model). Pull when we add that.

---

## 7. Data layout

```
finance/vae-paper/
├── pyproject.toml
├── docs/
│   └── DATA.md                          # this file
├── src/
│   ├── fetch_binance_eoh.py             # primary training data downloader
│   ├── clean.py                         # parse, filter, PCP forward, IV inversion
│   └── build_grid.py                    # surface gridding
├── notebooks/
│   ├── 01_eda.py                        # source-of-truth for the notebook
│   └── 01_eda.ipynb                     # executed, plots embedded
└── data/                                # gitignored
    ├── external/
    │   └── binance_eoh/                 # raw downloaded parquets
    ├── clean/
    │   └── binance_eoh/                 # post-clean, quality-flagged
    └── grid/
        └── binance_eoh/                 # 42-point gridded surfaces
```

---

## 8. Reproducibility notes

- All scripts run via `uv run python <script>` and rely only on `pyproject.toml` deps. Lockfile committed.
- All filtering thresholds and grid choices are **declared as constants at the top of each script** — no hidden magic.
- Cleaning is **idempotent**: re-running on the same raw input produces byte-identical output.
- Gridding is **deterministic** given clean input (no randomness in the solver).
- No proprietary or licensed data is used. Everything is public-archive or our own captured live data.

---

## 9. Open questions / decisions deferred

| Question | When to revisit |
|---|---|
| Use Binance perp-index as forward instead of PCP? | If the OTM-tail mark_iv disagreement hurts downstream metrics |
| Upgrade smile parameterization (quadratic → SVI)? | If arbitrage violations on gridded output exceed ~5% |
| Add bid-ask spread as a VAE feature? | After first VAE baseline is trained |
| Spot-conditional VAE (Lund's spot model extension)? | After unconditional VAE is solid |
| Foundation-model multi-asset pretraining? | If single-asset VAE works and we want scope expansion |
