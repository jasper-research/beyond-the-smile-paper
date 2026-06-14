"""Calendar-arbitrage post-projection for surface predictors.

Given a reconstructed (n_tenors, n_deltas) implied-vol surface, project it
onto the calendar-arbitrage-free set defined by

    w(T_i, delta_j) = sigma^2(T_i, delta_j) * T_i  non-decreasing in T_i
    at each fixed delta_j.

This is the calendar condition in the delta parameterisation used by
Ackerer, Tagasovska and Vatter (2020); strict no-calendar-arbitrage holds
at fixed log-moneyness rather than fixed delta, but the delta-fixed
condition is the standard practitioner approximation for delta-gridded
surfaces and is the same condition our deep-smoother baseline penalises.

Per delta column we solve the L2 monotone-regression problem
    min_{w' in R^{n_tenors}}  ||w' - w_recon||_2^2  s.t.  w'_{i+1} >= w'_i
by Pool-Adjacent-Violators (scipy.optimize.isotonic_regression).

We evaluate three predictors -- the joint ConvVAE, the hybrid (smile re-fit
+ ConvVAE under the routing rule), and the smile re-fit alone -- on the
BTC and ETH test sets at five random-mask rates. For each we report
  * hidden-cell RMSE before projection,
  * hidden-cell RMSE after projection,
  * fraction of test surfaces that violated calendar monotonicity,
  * mean per-cell vol-point shift induced by the projection.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import isotonic_regression
from scipy.stats import norm

from .baseline_smile import _load_k_grid, predict_one
from .config import RUNS_ROOT, list_runs, load_config
from .data_btc import DELTAS, N_CELLS, N_DELTAS, N_TENORS, TENORS, load_split
from .eval_hybrid import hybrid_predict
from .eval_vae import _build_masks, _device, _rmse, vae_predict
from .model_vae import MaskedVAE  # noqa: F401
from .train_vae import build_model_from_ckpt

TENOR_YEARS = TENORS / 365.25
RATES = (0.10, 0.20, 0.30, 0.40, 0.50)
VIOL_TOL = 1e-8         # ignore numerical-noise-level "violations" of dw>=0
BUTTERFLY_TOL = 1e-10   # second divided differences of C(K) vs K, normalised units


def project_calendar(surface: np.ndarray) -> np.ndarray:
    """Project one (n_tenors, n_deltas) vol surface onto the
    calendar-arbitrage-free set at fixed delta. Returns the projected vol
    surface in the same units."""
    w = (surface ** 2) * TENOR_YEARS[:, None]
    w_iso = np.empty_like(w)
    for j in range(w.shape[1]):
        res = isotonic_regression(w[:, j], increasing=True)
        w_iso[:, j] = res.x
    # Guard against numerical zeros before the sqrt; w cannot be negative
    # after isotonic regression but can be exactly zero in degenerate inputs.
    w_iso = np.maximum(w_iso, 0.0)
    return np.sqrt(w_iso / TENOR_YEARS[:, None])


def project_calendar_batch(surfaces: np.ndarray) -> np.ndarray:
    """Project a (n_surfaces, n_tenors, n_deltas) stack."""
    out = np.empty_like(surfaces)
    for i in range(len(surfaces)):
        out[i] = project_calendar(surfaces[i])
    return out


def violation_fraction(surfaces: np.ndarray) -> float:
    """Fraction of surfaces with any calendar violation > tolerance."""
    w = (surfaces ** 2) * TENOR_YEARS[None, :, None]
    dw = np.diff(w, axis=1)
    has_v = (dw < -VIOL_TOL).reshape(len(surfaces), -1).any(axis=1)
    return float(has_v.mean())


def mean_projection_shift(before: np.ndarray, after: np.ndarray) -> float:
    """Mean absolute per-cell vol shift induced by projection, in vol points."""
    return float(np.abs(after - before).mean()) * 100


# -- Butterfly arbitrage at the listed strikes --------------------------------

def _row_strikes_prices(row_sigma: np.ndarray, T: float) -> tuple[np.ndarray, np.ndarray]:
    """For one (n_deltas,) tenor row, return (k_sorted, c_sorted) with k = ln(K/F).

    Each cell's strike is recovered from delta via d_1 = N^{-1}(delta_j), then
    k = sigma^2 T / 2 - sigma sqrt(T) d_1. The Black-76 forward call price
    divided by the forward, c = C/F = N(d_1) - exp(k) N(d_2), is constructed
    cell-by-cell and the two arrays are returned sorted by ascending K.
    Convexity of c in K = exp(k) is the discrete butterfly-arbitrage check
    at the listed strikes; no continuous smile fit is required.
    """
    sqrtT = np.sqrt(T)
    d1 = norm.ppf(DELTAS)                                     # (n_deltas,)
    k = (row_sigma ** 2) * T / 2.0 - row_sigma * sqrtT * d1
    d2 = d1 - row_sigma * sqrtT
    c = norm.cdf(d1) - np.exp(k) * norm.cdf(d2)
    order = np.argsort(k)
    return np.exp(k[order]), c[order]


def butterfly_violation_one(surface: np.ndarray) -> tuple[bool, int, float]:
    """For a (n_tenors, n_deltas) vol surface, return
    (any_violation, n_violated_triplets, max_violation_magnitude).

    A violated triplet is an interior strike at which the second divided
    difference of C(K) drops below the numerical tolerance.
    """
    any_v = False
    n_v = 0
    max_v = 0.0
    for i in range(surface.shape[0]):
        K, C = _row_strikes_prices(surface[i], TENOR_YEARS[i])
        # Second divided differences at interior strikes 1..n-2
        dd_right = (C[2:] - C[1:-1]) / (K[2:] - K[1:-1])
        dd_left  = (C[1:-1] - C[:-2]) / (K[1:-1] - K[:-2])
        dd2 = dd_right - dd_left          # >= 0 if butterfly-free
        viol = dd2 < -BUTTERFLY_TOL
        if viol.any():
            any_v = True
            n_v += int(viol.sum())
            max_v = max(max_v, float(-dd2.min()))
    return any_v, n_v, max_v


def butterfly_stats(surfaces: np.ndarray) -> dict[str, float]:
    """Aggregate butterfly-violation statistics over a (n, n_T, n_d) stack."""
    any_flags = np.zeros(len(surfaces), dtype=bool)
    n_triplets = np.zeros(len(surfaces), dtype=int)
    max_mags = np.zeros(len(surfaces))
    for i, s in enumerate(surfaces):
        any_flags[i], n_triplets[i], max_mags[i] = butterfly_violation_one(s)
    # 24 interior triplets per surface (4 per tenor * 6 tenors), used to
    # normalise the count of violations.
    n_interior = (N_DELTAS - 2) * N_TENORS
    return {
        "any_frac":         float(any_flags.mean()),
        "mean_triplets":    float(n_triplets.mean()),
        "mean_triplet_frac": float(n_triplets.mean() / n_interior),
        "mean_max_magnitude": float(max_mags.mean()),
    }


def evaluate(vae_run: Path, target_symbol: str, seed: int = 0) -> dict:
    cfg = load_config(vae_run)
    ckpt = torch.load(vae_run / "best.pt", map_location="cpu", weights_only=True)
    device = _device()
    model = build_model_from_ckpt(ckpt, cfg).to(device)
    model.load_state_dict(ckpt["model"])

    bundle = load_split("test", symbol=target_symbol)
    train_bundle = load_split("train", symbol=target_symbol)
    mean, std = train_bundle.mean.reshape(-1), train_bundle.std.reshape(-1)
    fallback = train_bundle.surfaces.mean(axis=0)
    truths = bundle.surfaces.astype(np.float32)            # (n, 6, 7) physical units
    truths_flat = truths.reshape(-1, N_CELLS)
    truths_z = ((truths_flat - mean) / std).astype(np.float32)
    k_all, ts_all = _load_k_grid(target_symbol)
    ts_to_k = {t: k_all[i] for i, t in enumerate(ts_all)}
    k_split = np.stack([ts_to_k[t] for t in bundle.ts], axis=0)
    n = len(truths)

    # Arbitrage statistics of the raw ground-truth surfaces (properties
    # of the data, independent of any model). Reported once.
    truth_cal = violation_fraction(truths)
    truth_bfly = butterfly_stats(truths)

    out: dict = {
        "vae_run": vae_run.name,
        "target_symbol": target_symbol,
        "ckpt_epoch": int(ckpt["epoch"]),
        "n_test": int(n),
        "truth_calendar_viol_frac": truth_cal,
        "truth_butterfly_stats": truth_bfly,
        "rates": list(RATES),
        "methods": {},
    }

    # Match eval_vae's mask-seed pattern: a single rng is advanced across
    # rates so the BEFORE numbers reproduce the paper's existing tables.
    # We pre-build per-rate masks once and reuse across the three methods.
    mask_rng = np.random.default_rng(seed)
    masks_by_rate = {rate: _build_masks(n, rate, mask_rng) for rate in RATES}

    for method in ("convvae", "hybrid", "smile_refit"):
        per_rate: dict[str, dict[str, float]] = {}
        for rate in RATES:
            masks_flat = masks_by_rate[rate]
            masks_grid = masks_flat.reshape(n, N_TENORS, N_DELTAS)
            hidden = masks_flat < 0.5

            # Build the raw prediction surface (n, 6, 7) for this method
            if method == "convvae":
                recon_z = vae_predict(model, truths_z, masks_flat, device)
                pred = (recon_z * std + mean).astype(np.float32).reshape(n, N_TENORS, N_DELTAS)
            elif method == "hybrid":
                recon_z = vae_predict(model, truths_z, masks_flat, device)
                vae_recon = (recon_z * std + mean).astype(np.float32)
                pred = hybrid_predict(truths, k_split, masks_flat, fallback, vae_recon
                                      ).astype(np.float32).reshape(n, N_TENORS, N_DELTAS)
            elif method == "smile_refit":
                pred = np.full_like(truths, np.nan)
                for i in range(n):
                    pred[i] = predict_one(truths[i], k_split[i],
                                          masks_grid[i], fallback)
                pred = pred.astype(np.float32)
            else:
                raise ValueError(method)

            # The predictors return observed cells equal to truth; only
            # hidden cells need a model output. NaNs from rank-deficient
            # smile re-fit at low observed counts are handled by mask.
            pred_filled = np.where(np.isnan(pred), truths, pred)
            pred_proj = project_calendar_batch(pred_filled)

            rmse_before = _rmse(pred_filled.reshape(-1, N_CELLS), truths_flat, hidden)
            rmse_after  = _rmse(pred_proj.reshape(-1, N_CELLS),  truths_flat, hidden)
            cal_before = violation_fraction(pred_filled)
            cal_after  = violation_fraction(pred_proj)
            bfly_before = butterfly_stats(pred_filled)
            bfly_after  = butterfly_stats(pred_proj)
            shift_vp    = mean_projection_shift(pred_filled, pred_proj)

            per_rate[f"{rate:.2f}"] = {
                "rmse_before_vol": rmse_before,
                "rmse_after_vol":  rmse_after,
                "delta_rmse_vol":  rmse_after - rmse_before,
                "calendar_viol_frac_before": cal_before,
                "calendar_viol_frac_after":  cal_after,
                "butterfly_before": bfly_before,
                "butterfly_after":  bfly_after,
                "mean_shift_vol_pts": shift_vp,
            }

            print(f"  {method:11s} r={rate:.2f}  "
                  f"rmse {rmse_before*100:6.3f}->{rmse_after*100:6.3f}vp  "
                  f"cal {cal_before*100:5.1f}%->{cal_after*100:5.1f}%  "
                  f"bfly {bfly_before['any_frac']*100:5.1f}%"
                  f"->{bfly_after['any_frac']*100:5.1f}%  "
                  f"shift {shift_vp:.3f}vp")

        out["methods"][method] = per_rate

    return out


def _best_joint_conv_run() -> Path:
    """The joint-trained ConvVAE used throughout the paper."""
    for rd in list_runs():
        cfg_path = rd / "config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        if (cfg.get("arch") == "conv2d"
                and tuple(cfg.get("symbols", [])) == ("BTCUSDT", "ETHUSDT")
                and cfg.get("hidden") == 64):
            return rd
    raise SystemExit("joint ConvVAE run not found")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("vae_run", nargs="?", help="run dir (default = joint ConvVAE)")
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    args = ap.parse_args()

    src = Path(args.vae_run) if args.vae_run else _best_joint_conv_run()
    if args.vae_run and not src.is_absolute():
        src = RUNS_ROOT / args.vae_run
    if not src.exists():
        raise SystemExit(f"run dir not found: {src}")
    print(f"source VAE: {src.name}")

    all_results = {}
    for sym in args.symbols:
        print(f"\n=== {sym} ===")
        all_results[sym] = evaluate(src, sym)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_json = RUNS_ROOT / f"_arb_projection_{stamp}.json"
    out_json.write_text(json.dumps({
        "stamp": stamp,
        "source_vae_run": src.name,
        "by_symbol": all_results,
    }, indent=2))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
