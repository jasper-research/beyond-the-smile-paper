"""Functional-PCA baseline for surface completion.

Classical statistical approach in the spirit of Cont & da Fonseca (2002):
decompose the training surfaces into a mean plus a small number of
principal components, then complete a partially-observed surface by
solving for the latent code that best matches the observed cells.

Pipeline (z-normalisation matches the neural models):
  1. Take train surfaces, z-normalise per cell using training stats.
  2. SVD-PCA on the (n_train, 42) matrix of centred training surfaces.
  3. For each test (surface, mask):
       z = (B[obs]^T B[obs])^{-1} B[obs]^T (x[obs] - mu[obs])
       x_hat = mu + B z
  4. Evaluate hidden-cell RMSE in vol points.

A small ridge regulariser stabilises the regression when the number of
observed cells is comparable to k.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data_btc import N_CELLS, load_split
from .eval_structured import _build_structured
from .eval_vae import _build_masks

ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = ROOT / "runs" / "vae_btc"

RANDOM_RATES = (0.10, 0.20, 0.30, 0.40, 0.50)
STRUCT_SCHEMES = ("row_random", "col_random", "wing_put", "wing_call", "long_tenor")
DEFAULT_KS = (2, 4, 8, 16)


def fit_pca(symbols: tuple[str, ...]):
    """Fit PCA on per-symbol z-normalised concatenation of training surfaces.

    Returns (mu, B, std, mean_unnorm) where mu and B operate in z-space,
    and std/mean_unnorm are the per-cell statistics needed to map back.
    """
    Xs_z, mean_used, std_used = [], None, None
    for sym in symbols:
        train = load_split("train", symbol=sym)
        flat = train.surfaces.reshape(-1, N_CELLS).astype(np.float32)
        mean = train.mean.reshape(N_CELLS)
        std = train.std.reshape(N_CELLS)
        Xs_z.append((flat - mean) / std)
        # For inverse-normalisation we use the *first* symbol's stats by
        # convention; per-cell stats differ across symbols but we operate
        # in z-space, so the only thing that matters is consistency within
        # each evaluation.
        if mean_used is None:
            mean_used, std_used = mean, std
    X_z = np.concatenate(Xs_z, axis=0)
    mu_z = X_z.mean(axis=0)
    X_centred = X_z - mu_z
    U, S, Vt = np.linalg.svd(X_centred, full_matrices=False)
    B = Vt.T          # (42, k_full)
    return mu_z, B, std_used, mean_used


def reconstruct(x_z: np.ndarray, mask: np.ndarray, mu_z: np.ndarray,
                B_k: np.ndarray, ridge: float = 1e-4) -> np.ndarray:
    """Reconstruct one z-normalised surface from its observed cells.

    x_z   : (42,) z-normalised surface (observed cells only used)
    mask  : (42,) 1 = observed
    B_k   : (42, k) top-k principal components
    """
    obs = mask > 0.5
    B_obs = B_k[obs]                              # (n_obs, k)
    target = x_z[obs] - mu_z[obs]                 # (n_obs,)
    # Ridge least squares: (B^T B + lambda I)^{-1} B^T target
    A = B_obs.T @ B_obs + ridge * np.eye(B_obs.shape[1], dtype=np.float32)
    z = np.linalg.solve(A, B_obs.T @ target)      # (k,)
    return mu_z + B_k @ z                          # (42,)


def _rmse_vol(pred_z: np.ndarray, truth_z: np.ndarray, hidden: np.ndarray,
              mean: np.ndarray, std: np.ndarray) -> float:
    pred  = pred_z  * std + mean
    truth = truth_z * std + mean
    err   = (pred - truth)[hidden]
    err   = err[np.isfinite(err)]
    if len(err) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(err ** 2))) * 100  # vol points


def evaluate(symbol: str, ks: tuple[int, ...] = DEFAULT_KS) -> dict:
    mu_z, B_full, std, mean = fit_pca(symbols=(symbol,))

    bundle = load_split("test", symbol=symbol)
    test_flat = bundle.surfaces.reshape(-1, N_CELLS).astype(np.float32)
    test_z = (test_flat - mean) / std
    n = len(test_z)

    out = {"symbol": symbol, "n_test": n, "ks": list(ks),
           "random": {}, "structured": {}}

    for k in ks:
        B_k = B_full[:, :k]
        # Random masks
        rmses = {}
        rng = np.random.default_rng(0)
        for rate in RANDOM_RATES:
            masks = _build_masks(n, rate, rng)
            preds = np.zeros_like(test_z)
            for i in range(n):
                preds[i] = reconstruct(test_z[i], masks[i], mu_z, B_k)
            hidden = masks < 0.5
            rmses[f"{rate:.2f}"] = _rmse_vol(preds, test_z, hidden, mean, std)
        out["random"][f"k={k}"] = rmses
        # Structured masks
        srmses = {}
        rng = np.random.default_rng(0)
        for scheme in STRUCT_SCHEMES:
            masks = _build_structured(n, scheme, rng)
            preds = np.zeros_like(test_z)
            for i in range(n):
                preds[i] = reconstruct(test_z[i], masks[i], mu_z, B_k)
            hidden = masks < 0.5
            srmses[scheme] = _rmse_vol(preds, test_z, hidden, mean, std)
        out["structured"][f"k={k}"] = srmses
        # Variance explained at this k
        # (computed once on training set, fine to report)
        # not strictly needed; skip

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    args = ap.parse_args()

    res = evaluate(args.symbol)

    # Print a compact table
    print(f"\n=== PCA baseline on {args.symbol} ===")
    print("\nRandom-mask completion RMSE (vol points):")
    print(f"  {'k':>4s}  " + "  ".join(f"{float(r):>5.2f}" for r in RANDOM_RATES))
    for k in res["ks"]:
        row = res["random"][f"k={k}"]
        print(f"  {k:>4d}  " + "  ".join(f"{row[f'{r:.2f}']:>5.2f}" for r in RANDOM_RATES))

    print("\nStructured-hole RMSE (vol points):")
    print(f"  {'k':>4s}  " + "  ".join(f"{s:>11s}" for s in STRUCT_SCHEMES))
    for k in res["ks"]:
        row = res["structured"][f"k={k}"]
        print(f"  {k:>4d}  " + "  ".join(f"{row[s]:>11.2f}" for s in STRUCT_SCHEMES))

    out_json = RUNS_ROOT / f"_pca_{args.symbol.lower()}.json"
    out_json.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
