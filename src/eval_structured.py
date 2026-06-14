"""Evaluate existing models against ops-realistic structured-hole patterns.

Schemes (each hides a fixed pattern of cells per snapshot, not random):
  row_random   one random tenor row fully hidden  (7 cells = 16.7%)
  col_random   one random delta column fully hidden (6 cells = 14.3%)
  wing_put     OTM put wing dead: deltas {0.10, 0.20} all tenors (12 cells = 28.6%)
  wing_call    OTM call wing dead: deltas {0.80, 0.90} all tenors (12 cells = 28.6%)
  long_tenor   longest tenor (180d) row dead (7 cells = 16.7%)

For each scheme, compute RMSE for:
  * VAE-only       (current best VAE checkpoint)
  * Hybrid policy  (refit if row has ≥3 obs, else VAE)
  * Smile-refit    (the baseline)
  * Random control: random-mask result at the matching hidden-cell count

The "random control" is the apples-to-apples comparison: same number of cells
hidden, but iid random vs structured. Gap between structured and random
quantifies how much the *pattern* hurts (vs just losing that many cells).

Outputs (timestamped, no overwrites):
  runs/vae_btc/_structured_<stamp>.png
  runs/vae_btc/_structured_<stamp>.csv
  runs/vae_btc/_structured_<stamp>.json
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .baseline_smile import _load_k_grid, predict_one
from .config import RUNS_ROOT, list_runs, load_config
from .data_btc import N_CELLS, N_DELTAS, N_TENORS, load_split
from .eval_hybrid import MIN_OBS_FOR_REFIT, hybrid_predict
from .eval_vae import _build_masks, _device, _rmse, vae_predict
from .model_vae import MaskedVAE  # noqa: F401
from .train_vae import build_model_from_ckpt

SCHEMES = ("row_random", "col_random", "wing_put", "wing_call", "long_tenor")


def _build_structured(n: int, scheme: str, rng: np.random.Generator) -> np.ndarray:
    """Return (n, 42) mask array, 1=observed, 0=hidden, per the scheme."""
    masks = np.ones((n, N_TENORS, N_DELTAS), dtype=np.float32)
    if scheme == "row_random":
        rows = rng.integers(0, N_TENORS, size=n)
        for i, r in enumerate(rows):
            masks[i, r, :] = 0.0
    elif scheme == "col_random":
        cols = rng.integers(0, N_DELTAS, size=n)
        for i, c in enumerate(cols):
            masks[i, :, c] = 0.0
    elif scheme == "wing_put":
        masks[:, :, 0:2] = 0.0
    elif scheme == "wing_call":
        masks[:, :, 5:7] = 0.0
    elif scheme == "long_tenor":
        masks[:, -1, :] = 0.0  # 180d row
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    return masks.reshape(n, N_CELLS)


def _best_vae_run() -> Path:
    candidates = []
    for rd in list_runs():
        # Skip derived hybrid runs (no real best.pt of their own; their config
        # mirrors a VAE source but the dir intentionally has no checkpoint).
        if "hybrid" in rd.name:
            continue
        ckpt = rd / "best.pt"
        if not ckpt.exists():
            continue
        blob = torch.load(ckpt, map_location="cpu", weights_only=True)
        candidates.append((float(blob["val_loss"]), rd))
    if not candidates:
        raise SystemExit("no trained VAE runs found")
    candidates.sort()
    return candidates[0][1]


def evaluate(vae_run: Path, seed: int = 0) -> Path:
    cfg = load_config(vae_run)
    ckpt = torch.load(vae_run / "best.pt", map_location="cpu", weights_only=True)
    device = _device()
    model = build_model_from_ckpt(ckpt, cfg).to(device)
    model.load_state_dict(ckpt["model"])

    bundle = load_split("test")
    train_bundle = load_split("train")
    mean, std = train_bundle.mean.reshape(-1), train_bundle.std.reshape(-1)
    fallback = train_bundle.surfaces.mean(axis=0)
    surfaces = bundle.surfaces
    surfaces_flat = surfaces.reshape(-1, N_CELLS)
    surfaces_z = ((surfaces_flat - mean) / std).astype(np.float32)
    k_all, ts_all = _load_k_grid()
    ts_to_k = {t: k_all[i] for i, t in enumerate(ts_all)}
    k_split = np.stack([ts_to_k[t] for t in bundle.ts], axis=0)
    n = len(surfaces)

    results: dict[str, dict[str, float]] = {}

    for scheme in SCHEMES:
        rng = np.random.default_rng(seed)
        masks_flat = _build_structured(n, scheme, rng)
        n_hidden = int((masks_flat < 0.5).sum() / n)  # avg cells hidden per snap

        # VAE
        recon_z = vae_predict(model, surfaces_z, masks_flat, device)
        vae_recon = (recon_z * std + mean).astype(np.float32)
        hidden = masks_flat < 0.5
        vae_only = _rmse(vae_recon, surfaces_flat, hidden)

        # Smile refit
        refit_pred = np.full_like(surfaces, np.nan)
        for i in range(n):
            m2d = masks_flat[i].reshape(N_TENORS, N_DELTAS)
            refit_pred[i] = predict_one(surfaces[i], k_split[i], m2d, fallback)
        refit_rmse = _rmse(refit_pred.reshape(-1, N_CELLS), surfaces_flat, hidden)

        # Hybrid
        hybrid_pred = hybrid_predict(surfaces, k_split, masks_flat, fallback, vae_recon)
        hybrid_rmse = _rmse(hybrid_pred.reshape(-1, N_CELLS), surfaces_flat, hidden)

        # Random control at matching cell count
        rng_ctrl = np.random.default_rng(seed + 100)
        rate_match = n_hidden / N_CELLS
        masks_ctrl = _build_masks(n, rate_match, rng_ctrl)
        recon_z_c = vae_predict(model, surfaces_z, masks_ctrl, device)
        vae_recon_c = (recon_z_c * std + mean).astype(np.float32)
        hidden_c = masks_ctrl < 0.5
        vae_random = _rmse(vae_recon_c, surfaces_flat, hidden_c)
        refit_pred_c = np.full_like(surfaces, np.nan)
        for i in range(n):
            m2d = masks_ctrl[i].reshape(N_TENORS, N_DELTAS)
            refit_pred_c[i] = predict_one(surfaces[i], k_split[i], m2d, fallback)
        refit_random = _rmse(refit_pred_c.reshape(-1, N_CELLS), surfaces_flat, hidden_c)
        hybrid_pred_c = hybrid_predict(surfaces, k_split, masks_ctrl, fallback, vae_recon_c)
        hybrid_random = _rmse(hybrid_pred_c.reshape(-1, N_CELLS), surfaces_flat, hidden_c)

        results[scheme] = {
            "n_hidden_per_snap": float(n_hidden),
            "rate_match": float(rate_match),
            "vae_only": vae_only,
            "smile_refit": refit_rmse,
            "hybrid": hybrid_rmse,
            "vae_random_control": vae_random,
            "refit_random_control": refit_random,
            "hybrid_random_control": hybrid_random,
        }
        print(f"\n  scheme={scheme:11s}  hidden/snap={n_hidden:>2d}  "
              f"(equiv random rate {rate_match:.2f})")
        print(f"    {'method':<12s}  {'structured':>10s}  {'random ctrl':>11s}")
        print(f"    {'vae':<12s}  {vae_only:>10.4f}  {vae_random:>11.4f}")
        print(f"    {'smile_refit':<12s}  {refit_rmse:>10.4f}  {refit_random:>11.4f}")
        print(f"    {'hybrid':<12s}  {hybrid_rmse:>10.4f}  {hybrid_random:>11.4f}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_json = RUNS_ROOT / f"_structured_{stamp}.json"
    out_json.write_text(json.dumps({
        "source_vae_run": vae_run.name,
        "ckpt_epoch": int(ckpt["epoch"]),
        "ckpt_val_loss": float(ckpt["val_loss"]),
        "seed": seed,
        "results": results,
    }, indent=2))

    out_csv = RUNS_ROOT / f"_structured_{stamp}.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "method", "rmse_vol", "structured", "n_hidden_per_snap"])
        for scheme, r in results.items():
            for method, key in [("vae", "vae_only"), ("smile_refit", "smile_refit"),
                                 ("hybrid", "hybrid")]:
                w.writerow([scheme, method, r[key], 1, int(r["n_hidden_per_snap"])])
                w.writerow([scheme, method, r[key.replace("only", "random_control")
                                            .replace("smile_refit", "refit_random_control")
                                            .replace("hybrid", "hybrid_random_control")],
                            0, int(r["n_hidden_per_snap"])])

    _plot(results, vae_run, stamp)
    print(f"\nwrote {out_json}")
    print(f"wrote {out_csv}")
    return out_json


def _plot(results: dict, vae_run: Path, stamp: str) -> None:
    schemes = list(results)
    methods = [("vae", "vae_only", "vae_random_control", "tab:blue"),
               ("smile_refit", "smile_refit", "refit_random_control", "tab:orange"),
               ("hybrid", "hybrid", "hybrid_random_control", "tab:red")]
    x = np.arange(len(schemes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (label, struct_key, rand_key, color) in enumerate(methods):
        struct_vals = [results[s][struct_key] * 100 for s in schemes]
        rand_vals = [results[s][rand_key] * 100 for s in schemes]
        ax.bar(x + (i - 1) * width, struct_vals, width, color=color, label=f"{label} (structured)")
        ax.bar(x + (i - 1) * width, rand_vals, width, color=color, alpha=0.35,
               edgecolor=color, hatch="//", label=f"{label} (random ctrl, same #cells)" if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(schemes, rotation=15)
    ax.set_ylabel("RMSE on hidden cells (vol pts)")
    ax.set_title(f"Structured-hole eval — VAE source {vae_run.name[16:]}\n"
                 f"solid = structured pattern, hatched = random mask at same cell count")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out = RUNS_ROOT / f"_structured_{stamp}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("vae_run", nargs="?", help="VAE run dir; default = best by val_loss")
    args = ap.parse_args()
    if args.vae_run:
        src = Path(args.vae_run)
        if not src.is_absolute():
            src = RUNS_ROOT / args.vae_run
        if not src.exists():
            raise SystemExit(f"run dir not found: {src}")
    else:
        src = _best_vae_run()
        print(f"auto-selected best VAE: {src.name}")
    evaluate(src)


if __name__ == "__main__":
    main()
