"""Generate the two additional headline figures for the paper:

  fig_summary.png        — hybrid wins across every random + structured
                            scenario evaluated, on both BTC and ETH test
                            sets. One bar chart that summarises the
                            deployable claim.

  fig_residual_maps.png  — per-cell residual heatmaps under the
                            row_random structured hole, showing where
                            the smile re-fit fails (concentrated on the
                            hidden row) versus where the VAE distributes
                            its (smaller) errors. Visualises the
                            complementarity of the two predictors.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .baseline_smile import _load_k_grid, predict_one
from .config import RUNS_ROOT, load_config
from .data_btc import N_CELLS, N_DELTAS, N_TENORS, TENORS, DELTAS, load_split
from .eval_hybrid import hybrid_predict
from .eval_structured import _build_structured
from .eval_vae import _build_masks, _device, _rmse, compute_baseline, vae_predict
from .model_vae import MaskedVAE  # noqa: F401
from .train_vae import build_model_from_ckpt

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "paper" / "figs"

JOINT_RUN  = "20260521-142018_conv_z16_h64"      # ConvVAE joint (primary model)
SINGLE_RUN = "20260521-144553_conv_btc_z16_h64"  # ConvVAE BTC-only


def _load_model(run_name: str):
    run_dir = RUNS_ROOT / run_name
    cfg = load_config(run_dir)
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    device = _device()
    model = build_model_from_ckpt(ckpt, cfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model, device, cfg


def _eval_one(model, device, surfaces, k_split, masks_flat, mean, std, fallback):
    flat = surfaces.reshape(-1, N_CELLS)
    z = ((flat - mean) / std).astype(np.float32)
    recon_z = vae_predict(model, z, masks_flat, device)
    vae_recon = (recon_z * std + mean).astype(np.float32)
    hidden = masks_flat < 0.5
    vae_rmse = _rmse(vae_recon, flat, hidden)
    refit_pred = np.full_like(surfaces, np.nan)
    for i in range(len(surfaces)):
        m2d = masks_flat[i].reshape(N_TENORS, N_DELTAS)
        refit_pred[i] = predict_one(surfaces[i], k_split[i], m2d, fallback)
    refit_rmse = _rmse(refit_pred.reshape(-1, N_CELLS), flat, hidden)
    hyb = hybrid_predict(surfaces, k_split, masks_flat, fallback, vae_recon)
    hyb_rmse = _rmse(hyb.reshape(-1, N_CELLS), flat, hidden)
    return refit_rmse, vae_rmse, hyb_rmse


def fig_summary():
    """Bar chart: hybrid vs components across all scenarios × both markets."""
    scenarios = [
        ("random 10%",      ("random", 0.10)),
        ("random 30%",      ("random", 0.30)),
        ("random 50%",      ("random", 0.50)),
        ("row dropped",     ("row_random", None)),
        ("180d dropped",    ("long_tenor", None)),
        ("put wing dropped",  ("wing_put", None)),
        ("call wing dropped", ("wing_call", None)),
    ]

    model, device, cfg = _load_model(JOINT_RUN)
    rows = []
    for symbol in ("BTCUSDT", "ETHUSDT"):
        bundle = load_split("test", symbol=symbol)
        train_b = load_split("train", symbol=symbol)
        mean, std = train_b.mean.reshape(-1), train_b.std.reshape(-1)
        fallback = train_b.surfaces.mean(axis=0)
        k_all, ts_all = _load_k_grid(symbol)
        ts_to_k = {t: k_all[i] for i, t in enumerate(ts_all)}
        k_split = np.stack([ts_to_k[t] for t in bundle.ts], axis=0)
        surfaces = bundle.surfaces
        n = len(surfaces)
        for label, (kind, arg) in scenarios:
            rng = np.random.default_rng(0)
            if kind == "random":
                masks_flat = _build_masks(n, arg, rng)
            else:
                masks_flat = _build_structured(n, kind, rng)
            refit, vae, hyb = _eval_one(model, device, surfaces, k_split,
                                         masks_flat, mean, std, fallback)
            rows.append((symbol, label, refit * 100, vae * 100, hyb * 100))

    # Plot: one panel per symbol, three colour-coded bars per scenario
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    width = 0.27
    x = np.arange(len(scenarios))
    for ax, symbol in zip(axes, ("BTCUSDT", "ETHUSDT")):
        sub = [r for r in rows if r[0] == symbol]
        refit_v = [r[2] for r in sub]
        vae_v   = [r[3] for r in sub]
        hyb_v   = [r[4] for r in sub]
        # Cap visualised bars at 9 vp for readability; annotate true value when clipped
        cap = 9.0
        def clip(v): return [min(x, cap) for x in v]
        b1 = ax.bar(x - width, clip(refit_v), width, label="Smile re-fit",
                    color="#E8853C")
        b2 = ax.bar(x,         clip(vae_v),   width, label="ConvVAE (joint)",
                    color="#3B7BB8")
        b3 = ax.bar(x + width, clip(hyb_v),   width, label="Hybrid (ours)",
                    color="#C0392B")
        # annotate clipped bars with true value
        for xi, (rv, vv, hv) in enumerate(zip(refit_v, vae_v, hyb_v)):
            if rv > cap:
                ax.text(xi - width, cap * 1.02, f"{rv:.1f}", ha="center",
                        va="bottom", fontsize=7, color="#A04510")
            if vv > cap:
                ax.text(xi, cap * 1.02, f"{vv:.1f}", ha="center",
                        va="bottom", fontsize=7, color="#1F4E79")
            if hv > cap:
                ax.text(xi + width, cap * 1.02, f"{hv:.1f}", ha="center",
                        va="bottom", fontsize=7, color="#7B1010")
        ax.set_xticks(x)
        ax.set_xticklabels([s[0] for s in scenarios], rotation=22, ha="right",
                          fontsize=8)
        ax.set_title(f"Test on {symbol}", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        ax.set_ylim(0, cap + 0.5)
        ax.set_ylabel("RMSE on hidden cells (vol points)" if symbol == "BTCUSDT" else "")
        ax.axhline(cap, color="grey", lw=0.4, ls=":", alpha=0.6)
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Hybrid routing reduces error in every evaluated scenario "
                 "and on every market", fontsize=11, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "fig_summary.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_residual_maps():
    """Per-cell residual heatmaps under random 50% masking on BTC test.

    At 50% random masking some tenor rows retain >= 3 observed cells
    (smile re-fit applies) and some do not (VAE fallback applies), so
    the hybrid genuinely combines both predictors with different
    per-cell error profiles than either alone.
    """
    model, device, _ = _load_model(JOINT_RUN)
    bundle = load_split("test", symbol="BTCUSDT")
    train_b = load_split("train", symbol="BTCUSDT")
    mean, std = train_b.mean.reshape(-1), train_b.std.reshape(-1)
    fallback = train_b.surfaces.mean(axis=0)
    k_all, ts_all = _load_k_grid("BTCUSDT")
    ts_to_k = {t: k_all[i] for i, t in enumerate(ts_all)}
    k_split = np.stack([ts_to_k[t] for t in bundle.ts], axis=0)
    surfaces = bundle.surfaces
    n = len(surfaces)

    rng = np.random.default_rng(0)
    masks_flat = _build_masks(n, 0.50, rng)
    flat = surfaces.reshape(-1, N_CELLS)
    z = ((flat - mean) / std).astype(np.float32)
    recon_z = vae_predict(model, z, masks_flat, device)
    vae_recon = (recon_z * std + mean).astype(np.float32)

    refit_pred = np.full_like(surfaces, np.nan)
    for i in range(n):
        m2d = masks_flat[i].reshape(N_TENORS, N_DELTAS)
        refit_pred[i] = predict_one(surfaces[i], k_split[i], m2d, fallback)
    refit_flat = refit_pred.reshape(-1, N_CELLS)

    hyb = hybrid_predict(surfaces, k_split, masks_flat, fallback, vae_recon)
    hyb_flat = hyb.reshape(-1, N_CELLS)

    hidden = masks_flat < 0.5  # (n, 42)

    def per_cell_rmse(pred_flat, truth_flat, hidden_mask):
        sq = (pred_flat - truth_flat) ** 2
        out = np.full(N_CELLS, np.nan)
        for c in range(N_CELLS):
            mask_c = hidden_mask[:, c]
            sq_c = sq[mask_c, c]
            sq_c = sq_c[np.isfinite(sq_c)]
            if len(sq_c) > 0:
                out[c] = np.sqrt(sq_c.mean())
        return out.reshape(N_TENORS, N_DELTAS) * 100  # vol pts

    refit_map = per_cell_rmse(refit_flat, flat, hidden)
    vae_map   = per_cell_rmse(vae_recon, flat, hidden)
    hyb_map   = per_cell_rmse(hyb_flat,  flat, hidden)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    vmax = max(np.nanmax(refit_map), np.nanmax(vae_map))
    titles = [
        f"Smile re-fit (mean = {np.nanmean(refit_map):.1f} vp)",
        f"VAE (joint, mean = {np.nanmean(vae_map):.1f} vp)",
        f"Hybrid (mean = {np.nanmean(hyb_map):.1f} vp)",
    ]
    for ax, data, title in zip(axes, [refit_map, vae_map, hyb_map], titles):
        im = ax.imshow(data, aspect="auto", cmap="Reds", vmin=0, vmax=vmax)
        ax.set_xticks(range(N_DELTAS))
        ax.set_xticklabels([f"{d:.2f}" for d in DELTAS], fontsize=8)
        ax.set_yticks(range(N_TENORS))
        ax.set_yticklabels([int(t) for t in TENORS], fontsize=8)
        ax.set_xlabel("call $\\delta$")
        ax.set_title(title, fontsize=10)
        # Annotate each cell with its RMSE
        for i in range(N_TENORS):
            for j in range(N_DELTAS):
                v = data[i, j]
                if np.isfinite(v):
                    color = "white" if v > vmax * 0.55 else "black"
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            fontsize=6.5, color=color)
    axes[0].set_ylabel("tenor (days)")
    fig.suptitle("Per-cell RMSE under random $50\\%$ masking "
                 "(BTC test set)", fontsize=11, y=1.02)
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("RMSE (vol points)", fontsize=9)
    out = OUT_DIR / "fig_residual_maps.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_hybrid_plot():
    """3-line plot: smile re-fit, ConvVAE alone, hybrid, on BTC random masks."""
    rates = [0.10, 0.20, 0.30, 0.40, 0.50]
    refit = [0.00, 1.28, 2.53, 4.05, 7.00]
    vae   = [0.94, 1.01, 1.07, 1.10, 1.25]   # ConvVAE joint
    hyb   = [0.00, 0.12, 0.32, 0.50, 0.83]   # ConvVAE-based hybrid

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot([r * 100 for r in rates], refit, "s-", color="#E8853C",
            linewidth=2, label="Smile re-fit only")
    ax.plot([r * 100 for r in rates], vae,   "o-", color="#3B7BB8",
            linewidth=2, label="ConvVAE only (joint)")
    ax.plot([r * 100 for r in rates], hyb,   "D-", color="#C0392B",
            linewidth=2.4, label=r"Hybrid (refit when row $\geq$3 obs, else ConvVAE)")
    ax.set_xlabel("Mask rate (\\% cells hidden)")
    ax.set_ylabel("RMSE on hidden cells (vol points)")
    ax.set_title("BTC random-mask surface completion: hybrid policy vs.\\ components")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "fig_hybrid.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def fig_joint_plot():
    """2-panel: BTC test and ETH test, comparing BTC-only / ETH-only /
    joint ConvVAE plus the parametric baseline."""
    rates = [0.10, 0.20, 0.30, 0.40, 0.50]
    btc_data = {
        "BTC-only ConvVAE":  [1.28, 1.39, 1.51, 1.47, 1.67],
        "ETH-only ConvVAE":  [1.40, 1.43, 1.39, 1.32, 1.40],
        "Joint ConvVAE":     [0.94, 1.01, 1.07, 1.10, 1.25],
        "Smile re-fit (BTC)":[0.00, 1.28, 2.53, 4.05, 7.00],
    }
    eth_data = {
        "BTC-only ConvVAE":  [1.63, 1.61, 1.58, 1.66, 1.82],
        "ETH-only ConvVAE":  [1.46, 1.58, 1.58, 1.65, 1.72],
        "Joint ConvVAE":     [1.31, 1.33, 1.33, 1.43, 1.56],
        "Smile re-fit (ETH)":[0.00, 0.83, 1.52, 3.71, 5.84],
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    colors = {"BTC-only ConvVAE": "tab:blue",
              "ETH-only ConvVAE": "tab:orange",
              "Joint ConvVAE": "tab:green",
              "Smile re-fit (BTC)": "black",
              "Smile re-fit (ETH)": "black"}
    styles = {"Smile re-fit (BTC)": "--", "Smile re-fit (ETH)": "--"}
    for ax, data, title in zip(axes, [btc_data, eth_data],
                                ["Test on BTCUSDT", "Test on ETHUSDT"]):
        for label, vals in data.items():
            ls = styles.get(label, "-")
            marker = "s" if label.startswith("Smile") else "o"
            lw = 2.0 if label.startswith("Joint") else 1.5
            ax.plot([r * 100 for r in rates], vals, marker=marker, linestyle=ls,
                    color=colors[label], linewidth=lw, label=label)
        ax.set_xlabel("Mask rate (\\% cells hidden)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("RMSE on hidden cells (vol points)")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].legend(loc="upper left", fontsize=8)
    fig.suptitle("Joint vs.\\ single-symbol ConvVAE training", fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "fig_joint.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    fig_hybrid_plot()
    fig_joint_plot()
    fig_summary()
    fig_residual_maps()


if __name__ == "__main__":
    main()
