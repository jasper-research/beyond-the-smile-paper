"""Anomaly forensics for the BTC Sep–Oct 2023 reconstruction-error spikes.

Pure analysis. Uses the best trained VAE (joint by default — it's the most
accurate so its residuals are the cleanest "what doesn't fit the manifold"
signal). No new training.

Outputs (timestamped, never overwrite):
  runs/vae_btc/_anomaly_<stamp>_timeline.png    timeseries of recon error +
                                                  surface features + BTC spot
  runs/vae_btc/_anomaly_<stamp>_topN.png        top-N anomalous surfaces
                                                  (actual / reconstructed / residual)
  runs/vae_btc/_anomaly_<stamp>_latent.png      latent-space PCA colored by date
  runs/vae_btc/_anomaly_<stamp>.json            top-N timestamps + recon errors
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .config import RUNS_ROOT, list_runs, load_config
from .data_btc import N_CELLS, N_DELTAS, N_TENORS, TENORS, DELTAS, load_split
from .eval_vae import _device, vae_predict
from .model_vae import MaskedVAE  # noqa: F401
from .train_vae import build_model_from_ckpt

ROOT = Path(__file__).resolve().parent.parent
SPOT_PATH = ROOT / "data" / "processed" / "btc_spot_hourly.parquet"

# Known crypto events in the window 2023-05 → 2023-10. (Date, label, color).
EVENTS = [
    ("2023-08-17", "BTC flash crash $29k→$25.5k", "tab:red"),
    ("2023-08-29", "Grayscale wins SEC suit", "tab:green"),
    ("2023-09-13", "Mt Gox repayment news", "tab:purple"),
    ("2023-10-16", "Fake BlackRock ETF rumor", "tab:orange"),
    ("2023-10-23", "Real ETF rally begins", "tab:blue"),
]


def _resolve_run(cfg_name: str) -> Path | None:
    matches = []
    for rd in list_runs():
        try:
            if load_config(rd).name == cfg_name:
                matches.append(rd)
        except Exception:
            continue
    return matches[-1] if matches else None


def _load_model(vae_run: Path):
    cfg = load_config(vae_run)
    ckpt = torch.load(vae_run / "best.pt", map_location="cpu", weights_only=True)
    device = _device()
    model = build_model_from_ckpt(ckpt, cfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model, device, cfg


def _gather_surfaces(symbol: str = "BTCUSDT"):
    """Concatenate train+val+test into a single ordered (ts, surface) sequence."""
    bundles = [load_split(s, symbol=symbol) for s in ("train", "val", "test")]
    surfaces = np.concatenate([b.surfaces for b in bundles], axis=0)
    ts = np.concatenate([b.ts for b in bundles])
    # train bundle carries the train-set normalization (used at inference).
    mean = bundles[0].mean
    std = bundles[0].std
    splits = np.repeat(["train", "val", "test"],
                       [len(b.surfaces) for b in bundles])
    return surfaces, ts, mean, std, splits


def _full_recon(model, device, surfaces, mean, std):
    """Run VAE on every snapshot with mask=all-observed; return recon + per-snap RMSE."""
    flat = surfaces.reshape(-1, N_CELLS)
    z_mean = mean.reshape(-1)
    z_std = std.reshape(-1)
    surfaces_z = ((flat - z_mean) / z_std).astype(np.float32)
    masks = np.ones_like(surfaces_z, dtype=np.float32)
    recon_z = vae_predict(model, surfaces_z, masks, device)
    recon = (recon_z * z_std + z_mean).astype(np.float32)
    err = np.sqrt(((recon.reshape(-1, N_CELLS) - flat) ** 2).mean(axis=1))
    return recon.reshape(surfaces.shape), err


def _encode(model, device, surfaces, mean, std):
    """Encode all surfaces (no masking) and return latent means."""
    flat = surfaces.reshape(-1, N_CELLS)
    z_mean = mean.reshape(-1)
    z_std = std.reshape(-1)
    surfaces_z = ((flat - z_mean) / z_std).astype(np.float32)
    masks = np.ones_like(surfaces_z, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(surfaces_z).to(device)
        m = torch.from_numpy(masks).to(device)
        mu, _ = model.encode(x, m)
    return mu.cpu().numpy()


def _surface_features(surfaces):
    """Per-snapshot diagnostic features in vol units."""
    # Tenor indices: 14, 30, 60, 90, 120, 180. Use 60d (idx 2) for ATM/skew, 30/180 for term slope.
    atm = surfaces[:, 2, 3]                                # 60d, delta=0.50
    skew_25d = surfaces[:, 2, 1] - surfaces[:, 2, 5]       # 60d, 0.20Δ put-call IV diff
    term_slope = surfaces[:, 5, 3] - surfaces[:, 0, 3]     # 180d − 14d at delta=0.50
    return atm, skew_25d, term_slope


def timeline_plot(ts, err, atm, skew, slope, spot_df, splits, out_png: Path):
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    split_colors = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green"}
    for s in ("train", "val", "test"):
        m = splits == s
        axes[0].plot(ts[m], err[m] * 100, color=split_colors[s], lw=0.8, alpha=0.85, label=s)
    axes[0].set_ylabel("Recon RMSE (vol pts)")
    axes[0].set_title("Per-snapshot VAE reconstruction error (no masking)")
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].plot(ts, atm * 100, color="tab:purple", lw=0.8, label="ATM IV (60d, Δ=0.50)")
    axes[1].plot(ts, np.abs(skew) * 100, color="tab:brown", lw=0.8, alpha=0.7,
                 label="|25Δ skew| (60d)")
    axes[1].set_ylabel("Vol pts")
    axes[1].set_title("Surface shape features")
    axes[1].legend(loc="upper left", fontsize=8)

    axes[2].plot(ts, slope * 100, color="tab:olive", lw=0.8, label="Term slope (180d − 14d ATM)")
    axes[2].axhline(0, color="black", lw=0.4, alpha=0.5)
    axes[2].set_ylabel("Vol pts")
    axes[2].set_title("Term structure")
    axes[2].legend(loc="upper left", fontsize=8)

    if spot_df is not None:
        axes[3].plot(spot_df["ts"], spot_df["spot"], color="black", lw=0.8)
        axes[3].set_ylabel("BTC spot (USDT)")
        axes[3].set_title("BTC spot (proxied by shortest-tenor forward)")
        axes[3].yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter("$%d"))

    for ax in axes:
        ax.grid(alpha=0.3)
        for date_str, label, color in EVENTS:
            d = pd.Timestamp(date_str)
            ax.axvline(d, color=color, linestyle=":", alpha=0.6, lw=1)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Event legend in the spot panel for context.
    for date_str, label, color in EVENTS:
        axes[3].axvline(pd.Timestamp(date_str), color=color, linestyle=":",
                        alpha=0.8, lw=1.2, label=f"{date_str}  {label}")
    axes[3].legend(loc="upper left", fontsize=7, ncols=2)

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def topN_plot(ts, surfaces, recon, err, n: int, out_png: Path) -> list[dict]:
    ix = np.argsort(err)[::-1][:n]  # highest errors first
    fig, axes = plt.subplots(n, 3, figsize=(11, 2.7 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    # Shared vmin/vmax per-row for actual+recon; residual gets its own scale.
    rows: list[dict] = []
    for r, i in enumerate(ix):
        actual = surfaces[i] * 100
        rec = recon[i] * 100
        resid = (recon[i] - surfaces[i]) * 100
        vmin = min(actual.min(), rec.min())
        vmax = max(actual.max(), rec.max())
        for col, (data, title, cmap, vrange) in enumerate([
            (actual, "actual",       "viridis", (vmin, vmax)),
            (rec,    "reconstructed", "viridis", (vmin, vmax)),
            (resid,  "residual (recon-actual)", "RdBu_r",
                     (-abs(resid).max(), abs(resid).max()))]):
            im = axes[r, col].imshow(data, aspect="auto", cmap=cmap,
                                      vmin=vrange[0], vmax=vrange[1])
            axes[r, col].set_xticks(range(N_DELTAS))
            axes[r, col].set_xticklabels([f"{d:.2f}" for d in DELTAS], fontsize=7)
            axes[r, col].set_yticks(range(N_TENORS))
            axes[r, col].set_yticklabels([int(t) for t in TENORS], fontsize=7)
            if col == 0:
                axes[r, col].set_ylabel("tenor (d)", fontsize=8)
            axes[r, col].set_xlabel("call Δ", fontsize=8)
            ts_label = pd.Timestamp(ts[i]).strftime("%Y-%m-%d %H:%M")
            err_label = f"err={err[i]*100:.2f}vp"
            axes[r, col].set_title(f"{ts_label}  {title}  {err_label if col == 0 else ''}",
                                   fontsize=8)
            plt.colorbar(im, ax=axes[r, col], fraction=0.04)
        rows.append({"ts": str(pd.Timestamp(ts[i])),
                     "recon_rmse_vol": float(err[i])})
    fig.suptitle(f"Top {n} anomalous BTC surfaces — actual vs reconstructed vs residual",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return rows


def latent_plot(z, ts, err, out_png: Path) -> None:
    """2D scatter of latent means (first 2 PCs) colored by time, sized by recon error."""
    # PCA on z
    Zc = z - z.mean(axis=0)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    pcs = Zc @ Vt.T[:, :2]
    expl = (S ** 2 / (S ** 2).sum())[:2]

    # Map timestamp → numeric for color
    ts_pd = pd.to_datetime(ts)
    t_num = (ts_pd - ts_pd.min()).total_seconds().to_numpy()

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    # base scatter colored by time
    sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=t_num, cmap="viridis", s=10, alpha=0.6,
                    edgecolors="none")
    # overlay top 30 anomalies with size
    top = np.argsort(err)[::-1][:30]
    ax.scatter(pcs[top, 0], pcs[top, 1], s=80 + 600 * (err[top] / err.max()),
               facecolors="none", edgecolors="red", linewidths=1.2, label="top-30 anomalies")
    cbar = plt.colorbar(sc, ax=ax, label="time →")
    # Map a few cbar ticks back to dates
    ticks = np.linspace(t_num.min(), t_num.max(), 5)
    tick_labels = [pd.Timestamp(ts_pd.min() + pd.Timedelta(seconds=t)).strftime("%b %d")
                   for t in ticks]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)
    ax.set_xlabel(f"PC1 ({expl[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({expl[1]*100:.1f}% var)")
    ax.set_title("Latent-space PCA of BTC surfaces, colored by date\n"
                 "(red outlines = top 30 reconstruction anomalies)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae-run", default="conv_z16_h64",
                    help="cfg.name of run to analyze; default = joint (best)")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--top-n", type=int, default=5)
    args = ap.parse_args()

    vae_run = _resolve_run(args.vae_run)
    if vae_run is None:
        raise SystemExit(f"no run named {args.vae_run!r}")
    print(f"using {vae_run.name}")

    model, device, _ = _load_model(vae_run)
    surfaces, ts, mean, std, splits = _gather_surfaces(args.symbol)
    recon, err = _full_recon(model, device, surfaces, mean, std)
    atm, skew, slope = _surface_features(surfaces)
    spot_df = pd.read_parquet(SPOT_PATH) if SPOT_PATH.exists() else None

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_timeline = RUNS_ROOT / f"_anomaly_{stamp}_timeline.png"
    out_topn = RUNS_ROOT / f"_anomaly_{stamp}_top{args.top_n}.png"
    out_latent = RUNS_ROOT / f"_anomaly_{stamp}_latent.png"
    out_json = RUNS_ROOT / f"_anomaly_{stamp}.json"

    timeline_plot(ts, err, atm, skew, slope, spot_df, splits, out_timeline)
    print(f"wrote {out_timeline}")
    top_rows = topN_plot(ts, surfaces, recon, err, args.top_n, out_topn)
    print(f"wrote {out_topn}")
    z = _encode(model, device, surfaces, mean, std)
    latent_plot(z, ts, err, out_latent)
    print(f"wrote {out_latent}")

    out_json.write_text(json.dumps({
        "vae_run": vae_run.name,
        "symbol": args.symbol,
        "n_snapshots": len(ts),
        "err_mean": float(err.mean()),
        "err_p99": float(np.quantile(err, 0.99)),
        "err_max": float(err.max()),
        "top_anomalies": top_rows,
    }, indent=2))
    print(f"wrote {out_json}")

    print("\n=== summary ===")
    print(f"  mean recon RMSE: {err.mean()*100:.3f} vol pts")
    print(f"  p99 recon RMSE:  {np.quantile(err, 0.99)*100:.3f} vol pts")
    print(f"  max recon RMSE:  {err.max()*100:.3f} vol pts")
    print(f"  top {args.top_n} anomalies:")
    for row in top_rows:
        print(f"    {row['ts']}  err={row['recon_rmse_vol']*100:.2f} vp")


if __name__ == "__main__":
    main()
