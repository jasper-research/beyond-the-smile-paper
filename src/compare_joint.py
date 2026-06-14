"""Aggregate step-3b results: 3 models (BTC-only / ETH-only / joint) × 2 test
symbols (BTC / ETH). Walks the relevant runs, reads metrics.json +
metrics_<symbol>.json, plots a 2-panel figure (one panel per test symbol).
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from .config import RUNS_ROOT, list_runs, load_config


def _metrics_for_target(rd: Path, target: str, primary: str) -> dict | None:
    if target == primary:
        p = rd / "metrics.json"
    else:
        p = rd / f"metrics_{target.lower()}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _resolve_run(cfg_name: str) -> Path | None:
    """Match by cfg.name exactly (after timestamp prefix). Most recent wins."""
    matches = []
    for rd in list_runs():
        try:
            if load_config(rd).name == cfg_name:
                matches.append(rd)
        except Exception:
            continue
    return matches[-1] if matches else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--btc-run", default="z16_h128_ep500",
                    help="substring of BTC-only run name")
    ap.add_argument("--eth-run", default="eth_z16_h128")
    ap.add_argument("--joint-run", default="joint_z16_h128")
    args = ap.parse_args()

    runs = {
        "BTC-only": _resolve_run(args.btc_run),
        "ETH-only": _resolve_run(args.eth_run),
        "Joint BTC+ETH": _resolve_run(args.joint_run),
    }
    for k, v in runs.items():
        if v is None:
            raise SystemExit(f"no run matching {k}")
        print(f"  {k:14s} → {v.name}")

    test_symbols = ("BTCUSDT", "ETHUSDT")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    rows: list[dict] = []
    for ax, target in zip(axes, test_symbols):
        for label, rd in runs.items():
            cfg = load_config(rd)
            primary = cfg.symbols[0]
            m = _metrics_for_target(rd, target, primary)
            if m is None:
                print(f"missing metrics for {label} on {target}")
                continue
            rates = m["mask_rates"]
            vae = [m["vae_rmse_vol"][str(r)] for r in rates]
            ax.plot([r * 100 for r in rates], [v * 100 for v in vae],
                    "o-", label=label, linewidth=2)
            for r, v in zip(rates, vae):
                rows.append({"model": label, "test_symbol": target,
                             "mask_rate": r, "rmse_vol": v})
            if label == "BTC-only":  # baseline doesn't depend on model
                base = [m["baseline_rmse_vol"][str(r)] for r in rates]
                ax.plot([r * 100 for r in rates], [v * 100 for v in base],
                        "s--", label="Smile refit (baseline)",
                        color="black", linewidth=2, alpha=0.7)
                for r, v in zip(rates, base):
                    rows.append({"model": "_baseline", "test_symbol": target,
                                 "mask_rate": r, "rmse_vol": v})
        ax.set_xlabel("Mask rate (% cells hidden)")
        ax.set_title(f"Test on {target}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("RMSE on hidden cells (vol pts)")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Step 3b: joint vs single-symbol training (z=16 h=128)")
    fig.tight_layout()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_png = RUNS_ROOT / f"_joint_{stamp}.png"
    out_csv = RUNS_ROOT / f"_joint_{stamp}.csv"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "test_symbol", "mask_rate", "rmse_vol"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_png}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
