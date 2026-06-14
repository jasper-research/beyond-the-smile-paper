"""Aggregate metrics across all runs in runs/vae_btc/, write comparison plot + CSV.

Outputs are timestamped to avoid overwriting prior snapshots — research
artifacts (a comparison plot pinned to a specific set of runs) should be
preserved, not silently replaced.

  runs/vae_btc/_compare_<stamp>[_<label>].png
  runs/vae_btc/_compare_<stamp>[_<label>].csv

Pass --label to add a human-readable suffix. Use --overwrite if you really
want to replace an earlier file (rare; intended for fixing typos).
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from .config import RUNS_ROOT, list_runs, load_config


def _load_metrics(run_dir: Path) -> dict | None:
    p = run_dir / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="",
                    help="suffix appended to output filenames (e.g. step1b_hybrid)")
    ap.add_argument("--overwrite", default="",
                    help="exact basename to overwrite (use sparingly)")
    args = ap.parse_args()

    runs = list_runs()
    if not runs:
        print("no runs found")
        return
    rows: list[dict] = []
    baseline = None
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for rd in runs:
        m = _load_metrics(rd)
        if m is None:
            print(f"skip {rd.name}: no metrics.json")
            continue
        cfg = load_config(rd)
        rates = m["mask_rates"]
        vae = [m["vae_rmse_vol"][str(r)] for r in rates]
        ax.plot([r * 100 for r in rates], [v * 100 for v in vae], "o-",
                label=f"{cfg.name} (z={cfg.z_dim} h={cfg.hidden})")
        for r, v in zip(rates, vae):
            rows.append({"run": rd.name, "mask_rate": r,
                         "rmse_vol": v, "source": "vae"})
        # All runs share the same baseline; capture it once.
        if baseline is None:
            baseline = [m["baseline_rmse_vol"][str(r)] for r in rates]
            base_rates = rates
        plotted += 1
    if baseline is not None:
        ax.plot([r * 100 for r in base_rates], [b * 100 for b in baseline],
                "s--", color="black", label="smile refit (baseline)", linewidth=2)
        for r, b in zip(base_rates, baseline):
            rows.append({"run": "_baseline", "mask_rate": r,
                         "rmse_vol": b, "source": "baseline"})

    ax.set_xlabel("Mask rate (% cells hidden)")
    ax.set_ylabel("RMSE on hidden cells (vol pts)")
    ax.set_title(f"BTC completion: {plotted} VAE run(s) vs smile-refit baseline")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if args.overwrite:
        base = args.overwrite
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = f"_{args.label}" if args.label else ""
        base = f"_compare_{stamp}{suffix}"
    out_png = RUNS_ROOT / f"{base}.png"
    if out_png.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {out_png}; pass --label or --overwrite")
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {out_png}")

    out_csv = RUNS_ROOT / f"{base}.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "mask_rate", "rmse_vol", "source"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
