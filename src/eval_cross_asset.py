"""Cross-asset OOD eval: take a source-trained VAE and run it on a target symbol.

Default: best BTC-trained VAE → ETH test set.

Two normalizations evaluated side-by-side:
  target-norm: z-score the target with target's own train stats — the realistic
               production setup (you have ETH data and want predictions on it).
  source-norm: z-score the target with the source's train stats — tests
               whether absolute IV level matters or only the shape does.

Outputs (timestamped, never overwrite):
  runs/vae_btc/_cross_<stamp>_<source>_to_<target>.json
  runs/vae_btc/_cross_<stamp>_<source>_to_<target>.png
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import RUNS_ROOT, list_runs, load_config
from .data_btc import DEFAULT_SYMBOL, N_CELLS, load_split
from .eval_vae import _build_masks, _device, _rmse, compute_baseline, vae_predict
from .model_vae import MaskedVAE  # noqa: F401
from .train_vae import build_model_from_ckpt


def _best_vae_run() -> Path:
    candidates = []
    for rd in list_runs():
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


def _eval_at_rates(model, surfaces, mean, std, mask_rates, seed, device
                   ) -> dict[float, float]:
    surfaces_flat = surfaces.reshape(-1, N_CELLS)
    surfaces_z = ((surfaces_flat - mean) / std).astype(np.float32)
    rng = np.random.default_rng(seed)
    out: dict[float, float] = {}
    for rate in mask_rates:
        masks_flat = _build_masks(len(surfaces), rate, rng)
        recon_z = vae_predict(model, surfaces_z, masks_flat, device)
        recon = recon_z * std + mean
        out[rate] = _rmse(recon, surfaces_flat, masks_flat < 0.5)
    return out


def evaluate(vae_run: Path, source_symbol: str = DEFAULT_SYMBOL,
             target_symbol: str = "ETHUSDT") -> Path:
    cfg = load_config(vae_run)
    ckpt = torch.load(vae_run / "best.pt", map_location="cpu", weights_only=True)
    device = _device()
    model = build_model_from_ckpt(ckpt, cfg).to(device)
    model.load_state_dict(ckpt["model"])

    target_test = load_split("test", symbol=target_symbol)
    target_train = load_split("train", symbol=target_symbol)
    source_train = load_split("train", symbol=source_symbol)
    mean_t = target_train.mean.reshape(-1)
    std_t = target_train.std.reshape(-1)
    mean_s = source_train.mean.reshape(-1)
    std_s = source_train.std.reshape(-1)

    rates = cfg.eval_mask_rates
    seed = cfg.eval_seed

    vae_target_norm = _eval_at_rates(model, target_test.surfaces, mean_t, std_t,
                                     rates, seed, device)
    vae_source_norm = _eval_at_rates(model, target_test.surfaces, mean_s, std_s,
                                     rates, seed, device)
    target_baseline = compute_baseline(rates, seed, symbol=target_symbol)

    # In-dist reference: the source VAE's own random-mask metrics.
    src_metrics_path = vae_run / "metrics.json"
    in_dist = None
    if src_metrics_path.exists():
        m = json.loads(src_metrics_path.read_text())
        in_dist = {float(r): m["vae_rmse_vol"][r] for r in m["vae_rmse_vol"]}

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    label = f"{source_symbol[:3]}_to_{target_symbol[:3]}"
    out_json = RUNS_ROOT / f"_cross_{stamp}_{label}.json"
    out_png = RUNS_ROOT / f"_cross_{stamp}_{label}.png"
    payload = {
        "source_vae_run": vae_run.name,
        "source_symbol": source_symbol,
        "target_symbol": target_symbol,
        "ckpt_epoch": int(ckpt["epoch"]),
        "ckpt_val_loss": float(ckpt["val_loss"]),
        "mask_rates": list(rates),
        "vae_target_norm_rmse_vol": {str(r): vae_target_norm[r] for r in rates},
        "vae_source_norm_rmse_vol": {str(r): vae_source_norm[r] for r in rates},
        "target_baseline_rmse_vol": {str(r): target_baseline[r] for r in rates},
        "source_in_dist_rmse_vol": (
            {str(r): in_dist[r] for r in rates} if in_dist else None),
    }
    out_json.write_text(json.dumps(payload, indent=2))

    fig, ax = plt.subplots(figsize=(8, 5))
    if in_dist:
        ax.plot([r * 100 for r in rates], [in_dist[r] * 100 for r in rates],
                "o-", label=f"VAE on {source_symbol} test (in-dist reference)",
                color="tab:blue", linewidth=2)
    ax.plot([r * 100 for r in rates], [vae_target_norm[r] * 100 for r in rates],
            "D-", label=f"VAE on {target_symbol} (target-norm — realistic)",
            color="tab:red", linewidth=2.5)
    ax.plot([r * 100 for r in rates], [vae_source_norm[r] * 100 for r in rates],
            "v--", label=f"VAE on {target_symbol} (source-norm)",
            color="tab:purple", linewidth=1.5, alpha=0.7)
    ax.plot([r * 100 for r in rates], [target_baseline[r] * 100 for r in rates],
            "s-", label=f"Smile refit on {target_symbol} (target oracle)",
            color="tab:orange", linewidth=2)
    ax.set_xlabel("Mask rate (% cells hidden)")
    ax.set_ylabel("RMSE on hidden cells (vol pts)")
    ax.set_title(f"Cross-asset OOD: VAE trained on {source_symbol}, evaluated on {target_symbol}\n"
                 f"source = {vae_run.name[16:]}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

    print(f"\n[{vae_run.name}]  {source_symbol} → {target_symbol}")
    print(f"  {'rate':>5s}  {'in-dist':>8s}  {'OOD-tgt':>8s}  {'OOD-src':>8s}  {'tgt base':>8s}")
    for r in rates:
        idv = f"{in_dist[r]:.4f}" if in_dist else "    —"
        print(f"  {r:>5.2f}  {idv:>8s}  {vae_target_norm[r]:>8.4f}  "
              f"{vae_source_norm[r]:>8.4f}  {target_baseline[r]:>8.4f}")
    print(f"\nwrote {out_json}")
    print(f"wrote {out_png}")
    return out_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("vae_run", nargs="?", help="source VAE run; default = best by val_loss")
    ap.add_argument("--source-symbol", default=DEFAULT_SYMBOL)
    ap.add_argument("--target-symbol", default="ETHUSDT")
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
    evaluate(src, args.source_symbol, args.target_symbol)


if __name__ == "__main__":
    main()
