"""Architecture-ablation comparison figure: MLP / Conv / Attention across
the seven evaluation scenarios used in the paper.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .baseline_smile import _load_k_grid, predict_one
from .config import RUNS_ROOT, load_config
from .data_btc import N_CELLS, N_DELTAS, N_TENORS, load_split
from .eval_hybrid import hybrid_predict
from .eval_structured import _build_structured
from .eval_vae import _build_masks, _device, _rmse, vae_predict
from .train_vae import build_model_from_ckpt

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "paper" / "figs"

RUNS = {
    "MLP (joint)":      "20260521-103848_joint_z16_h128",
    "Conv2D (h=64)":    "20260521-142018_conv_z16_h64",
    "Attention (h=64)": "20260521-143511_attn_z16_h64",
}

SCENARIOS = [
    ("random 10%",       ("random", 0.10)),
    ("random 30%",       ("random", 0.30)),
    ("random 50%",       ("random", 0.50)),
    ("row dropped",      ("row_random", None)),
    ("180d dropped",     ("long_tenor", None)),
    ("put wing dropped", ("wing_put", None)),
    ("call wing dropped",("wing_call", None)),
]


def _load_model(run_name):
    rd = RUNS_ROOT / run_name
    cfg = load_config(rd)
    ckpt = torch.load(rd / "best.pt", map_location="cpu", weights_only=True)
    device = _device()
    m = build_model_from_ckpt(ckpt, cfg).to(device)
    m.load_state_dict(ckpt["model"])
    return m, device


def _eval_vae_rmse(model, device, surfaces, masks_flat, mean, std):
    flat = surfaces.reshape(-1, N_CELLS)
    z = ((flat - mean) / std).astype(np.float32)
    recon_z = vae_predict(model, z, masks_flat, device)
    recon = (recon_z * std + mean).astype(np.float32)
    hidden = masks_flat < 0.5
    return _rmse(recon, flat, hidden) * 100  # vol points


def main():
    symbol = "BTCUSDT"
    bundle = load_split("test", symbol=symbol)
    train_b = load_split("train", symbol=symbol)
    mean, std = train_b.mean.reshape(-1), train_b.std.reshape(-1)
    surfaces = bundle.surfaces
    n = len(surfaces)

    results = {}  # arch_label -> [vol pts per scenario]
    for label, run in RUNS.items():
        model, device = _load_model(run)
        per_scenario = []
        for scen_label, (kind, arg) in SCENARIOS:
            rng = np.random.default_rng(0)
            if kind == "random":
                masks_flat = _build_masks(n, arg, rng)
            else:
                masks_flat = _build_structured(n, kind, rng)
            per_scenario.append(_eval_vae_rmse(model, device, surfaces,
                                                masks_flat, mean, std))
        results[label] = per_scenario
        print(f"{label}: {[f'{v:.2f}' for v in per_scenario]}")

    # Plot
    x = np.arange(len(SCENARIOS))
    width = 0.27
    colors = {"MLP (joint)": "#888888",
              "Conv2D (h=64)": "#C0392B",
              "Attention (h=64)": "#3B7BB8"}
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, (label, vals) in enumerate(results.items()):
        ax.bar(x + (i - 1) * width, vals, width, label=label, color=colors[label])
        for xi, v in enumerate(vals):
            ax.text(xi + (i - 1) * width, v + 0.08, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in SCENARIOS], rotation=18, ha="right",
                        fontsize=9)
    ax.set_ylabel("RMSE on hidden cells (vol points)")
    ax.set_title("Architecture comparison on the BTC test set: explicit "
                 "2D structure (Conv) closes the row-hole gap",
                 fontsize=11)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    out = OUT_DIR / "fig_arch.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
