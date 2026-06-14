"""Run a list of TrainConfig variants end-to-end (train + eval).

Each entry is a TrainConfig. Edit `ABLATIONS` to add/remove sweeps. To run a
single ablation by name:  python -m src.ablate <name>
To run all:               python -m src.ablate
"""

from __future__ import annotations

import argparse
import sys

from .config import TrainConfig
from .eval_vae import eval_vae
from .train_vae import train

ABLATIONS: list[TrainConfig] = [
    TrainConfig(name="default",
                notes="MVP baseline: z=8 h=128 beta=1e-3, 200 epochs"),
    # Step 1: capacity sweep (z, hidden) at longer training to see if we can
    # tighten the low-mask gap to smile-refit.
    TrainConfig(name="z4_h128_ep500",  z_dim=4,  hidden=128, epochs=500, patience=50,
                notes="step1: smaller latent, baseline width, 500 epochs"),
    TrainConfig(name="z4_h256_ep500",  z_dim=4,  hidden=256, epochs=500, patience=50,
                notes="step1: smaller latent, wider hidden"),
    TrainConfig(name="z8_h128_ep500",  z_dim=8,  hidden=128, epochs=500, patience=50,
                notes="step1: default latent + width, longer training"),
    TrainConfig(name="z8_h256_ep500",  z_dim=8,  hidden=256, epochs=500, patience=50,
                notes="step1: default latent, wider hidden"),
    TrainConfig(name="z16_h128_ep500", z_dim=16, hidden=128, epochs=500, patience=50,
                notes="step1: larger latent, baseline width"),
    TrainConfig(name="z16_h256_ep500", z_dim=16, hidden=256, epochs=500, patience=50,
                notes="step1: larger latent, wider hidden"),
    # Step 2b: mixed-mask training. 50% random + 50% structured rotation.
    # Hypothesis: closes the structured-row gap (random-trained: 4 vp on
    # row_random vs 1.85 on equiv random) without hurting random eval.
    TrainConfig(name="mixed_z16_h128", z_dim=16, hidden=128, epochs=500, patience=50,
                mask_scheme="mixed",
                notes="step2b: mixed random+structured masking, z=16 h=128"),
    TrainConfig(name="mixed_z16_h256", z_dim=16, hidden=256, epochs=500, patience=50,
                mask_scheme="mixed",
                notes="step2b: mixed random+structured masking, z=16 h=256"),
    # Step 3b: multi-symbol training. Comparing against BTC-only baseline
    # (z16_h128_ep500) and the BTC→ETH OOD result already on disk.
    TrainConfig(name="eth_z16_h128", z_dim=16, hidden=128, epochs=500, patience=50,
                symbols=("ETHUSDT",),
                notes="step3b: ETH-only, same capacity as BTC-only best"),
    TrainConfig(name="joint_z16_h128", z_dim=16, hidden=128, epochs=500, patience=50,
                symbols=("BTCUSDT", "ETHUSDT"),
                notes="step3b: BTC+ETH joint, per-symbol z-norm then concat"),
    # Architectural exploration (separate from main paper): does explicit
    # tenor x delta structure close the row-shaped-hole gap?
    TrainConfig(name="conv_z16_h32", arch="conv2d", z_dim=16, hidden=32,
                epochs=500, patience=50, symbols=("BTCUSDT", "ETHUSDT"),
                notes="arch: 2D conv VAE on 6x7 grid, h=32 channels"),
    TrainConfig(name="conv_z16_h64", arch="conv2d", z_dim=16, hidden=64,
                epochs=500, patience=50, symbols=("BTCUSDT", "ETHUSDT"),
                notes="arch: 2D conv VAE on 6x7 grid, h=64 channels"),
    TrainConfig(name="attn_z16_h64", arch="attention", z_dim=16, hidden=64,
                epochs=500, patience=50, symbols=("BTCUSDT", "ETHUSDT"),
                notes="arch: per-cell self-attention VAE, h=64 d_model, 2 layers"),
    # Single-symbol conv variants for the joint-vs-single comparison
    TrainConfig(name="conv_btc_z16_h64", arch="conv2d", z_dim=16, hidden=64,
                epochs=500, patience=50, symbols=("BTCUSDT",),
                notes="conv variant trained on BTC only"),
    TrainConfig(name="conv_eth_z16_h64", arch="conv2d", z_dim=16, hidden=64,
                epochs=500, patience=50, symbols=("ETHUSDT",),
                notes="conv variant trained on ETH only"),
    # Ackerer 2020 Deep Smoothing-style baseline: deterministic MLP
    # autoencoder with a soft calendar-arbitrage penalty.
    TrainConfig(name="smoother_z32_h128", arch="smoother", z_dim=32, hidden=128,
                epochs=500, patience=50, symbols=("BTCUSDT", "ETHUSDT"),
                notes="Ackerer-style deterministic smoother, joint BTC+ETH"),
]


def run_one(cfg: TrainConfig) -> None:
    run_dir = train(cfg)
    eval_vae(run_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?",
                    help="Single ablation name to run; default = run all")
    args = ap.parse_args()
    if args.name:
        target = [c for c in ABLATIONS if c.name == args.name]
        if not target:
            print(f"no ablation named {args.name!r}", file=sys.stderr)
            print(f"known: {[c.name for c in ABLATIONS]}", file=sys.stderr)
            sys.exit(1)
        run_one(target[0])
    else:
        for cfg in ABLATIONS:
            run_one(cfg)


if __name__ == "__main__":
    main()
