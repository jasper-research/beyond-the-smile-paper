# Beyond the Smile: A Hybrid Convolutional VAE for Crypto Volatility Surfaces

Reproduction code and processed data for the paper
**"Beyond the Smile: A Hybrid Convolutional VAE for Crypto Volatility Surfaces"**
(Singh, Reddy, Chopra — Jasper Research).

- **Paper (arXiv):** `<ARXIV-ID-TO-INSERT>`
- **Archived code + processed data + run artifacts (Zenodo):** [10.5281/zenodo.20693546](https://doi.org/10.5281/zenodo.20693546)

This repository contains the full pipeline that turns the public Binance
options archive into fixed-shape implied-volatility surfaces, trains a
convolutional VAE with a deterministic hybrid reconstruction rule, and
reproduces every figure, table, and reported number in the manuscript.

---

## What's here vs. on Zenodo

| Artifact | Location |
|---|---|
| All source code (`src/`), notebooks, pipeline | **this repo** |
| Processed `6 × 7` gridded surfaces (`data/grid/`) | **Zenodo** ([10.5281/zenodo.20693546](https://doi.org/10.5281/zenodo.20693546)) |
| Trained runs: configs, checkpoints, metrics, plots (`runs/`) | **Zenodo** ([10.5281/zenodo.20693546](https://doi.org/10.5281/zenodo.20693546)) |
| Raw Binance EOH inputs | **public source** (see below); regenerable via `fetch_binance_eoh.py` |

`data/` and `runs/` are git-ignored. To reproduce *without* re-running the
multi-hour download/clean steps, download the Zenodo archive and unzip its
`data/grid/` and `runs/` folders into the repo root.

---

## Setup

Requires Python ≥ 3.11 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync          # installs locked dependencies from pyproject.toml + uv.lock
```

---

## Reproducing the paper from raw public data

All scripts run as `uv run python src/<script>.py`. Thresholds and grid
choices are declared as constants at the top of each script; the data
pipeline is deterministic and idempotent (see `docs/DATA.md` for the full
data provenance record).

```bash
# 1. Download the primary training archive (Binance Options EOH, public, no auth)
uv run python src/fetch_binance_eoh.py            # -> data/external/binance_eoh/

# 2. Clean: parse, put-call-parity forward, Black-76 IV inversion, quality flags
uv run python src/clean.py                        # -> data/clean/binance_eoh/

# 3. Grid: irregular chains -> fixed 6-tenor x 7-delta surfaces
uv run python src/build_grid.py                   # -> data/grid/binance_eoh/

# 4. Train the baseline VAE
uv run python src/train_vae.py                    # -> runs/vae_btc/<timestamp>_default/

# 5. Run the full ablation grid (train + eval; ~5 min GPU on a consumer machine)
uv run python src/ablate.py                       # all variants; or pass a single name

# 6. Regenerate the paper figures and comparison tables
uv run python src/make_paper_figs.py
```

### Evaluation / analysis entry points

```bash
uv run python src/eval_hybrid.py        # hybrid reconstruction metrics
uv run python src/eval_structured.py    # structured (sparse-strike) completion
uv run python src/eval_cross_asset.py   # BTC -> ETH transfer
uv run python src/arb_project.py        # calendar/butterfly arbitrage check
uv run python src/analyze_anomaly.py    # reconstruction-error anomaly timeline
uv run python src/baseline_pca.py       # PCA / parametric-smile baselines
```

The out-of-distribution test reported in the paper is **zero-shot BTC→ETH
transfer**, both within the Binance dataset (`src/eval_cross_asset.py`).

---

## Data source

- **Binance Options end-of-hour archive** — the single training/evaluation
  dataset. `https://data.binance.vision/data/option/daily/EOHSummary/` (no auth).

No proprietary or licensed data is used. Full provenance, schema, cleaning
decisions, and grid rationale are documented in [`docs/DATA.md`](docs/DATA.md).

---

## Citation

If you use this code or data, please cite the paper (see `CITATION.cff`):

```bibtex
@article{singh2026beyondthesmile,
  title  = {Beyond the Smile: A Hybrid Convolutional VAE for Crypto Volatility Surfaces},
  author = {Singh, Sadanand and Reddy, Allam and Chopra, Manan},
  year   = {2026},
  eprint = {<ARXIV-ID-TO-INSERT>},
  archivePrefix = {arXiv}
}
```

## License

Code is released under the MIT License (see `LICENSE`). The processed data
deposited on Zenodo is released under CC BY 4.0.
