# Jane Street Real-Time Market Data Forecasting: 8th-place solution, refactored

[![CI](https://github.com/JosephZhou27/janestreet-8th-refactor/actions/workflows/ci.yml/badge.svg)](https://github.com/JosephZhou27/janestreet-8th-refactor/actions/workflows/ci.yml)

This project predicts short-term returns of financial instruments from anonymised market features, for the [Jane Street Real-Time Market Data Forecasting](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/overview) Kaggle competition (2024–2025). Predictions are made in real time through Kaggle's API, one time step at a time, and the true values of each day are revealed the next day.

The modelling solution is by **Evgeniia Volkova** ([original repository](https://github.com/evgeniavolkova/kagglejanestreet)) and finished **8th on the private leaderboard**. All modelling ideas are hers.

**This fork is my engineering refactor of her code.** The goal is to turn competition code into a project a team could maintain: reproducible environment, configuration without hard-coded paths, logging, type checks, automated tests, CI, and a mode that runs on a laptop. Every change went through a pull request.

## Results

| | Value | Source |
|---|---|---|
| Private leaderboard | 8th place | Original solution |
| CV R² (2 GRU models × 3 seeds, 2 folds) | 0.0222 | Original solution, [`solution.md`](solution.md) |
| Full training requirements | ~100 GB RAM, 12 GB GPU | Original solution |
| `--sample` run (last 30 days, small model) | ~1 minute on a MacBook Pro M4, 24 GB | This fork |
| Automated checks on every PR | ruff, mypy, 19 tests in ~5 s | This fork |

The `--sample` run checks that the whole pipeline works end to end. Its score is not meaningful: it trains a small model for 2 epochs on 10–20 days, so its R² is close to zero.

## What I changed

| PR | Change | Why it matters |
|---|---|---|
| [#5](https://github.com/JosephZhou27/janestreet-8th-refactor/pull/5) | Poetry → uv, Python 3.12, locked dependencies | Same environment on every machine with one command |
| [#6](https://github.com/JosephZhou27/janestreet-8th-refactor/pull/6) | ruff for formatting and linting | Consistent style; catches common bugs |
| [#7](https://github.com/JosephZhou27/janestreet-8th-refactor/pull/7) | Paths from environment variables / `.env` (pydantic-settings) | No hard-coded paths; runs locally and on Kaggle |
| [#8](https://github.com/JosephZhou27/janestreet-8th-refactor/pull/8) | `logging` instead of `print` | Timestamps and levels; output can go to files |
| [#9](https://github.com/JosephZhou27/janestreet-8th-refactor/pull/9) | Type hints checked by mypy | Catches wrong types before running |
| [#10](https://github.com/JosephZhou27/janestreet-8th-refactor/pull/10) | Unit tests on synthetic data (metric, scaling, features, no data leakage) | Refactoring is safe; tests run in seconds |
| [#12](https://github.com/JosephZhou27/janestreet-8th-refactor/pull/12) | GitHub Actions CI | Every PR is checked automatically |
| [#13](https://github.com/JosephZhou27/janestreet-8th-refactor/pull/13) | pre-commit hooks | Formatting problems are fixed before each commit |
| [#14](https://github.com/JosephZhou27/janestreet-8th-refactor/pull/14) | `--sample` mode and an end-to-end smoke test | Runs on a laptop in a minute; CI runs the whole pipeline on synthetic data |

## Project structure

```
janestreet/            Python package
  config.py            settings (paths) read from environment variables or .env
  data_processor.py    loads data and builds features
  models/nn.py         GRU model, training and online updates
  pipeline.py          training pipeline and time-series cross-validation
  metrics.py           weighted R², the competition metric
scripts/python/        entry points (run_cv.py, run_full_2.py, ...)
tests/                 pytest tests on synthetic data
solution.md            detailed description of the modelling approach
```

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then install the environment and the git hooks:

   ```bash
   uv sync
   uv run pre-commit install
   ```

2. Download the data from [Kaggle](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/data) into `data/`, so that the folder `data/train.parquet/` exists. For a quick run, one partition is enough.
3. Optional: to keep the data elsewhere, copy [`.env.example`](.env.example) to `.env` and set `JANESTREET_DATA_DIR`.

## Usage

```bash
# Quick run: last 30 days, small model, CPU is fine
uv run python scripts/python/run_cv.py --sample

# Full cross-validation (needs ~100 GB RAM and a GPU)
uv run python scripts/python/run_cv.py
```

Other scripts in [`scripts/python/`](scripts/python/):

- [`run_full_2.py`](scripts/python/run_full_2.py), [`run_full_3.py`](scripts/python/run_full_3.py): train the final models on all data
- [`run_ensemble.py`](scripts/python/run_ensemble.py): evaluate the ensemble on CV
- [`run_test_gap.py`](scripts/python/run_test_gap.py): test on the last 200 dates with a 200-date gap

## Checks

The same checks run in CI on every pull request:

```bash
uv run ruff check
uv run ruff format --check
uv run mypy
uv run pytest
```

## Method in brief

Features (79 raw features, rolling and market-average features) → standardisation → GRU over one trading day per sequence, with auxiliary targets → time-series cross-validation → online learning: after each day, the model is updated once on that day's data.

Details and ablation scores are in [`solution.md`](solution.md) and the author's [Kaggle write-up](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/556542).
