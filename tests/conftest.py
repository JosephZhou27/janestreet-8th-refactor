"""Shared pytest fixtures."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from janestreet.config import Settings, get_settings


def make_synthetic_data(
    n_symbols: int = 3, n_dates: int = 4, n_times: int = 5, seed: int = 0
) -> pl.DataFrame:
    """Builds a small random table with the same columns as the competition data."""
    rng = np.random.default_rng(seed)
    n_rows = n_symbols * n_dates * n_times

    # One row per (date, time, symbol), sorted like the real data
    date_id, time_id, symbol_id = np.meshgrid(
        np.arange(n_dates), np.arange(n_times), np.arange(n_symbols), indexing="ij"
    )
    columns: dict[str, np.ndarray] = {
        "date_id": date_id.ravel(),
        "time_id": time_id.ravel(),
        "symbol_id": symbol_id.ravel(),
        "weight": rng.uniform(0.5, 2.0, n_rows),
    }
    for i in range(79):
        columns[f"feature_{i:02d}"] = rng.normal(size=n_rows)
    for i in range(9):
        columns[f"responder_{i}"] = rng.normal(size=n_rows)

    return pl.DataFrame(columns).with_columns(
        pl.col("date_id", "time_id").cast(pl.Int16),
        pl.col("symbol_id").cast(pl.Int8),
    )


@pytest.fixture
def synthetic_df() -> pl.DataFrame:
    """60 rows: 3 symbols x 4 dates x 5 time steps."""
    return make_synthetic_data()


@pytest.fixture
def tmp_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    """Points data_dir and models_dir to a temporary folder for one test."""
    monkeypatch.setenv("JANESTREET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("JANESTREET_MODELS_DIR", str(tmp_path / "models"))
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()