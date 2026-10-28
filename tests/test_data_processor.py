"""Tests for DataProcessor feature engineering."""

import polars as pl
import pytest

from janestreet.config import Settings
from janestreet.data_processor import DataProcessor
from janestreet.transformers import PolarsTransformer


@pytest.fixture
def short_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Uses a rolling window of 2 rows instead of 1000, so 60 rows are enough."""
    monkeypatch.setattr(DataProcessor, "T", 2)


def test_feature_list() -> None:
    dp = DataProcessor("test")

    # 79 raw - 3 categorical + 16 x 3 engineered + feature_time_id
    assert len(dp.features) == 79 - 3 + 16 * 3 + 1
    assert "feature_09" not in dp.features
    assert "feature_06_rolling_std_1000" in dp.features


def test_rolling_features_are_computed_per_symbol(
    synthetic_df: pl.DataFrame, short_window: None
) -> None:
    dp = DataProcessor("test")

    result = dp._add_features(synthetic_df)

    one_symbol = synthetic_df.filter(pl.col("symbol_id") == 0)
    expected_std = one_symbol["feature_06"].rolling_std(window_size=2)
    actual_std = result.filter(pl.col("symbol_id") == 0)["feature_06_rolling_std_2"]
    assert actual_std.to_list() == pytest.approx(expected_std.to_list(), nan_ok=True)


def test_market_average_is_mean_per_date_and_time(synthetic_df: pl.DataFrame) -> None:
    dp = DataProcessor("test")

    result = dp._add_features(synthetic_df)

    expected = synthetic_df.group_by("date_id", "time_id").agg(pl.col("feature_06").mean())
    first_row = result.row(0, named=True)
    expected_value = expected.filter(
        (pl.col("date_id") == first_row["date_id"]) & (pl.col("time_id") == first_row["time_id"])
    )["feature_06"].item()
    assert first_row["feature_06_avg_per_date_time"] == pytest.approx(expected_value)


@pytest.mark.parametrize("skip_days", [None, 2])
def test_get_train_data_end_to_end(
    synthetic_df: pl.DataFrame, tmp_settings: Settings, short_window: None, skip_days: int | None
) -> None:
    tmp_settings.data_dir.mkdir(parents=True)
    synthetic_df.with_columns(pl.lit(0).alias("partition_id")).write_parquet(
        tmp_settings.data_dir / "train.parquet"
    )
    dp = DataProcessor("test", skip_days=skip_days, transformer=PolarsTransformer())

    df = dp.get_train_data()

    assert df["date_id"].min() == (skip_days or 0)
    assert {"responder_9", "responder_10", *dp.features} <= set(df.columns)
    saved = tmp_settings.models_dir / "data_processors" / "test.joblib"
    assert saved.exists()
    assert dp.load().features == dp.features