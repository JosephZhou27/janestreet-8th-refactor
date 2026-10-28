"""Tests for PolarsTransformer."""

import polars as pl
import pytest

from janestreet.transformers import PolarsTransformer


@pytest.fixture
def train_df() -> pl.DataFrame:
    return pl.DataFrame({"a": [1.0, 2.0, None, 4.0], "feature_time_id": [0, 1, 2, 3]})


def test_fit_transform_uses_statistics_before_filling_nulls(train_df: pl.DataFrame) -> None:
    transformer = PolarsTransformer(["a"])

    result = transformer.fit_transform(train_df)

    # Mean and std are computed on [1, 2, 4] only; the null becomes 0 and is then scaled
    mean, std = 7 / 3, train_df["a"].std()
    expected = [(v - mean) / std for v in [1.0, 2.0, 0.0, 4.0]]
    assert result["a"].to_list() == pytest.approx(expected)
    assert transformer.statistics_mean_std["a"]["mean"] == pytest.approx(mean)


def test_transform_reuses_training_statistics(train_df: pl.DataFrame) -> None:
    transformer = PolarsTransformer(["a"])
    transformer.fit_transform(train_df)
    stats_before = transformer.statistics_mean_std["a"].copy()

    new_df = pl.DataFrame({"a": [100.0, 200.0], "feature_time_id": [0, 1]})
    result = transformer.transform(new_df)

    # No leakage: statistics are not recomputed on new data
    assert transformer.statistics_mean_std["a"] == stats_before
    expected = [(v - stats_before["mean"]) / stats_before["std"] for v in [100.0, 200.0]]
    assert result["a"].to_list() == pytest.approx(expected)


def test_time_id_is_clipped_to_training_range(train_df: pl.DataFrame) -> None:
    transformer = PolarsTransformer(["a"], scale=False)
    transformer.fit_transform(train_df)

    new_df = pl.DataFrame({"a": [1.0, 1.0], "feature_time_id": [-5, 99]})
    result = transformer.transform(new_df)

    assert result["feature_time_id"].to_list() == [0, 3]


def test_missing_features_raise_clear_error(train_df: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="Features are not set"):
        PolarsTransformer().fit_transform(train_df)
