"""Smoke test: the whole CV pipeline runs end to end on a tiny synthetic dataset."""

import math
from pathlib import Path

import pytest

from janestreet.config import Settings
from janestreet.data_processor import DataProcessor
from janestreet.models.nn import NN, CustomTensorDataset
from janestreet.pipeline import FullPipeline, PipelineCV
from janestreet.transformers import PolarsTransformer
from tests.conftest import make_synthetic_data


@pytest.fixture
def small_data(tmp_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Writes 12 days x 50 time steps x 3 symbols in the same folder layout as Kaggle.

    The model needs every day to have the same number of time steps (968 in the real data),
    and responder_10 looks 40 steps ahead, so a day must be longer than 40 steps.
    """
    monkeypatch.setattr(DataProcessor, "T", 5)
    monkeypatch.setattr(CustomTensorDataset, "T", 50)

    partition = tmp_settings.data_dir / "train.parquet" / "partition_id=0"
    partition.mkdir(parents=True)
    make_synthetic_data(n_symbols=3, n_dates=12, n_times=50).write_parquet(
        partition / "part-0.parquet"
    )
    return tmp_settings.data_dir


def test_last_n_days_keeps_only_the_most_recent_dates(small_data: Path) -> None:
    df = DataProcessor("test", last_n_days=4).get_train_data()

    assert sorted(df["date_id"].unique().to_list()) == [8, 9, 10, 11]


def test_cv_runs_end_to_end(small_data: Path) -> None:
    data_processor = DataProcessor("test", last_n_days=9)
    df = data_processor.get_train_data()
    features = data_processor.features

    model = NN(
        model_type="gru",
        hidden_sizes=[8],
        dropout_rates=[0.0],
        hidden_sizes_linear=[8],
        dropout_rates_linear=[0.0],
        epochs=1,
    )
    pipeline = FullPipeline(
        model,
        preprocessor=PolarsTransformer(features),
        run_name="full",
        name="test",
        features=features,
        refit=True,
        change_lr=True,
    )
    scores = PipelineCV(pipeline, None, n_splits=2).fit(df)

    assert len(scores) == 2
    assert all(math.isfinite(score) for score in scores)
