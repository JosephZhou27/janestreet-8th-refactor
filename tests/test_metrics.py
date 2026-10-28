"""Tests for the weighted R² metric and loss."""

import numpy as np
import pytest
import torch

from janestreet.metrics import WeightedR2Loss, r2_weighted, r2_weighted_torch


def test_perfect_prediction_scores_one() -> None:
    y = np.array([0.5, -1.0, 2.0])
    w = np.array([1.0, 2.0, 1.0])
    assert r2_weighted(y, y, w) == pytest.approx(1.0)


def test_predicting_zero_scores_zero() -> None:
    # The competition's R² has no mean term, so predicting 0 gives exactly 0
    y = np.array([0.5, -1.0, 2.0])
    w = np.array([1.0, 2.0, 1.0])
    assert r2_weighted(y, np.zeros_like(y), w) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("y_pred", "expected"),
    [
        ([1.0, 2.0], 1.0),
        ([0.0, 0.0], 0.0),
        ([2.0, 4.0], 0.0),
        ([1.0, 0.0], 0.2),
    ],
)
def test_matches_hand_calculation(y_pred: list[float], expected: float) -> None:
    # y_true = [1, 2], equal weights: R² = 1 - sum((pred - y)²) / sum(y²) = 1 - SSE / 5
    y_true = np.array([1.0, 2.0])
    w = np.array([1.0, 1.0])
    assert r2_weighted(y_true, np.array(y_pred), w) == pytest.approx(expected)


def test_numpy_and_torch_versions_agree() -> None:
    rng = np.random.default_rng(0)
    y_true, y_pred = rng.normal(size=100), rng.normal(size=100)
    w = rng.uniform(0.5, 2.0, size=100)

    r2_np = r2_weighted(y_true, y_pred, w)
    r2_torch = r2_weighted_torch(torch.tensor(y_true), torch.tensor(y_pred), torch.tensor(w))

    assert r2_torch.item() == pytest.approx(r2_np)


def test_loss_is_one_minus_r2() -> None:
    y_true = torch.tensor([1.0, 2.0, 3.0])
    y_pred = torch.tensor([1.5, 1.5, 2.5])
    w = torch.tensor([1.0, 1.0, 2.0])

    loss = WeightedR2Loss()(y_pred, y_true, w)

    assert loss.item() == pytest.approx(1 - r2_weighted_torch(y_true, y_pred, w).item())