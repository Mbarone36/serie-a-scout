import numpy as np
import pandas as pd
import pytest
from src import baselines


def _frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    f1 = rng.random(n)
    return pd.DataFrame({
        "player": [f"p{i}" for i in range(n)],
        "f1": f1,
        "f2": rng.random(n),
        "minutes": f1 * 0.8 + rng.random(n) * 0.2,
        "output": f1 * 0.5 + rng.random(n) * 0.5,
        "output_share": f1 * 0.5 + rng.random(n) * 0.5,
        "retention": rng.choice([0.0, 0.33, 0.66, 1.0], n),
    })


def test_spearman_of_perfect_ranking_is_one():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert baselines.spearman(y, y * 10) == pytest.approx(1.0)


def test_spearman_of_reversed_ranking_is_minus_one():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert baselines.spearman(y, -y) == pytest.approx(-1.0)


def test_spearman_of_constant_prediction_is_zero_not_nan():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert baselines.spearman(y, np.zeros(4)) == 0.0


def test_models_registry_has_ridge_and_gbm():
    assert set(baselines.MODELS) == {"ridge", "gbm"}


def test_evaluate_reports_every_target():
    result = baselines.evaluate(_frame(), ["f1", "f2"], "ridge")
    assert set(result) == {"minutes", "output", "output_share", "retention"}
    for metrics in result.values():
        assert set(metrics) == {"spearman", "mae"}


def test_evaluate_recovers_a_planted_signal():
    result = baselines.evaluate(_frame(n=120), ["f1", "f2"], "ridge")
    assert result["minutes"]["spearman"] > 0.5
