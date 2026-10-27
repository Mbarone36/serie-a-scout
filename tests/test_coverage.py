import numpy as np
import pandas as pd
from src import coverage


def _frame(npxg_nulls: int, n: int = 10) -> pd.DataFrame:
    npxg = [1.0] * n
    for i in range(npxg_nulls):
        npxg[i] = np.nan
    return pd.DataFrame({
        "player": [f"p{i}" for i in range(n)],
        "expected_npxg": npxg,
        "progression_prgp": [5.0] * n,
        "possession_take_ons_succ": [1.0] * n,
        "minutes": [900] * n,
    })


def test_assess_reports_full_coverage():
    result = coverage.assess(_frame(npxg_nulls=0))
    assert result["coverage"] == 1.0
    assert result["n_players"] == 10


def test_assess_reports_partial_coverage():
    result = coverage.assess(_frame(npxg_nulls=5))
    assert result["coverage"] < 1.0


def test_assess_handles_missing_column_entirely():
    frame = _frame(0).drop(columns=["expected_npxg"])
    result = coverage.assess(frame)
    assert result["coverage"] < 1.0
    assert "expected_npxg" in result["missing_columns"]


def test_trusted_leagues_filters_below_threshold():
    report = pd.DataFrame({
        "league": ["A", "A", "B", "B"],
        "season": ["2223", "2324", "2223", "2324"],
        "coverage": [1.0, 0.95, 0.2, 0.1],
        "n_players": [500, 500, 400, 400],
    })
    assert coverage.trusted_leagues(report, min_coverage=0.9) == ["A"]


def test_trusted_leagues_excludes_empty_report():
    report = pd.DataFrame(columns=["league", "season", "coverage", "n_players"])
    assert coverage.trusted_leagues(report) == []
