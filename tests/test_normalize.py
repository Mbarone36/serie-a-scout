import numpy as np
import pandas as pd
import pytest
from src import normalize


def test_position_group_maps_primary_position():
    assert normalize.position_group("FW") == "attacker"
    assert normalize.position_group("FW,MF") == "attacker"
    assert normalize.position_group("DF,MF") == "defender"
    assert normalize.position_group("") == "unknown"


def test_per_90_scales_by_minutes():
    frame = pd.DataFrame({"minutes": [900, 450], "goals": [10, 10]})
    out = normalize.per_90(frame, ["goals"])
    assert out["goals_p90"].tolist() == [1.0, 2.0]


def test_per_90_guards_against_zero_minutes():
    frame = pd.DataFrame({"minutes": [0], "goals": [3]})
    out = normalize.per_90(frame, ["goals"])
    assert pd.isna(out["goals_p90"].iloc[0])


def test_percentile_within_group_is_zero_to_one():
    frame = pd.DataFrame({
        "league": ["A"] * 4, "season": ["2223"] * 4, "position_group": ["attacker"] * 4,
        "x": [1.0, 2.0, 3.0, 4.0],
    })
    out = normalize.percentile_within(frame, ["x"])
    assert out["x_pct"].min() >= 0.0
    assert out["x_pct"].max() <= 1.0
    assert out["x_pct"].is_monotonic_increasing


def test_percentile_is_computed_separately_per_league():
    frame = pd.DataFrame({
        "league": ["A", "A", "B", "B"],
        "season": ["2223"] * 4,
        "position_group": ["attacker"] * 4,
        "x": [1.0, 2.0, 100.0, 200.0],
    })
    out = normalize.percentile_within(frame, ["x"])
    a = out[out.league == "A"]["x_pct"].tolist()
    b = out[out.league == "B"]["x_pct"].tolist()
    assert a == b  # identical ranks despite wildly different raw values


def test_percentile_does_not_pool_across_seasons():
    frame = pd.DataFrame({
        "league": ["A"] * 4,
        "season": ["2122", "2122", "2223", "2223"],
        "position_group": ["attacker"] * 4,
        "x": [1.0, 2.0, 50.0, 100.0],
    })
    out = normalize.percentile_within(frame, ["x"])
    assert out[out.season == "2122"]["x_pct"].tolist() == out[out.season == "2223"]["x_pct"].tolist()


def test_percentile_handles_single_member_group():
    frame = pd.DataFrame({
        "league": ["A"], "season": ["2223"], "position_group": ["keeper"], "x": [5.0],
    })
    out = normalize.percentile_within(frame, ["x"])
    assert out["x_pct"].iloc[0] == 0.5


def test_percentile_preserves_nulls():
    frame = pd.DataFrame({
        "league": ["A"] * 3, "season": ["2223"] * 3, "position_group": ["attacker"] * 3,
        "x": [1.0, np.nan, 3.0],
    })
    out = normalize.percentile_within(frame, ["x"])
    assert pd.isna(out["x_pct"].iloc[1])


def test_percentile_single_member_rule_counts_non_null_only():
    """Single-member rule should count only non-null values.

    Group [NaN, 5.0, NaN] has 1 non-null value, so 5.0 should get 0.5 (not 1.0).
    """
    frame = pd.DataFrame({
        "league": ["A"] * 3, "season": ["2223"] * 3, "position_group": ["attacker"] * 3,
        "x": [np.nan, 5.0, np.nan],
    })
    out = normalize.percentile_within(frame, ["x"])
    assert out["x_pct"].iloc[1] == 0.5
    assert pd.isna(out["x_pct"].iloc[0])
    assert pd.isna(out["x_pct"].iloc[2])


def test_percentile_ignores_nulls_in_ranking():
    """Multiple real values with nulls should rank only the real values.

    Group [NaN, 1.0, 2.0] has 2 non-null values, so they rank as 0.5 and 1.0.
    """
    frame = pd.DataFrame({
        "league": ["A"] * 3, "season": ["2223"] * 3, "position_group": ["attacker"] * 3,
        "x": [np.nan, 1.0, 2.0],
    })
    out = normalize.percentile_within(frame, ["x"])
    assert out["x_pct"].iloc[1] == 0.5
    assert out["x_pct"].iloc[2] == 1.0
    assert pd.isna(out["x_pct"].iloc[0])
