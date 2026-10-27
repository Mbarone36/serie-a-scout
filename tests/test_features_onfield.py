import pandas as pd
import pytest
from src import features_onfield


def _seasons(n=6):
    return pd.DataFrame({
        "player": [f"p{i}" for i in range(n)],
        "team": ["T"] * n,
        "league": ["ITA-Serie A"] * n,
        "season": ["2223"] * n,
        "pos": ["FW"] * n,
        "minutes": [900] * n,
        "age": [24] * n,
        "int": [float(i) for i in range(n)],
        "tklw": [float(i) for i in range(n)],
        "crs": [float(i) * 2 for i in range(n)],
        "fld": [float(i) for i in range(n)],
        "af_tackles": [float(i) for i in range(n)],
        "af_duels_won": [float(i) * 3 for i in range(n)],
        "af_duels_won_pct": [0.5] * n,
        "performance_crdy": [float(i) for i in range(n)],
        "performance_fls": [float(i) for i in range(n)],
        "performance_gls": [float(i) for i in range(n)],
        "performance_ast": [float(i) * 0.5 for i in range(n)],
        "standard_sh": [float(i) * 2 for i in range(n)],
        "standard_sot": [float(i) for i in range(n)],
    })


def test_build_adds_position_group():
    out = features_onfield.build(_seasons())
    assert set(out["position_group"]) == {"attacker"}


def test_build_emits_percentile_columns_for_every_feature():
    out = features_onfield.build(_seasons())
    for column in features_onfield.FEATURE_COLUMNS:
        assert column in out.columns, column


def test_percentile_features_are_bounded():
    out = features_onfield.build(_seasons())
    values = out[features_onfield.FEATURE_COLUMNS].to_numpy()
    assert ((values >= 0.0) & (values <= 1.0)).all()


def test_missing_optional_column_is_filled_not_fatal():
    frame = _seasons().drop(columns=["crs"])
    out = features_onfield.build(frame)
    assert "crs_p90_pct" in out.columns
    assert out["crs_p90_pct"].notna().all()


def test_build_is_row_preserving():
    frame = _seasons(n=7)
    assert len(features_onfield.build(frame)) == 7


def test_live_shaped_frame_yields_real_percentiles_for_basics_only():
    """FBref's live pages (seasons "2324"+) carry only the basic columns —
    no expected/progression/possession/tackles/int at all. The four basics
    that ARE available in both eras (performance_gls, performance_ast,
    standard_sh, standard_sot) must still rank for real; everything else
    has zero signal and must centre at 0.5 rather than look like a real
    (and misleadingly average) percentile.
    """
    n = 6
    frame = pd.DataFrame({
        "player": [f"p{i}" for i in range(n)],
        "team": ["T"] * n,
        "league": ["ITA-Serie A"] * n,
        "season": ["2324"] * n,
        "pos": ["FW"] * n,
        "minutes": [900] * n,
        "age": [24] * n,
        "performance_gls": [float(i) for i in range(n)],
        "performance_ast": [float(i) * 2 for i in range(n)],
        "standard_sh": [float(i) * 3 for i in range(n)],
        "standard_sot": [float(i) for i in range(n)],
        # No expected_*, progression_*, possession_*, tackles_tkl, int,
        # aerial_duels_won_pct columns at all -- the live-page shape.
    })
    out = features_onfield.build(frame)

    basic_pct_cols = [
        "performance_gls_p90_pct", "performance_ast_p90_pct",
        "standard_sh_p90_pct", "standard_sot_p90_pct",
    ]
    for col in basic_pct_cols:
        values = out[col].to_numpy()
        assert len(set(values)) > 1, f"{col} should show real spread, not a constant fill"
        assert not (values == 0.5).all()

    advanced_pct_cols = [
        c for c in features_onfield.FEATURE_COLUMNS if c not in basic_pct_cols and c != "age_pct"
    ]
    for col in advanced_pct_cols:
        assert (out[col] == 0.5).all(), f"{col} should be neutral-filled with no signal at all"


def test_percentiles_isolated_by_league():
    """Percentiles must rank within league, not globally.

    Two leagues with identical rank order but 100x scale should yield
    identical percentiles row-for-row.
    """
    # League A: base values
    league_a = pd.DataFrame({
        "player": ["a1", "a2", "a3", "a4"],
        "team": ["T"] * 4,
        "league": ["ITA-Serie A"] * 4,
        "season": ["2223"] * 4,
        "pos": ["FW"] * 4,
        "minutes": [900] * 4,
        "age": [24] * 4,
        "expected_npxg": [1.0, 2.0, 3.0, 4.0],
        "expected_xag": [0.5, 1.0, 1.5, 2.0],
        "progression_prgp": [2.0, 4.0, 6.0, 8.0],
        "progression_prgc": [3.0, 6.0, 9.0, 12.0],
        "possession_take_ons_succ": [1.0, 2.0, 3.0, 4.0],
        "possession_touches_att_pen": [4.0, 8.0, 12.0, 16.0],
        "possession_carries_mis": [1.0, 2.0, 3.0, 4.0],
        "possession_carries_dis": [1.0, 2.0, 3.0, 4.0],
        "tackles_tkl": [1.0, 2.0, 3.0, 4.0],
        "int": [1.0, 2.0, 3.0, 4.0],
        "aerial_duels_won_pct": [50.0] * 4,
        "performance_crdy": [1.0, 2.0, 3.0, 4.0],
        "performance_fls": [1.0, 2.0, 3.0, 4.0],
        "performance_gls": [1.0, 2.0, 3.0, 4.0],
        "performance_ast": [1.0, 2.0, 3.0, 4.0],
        "standard_sh": [1.0, 2.0, 3.0, 4.0],
        "standard_sot": [1.0, 2.0, 3.0, 4.0],
    })

    # League B: 100x scale, same rank order
    league_b = pd.DataFrame({
        "player": ["b1", "b2", "b3", "b4"],
        "team": ["T"] * 4,
        "league": ["NED-Eredivisie"] * 4,
        "season": ["2223"] * 4,
        "pos": ["FW"] * 4,
        "minutes": [900] * 4,
        "age": [24] * 4,
        "expected_npxg": [100.0, 200.0, 300.0, 400.0],
        "expected_xag": [50.0, 100.0, 150.0, 200.0],
        "progression_prgp": [200.0, 400.0, 600.0, 800.0],
        "progression_prgc": [300.0, 600.0, 900.0, 1200.0],
        "possession_take_ons_succ": [100.0, 200.0, 300.0, 400.0],
        "possession_touches_att_pen": [400.0, 800.0, 1200.0, 1600.0],
        "possession_carries_mis": [100.0, 200.0, 300.0, 400.0],
        "possession_carries_dis": [100.0, 200.0, 300.0, 400.0],
        "tackles_tkl": [100.0, 200.0, 300.0, 400.0],
        "int": [100.0, 200.0, 300.0, 400.0],
        "aerial_duels_won_pct": [50.0] * 4,
        "performance_crdy": [100.0, 200.0, 300.0, 400.0],
        "performance_fls": [100.0, 200.0, 300.0, 400.0],
        "performance_gls": [100.0, 200.0, 300.0, 400.0],
        "performance_ast": [100.0, 200.0, 300.0, 400.0],
        "standard_sh": [100.0, 200.0, 300.0, 400.0],
        "standard_sot": [100.0, 200.0, 300.0, 400.0],
    })

    frame = pd.concat([league_a, league_b], ignore_index=True)
    out = features_onfield.build(frame)

    # Extract percentiles for league A and B
    league_a_out = out[out["league"] == "ITA-Serie A"].reset_index(drop=True)
    league_b_out = out[out["league"] == "NED-Eredivisie"].reset_index(drop=True)

    # Same rank order in both leagues should yield identical percentiles
    for col in features_onfield.FEATURE_COLUMNS:
        pd.testing.assert_series_equal(
            league_a_out[col].reset_index(drop=True),
            league_b_out[col].reset_index(drop=True),
            check_names=False,
            check_exact=False,
            atol=1e-10,
        )


def test_percentiles_isolated_by_season():
    """Percentiles must rank within season, not globally.

    Two seasons with identical rank order but different scales should yield
    identical percentiles row-for-row.
    """
    # Season 2223: base values
    season_a = pd.DataFrame({
        "player": ["a1", "a2", "a3", "a4"],
        "team": ["T"] * 4,
        "league": ["ITA-Serie A"] * 4,
        "season": ["2223"] * 4,
        "pos": ["FW"] * 4,
        "minutes": [900] * 4,
        "age": [24] * 4,
        "expected_npxg": [1.0, 2.0, 3.0, 4.0],
        "expected_xag": [0.5, 1.0, 1.5, 2.0],
        "progression_prgp": [2.0, 4.0, 6.0, 8.0],
        "progression_prgc": [3.0, 6.0, 9.0, 12.0],
        "possession_take_ons_succ": [1.0, 2.0, 3.0, 4.0],
        "possession_touches_att_pen": [4.0, 8.0, 12.0, 16.0],
        "possession_carries_mis": [1.0, 2.0, 3.0, 4.0],
        "possession_carries_dis": [1.0, 2.0, 3.0, 4.0],
        "tackles_tkl": [1.0, 2.0, 3.0, 4.0],
        "int": [1.0, 2.0, 3.0, 4.0],
        "aerial_duels_won_pct": [50.0] * 4,
        "performance_crdy": [1.0, 2.0, 3.0, 4.0],
        "performance_fls": [1.0, 2.0, 3.0, 4.0],
        "performance_gls": [1.0, 2.0, 3.0, 4.0],
        "performance_ast": [1.0, 2.0, 3.0, 4.0],
        "standard_sh": [1.0, 2.0, 3.0, 4.0],
        "standard_sot": [1.0, 2.0, 3.0, 4.0],
    })

    # Season 2324: 50x scale, same rank order
    season_b = pd.DataFrame({
        "player": ["b1", "b2", "b3", "b4"],
        "team": ["T"] * 4,
        "league": ["ITA-Serie A"] * 4,
        "season": ["2324"] * 4,
        "pos": ["FW"] * 4,
        "minutes": [900] * 4,
        "age": [24] * 4,
        "expected_npxg": [50.0, 100.0, 150.0, 200.0],
        "expected_xag": [25.0, 50.0, 75.0, 100.0],
        "progression_prgp": [100.0, 200.0, 300.0, 400.0],
        "progression_prgc": [150.0, 300.0, 450.0, 600.0],
        "possession_take_ons_succ": [50.0, 100.0, 150.0, 200.0],
        "possession_touches_att_pen": [200.0, 400.0, 600.0, 800.0],
        "possession_carries_mis": [50.0, 100.0, 150.0, 200.0],
        "possession_carries_dis": [50.0, 100.0, 150.0, 200.0],
        "tackles_tkl": [50.0, 100.0, 150.0, 200.0],
        "int": [50.0, 100.0, 150.0, 200.0],
        "aerial_duels_won_pct": [50.0] * 4,
        "performance_crdy": [50.0, 100.0, 150.0, 200.0],
        "performance_fls": [50.0, 100.0, 150.0, 200.0],
        "performance_gls": [50.0, 100.0, 150.0, 200.0],
        "performance_ast": [50.0, 100.0, 150.0, 200.0],
        "standard_sh": [50.0, 100.0, 150.0, 200.0],
        "standard_sot": [50.0, 100.0, 150.0, 200.0],
    })

    frame = pd.concat([season_a, season_b], ignore_index=True)
    out = features_onfield.build(frame)

    # Extract percentiles for season A and B
    season_a_out = out[out["season"] == "2223"].reset_index(drop=True)
    season_b_out = out[out["season"] == "2324"].reset_index(drop=True)

    # Same rank order in both seasons should yield identical percentiles
    for col in features_onfield.FEATURE_COLUMNS:
        pd.testing.assert_series_equal(
            season_a_out[col].reset_index(drop=True),
            season_b_out[col].reset_index(drop=True),
            check_names=False,
            check_exact=False,
            atol=1e-10,
        )


UNDERSTAT_COUNTING = ("us_npxg", "us_xa", "us_xgchain", "us_xgbuildup", "us_key_passes")


def test_understat_columns_are_counting_features():
    for column in UNDERSTAT_COUNTING:
        assert column in features_onfield.COUNTING_COLUMNS, column
        assert f"{column}_p90_pct" in features_onfield.FEATURE_COLUMNS
    # 10 FBref + 5 Understat + 6 API-Football counting, 1 rate, age.
    assert len(features_onfield.COUNTING_COLUMNS) == 21
    assert len(features_onfield.FEATURE_COLUMNS) == 23


def test_understat_columns_rank_for_real_in_the_live_era():
    """A live-era frame (FBref basics only) that carries Understat columns
    must get real percentiles for them -- the point of the Understat
    source -- while a column absent from the frame stays neutral."""
    n = 6
    frame = pd.DataFrame({
        "player": [f"p{i}" for i in range(n)],
        "team": ["T"] * n, "league": ["ITA-Serie A"] * n, "season": ["2324"] * n,
        "pos": ["FW"] * n, "minutes": [900] * n, "age": [24] * n,
        "performance_gls": [float(i) for i in range(n)],
        **{c: [float(i) * 0.3 for i in range(n)] for c in UNDERSTAT_COUNTING},
    })
    out = features_onfield.build(frame)
    for column in UNDERSTAT_COUNTING:
        assert out[f"{column}_p90_pct"].nunique() > 1, column
    assert (out["af_tackles_p90_pct"] == 0.5).all()
