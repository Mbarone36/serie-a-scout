import numpy as np
import pandas as pd
import pytest
from src import clubs, presets


def _team_stats(n=4):
    return pd.DataFrame({
        "team": [f"T{i}" for i in range(n)],
        "league": ["ITA-Serie A"] * n,
        "season": ["2223"] * n,
        "possession": [40.0 + i * 5 for i in range(n)],
        "team_passes": [400.0 + i * 50 for i in range(n)],
        "team_deep": [5.0 + i for i in range(n)],
        "team_crs": [15.0 + i for i in range(n)],
        "team_press": [0.07 + i * 0.01 for i in range(n)],
        "team_npxg": [1.0 + i * 0.3 for i in range(n)],
    })


def test_style_vectors_are_percentiles():
    out = clubs.build_style_vectors(_team_stats())
    values = out[[f"{c}_pct" for c in clubs.STYLE_COLUMNS]].to_numpy()
    assert ((values >= 0.0) & (values <= 1.0)).all()


def test_vector_for_returns_correct_length():
    out = clubs.build_style_vectors(_team_stats())
    vector = clubs.vector_for(out, "T0", "2223")
    assert vector.shape == (len(clubs.STYLE_COLUMNS),)


def test_vector_for_raises_on_unknown_club():
    out = clubs.build_style_vectors(_team_stats())
    with pytest.raises(KeyError):
        clubs.vector_for(out, "Nonexistent", "2223")


def test_every_preset_matches_the_club_vector_space():
    for name, vector in presets.PRESETS.items():
        assert vector.shape == (len(clubs.STYLE_COLUMNS),), name
        assert ((vector >= 0.0) & (vector <= 1.0)).all(), name


def test_presets_cover_the_five_named_in_the_spec():
    assert set(presets.PRESETS) == {
        "high-press-possession", "low-block-counter", "direct-vertical",
        "positional-control", "wide-overload",
    }


def test_preset_get_rejects_unknown_name():
    with pytest.raises(KeyError):
        presets.get("tiki-taka")


def _players():
    # Two Roma players (one missing API-Football passes), one Lazio player.
    return pd.DataFrame({
        "league": ["ITA-Serie A"] * 3, "season": ["2223"] * 3,
        "team": ["Roma", "Roma", "Lazio"], "player": ["a", "b", "c"],
        "minutes": [990.0, 990.0, 990.0],
        "af_passes": [500.0, np.nan, 300.0],
        "crs": [10.0, 20.0, 5.0],
    })


def test_profile_rates_use_only_rows_that_report_the_stat():
    team_stats = pd.DataFrame({"team": ["Roma", "Lazio"], "league": ["ITA-Serie A"] * 2,
                               "season": ["2223"] * 2, "possession": [55.0, 45.0]})
    out = clubs.add_profile_columns(team_stats, _players()).set_index("team")
    # Roma: 500 passes over 990 covered minutes -> 500 per 990'; crosses 30 over 1980'.
    assert out.loc["Roma", "team_passes"] == pytest.approx(500.0)
    assert out.loc["Roma", "team_crs"] == pytest.approx(15.0)
    assert np.isnan(out.loc["Roma", "team_npxg"])  # no Understat given


def test_profile_takes_understat_team_stats_through_the_name_map():
    team_stats = pd.DataFrame({"team": ["Milan"], "league": ["ITA-Serie A"],
                               "season": ["2223"], "possession": [52.0]})
    us_teams = pd.DataFrame({"league": ["ITA-Serie A"], "season": ["2223"],
                             "us_team": ["AC Milan"], "matches": [38], "team_npxg": [1.6],
                             "team_deep": [8.0], "team_press": [0.09], "team_xpts": [1.8]})
    name_map = pd.DataFrame({"league": ["ITA-Serie A"], "season": ["2223"],
                             "us_team": ["AC Milan"], "team": ["Milan"]})
    out = clubs.add_profile_columns(team_stats, _players().iloc[0:0], us_teams, name_map)
    assert out.iloc[0]["team_npxg"] == pytest.approx(1.6)
    assert out.iloc[0]["team_press"] == pytest.approx(0.09)


def test_every_style_column_is_available_for_live_era_clubs():
    """The profile must not depend on FBref tables the live pages dropped."""
    for column in clubs.STYLE_COLUMNS:
        assert column in {"possession", "team_passes", "team_deep", "team_crs",
                          "team_press", "team_npxg"}
