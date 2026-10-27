import pandas as pd
import pytest
from src import features_offfield as off


def test_language_family_classifies_romance_nations():
    for nat in ("ESP", "ARG", "BRA", "FRA", "POR", "ROU"):
        assert off.LANGUAGE_FAMILY[nat] == "romance"


def test_language_family_defaults_to_other():
    assert off.LANGUAGE_FAMILY.get("XXX", "other") == "other"


def test_adaptation_flags_first_move_abroad():
    arrivals = pd.DataFrame({
        "player": ["A"], "arrival_season": ["2223"], "to_team": ["Napoli"],
        "from_league": ["GEO-Erovnuli"], "nationality": ["GEO"],
    })
    seasons = pd.DataFrame({
        "player": ["A", "A"], "league": ["GEO-Erovnuli", "GEO-Erovnuli"],
        "season": ["2021", "2122"], "team": ["Batumi", "Batumi"], "minutes": [1800, 2000],
    })
    squads = pd.DataFrame({"team": ["Napoli"], "season": ["2223"], "nationality": ["ITA"], "n": [20]})
    out = off.build_adaptation(arrivals, seasons, squads)
    assert out.loc[0, "first_move_abroad"] == 1.0
    assert out.loc[0, "n_countries"] == 1.0
    assert out.loc[0, "compatriots_in_squad"] == 0.0


def test_adaptation_counts_compatriots():
    arrivals = pd.DataFrame({
        "player": ["A"], "arrival_season": ["2223"], "to_team": ["Napoli"],
        "from_league": ["GEO-Erovnuli"], "nationality": ["GEO"],
    })
    seasons = pd.DataFrame({
        "player": ["A"], "league": ["GEO-Erovnuli"], "season": ["2122"],
        "team": ["Batumi"], "minutes": [2000],
    })
    squads = pd.DataFrame({
        "team": ["Napoli", "Napoli"], "season": ["2223", "2223"],
        "nationality": ["GEO", "ITA"], "n": [2, 18],
    })
    out = off.build_adaptation(arrivals, seasons, squads)
    # n=2 for GEO includes the arriving player's own row; he is excluded.
    assert out.loc[0, "compatriots_in_squad"] == 1.0


def test_trajectory_detects_loan_round_trip():
    arrivals = pd.DataFrame({"player": ["A"], "arrival_season": ["2324"]})
    seasons = pd.DataFrame({
        "player": ["A", "A", "A"],
        "team": ["Home", "Away", "Home"],
        "league": ["ITA-Serie A"] * 3,
        "season": ["2021", "2122", "2223"],
        "minutes": [1000, 1500, 1200],
    })
    out = off.build_trajectory(arrivals, seasons)
    assert out.loc[0, "loan_count"] == 1.0


def test_injury_features_exclude_seasons_after_the_move():
    arrivals = pd.DataFrame({"player": ["A"], "arrival_season": ["2223"]})
    injuries = pd.DataFrame({
        "player": ["A", "A"],
        "season": ["2122", "2324"],
        "days_missed": [30, 200],
        "games_missed": [5, 40],
    })
    out = off.build_injury(arrivals, injuries)
    assert out.loc[0, "injury_days_prior"] == 30.0
    assert out.loc[0, "injury_longest_absence"] == 30.0


def test_availability_separates_missed_while_fit_from_injury():
    arrivals = pd.DataFrame({"player": ["A"], "arrival_season": ["2223"]})
    seasons = pd.DataFrame({
        "player": ["A"], "season": ["2122"], "team": ["X"], "league": ["ITA-Serie A"],
        "minutes": [900], "playing_time_mp": [10], "team_matches": [38],
    })
    injuries = pd.DataFrame({"player": ["A"], "season": ["2122"], "days_missed": [30], "games_missed": [5]})
    out = off.build_availability(arrivals, seasons, injuries)
    assert out.loc[0, "appearance_share"] == pytest.approx(10 / 38)
    assert out.loc[0, "fixtures_missed_while_fit"] == 23.0


def test_value_uses_most_recent_valuation_before_the_move():
    arrivals = pd.DataFrame({"player": ["A"], "arrival_season": ["2223"]})
    values = pd.DataFrame({
        "player": ["A", "A", "A"],
        "date": pd.to_datetime(["2020-07-01", "2022-05-01", "2023-01-01"]),
        "value_eur": [1e6, 8e6, 40e6],
    })
    out = off.build_value(arrivals, values)
    assert out.loc[0, "value_at_move"] == 8e6
    assert out.loc[0, "value_trend_2y"] > 1.0


def test_bundles_partition_all_off_field_columns():
    all_columns = [c for cols in off.BUNDLES.values() for c in cols]
    assert len(all_columns) == len(set(all_columns))


def test_prior_handles_int_season_dtype():
    """A season column read back from parquet may come back as int64, not str."""
    seasons = pd.DataFrame({
        "player": ["A", "A", "A"],
        "season": [2021, 2122, 2223],
        "team": ["X", "X", "X"],
        "league": ["ITA-Serie A"] * 3,
        "minutes": [1000, 1200, 1300],
    })
    out = off._prior(seasons, "A", "2324")
    assert sorted(out["season"].tolist()) == [2021, 2122, 2223]

    out_partial = off._prior(seasons, "A", "2223")
    assert sorted(out_partial["season"].tolist()) == [2021, 2122]


def test_availability_ignores_nan_team_matches_in_sum():
    """A NaN team_matches on one prior row must not zero the availability features."""
    arrivals = pd.DataFrame({"player": ["A"], "arrival_season": ["2223"]})
    seasons = pd.DataFrame({
        "player": ["A", "A"],
        "season": ["2021", "2122"],
        "team": ["X", "X"],
        "league": ["ITA-Serie A"] * 2,
        "minutes": [900, 900],
        "playing_time_mp": [10, 10],
        "team_matches": [38.0, float("nan")],
    })
    # The player IS in the injury source (with a post-cutoff row only), so
    # `fixtures_missed_while_fit` is a measured zero rather than missing data.
    injuries = pd.DataFrame({"player": ["A"], "season": ["2324"],
                              "days_missed": [10.0], "games_missed": [2.0]})
    out = off.build_availability(arrivals, seasons, injuries)
    assert out.loc[0, "appearance_share"] == pytest.approx(20 / 38)
    assert out.loc[0, "fixtures_missed_total"] == 18.0
    assert out.loc[0, "fixtures_missed_while_fit"] == 18.0


def test_bundles_cover_every_produced_feature_column():
    arrivals = pd.DataFrame({
        "player": ["A"], "arrival_season": ["2223"], "to_team": ["Napoli"],
        "from_league": ["GEO-Erovnuli"], "nationality": ["GEO"],
    })
    seasons = pd.DataFrame({
        "player": ["A", "A"], "league": ["GEO-Erovnuli", "GEO-Erovnuli"],
        "season": ["2021", "2122"], "team": ["Batumi", "Batumi"], "minutes": [1800, 2000],
        "playing_time_mp": [20, 22], "team_matches": [30, 32],
    })
    squads = pd.DataFrame({"team": ["Napoli"], "season": ["2223"], "nationality": ["ITA"], "n": [20]})
    injuries = pd.DataFrame({
        "player": ["A"], "season": ["2122"], "days_missed": [30], "games_missed": [5],
    })
    values = pd.DataFrame({
        "player": ["A"], "date": pd.to_datetime(["2022-05-01"]), "value_eur": [8e6],
    })

    produced = set()
    for frame in (
        off.build_adaptation(arrivals, seasons, squads),
        off.build_trajectory(arrivals, seasons),
        off.build_injury(arrivals, injuries),
        off.build_availability(arrivals, seasons, injuries),
        off.build_value(arrivals, values),
    ):
        produced |= set(frame.columns) - {"player", "arrival_season"}

    bundled = {c for cols in off.BUNDLES.values() for c in cols}
    assert produced == bundled


# ---------------------------------------------------------------------------
# Finding #4: absent from the injury source is missing data, not durability
# ---------------------------------------------------------------------------


def test_injury_features_are_nan_for_a_player_absent_from_the_injury_source():
    """The spec's own Known Limitations: "Absence of injury data is not evidence
    of durability." A player Transfermarkt has never been scraped for must not
    read as an ever-fit player with zero days missed."""
    arrivals = pd.DataFrame({"player": ["A"], "arrival_season": ["2223"]})
    injuries = pd.DataFrame({
        "player": ["SomeoneElse"], "season": ["2122"],
        "days_missed": [30.0], "games_missed": [5.0],
    })

    out = off.build_injury(arrivals, injuries)

    assert out.loc[0, ["injury_days_prior", "injury_spells_prior",
                        "injury_longest_absence"]].isna().all()


def test_injury_features_are_zero_when_the_player_is_known_but_never_injured():
    """The other side of the same distinction: rows exist for this player, just
    none before the move. That is a measured zero, and the model should see 0.0
    rather than have it median-filled away."""
    arrivals = pd.DataFrame({"player": ["A"], "arrival_season": ["2223"]})
    injuries = pd.DataFrame({
        "player": ["A"], "season": ["2324"],
        "days_missed": [30.0], "games_missed": [5.0],
    })

    out = off.build_injury(arrivals, injuries)

    assert out.loc[0, "injury_days_prior"] == 0.0
    assert out.loc[0, "injury_spells_prior"] == 0.0
    assert out.loc[0, "injury_longest_absence"] == 0.0


def test_injury_features_are_all_nan_when_the_injury_frame_is_empty():
    """The bulk Transfermarkt pull may not have run. Nothing was measured, so
    nothing is reported — the pipeline's median fill takes it from there."""
    arrivals = pd.DataFrame({"player": ["A", "B"], "arrival_season": ["2223", "2223"]})
    injuries = pd.DataFrame(columns=["player", "season", "days_missed", "games_missed"])

    out = off.build_injury(arrivals, injuries)

    assert out[["injury_days_prior", "injury_spells_prior",
                 "injury_longest_absence"]].isna().all().all()


def test_missed_while_fit_is_nan_for_a_player_absent_from_the_injury_source():
    """`fixtures_missed_while_fit` is the cross of two sources. Without injury
    records every missed fixture would be blamed on being out of favour — the
    exact confusion the feature exists to resolve."""
    arrivals = pd.DataFrame({"player": ["A"], "arrival_season": ["2223"]})
    seasons = pd.DataFrame({
        "player": ["A"], "season": ["2122"], "team": ["X"], "league": ["ITA-Serie A"],
        "minutes": [900], "playing_time_mp": [10], "team_matches": [38],
    })
    injuries = pd.DataFrame({
        "player": ["SomeoneElse"], "season": ["2122"],
        "days_missed": [30.0], "games_missed": [5.0],
    })

    out = off.build_availability(arrivals, seasons, injuries)

    assert pd.isna(out.loc[0, "fixtures_missed_while_fit"])
    # FBref alone still yields the total — only the crossed feature is unknown.
    assert out.loc[0, "fixtures_missed_total"] == 28.0
