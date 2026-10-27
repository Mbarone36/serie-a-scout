import pandas as pd
import pytest

from src import labels


@pytest.fixture(autouse=True)
def _rank_tiny_groups(monkeypatch):
    """Most tests here rank 2-4 player groups; disable the tiny-group neutral
    rule for them (the dedicated MIN_RANK_GROUP tests restore the real value)."""
    monkeypatch.setattr(labels, "MIN_RANK_GROUP", 1)


def test_window_is_two_seasons_after_arrival():
    assert labels.window_seasons("2223") == ["2223", "2324"]


def test_minutes_target_is_share_of_available():
    value = labels.minutes_target(minutes_played=3420, team_matches=76)
    assert value == pytest.approx(0.5)


def test_minutes_target_clamps_at_one():
    assert labels.minutes_target(minutes_played=99999, team_matches=76) == 1.0


def test_minutes_outside_serie_a_do_not_count():
    seasons = pd.DataFrame({
        "player": ["A", "A"],
        "season": ["2223", "2324"],
        "league": ["ITA-Serie A", "ENG-Premier League"],
        "team": ["Napoli", "Chelsea"],
        "minutes": [2000, 3000],
        "team_matches": [38, 38],
    })
    played = labels.serie_a_minutes(seasons, "A", ["2223", "2324"])
    assert played == 2000


def test_retention_stayed_at_signing_club():
    assert labels.retention_target(
        end_team="Napoli", signing_team="Napoli", end_league="ITA-Serie A",
        fee_in=10e6, fee_out=None, club_value_from=500e6, club_value_to=None,
    ) == 1.0


def test_retention_step_up_exit_scores_full_marks():
    # Kvaratskhelia: Napoli -> PSG at a profit. Must not read as failure.
    assert labels.retention_target(
        end_team="Paris Saint-Germain", signing_team="Napoli", end_league="FRA-Ligue 1",
        fee_in=10e6, fee_out=70e6, club_value_from=500e6, club_value_to=1100e6,
    ) == 1.0


def test_retention_move_within_serie_a():
    assert labels.retention_target(
        end_team="Torino", signing_team="Napoli", end_league="ITA-Serie A",
        fee_in=10e6, fee_out=8e6, club_value_from=500e6, club_value_to=200e6,
    ) == pytest.approx(0.66, abs=0.01)


def test_retention_lateral_exit_scores_partial():
    assert labels.retention_target(
        end_team="Fenerbahce", signing_team="Napoli", end_league="TUR-Super Lig",
        fee_in=10e6, fee_out=10e6, club_value_from=500e6, club_value_to=200e6,
    ) == pytest.approx(0.33, abs=0.01)


def test_retention_sold_at_a_loss_to_lesser_club_is_zero():
    assert labels.retention_target(
        end_team="Salernitana B", signing_team="Napoli", end_league="ESP-Segunda",
        fee_in=30e6, fee_out=3e6, club_value_from=500e6, club_value_to=40e6,
    ) == 0.0


def test_retention_free_exit_is_zero_regardless_of_destination():
    assert labels.retention_target(
        end_team="Real Madrid", signing_team="Napoli", end_league="ESP-La Liga",
        fee_in=30e6, fee_out=0.0, club_value_from=500e6, club_value_to=1200e6,
    ) == 0.0


def test_post_output_ranks_attackers_against_attackers():
    arrivals = pd.DataFrame({
        "player": ["A", "B"], "arrival_season": ["2223", "2223"],
    })
    seasons = pd.DataFrame({
        "player": ["A", "B"], "season": ["2223", "2223"],
        "league": ["ITA-Serie A", "ITA-Serie A"], "pos": ["FW", "FW"],
        "minutes": [900, 900],
        "expected_npxg": [10.0, 1.0], "expected_xag": [5.0, 1.0],
        "progression_prgp": [0.0, 0.0], "progression_prgc": [0.0, 0.0],
    })
    out = labels.post_output_percentile(seasons, arrivals)
    a = out.loc[out.player == "A", "post_output_pct"].item()
    b = out.loc[out.player == "B", "post_output_pct"].item()
    assert a > b


def test_post_output_is_zero_when_player_never_played_in_serie_a():
    arrivals = pd.DataFrame({"player": ["A"], "arrival_season": ["2223"]})
    seasons = pd.DataFrame(columns=["player", "season", "league", "pos", "minutes",
                                    "expected_npxg", "expected_xag",
                                    "progression_prgp", "progression_prgc"])
    out = labels.post_output_percentile(seasons, arrivals)
    assert out["post_output_pct"].iloc[0] == 0.0


def test_post_output_midfielder_uses_xgchain():
    # A has weaker npxG+xA but more xGChain than B: if A ranks ahead, xGChain
    # (not xG) was used. Progressive stats are ignored even when present.
    arrivals = pd.DataFrame({
        "player": ["A", "B"], "arrival_season": ["2223", "2223"],
    })
    seasons = pd.DataFrame({
        "player": ["A", "B"], "season": ["2223", "2223"],
        "league": ["ITA-Serie A", "ITA-Serie A"], "pos": ["MF", "MF"],
        "minutes": [900, 900],
        "us_npxg": [1.0, 10.0], "us_xa": [1.0, 10.0],
        "us_xgchain": [12.0, 3.0],
        "progression_prgp": [1.0, 10.0], "progression_prgc": [1.0, 10.0],
    })
    out = labels.post_output_percentile(seasons, arrivals)
    a = out.loc[out.player == "A", "post_output_pct"].item()
    b = out.loc[out.player == "B", "post_output_pct"].item()
    assert a > b
    assert not out["output_fallback"].any()


def test_post_output_attacker_with_xg_uses_xg_basis():
    arrivals = pd.DataFrame({"player": ["A"], "arrival_season": ["2223"]})
    seasons = pd.DataFrame({
        "player": ["A"], "season": ["2223"], "league": ["ITA-Serie A"],
        "pos": ["FW"], "minutes": [900],
        "expected_npxg": [10.0], "expected_xag": [5.0],
        "performance_gls": [1.0], "performance_ast": [0.0],
    })
    out = labels.post_output_percentile(seasons, arrivals)
    assert out.loc[0, "output_basis"] == "xg"
    assert not out.loc[0, "output_fallback"]


def test_post_output_attacker_without_xg_falls_back_to_goals_and_assists():
    # Two attackers with no npxG/xAG signal at all (the live-scrape era):
    # they must rank against each other on goals+assists, not xG.
    arrivals = pd.DataFrame({
        "player": ["A", "B"], "arrival_season": ["2223", "2223"],
    })
    seasons = pd.DataFrame({
        "player": ["A", "B"], "season": ["2223", "2223"],
        "league": ["ITA-Serie A", "ITA-Serie A"], "pos": ["FW", "FW"],
        "minutes": [900, 900],
        "performance_gls": [10.0, 1.0], "performance_ast": [5.0, 1.0],
    })
    out = labels.post_output_percentile(seasons, arrivals)
    assert (out["output_basis"] == "ga").all()
    assert out["output_fallback"].all()
    a = out.loc[out.player == "A", "post_output_pct"].item()
    b = out.loc[out.player == "B", "post_output_pct"].item()
    assert a > b


def test_post_output_ga_attackers_ranked_only_among_ga_attackers():
    # A "xg" attacker with a middling xG figure must not be compared against
    # a "ga" attacker with huge raw goals -- they are on different scales and
    # must be ranked within their own (position_group, output_basis) group.
    arrivals = pd.DataFrame({
        "player": ["xg_hi", "ga_lo", "ga_hi"],
        "arrival_season": ["2223", "2223", "2223"],
    })
    seasons = pd.DataFrame({
        "player": ["xg_hi", "ga_lo", "ga_hi"],
        "season": ["2223", "2223", "2223"],
        "league": ["ITA-Serie A"] * 3, "pos": ["FW"] * 3,
        "minutes": [900, 900, 900],
        "expected_npxg": [5.0, None, None], "expected_xag": [5.0, None, None],
        "performance_gls": [0.0, 1.0, 20.0], "performance_ast": [0.0, 0.0, 10.0],
    })
    out = labels.post_output_percentile(seasons, arrivals)
    assert out.loc[out.player == "xg_hi", "output_basis"].item() == "xg"
    assert out.loc[out.player == "xg_hi", "post_output_pct"].item() == 1.0  # sole "xg" attacker: single-member rank
    ga_lo = out.loc[out.player == "ga_lo", "post_output_pct"].item()
    ga_hi = out.loc[out.player == "ga_hi", "post_output_pct"].item()
    assert ga_hi > ga_lo


def test_post_output_midfielder_with_neither_prog_nor_xg_falls_back_to_ga():
    arrivals = pd.DataFrame({"player": ["A"], "arrival_season": ["2223"]})
    seasons = pd.DataFrame({
        "player": ["A"], "season": ["2223"], "league": ["ITA-Serie A"],
        "pos": ["MF"], "minutes": [900],
        "performance_gls": [3.0], "performance_ast": [4.0],
    })
    out = labels.post_output_percentile(seasons, arrivals)
    assert out.loc[0, "output_basis"] == "ga"
    assert out.loc[0, "output_fallback"]


def test_post_output_no_window_rows_yields_none_basis():
    arrivals = pd.DataFrame({"player": ["A"], "arrival_season": ["2223"]})
    seasons = pd.DataFrame(columns=["player", "season", "league", "pos", "minutes"])
    out = labels.post_output_percentile(seasons, arrivals)
    assert out.loc[0, "output_basis"] == "none"
    assert out.loc[0, "post_output_pct"] == 0.0


def test_post_output_midfielder_never_falls_back_to_xg():
    # No xGChain (unmatched in Understat): goals + assists, not xG -- an xG
    # fallback would judge the same position on a different yardstick.
    arrivals = pd.DataFrame({
        "player": ["A", "B"], "arrival_season": ["2223", "2223"],
    })
    seasons = pd.DataFrame({
        "player": ["A", "B"], "season": ["2223", "2223"],
        "league": ["ITA-Serie A", "ITA-Serie A"], "pos": ["MF", "MF"],
        "minutes": [900, 900],
        "expected_npxg": [10.0, 1.0], "expected_xag": [10.0, 1.0],
        "performance_gls": [1.0, 4.0], "performance_ast": [0.0, 2.0],
    })
    out = labels.post_output_percentile(seasons, arrivals)
    assert (out["output_basis"] == "ga").all()
    a = out.loc[out.player == "A", "post_output_pct"].item()
    b = out.loc[out.player == "B", "post_output_pct"].item()
    assert b > a
    assert out["output_fallback"].all()


def test_post_output_keeper_who_played_is_neutral_and_one_who_did_not_is_zero():
    arrivals = pd.DataFrame({"player": ["K", "J"], "arrival_season": ["2223", "2223"]})
    seasons = pd.DataFrame({
        "player": ["K"], "season": ["2223"], "league": ["ITA-Serie A"],
        "pos": ["GK"], "minutes": [3000], "us_xgchain": [0.4],
    })
    out = labels.post_output_percentile(seasons, arrivals).set_index("player")
    assert out.loc["K", "output_basis"] == "keeper"
    assert out.loc["K", "post_output_pct"] == 0.5
    assert out.loc["J", "post_output_pct"] == 0.0


# ---------------------------------------------------------------------------
# Review finding (commit d48ca6e): per-90 rate must use only the window rows
# that carry the basis columns, not the whole window's minutes.
# ---------------------------------------------------------------------------


def test_post_output_xg_rate_uses_only_the_season_that_has_xg():
    # A arrives 2223: xG is present in the 2223 row only (2324 comes from the
    # no-xG live scraper). B arrives 2223 too, with the SAME per-90 xG rate in
    # season 2223 but xG present in BOTH seasons. If the denominator wrongly
    # used the whole two-season window for A, A's rate would be silently
    # halved relative to B even though both players produced identically in
    # the one season that has data.
    arrivals = pd.DataFrame({
        "player": ["A", "B"], "arrival_season": ["2223", "2223"],
    })
    seasons = pd.DataFrame({
        "player": ["A", "A", "B", "B"],
        "season": ["2223", "2324", "2223", "2324"],
        "league": ["ITA-Serie A"] * 4, "pos": ["FW"] * 4,
        "minutes": [900, 900, 900, 900],
        "expected_npxg": [5.0, None, 5.0, 5.0],
        "expected_xag": [4.0, None, 4.0, 4.0],
    })
    out = labels.post_output_percentile(seasons, arrivals)
    a = out.loc[out.player == "A"].iloc[0]
    b = out.loc[out.player == "B"].iloc[0]

    assert a["output_basis"] == "xg"
    assert b["output_basis"] == "xg"
    assert a["output_rows_used"] == 1
    assert b["output_rows_used"] == 2
    assert a["output_window_share"] == pytest.approx(0.5)
    assert b["output_window_share"] == pytest.approx(1.0)
    # Same per-season rate in both cases (9 per 900 minutes, per 90) -> the
    # windowed rate must come out equal, not halved for A.
    assert a["post_output_pct"] == pytest.approx(b["post_output_pct"])


def test_post_output_falls_to_ga_when_the_only_basis_row_has_zero_minutes():
    # The only row carrying progression data logged zero minutes (a bench
    # cameo/unused-window row); the other row has real minutes but no
    # progression or xG data at all. Masked minutes for "prog" is 0, so the
    # basis must fall through to "xg" (unusable: no xG columns present) and
    # then to "ga".
    arrivals = pd.DataFrame({"player": ["A"], "arrival_season": ["2223"]})
    seasons = pd.DataFrame({
        "player": ["A", "A"], "season": ["2223", "2324"],
        "league": ["ITA-Serie A", "ITA-Serie A"], "pos": ["MF", "MF"],
        "minutes": [0, 900],
        "progression_prgp": [5.0, None], "progression_prgc": [5.0, None],
        "performance_gls": [None, 2.0], "performance_ast": [None, 1.0],
    })
    out = labels.post_output_percentile(seasons, arrivals)
    row = out.iloc[0]
    assert row["output_basis"] == "ga"
    assert row["output_rows_used"] == 1
    assert row["output_window_share"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Round 1 fixes: loan-aware retention (Finding A)
# ---------------------------------------------------------------------------


def test_retention_loan_out_of_serie_a_is_zero():
    # A loan is not a sale, even with a disclosed loan fee -- a departing
    # loan out of the league scores like any other non-sale exit.
    assert labels.retention_target(
        end_team="Chelsea", signing_team="Napoli", end_league="ENG-Premier League",
        fee_in=10e6, fee_out=2e6, club_value_from=500e6, club_value_to=1200e6,
        is_loan=True,
    ) == 0.0


def test_retention_loan_within_serie_a_scores_like_a_league_move():
    assert labels.retention_target(
        end_team="Torino", signing_team="Napoli", end_league="ITA-Serie A",
        fee_in=10e6, fee_out=1e6, club_value_from=500e6, club_value_to=200e6,
        is_loan=True,
    ) == pytest.approx(0.66, abs=0.01)


# ---------------------------------------------------------------------------
# Round 2 fix: is_loan must be checked before the free-exit rule (Finding)
# ---------------------------------------------------------------------------


def test_retention_zero_fee_loan_within_serie_a_is_moved_within_league():
    # Most real loans carry no disclosed fee ("loan transfer" -> fee_eur 0.0).
    # If left_for_free ran before is_loan, this would wrongly score 0.0.
    assert labels.retention_target(
        end_team="Torino", signing_team="Napoli", end_league="ITA-Serie A",
        fee_in=10e6, fee_out=0.0, club_value_from=500e6, club_value_to=200e6,
        is_loan=True,
    ) == pytest.approx(0.66, abs=0.01)


def test_retention_zero_fee_loan_out_of_serie_a_is_zero():
    assert labels.retention_target(
        end_team="Chelsea", signing_team="Napoli", end_league="ENG-Premier League",
        fee_in=10e6, fee_out=0.0, club_value_from=500e6, club_value_to=1200e6,
        is_loan=True,
    ) == 0.0


def test_retention_zero_fee_permanent_free_transfer_out_is_still_zero():
    # Same zero fee_out, but NOT a loan: a genuine free transfer out of
    # Serie A must still hit the free-exit rule and score 0.0.
    assert labels.retention_target(
        end_team="Chelsea", signing_team="Napoli", end_league="ENG-Premier League",
        fee_in=10e6, fee_out=0.0, club_value_from=500e6, club_value_to=1200e6,
        is_loan=False,
    ) == 0.0


def test_retention_kvaratskhelia_case_still_scores_full_marks_after_round_2():
    assert labels.retention_target(
        end_team="Paris Saint-Germain", signing_team="Napoli", end_league="FRA-Ligue 1",
        fee_in=10e6, fee_out=70e6, club_value_from=500e6, club_value_to=1100e6,
    ) == 1.0


# ---------------------------------------------------------------------------
# Round 1 fixes: strict step-up profit and NaN-safe fees (Finding B)
# ---------------------------------------------------------------------------


def test_retention_break_even_sale_to_bigger_club_is_lateral_not_step_up():
    # Breaking even is not PROFIT -- the 1.0 tier requires a strict gain.
    assert labels.retention_target(
        end_team="Bayern Munich", signing_team="Napoli", end_league="GER-Bundesliga",
        fee_in=10e6, fee_out=10e6, club_value_from=500e6, club_value_to=900e6,
    ) == pytest.approx(0.33, abs=0.01)


def test_retention_step_up_with_unknown_inbound_fee_scores_full_marks():
    # fee_in undisclosed (NaN): profit is unknowable, so a bigger-or-equal
    # destination club alone earns the top rung.
    assert labels.retention_target(
        end_team="Paris Saint-Germain", signing_team="Napoli", end_league="FRA-Ligue 1",
        fee_in=float("nan"), fee_out=70e6, club_value_from=500e6, club_value_to=1100e6,
    ) == 1.0


def test_retention_lateral_with_unknown_inbound_fee_to_smaller_club():
    assert labels.retention_target(
        end_team="Fenerbahce", signing_team="Napoli", end_league="TUR-Super Lig",
        fee_in=float("nan"), fee_out=10e6, club_value_from=500e6, club_value_to=200e6,
    ) == pytest.approx(0.33, abs=0.01)


def test_retention_step_up_exit_still_scores_full_marks_after_round_1_fixes():
    # The Kvaratskhelia case must never regress: strict profit + bigger club.
    assert labels.retention_target(
        end_team="Paris Saint-Germain", signing_team="Napoli", end_league="FRA-Ligue 1",
        fee_in=10e6, fee_out=70e6, club_value_from=500e6, club_value_to=1100e6,
    ) == 1.0


# ---------------------------------------------------------------------------
# Round 1 fixes: position group by most-played row (Finding C)
# ---------------------------------------------------------------------------


def test_post_output_position_group_uses_most_played_row():
    # A brief FW cameo (200 min) followed by a season as a MF (1500 min):
    # the position group must be driven by where he actually played, not
    # whichever row happens to come first.
    arrivals = pd.DataFrame({"player": ["A"], "arrival_season": ["2223"]})
    seasons = pd.DataFrame({
        "player": ["A", "A"],
        "season": ["2223", "2324"],
        "league": ["ITA-Serie A", "ITA-Serie A"],
        "pos": ["FW", "MF"],
        "minutes": [200, 1500],
        "expected_npxg": [1.0, 1.0],
        "expected_xag": [1.0, 1.0],
        "progression_prgp": [1.0, 20.0],
        "progression_prgc": [1.0, 20.0],
    })
    out = labels.post_output_percentile(seasons, arrivals)
    assert out.loc[out.player == "A", "position_group"].item() == "midfielder"


# ---------------------------------------------------------------------------
# Finding #3: an unobserved exit is the bottom rung, not a lateral move
# ---------------------------------------------------------------------------


def _build_inputs(fees):
    """One arrival with a first-window-season row and NO final-window row —
    the player vanishes from the covered leagues."""
    arrivals = pd.DataFrame({
        "player": ["A"], "arrival_season": ["2223"], "to_team": ["Napoli"],
    })
    seasons = pd.DataFrame({
        "player": ["A"], "season": ["2223"], "league": ["ITA-Serie A"],
        "team": ["Napoli"], "pos": ["FW"], "minutes": [900.0], "team_matches": [38.0],
        "expected_npxg": [3.0], "expected_xag": [2.0],
        "progression_prgp": [10.0], "progression_prgc": [10.0],
    })
    club_values = pd.DataFrame({"team": ["Napoli"], "squad_value_eur": [500e6]})
    return arrivals, seasons, club_values, fees


def test_build_scores_an_unobserved_exit_with_unknown_fees_as_zero():
    """Previously this landed on 0.33 via the "profit unknowable, judge on club
    size" fallback, with `end_team=""` and no club value — which is why 609 of
    1284 rows sat at 0.33 and not one row reached the ladder's bottom rung."""
    fees = pd.DataFrame({"player": ["A"], "season": ["2223"],
                          "fee_eur": [float("nan")], "is_loan": [False]})
    out = labels.build(*_build_inputs(fees))
    assert out.loc[0, "retention"] == 0.0


def test_build_scores_an_unobserved_exit_with_a_known_profitable_sale_as_lateral():
    """Positive evidence to the contrary: the club banked money on him and he
    moved somewhere the covered leagues do not see. That is 0.33, not failure."""
    fees = pd.DataFrame({
        "player": ["A", "A"], "season": ["2223", "2425"],
        "fee_eur": [5e6, 20e6], "is_loan": [False, False],
    })
    out = labels.build(*_build_inputs(fees))
    assert out.loc[0, "retention"] == pytest.approx(0.33)


def test_unobserved_exit_target_ladder():
    assert labels.unobserved_exit_target(None, None) == 0.0
    assert labels.unobserved_exit_target(float("nan"), float("nan")) == 0.0
    # a known LOSS is still the bottom rung
    assert labels.unobserved_exit_target(20e6, 5e6) == 0.0
    # break-even is not profit
    assert labels.unobserved_exit_target(10e6, 10e6) == 0.0
    assert labels.unobserved_exit_target(5e6, 20e6) == pytest.approx(0.33)


def test_build_end_team_is_the_most_played_club_of_the_final_window_season():
    """Finding #11: `after.iloc[0]` took whichever club row sat first on disk.
    Here the player's final-window season is split — a 200-minute cameo back at
    the signing club, then 1500 minutes elsewhere. Row order says he stayed;
    minutes say he left."""
    arrivals = pd.DataFrame({
        "player": ["A"], "arrival_season": ["2223"], "to_team": ["Napoli"],
    })
    seasons = pd.DataFrame({
        "player": ["A", "A"], "season": ["2324", "2324"],
        "league": ["ITA-Serie A", "ITA-Serie A"],
        "team": ["Napoli", "Torino"], "pos": ["FW", "FW"],
        "minutes": [200.0, 1500.0], "team_matches": [38.0, 38.0],
        "expected_npxg": [1.0, 3.0], "expected_xag": [1.0, 2.0],
        "progression_prgp": [5.0, 10.0], "progression_prgc": [5.0, 10.0],
    })
    club_values = pd.DataFrame({
        "team": ["Napoli", "Torino"], "squad_value_eur": [500e6, 200e6],
    })
    fees = pd.DataFrame(columns=["player", "season", "fee_eur", "is_loan"])

    out = labels.build(arrivals, seasons, club_values, fees)

    # Torino, not Napoli: a permanent move within Serie A, not "stayed put".
    assert out.loc[0, "retention"] == pytest.approx(0.66)


# ---------------------------------------------------------------------------
# Understat xG: preferred over FBref xG for the "xg" basis when present.
# ---------------------------------------------------------------------------


def test_post_output_xg_prefers_understat_over_fbref():
    # FBref xG ranks A above B; Understat ranks B above A. With both present
    # the Understat figures must decide the ranking.
    arrivals = pd.DataFrame({"player": ["A", "B"], "arrival_season": ["2223", "2223"]})
    seasons = pd.DataFrame({
        "player": ["A", "B"], "season": ["2223", "2223"],
        "league": ["ITA-Serie A"] * 2, "pos": ["FW", "FW"],
        "minutes": [900, 900],
        "expected_npxg": [10.0, 1.0], "expected_xag": [5.0, 1.0],
        "us_npxg": [1.0, 10.0], "us_xa": [1.0, 5.0],
    })
    out = labels.post_output_percentile(seasons, arrivals)
    assert (out["output_basis"] == "xg").all()
    assert (out["output_xg_source"] == "understat").all()
    a = out.loc[out.player == "A", "post_output_pct"].item()
    b = out.loc[out.player == "B", "post_output_pct"].item()
    assert b > a


def test_post_output_xg_falls_back_to_fbref_when_understat_absent():
    arrivals = pd.DataFrame({"player": ["A", "B"], "arrival_season": ["2223", "2223"]})
    seasons = pd.DataFrame({
        "player": ["A", "B"], "season": ["2223", "2223"],
        "league": ["ITA-Serie A"] * 2, "pos": ["FW", "FW"],
        "minutes": [900, 900],
        "expected_npxg": [10.0, 1.0], "expected_xag": [5.0, 1.0],
        "us_npxg": [None, None], "us_xa": [None, None],
    })
    out = labels.post_output_percentile(seasons, arrivals)
    assert (out["output_basis"] == "xg").all()
    assert (out["output_xg_source"] == "fbref").all()
    a = out.loc[out.player == "A", "post_output_pct"].item()
    b = out.loc[out.player == "B", "post_output_pct"].item()
    assert a > b


def test_post_output_understat_rescues_the_no_fbref_xg_era():
    # A live-era row (no FBref xG) with Understat present ranks on "xg", not "ga".
    arrivals = pd.DataFrame({"player": ["A"], "arrival_season": ["2324"]})
    seasons = pd.DataFrame({
        "player": ["A"], "season": ["2324"], "league": ["ITA-Serie A"],
        "pos": ["FW"], "minutes": [900],
        "us_npxg": [4.0], "us_xa": [2.0],
        "performance_gls": [5.0], "performance_ast": [1.0],
    })
    out = labels.post_output_percentile(seasons, arrivals)
    assert out.loc[0, "output_basis"] == "xg"
    assert out.loc[0, "output_xg_source"] == "understat"


def test_post_output_understat_rate_masks_to_rows_carrying_it():
    # Understat present for one of two window seasons: the rate uses only
    # that season's minutes (same masking rule as FBref xG).
    arrivals = pd.DataFrame({"player": ["A"], "arrival_season": ["2223"]})
    seasons = pd.DataFrame({
        "player": ["A", "A"], "season": ["2223", "2324"],
        "league": ["ITA-Serie A"] * 2, "pos": ["FW", "FW"],
        "minutes": [900, 900],
        "us_npxg": [5.0, None], "us_xa": [4.0, None],
        "expected_npxg": [3.0, None], "expected_xag": [3.0, None],
    })
    out = labels.post_output_percentile(seasons, arrivals)
    assert out.loc[0, "output_xg_source"] == "understat"  # equal coverage: tie
    assert out.loc[0, "output_rows_used"] == 1
    assert out.loc[0, "output_window_share"] == pytest.approx(0.5)


def test_post_output_xg_source_empty_for_non_xg_bases():
    arrivals = pd.DataFrame({"player": ["M", "G", "N"],
                             "arrival_season": ["2223", "2223", "2223"]})
    seasons = pd.DataFrame({
        "player": ["M", "G"], "season": ["2223", "2223"],
        "league": ["ITA-Serie A"] * 2, "pos": ["MF", "FW"],
        "minutes": [900, 900],
        "us_xgchain": [10.0, None],
        "us_npxg": [1.0, None], "us_xa": [1.0, None],
        "performance_gls": [1.0, 3.0], "performance_ast": [1.0, 1.0],
    })
    out = labels.post_output_percentile(seasons, arrivals).set_index("player")
    assert out.loc["M", "output_basis"] == "chain"
    assert out.loc["G", "output_basis"] == "ga"
    assert out.loc["N", "output_basis"] == "none"
    assert (out["output_xg_source"] == "").all()


def test_build_carries_output_xg_source():
    arrivals = pd.DataFrame({"player": ["A"], "arrival_season": ["2223"], "to_team": ["Napoli"]})
    seasons = pd.DataFrame({
        "player": ["A", "A"], "season": ["2223", "2324"],
        "league": ["ITA-Serie A"] * 2, "team": ["Napoli", "Napoli"],
        "pos": ["FW", "FW"], "minutes": [900.0, 900.0], "team_matches": [38.0, 38.0],
        "us_npxg": [3.0, 3.0], "us_xa": [1.0, 1.0],
    })
    club_values = pd.DataFrame(columns=["team", "squad_value_eur"])
    fees = pd.DataFrame(columns=["player", "season", "fee_eur", "is_loan"])
    out = labels.build(arrivals, seasons, club_values, fees)
    assert out.loc[0, "output_xg_source"] == "understat"


def test_post_output_xg_picks_source_with_greater_window_coverage():
    # Understat covers 1 of the 2 window seasons, FBref covers both: FBref
    # observes more of the window, so it is the source (never summed).
    arrivals = pd.DataFrame({"player": ["A"], "arrival_season": ["2223"]})
    seasons = pd.DataFrame({
        "player": ["A", "A"], "season": ["2223", "2324"],
        "league": ["ITA-Serie A"] * 2, "pos": ["FW", "FW"],
        "minutes": [900, 900],
        "us_npxg": [5.0, None], "us_xa": [4.0, None],
        "expected_npxg": [3.0, 3.0], "expected_xag": [1.0, 1.0],
    })
    out = labels.post_output_percentile(seasons, arrivals)
    assert out.loc[0, "output_xg_source"] == "fbref"
    assert out.loc[0, "output_rows_used"] == 2
    assert out.loc[0, "output_window_share"] == pytest.approx(1.0)


def test_post_output_xg_equal_coverage_prefers_understat():
    arrivals = pd.DataFrame({"player": ["A"], "arrival_season": ["2223"]})
    seasons = pd.DataFrame({
        "player": ["A", "A"], "season": ["2223", "2324"],
        "league": ["ITA-Serie A"] * 2, "pos": ["FW", "FW"],
        "minutes": [900, 900],
        "us_npxg": [5.0, 5.0], "us_xa": [4.0, 4.0],
        "expected_npxg": [3.0, 3.0], "expected_xag": [1.0, 1.0],
    })
    out = labels.post_output_percentile(seasons, arrivals)
    assert out.loc[0, "output_xg_source"] == "understat"


def test_post_output_percentiles_are_ranked_within_xg_source():
    # U1/U2 are Understat; F1/F2 are FBref (a different xG model). Each pair
    # is ranked only against its own source: top of each pair is 1.0, not
    # spread over a 4-way ranking.
    arrivals = pd.DataFrame({"player": ["U1", "U2", "F1", "F2"],
                             "arrival_season": ["2223"] * 4})
    seasons = pd.DataFrame({
        "player": ["U1", "U2", "F1", "F2"], "season": ["2223"] * 4,
        "league": ["ITA-Serie A"] * 4, "pos": ["FW"] * 4,
        "minutes": [900] * 4,
        "us_npxg": [1.0, 2.0, None, None], "us_xa": [0.0, 0.0, None, None],
        "expected_npxg": [None, None, 10.0, 20.0], "expected_xag": [None, None, 0.0, 0.0],
    })
    out = labels.post_output_percentile(seasons, arrivals).set_index("player")
    assert out.loc[["U1", "U2"], "output_xg_source"].eq("understat").all()
    assert out.loc[["F1", "F2"], "output_xg_source"].eq("fbref").all()
    assert out.loc["U2", "post_output_pct"] == pytest.approx(1.0)
    assert out.loc["F2", "post_output_pct"] == pytest.approx(1.0)
    assert out.loc["U1", "post_output_pct"] == pytest.approx(0.5)
    assert out.loc["F1", "post_output_pct"] == pytest.approx(0.5)


def _fw_frame(n_by_group):
    """Attackers ranked on Understat xG, one group per (name, size)."""
    arrivals, rows = [], []
    for g, n in enumerate(n_by_group):
        for k in range(n):
            name = f"P{g}_{k}"
            arrivals.append({"player": name, "arrival_season": "2223"})
            rows.append({"player": name, "season": "2223", "league": "ITA-Serie A",
                         "pos": "FW", "minutes": 900, "us_npxg": float(k + 1),
                         "us_xa": 0.0})
    return pd.DataFrame(arrivals), pd.DataFrame(rows)


def test_tiny_rank_groups_get_neutral_percentile(monkeypatch):
    monkeypatch.setattr(labels, "MIN_RANK_GROUP", 5)
    # a 1-member and a 3-member group (different position groups)
    arrivals = pd.DataFrame({"player": ["S", "A", "B", "C"],
                             "arrival_season": ["2223"] * 4})
    seasons = pd.DataFrame({
        "player": ["S", "A", "B", "C"], "season": ["2223"] * 4,
        "league": ["ITA-Serie A"] * 4, "pos": ["GK", "FW", "FW", "FW"],
        "minutes": [900] * 4, "us_npxg": [1.0, 1.0, 2.0, 3.0], "us_xa": [0.0] * 4,
        "performance_gls": [1.0] * 4, "performance_ast": [0.0] * 4,
    })
    out = labels.post_output_percentile(seasons, arrivals).set_index("player")
    assert out.loc["S", "post_output_pct"] == pytest.approx(0.5)
    assert (out.loc[["A", "B", "C"], "post_output_pct"] == 0.5).all()


def test_six_member_rank_group_is_ranked_as_before(monkeypatch):
    monkeypatch.setattr(labels, "MIN_RANK_GROUP", 5)
    arrivals, seasons = _fw_frame([6])
    out = labels.post_output_percentile(seasons, arrivals).set_index("player")
    assert out["post_output_pct"].tolist() == pytest.approx(
        [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0])


def _team_rows(team, n_players, npxg_each, season="2223"):
    """Teammates whose Understat npxG sets the team's attack rate."""
    return pd.DataFrame({
        "player": [f"{team}{i}" for i in range(n_players)], "season": [season] * n_players,
        "league": ["ITA-Serie A"] * n_players, "team": [team] * n_players,
        "pos": ["FW"] * n_players, "minutes": [3420] * n_players,
        "us_npxg": [npxg_each] * n_players,
    })


def test_team_attack_rate_is_team_npxg_per_match():
    rows = _team_rows("Inter", 11, 7.6)   # 11 x 7.6 = 83.6 npxG over 38 full matches
    rates = labels.team_attack_rates(rows)
    assert rates[("ITA-Serie A", "Inter", "2223")] == pytest.approx(83.6 / 38)


def test_output_is_share_of_team_attack_not_raw_volume():
    # A at strong Inter and B at weak Lecce take part in the same SHARE of
    # their team's attack; C at Inter is involved in less. A raw per-90 label
    # would rank A far above B; the share ranks them equal, both above C.
    seasons = pd.concat([
        _team_rows("Inter", 11, 7.6), _team_rows("Lecce", 11, 3.8),
        pd.DataFrame({
            "player": ["A", "B", "C"], "season": ["2223"] * 3,
            "league": ["ITA-Serie A"] * 3, "team": ["Inter", "Lecce", "Inter"],
            "pos": ["MF"] * 3, "minutes": [3420] * 3,
            "us_xgchain": [40.0, 20.0, 20.0],
        }),
    ], ignore_index=True)
    arrivals = pd.DataFrame({"player": ["A", "B", "C"], "arrival_season": ["2223"] * 3})
    out = labels.post_output_percentile(seasons, arrivals).set_index("player")
    assert out.loc["A", "post_output_pct"] == out.loc["B", "post_output_pct"]
    assert out.loc["C", "post_output_pct"] < out.loc["A", "post_output_pct"]


def test_output_share_weights_each_spell_by_its_own_team():
    # Half a window at Inter (2.2 npxG/match), half at Lecce (1.1): the team
    # output he played through is 19 x 2.2 + 19 x 1.1 = 62.7.
    seasons = pd.concat([
        _team_rows("Inter", 11, 7.6), _team_rows("Lecce", 11, 3.8),
        pd.DataFrame({
            "player": ["A", "A"], "season": ["2223", "2223"],
            "league": ["ITA-Serie A"] * 2, "team": ["Inter", "Lecce"],
            "pos": ["MF", "MF"], "minutes": [1710, 1710], "us_xgchain": [10.0, 5.0],
        }),
    ], ignore_index=True)
    team_rates = labels.team_attack_rates(seasons)
    played = seasons[seasons["player"] == "A"]
    rate, _, _, _ = labels._basis_rate(played, "chain", team_rates)
    assert rate == pytest.approx(15.0 / (19 * 2.2 + 19 * 1.1))


def test_build_gives_raw_output_and_share_of_team_output():
    # A and B produce the same raw xG chain, but B's team attacks half as much:
    # equal on `output`, B ahead on `output_share`.
    teammates = pd.concat([_team_rows("Inter", 11, 7.6, "2223"), _team_rows("Inter", 11, 7.6, "2324"),
                           _team_rows("Lecce", 11, 3.8, "2223"), _team_rows("Lecce", 11, 3.8, "2324")])
    players = pd.DataFrame({
        "player": ["A", "A", "B", "B"], "season": ["2223", "2324"] * 2,
        "league": ["ITA-Serie A"] * 4, "team": ["Inter", "Inter", "Lecce", "Lecce"],
        "pos": ["MF"] * 4, "minutes": [3420.0] * 4, "team_matches": [38.0] * 4,
        "us_xgchain": [20.0] * 4,
    })
    seasons = pd.concat([teammates.assign(team_matches=38.0), players], ignore_index=True)
    arrivals = pd.DataFrame({"player": ["A", "B"], "arrival_season": ["2223", "2223"],
                             "to_team": ["Inter", "Lecce"]})
    out = labels.build(arrivals, seasons, pd.DataFrame(columns=["team", "squad_value_eur"]),
                       pd.DataFrame(columns=["player", "season", "fee_eur", "is_loan"])).set_index("player")
    assert out.loc["A", "output"] == out.loc["B", "output"]
    assert out.loc["B", "output_share"] > out.loc["A", "output_share"]
