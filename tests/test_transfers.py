import pandas as pd
import pytest
from src import transfers


def _rows(records):
    frame = pd.DataFrame(records, columns=["player", "team", "league", "season", "minutes", "pos"])
    frame["nationality"] = "ITA"
    return frame


def test_previous_season_rolls_back_correctly():
    assert transfers.previous_season("2223") == "2122"
    assert transfers.previous_season("2021") == "1920"
    assert transfers.previous_season("1819") == "1718"


def test_detects_arrival_from_another_league():
    frame = _rows([
        ("Kvara", "Dinamo Batumi", "GEO-Erovnuli", "2122", 2000, "FW"),
        ("Kvara", "Napoli", "ITA-Serie A", "2223", 2500, "FW"),
    ])
    out = transfers.detect_arrivals(frame)
    assert len(out) == 1
    assert out.iloc[0]["to_team"] == "Napoli"
    assert out.iloc[0]["from_team"] == "Dinamo Batumi"
    assert out.iloc[0]["arrival_season"] == "2223"


def test_ignores_player_who_stayed_at_the_same_club():
    frame = _rows([
        ("Barella", "Inter", "ITA-Serie A", "2122", 2500, "MF"),
        ("Barella", "Inter", "ITA-Serie A", "2223", 2600, "MF"),
    ])
    assert transfers.detect_arrivals(frame).empty


def test_detects_intra_serie_a_transfer():
    frame = _rows([
        ("Player", "Sassuolo", "ITA-Serie A", "2122", 2000, "MF"),
        ("Player", "Juventus", "ITA-Serie A", "2223", 1800, "MF"),
    ])
    out = transfers.detect_arrivals(frame)
    assert len(out) == 1
    assert out.iloc[0]["from_league"] == "ITA-Serie A"


def test_excludes_arrival_with_too_few_prior_minutes():
    frame = _rows([
        ("Kid", "Ajax", "NED-Eredivisie", "2122", 100, "FW"),
        ("Kid", "Milan", "ITA-Serie A", "2223", 400, "FW"),
    ])
    assert transfers.detect_arrivals(frame).empty


def test_excludes_player_with_no_prior_season_on_record():
    frame = _rows([("Ghost", "Roma", "ITA-Serie A", "2223", 1500, "FW")])
    assert transfers.detect_arrivals(frame).empty


def test_arrival_carries_nationality_for_the_adaptation_bundle():
    frame = _rows([
        ("Kvara", "Batumi", "GEO-Erovnuli", "2122", 2000, "FW"),
        ("Kvara", "Napoli", "ITA-Serie A", "2223", 2500, "FW"),
    ])
    frame.loc[frame.season == "2122", "nationality"] = "GEO"
    out = transfers.detect_arrivals(frame)
    assert out.iloc[0]["nationality"] == "GEO"


def test_emits_one_row_per_arrival_not_per_prior_season():
    frame = _rows([
        ("P", "Porto", "POR-Liga", "2021", 2000, "FW"),
        ("P", "Porto", "POR-Liga", "2122", 2100, "FW"),
        ("P", "Lazio", "ITA-Serie A", "2223", 1900, "FW"),
    ])
    out = transfers.detect_arrivals(frame)
    assert len(out) == 1


def test_mid_season_split_between_two_serie_a_clubs_picks_max_minutes_team():
    # Adam Ounas 2021 shape: absent from Nice in 1920, then split 2021
    # between Cagliari (308') and Crotone (1199'). Must be one arrival, at
    # the higher-minutes club, with arrival_clubs recording the split.
    frame = _rows([
        ("Ounas", "Nice", "FRA-Ligue 1", "1920", 1500, "FW"),
        ("Ounas", "Cagliari", "ITA-Serie A", "2021", 308, "FW"),
        ("Ounas", "Crotone", "ITA-Serie A", "2021", 1199, "FW"),
    ])
    out = transfers.detect_arrivals(frame)
    assert len(out) == 1
    assert out.iloc[0]["to_team"] == "Crotone"
    assert out.iloc[0]["arrival_clubs"] == 2


def test_normal_single_club_arrival_has_arrival_clubs_one():
    frame = _rows([
        ("Kvara", "Dinamo Batumi", "GEO-Erovnuli", "2122", 2000, "FW"),
        ("Kvara", "Napoli", "ITA-Serie A", "2223", 2500, "FW"),
    ])
    out = transfers.detect_arrivals(frame)
    assert out.iloc[0]["arrival_clubs"] == 1


def test_born_disambiguates_same_name_players_when_matching_prior_season():
    # Two different real people named "Danny Ward". Only the one born 1993
    # moved to Serie A; his prior club must be attributed correctly even
    # though a same-named 1985-born player also has a prior-season row.
    frame = pd.DataFrame([
        {"player": "Danny Ward", "team": "Leicester", "league": "ENG-Premier League",
         "season": "2122", "minutes": 2000, "pos": "GK", "nationality": "WAL", "born": 1993},
        {"player": "Danny Ward", "team": "Rangers", "league": "SCO-Premiership",
         "season": "2122", "minutes": 1800, "pos": "GK", "nationality": "SCO", "born": 1985},
        {"player": "Danny Ward", "team": "Watford", "league": "ITA-Serie A",
         "season": "2223", "minutes": 1500, "pos": "GK", "nationality": "WAL", "born": 1993},
    ])
    out = transfers.detect_arrivals(frame)
    assert len(out) == 1
    assert out.iloc[0]["from_team"] == "Leicester"
    assert out.iloc[0]["from_league"] == "ENG-Premier League"


def test_identities_keep_distinct_born_per_name_not_collapsed():
    combined = pd.DataFrame([
        {"player": "Danny Ward", "born": 1993, "nationality": "WAL",
         "team": "Leicester", "league": "ENG-Premier League", "season": "2122",
         "minutes": 2000, "pos": "GK"},
        {"player": "Danny Ward", "born": 1985, "nationality": "SCO",
         "team": "Rangers", "league": "SCO-Premiership", "season": "2122",
         "minutes": 1800, "pos": "GK"},
    ])
    identities = transfers.derive_identities(combined)
    assert len(identities) == 2
    assert set(identities["born"]) == {1993, 1985}


def test_prior_minutes_is_season_total_across_prior_clubs():
    # Prior season split 300/300 across two clubs: neither club alone
    # clears the 450-minute floor, but the season total (600) does.
    frame = _rows([
        ("Player", "ClubA", "FRA-Ligue 1", "2122", 300, "MF"),
        ("Player", "ClubB", "GER-Bundesliga", "2122", 300, "MF"),
        ("Player", "Roma", "ITA-Serie A", "2223", 1000, "MF"),
    ])
    out = transfers.detect_arrivals(frame)
    assert len(out) == 1
    assert out.iloc[0]["prior_minutes"] == 600
    assert out.iloc[0]["from_team"] in {"ClubA", "ClubB"}


def test_intra_serie_a_mid_season_move_is_arrival_to_new_club_not_dropped():
    # Vlahovic shape: at Fiorentina last season, and still registered there
    # this season too, but also moves mid-season to Juventus. The still-held
    # old club must not block the genuinely new club from being an arrival.
    frame = _rows([
        ("Vlahovic", "Fiorentina", "ITA-Serie A", "2021", 2500, "FW"),
        ("Vlahovic", "Fiorentina", "ITA-Serie A", "2122", 1800, "FW"),
        ("Vlahovic", "Juventus", "ITA-Serie A", "2122", 900, "FW"),
    ])
    out = transfers.detect_arrivals(frame)
    assert len(out) == 1
    assert out.iloc[0]["to_team"] == "Juventus"
    assert out.iloc[0]["from_team"] == "Fiorentina"
    assert out.iloc[0]["from_league"] == "ITA-Serie A"
    assert out.iloc[0]["arrival_clubs"] == 1


def test_no_new_club_means_no_arrival_even_with_multiple_prior_clubs():
    # Player held two Serie A clubs last season (e.g. a mid-season loan) and
    # is only at one of those same clubs this season -- no new club, so no
    # arrival, even though he has more than one row on record last season.
    frame = _rows([
        ("Stayer", "Inter", "ITA-Serie A", "2122", 1200, "MF"),
        ("Stayer", "Milan", "ITA-Serie A", "2122", 800, "MF"),
        ("Stayer", "Inter", "ITA-Serie A", "2223", 2000, "MF"),
    ])
    assert transfers.detect_arrivals(frame).empty
