import json
import math
from pathlib import Path

import pandas as pd
import pytest

from src import config, understat

FIXTURE = Path(__file__).parent / "fixtures" / "understat_league.json"
SA = "ITA-Serie A"


def _us(rows):
    """Understat frame from (us_id, player, minutes, npxg) tuples."""
    frame = pd.DataFrame(rows, columns=["us_id", "player", "minutes", "us_npxg"])
    frame["league"] = SA
    frame["season"] = "2223"
    frame["team"] = "T"
    for column in understat.AUGMENT_COLUMNS:
        if column != "us_npxg":
            frame[column] = frame["us_npxg"] * 2
    frame["us_games"] = 30.0
    return frame


def _fb(rows):
    """FBref frame from (player, team, minutes[, born]) tuples (born defaults
    to one shared value, i.e. same person across clubs)."""
    rows = [tuple(r) + ("2000",) if len(r) == 3 else tuple(r) for r in rows]
    frame = pd.DataFrame(rows, columns=["player", "team", "minutes", "born"])
    frame["league"] = SA
    frame["season"] = "2223"
    return frame


def _no_network(*args, **kwargs):
    raise AssertionError("network called")


# --- parsing / fetching ----------------------------------------------------


def test_season_to_start_year():
    assert understat.season_to_start_year("2324") == 2023
    assert understat.season_to_start_year("1718") == 2017


def test_league_slugs_cover_candidate_leagues():
    assert set(config.CANDIDATE_LEAGUES) <= set(understat.LEAGUE_SLUGS)


def test_parse_players_types_and_columns():
    out = understat.parse_players(FIXTURE.read_text(), SA, "2223")
    assert len(out) == 3
    for column in understat.PLAYER_COLUMNS:
        assert column in out.columns, column
    assert out["minutes"].dtype.kind == "i"
    for column in ["us_npxg", "us_xa", "us_xgchain", "us_xgbuildup",
                   "us_key_passes", "us_shots", "us_games"]:
        assert out[column].dtype == float, column
    kvara = out[out["player"] == "Khvicha Kvaratskhelia"].iloc[0]
    assert kvara["minutes"] == 2577
    assert kvara["us_npxg"] == pytest.approx(9.481)
    assert kvara["us_xa"] == pytest.approx(7.802)
    assert kvara["us_xgchain"] == pytest.approx(17.204)
    assert kvara["us_xgbuildup"] == pytest.approx(6.113)
    assert kvara["us_key_passes"] == 66.0
    assert kvara["us_shots"] == 84.0
    assert kvara["us_games"] == 34.0
    assert (out["league"] == SA).all() and (out["season"] == "2223").all()


def test_parse_players_unescapes_names_and_keeps_joined_teams():
    out = understat.parse_players(FIXTURE.read_text(), SA, "2223")
    assert "M'Bala Nzola" in set(out["player"])
    assert out.loc[out["player"] == "Tanguy Ndombele", "team"].item() == "Napoli,Roma"


def test_fetch_league_season_reads_cache_key_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    (tmp_path / "understat").mkdir()
    (tmp_path / "understat" / "Serie_A_2022.html").write_text(FIXTURE.read_text())
    monkeypatch.setattr(understat.requests, "get", _no_network)
    assert "Kvaratskhelia" in understat.fetch_league_season(SA, "2223")


def test_fetch_rejects_non_json_without_caching(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)

    class Resp:
        text = "<html>blocked</html>"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(understat.requests, "get", lambda *a, **k: Resp())
    monkeypatch.setattr(understat._limiter, "wait", lambda: None)
    with pytest.raises(ValueError):
        understat.fetch_league_season(SA, "2223")
    assert not (tmp_path / "understat" / "Serie_A_2022.html").exists()


def test_pull_all_writes_parquet_from_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    (tmp_path / "understat").mkdir()
    (tmp_path / "understat" / "Serie_A_2022.html").write_text(FIXTURE.read_text())
    monkeypatch.setattr(understat.requests, "get", _no_network)
    out = understat.pull_all([SA], ["2223"])
    assert len(out) == 3
    assert len(pd.read_parquet(tmp_path / "understat.parquet")) == 3


# --- matching --------------------------------------------------------------


def test_match_accent_and_variant_names():
    fb = _fb([("Khvicha Kvaratskhelia", "Napoli", 2580),
              ("Filip Đuričić", "Sampdoria", 1500),
              ("Rafael Leão", "Milan", 2700),
              ("M'Bala Nzola", "Spezia", 2250)])
    us = _us([("1", "Khvicha Kvaratskhelia", 2577, 9.0),
              ("2", "Filip Djuricic", 1480, 2.0),
              ("3", "Rafael Leao", 2690, 8.0),
              ("4", "M'Bala Nzola", 2250, 8.9)])
    mapping = understat.match_to_fbref(fb, us)
    pairs = dict(zip(mapping["player"], mapping["us_id"]))
    assert pairs == {"Khvicha Kvaratskhelia": "1", "Filip Đuričić": "2",
                     "Rafael Leão": "3", "M'Bala Nzola": "4"}


def test_match_subset_names():
    # Understat carries a middle name FBref drops: a token-set subset.
    fb = _fb([("Nicolás González", "Fiorentina", 1800)])
    us = _us([("7", "Nicolás Iván Martínez González", 1790, 5.0)])
    mapping = understat.match_to_fbref(fb, us)
    assert mapping["us_id"].tolist() == ["7"]
    assert mapping["name_rule"].item() == "subset"


def test_unrelated_names_never_match_even_with_equal_minutes():
    fb = _fb([("Paulo Dybala", "Roma", 2000)])
    us = _us([("1", "Lorenzo Pellegrini", 2000, 1.0)])
    assert understat.match_to_fbref(fb, us).empty


def test_minutes_gate_rejects_namesake_with_very_different_minutes():
    fb = _fb([("Danilo", "Juventus", 2900)])
    us = _us([("10", "Danilo", 400, 0.5)])
    assert understat.match_to_fbref(fb, us).empty


def test_minutes_tolerance_is_max_of_90_and_ten_percent():
    fb = _fb([("Alpha Player", "T", 300), ("Beta Player", "T", 3000)])
    us = _us([("1", "Alpha Player", 389, 1.0),     # within 90
              ("2", "Beta Player", 2710, 1.0)])    # within 10% of 3000
    assert set(understat.match_to_fbref(fb, us)["us_id"]) == {"1", "2"}
    fb2 = _fb([("Alpha Player", "T", 300)])
    us2 = _us([("1", "Alpha Player", 392, 1.0)])   # 92 > max(90, 39.2)
    assert understat.match_to_fbref(fb2, us2).empty


def test_one_to_one_best_name_score_wins():
    # Two FBref players compete for one Understat row: the exact-name one
    # takes it, and the Understat row is never assigned twice.
    fb = _fb([("Luis Alberto", "Lazio", 2000),
              ("Luis Alberto Romero", "Lazio", 2000)])
    us = _us([("5", "Luis Alberto", 2010, 4.0)])
    mapping = understat.match_to_fbref(fb, us)
    assert len(mapping) == 1
    assert mapping["player"].item() == "Luis Alberto"


def test_one_to_one_ties_on_name_break_on_closest_minutes():
    fb = _fb([("Same Name", "X", 1000)])
    us = _us([("a", "Same Name", 1080, 1.0), ("b", "Same Name", 1010, 1.0)])
    mapping = understat.match_to_fbref(fb, us)
    assert mapping["us_id"].tolist() == ["b"]


def test_one_to_one_across_many_pairs():
    fb = _fb([("Same Name", "X", 1000), ("Same Name Junior", "Y", 1000)])
    us = _us([("a", "Same Name", 1000, 1.0), ("b", "Same Name Junior", 1000, 1.0)])
    mapping = understat.match_to_fbref(fb, us)
    assert dict(zip(mapping["player"], mapping["us_id"])) == {
        "Same Name": "a", "Same Name Junior": "b"}
    assert not mapping["us_id"].duplicated().any()
    assert not mapping["player"].duplicated().any()


def test_match_is_scoped_to_league_season():
    fb = _fb([("Same Name", "X", 1000)])
    us = _us([("a", "Same Name", 1000, 1.0)])
    us["season"] = "2324"
    assert understat.match_to_fbref(fb, us).empty


def test_split_season_minutes_are_summed_before_matching():
    fb = _fb([("Tanguy Ndombele", "Napoli", 700), ("Tanguy Ndombele", "Roma", 600)])
    us = _us([("6447", "Tanguy Ndombele", 1300, 1.06)])
    mapping = understat.match_to_fbref(fb, us)
    assert mapping["fb_minutes"].item() == 1300


def test_aboubacar_ali_matches_full_name_not_fuzzy_lookalike():
    # "aboubacar ali" vs "aboubacar lo" is a token_sort 88 fuzzy hit with a
    # different surname; the real match is the subset "... ali abdallah".
    fb = _fb([("Aboubacar Ali", "Spezia", 1500)])
    us = _us([("1", "Aboubacar Lô", 1500, 1.0),
              ("2", "Aboubacar Ali Abdallah", 1450, 1.0)])
    mapping = understat.match_to_fbref(fb, us)
    assert mapping["us_id"].tolist() == ["2"]
    assert mapping["name_rule"].item() == "subset"


def test_fuzzy_with_different_surnames_is_rejected():
    assert understat._name_verdict("aboubacar ali", "aboubacar lo") is None
    fb = _fb([("Aboubacar Ali", "Spezia", 1500)])
    us = _us([("1", "Aboubacar Lô", 1500, 1.0)])
    assert understat.match_to_fbref(fb, us).empty


def test_fuzzy_with_similar_last_token_still_matches():
    fb = _fb([("Marcus Thuram", "Inter", 2000)])
    us = _us([("1", "Marcus Thuramm", 2000, 1.0)])
    mapping = understat.match_to_fbref(fb, us)
    assert mapping["name_rule"].item() == "fuzzy"


def test_rule_precedence_exact_over_subset_over_fuzzy_regardless_of_score():
    # Same minutes for all: the exact name must win over higher-scoring-or-not
    # subset/fuzzy lookalikes.
    fb = _fb([("Luis Alberto", "Lazio", 2000)])
    us = _us([("fuzzy", "Luis Albertoo", 2000, 1.0),
              ("subset", "Luis Alberto Romero Alconchel", 2000, 1.0),
              ("exact", "Luis Alberto", 2000, 1.0)])
    mapping = understat.match_to_fbref(fb, us)
    assert mapping["us_id"].tolist() == ["exact"]
    fb = _fb([("Luis Alberto", "Lazio", 2000)])
    us = _us([("fuzzy", "Luis Albertoo", 2000, 1.0),
              ("subset", "Luis Alberto Romero Alconchel", 2000, 1.0)])
    assert understat.match_to_fbref(fb, us)["us_id"].tolist() == ["subset"]


def test_subset_requires_shorter_last_token_shared():
    # shares only the FIRST name: rejected
    assert understat._name_verdict("luis alberto", "luis miguel romero") is None
    assert understat._name_verdict("luis alberto", "luis alberto romero alconchel")[1] == "subset"
    assert understat._name_verdict("aboubacar ali", "aboubacar ali abdallah")[1] == "subset"


def test_mononym_subset_matches_with_tight_minutes_gate():
    for short, long in [("Kepa", "Kepa Arrizabalaga"), ("Yuri", "Yuri Alberto"),
                        ("Bryan", "Bryan Cristante"), ("Danilo", "Danilo Luiz da Silva")]:
        fb = _fb([(short, "T", 2000)])
        us = _us([("1", long, 1950, 1.0)])     # 50 <= max(45, 100)
        mapping = understat.match_to_fbref(fb, us)
        assert mapping["us_id"].tolist() == ["1"], short
        assert mapping["name_rule"].item() == "subset"
    # 60 minutes apart: fine for a normal subset (<= 90) but not a mononym
    # (> max(45, 5% of 1000) = 50).
    fb = _fb([("Kepa", "T", 1000)])
    us = _us([("1", "Kepa Arrizabalaga", 940, 1.0)])
    assert understat.match_to_fbref(fb, us).empty
    fb = _fb([("Kepa Arrizabalaga", "T", 1000)])
    us = _us([("1", "Kepa Arrizabalaga", 940, 1.0)])
    assert understat.match_to_fbref(fb, us)["us_id"].tolist() == ["1"]


# --- namesakes ---------------------------------------------------------------


def test_cameo_and_starter_namesakes_are_skipped():
    # 20 + 2000 min namesakes on different clubs (different birth years) sum
    # to the starter's Understat row; neither may receive Understat data.
    fb = _fb([("Marco Rossi", "Spezia", 20, "1999"),
              ("Marco Rossi", "Torino", 2000, "1990"),
              ("Other Player", "Roma", 1500)])
    us = _us([("1", "Marco Rossi", 2020, 5.0), ("2", "Other Player", 1500, 3.0)])
    assert understat.match_to_fbref(fb, us)["player"].tolist() == ["Other Player"]
    out = understat.augment(fb, us, verbose=False)
    rossi = out[out["player"] == "Marco Rossi"]
    for column in understat.AUGMENT_COLUMNS:
        assert rossi[column].isna().all(), column
    assert out.loc[out["player"] == "Other Player", "us_npxg"].notna().all()


def test_namesakes_distinguished_by_nationality_when_born_missing():
    fb = _fb([("Marco Rossi", "Spezia", 20), ("Marco Rossi", "Torino", 2000)])
    fb = fb.drop(columns="born")
    fb["nationality"] = ["BRA", "ITA"]
    us = _us([("1", "Marco Rossi", 2020, 5.0)])
    assert understat.match_to_fbref(fb, us).empty
    fb["nationality"] = ["ITA", "ITA"]
    assert understat.match_to_fbref(fb, us)["us_id"].tolist() == ["1"]


def test_partial_born_is_ambiguous_even_with_same_nationality():
    us = _us([("1", "Marco Rossi", 2020, 5.0)])
    fb = _fb([("Marco Rossi", "Spezia", 20, "1999"), ("Marco Rossi", "Torino", 2000, None)])
    fb["nationality"] = ["ITA", "ITA"]
    assert understat.match_to_fbref(fb, us).empty
    assert ("ITA-Serie A", "2223", "Marco Rossi") in understat.ambiguous_namesakes(fb)
    # born NaN on EVERY row + same nationality: same person, aggregated
    fb = _fb([("Marco Rossi", "Spezia", 20, None), ("Marco Rossi", "Torino", 2000, None)])
    fb["nationality"] = ["ITA", "ITA"]
    assert understat.match_to_fbref(fb, us)["us_id"].tolist() == ["1"]


def test_multi_team_name_without_born_or_nationality_is_ambiguous():
    fb = _fb([("Marco Rossi", "Spezia", 20), ("Marco Rossi", "Torino", 2000)])
    fb = fb.drop(columns="born")
    us = _us([("1", "Marco Rossi", 2020, 5.0)])
    assert understat.match_to_fbref(fb, us).empty


def test_midseason_mover_with_same_born_stays_aggregated():
    fb = _fb([("Tanguy Ndombele", "Napoli", 900, "1996-12-28"),
              ("Tanguy Ndombele", "Roma", 300, "1996-12-28")])
    us = _us([("6447", "Tanguy Ndombele", 1200, 2.4)])
    assert understat.match_to_fbref(fb, us)["us_id"].tolist() == ["6447"]
    out = understat.augment(fb, us, verbose=False)
    assert out["us_npxg"].tolist() == pytest.approx([2.4 * 900 / 1200, 2.4 * 300 / 1200])


def test_match_rate_printout_counts_ambiguous_namesakes(capsys):
    fb = _fb([("Marco Rossi", "Spezia", 20, "1999"),
              ("Marco Rossi", "Torino", 2000, "1990"),
              ("Other Player", "Roma", 1500)])
    us = _us([("1", "Marco Rossi", 2020, 5.0), ("2", "Other Player", 1500, 3.0)])
    understat.augment(fb, us)
    printed = capsys.readouterr().out
    assert "ITA-Serie A 2223: 1/2 (50.0%), 1 ambiguous namesakes" in printed


# --- augment ---------------------------------------------------------------


def test_augment_split_rows_get_proportional_values_and_unmatched_nan():
    fb = _fb([("Tanguy Ndombele", "Napoli", 900), ("Tanguy Ndombele", "Roma", 300),
              ("Nobody Here", "Roma", 1000)])
    fb.index = [10, 11, 12]
    us = _us([("6447", "Tanguy Ndombele", 1200, 2.4)])  # 0.18 npxG per 90
    out = understat.augment(fb, us, verbose=False)

    assert list(out.index) == [10, 11, 12]
    assert list(out["team"]) == ["Napoli", "Roma", "Roma"]
    napoli, roma, nobody = out.loc[10], out.loc[11], out.loc[12]
    assert napoli["us_npxg"] == pytest.approx(2.4 * 900 / 1200)
    assert roma["us_npxg"] == pytest.approx(2.4 * 300 / 1200)
    assert napoli["us_npxg"] + roma["us_npxg"] == pytest.approx(2.4)
    # per-90 on each row recovers the Understat rate
    assert napoli["us_npxg"] / 900 * 90 == pytest.approx(0.18)
    assert roma["us_xa"] / 300 * 90 == pytest.approx(4.8 / 1200 * 90)
    for column in understat.AUGMENT_COLUMNS:
        assert math.isnan(nobody[column]), column


def test_augment_rate_uses_understat_minutes():
    # FBref 1250 vs Understat 1200 minutes (within tolerance): per_90 on the
    # row equals Understat's own per-90 rate.
    fb = _fb([("Alpha Player", "T", 1250)])
    us = _us([("1", "Alpha Player", 1200, 6.0)])
    out = understat.augment(fb, us, verbose=False)
    assert out["us_npxg"].iloc[0] / 1250 * 90 == pytest.approx(6.0 / 1200 * 90)


def test_augment_prints_match_rate(capsys):
    fb = _fb([("Alpha Player", "T", 1000), ("Beta Player", "T", 900),
              ("Gamma Bench", "T", 100)])
    us = _us([("1", "Alpha Player", 1000, 1.0)])
    understat.augment(fb, us)
    printed = capsys.readouterr().out
    # Gamma Bench (< 450 min) is outside the denominator.
    assert "ITA-Serie A 2223: 1/2 (50.0%)" in printed


def test_augment_missing_file_warns_and_adds_nan_columns(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    fb = _fb([("Alpha Player", "T", 1000)])
    out = understat.augment(fb)
    assert "WARNING" in capsys.readouterr().out
    assert len(out) == 1
    for column in understat.AUGMENT_COLUMNS:
        assert out[column].isna().all()


def test_augment_is_idempotent():
    fb = _fb([("Alpha Player", "T", 900)])
    us = _us([("1", "Alpha Player", 900, 3.0)])
    once = understat.augment(fb, us, verbose=False)
    twice = understat.augment(once, us, verbose=False)
    assert sorted(twice.columns) == sorted(once.columns)
    assert twice["us_npxg"].item() == pytest.approx(3.0)



def _team_payload():
    match = lambda npxg, deep, att, deff, xpts: {
        "npxG": npxg, "deep": deep, "ppda": {"att": att, "def": deff}, "xpts": xpts}
    return json.dumps({"players": [], "teams": {
        "1": {"id": "1", "title": "AC Milan",
              "history": [match(1.0, 6, 200, 20, 1.5), match(2.0, 10, 100, 20, 2.5)]}}})


def test_parse_teams_averages_per_match_and_inverts_ppda():
    row = understat.parse_teams(_team_payload(), SA, "2223").iloc[0]
    assert row["us_team"] == "AC Milan" and row["matches"] == 2
    assert row["team_npxg"] == pytest.approx(1.5)
    assert row["team_deep"] == pytest.approx(8.0)
    assert row["team_press"] == pytest.approx(40 / 300)  # 1 / PPDA over the season
    assert row["team_xpts"] == pytest.approx(2.0)


def test_team_name_map_follows_the_matched_players():
    fb = _fb([("Rafael Leão", "Milan", 2500), ("Theo Hernández", "Milan", 2800),
              ("Mattia Zaccagni", "Lazio", 2400)])
    us = _us([("1", "Rafael Leão", 2510, 9.0), ("2", "Theo Hernández", 2790, 3.0),
              ("3", "Mattia Zaccagni", 2400, 6.0)])
    us["team"] = ["AC Milan", "AC Milan", "Lazio"]
    out = understat.team_name_map(fb, us).set_index("us_team")["team"]
    assert out["AC Milan"] == "Milan"
    assert out["Lazio"] == "Lazio"
