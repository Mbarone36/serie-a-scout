import json

import numpy as np
import pandas as pd
import pytest

from src import api_football, config

SA = "ITA-Serie A"


def _item(pid, first, last, short, minutes, team="Roma", league_id=135,
          tackles=10, duels=(100, 60), born="1997-08-22"):
    return {
        "player": {"id": pid, "name": short, "firstname": first, "lastname": last,
                   "birth": {"date": born}},
        "statistics": [{
            "team": {"name": team},
            "league": {"id": league_id, "season": 2023},
            "games": {"minutes": minutes},
            "tackles": {"total": tackles, "blocks": 2, "interceptions": 5},
            "duels": {"total": duels[0], "won": duels[1]},
            "dribbles": {"attempts": 20, "success": 12, "past": 7},
            "passes": {"key": 15},
        }],
    }


def _page(items, current=1, total=1):
    return json.dumps({"errors": [], "paging": {"current": current, "total": total},
                       "response": items})


def _fb(rows):
    frame = pd.DataFrame(rows, columns=["player", "team", "minutes", "born"])
    frame["league"] = SA
    frame["season"] = "2324"
    return frame


# --- parsing ------------------------------------------------------------------


def test_parse_keeps_full_and_display_names_and_stats():
    frame = api_football.parse_pages(
        [_page([_item(1, "Lautaro Javier", "Martínez", "L. Martínez", 2700)])], SA, "2324")
    row = frame.iloc[0]
    assert row["player"] == "Lautaro Javier Martínez"
    assert row["short_name"] == "L. Martínez"
    assert row["born"] == 1997.0
    assert row["minutes"] == 2700
    assert row["af_tackles"] == 10 and row["af_duels_won"] == 60


def test_parse_drops_other_competitions_and_sums_a_within_league_move():
    item = _item(1, "A", "B", "A. B", 900, team="Roma")
    item["statistics"].append({**item["statistics"][0], "team": {"name": "Lazio"}})
    item["statistics"].append({**item["statistics"][0], "league": {"id": 2}})  # Champions League
    frame = api_football.parse_pages([_page([item])], SA, "2324")
    assert len(frame) == 1
    assert frame.iloc[0]["minutes"] == 1800
    assert frame.iloc[0]["af_tackles"] == 20
    assert frame.iloc[0]["team"] == "Roma/Lazio"


def test_parse_keeps_null_stats_as_nan():
    item = _item(1, "A", "B", "A. B", 900)
    item["statistics"][0]["tackles"]["total"] = None
    frame = api_football.parse_pages([_page([item])], SA, "2324")
    assert np.isnan(frame.iloc[0]["af_tackles"])


# --- error payloads -------------------------------------------------------------


def test_quota_error_raises_quota_exceeded():
    with pytest.raises(api_football.QuotaExceeded):
        api_football.check_payload({"errors": {"requests": "You have reached the request limit for the day"},
                                    "response": []})


def test_plan_error_raises_plan_restricted():
    with pytest.raises(api_football.PlanRestricted):
        api_football.check_payload({"errors": {"plan": "Free plans do not have access to this season"},
                                    "response": []})


def test_clean_payload_passes():
    api_football.check_payload({"errors": [], "response": []})


def test_missing_key_fails_before_any_request(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    monkeypatch.delenv(api_football.KEY_ENV, raising=False)
    monkeypatch.setattr(api_football.requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("network called")))
    with pytest.raises(RuntimeError, match=api_football.KEY_ENV):
        api_football.fetch_page(SA, "2324", 1)


def test_fetch_league_season_reads_every_page(monkeypatch):
    pages = {1: _page([], 1, 3), 2: _page([], 2, 3), 3: _page([], 3, 3)}
    seen = []
    monkeypatch.setattr(api_football, "fetch_page",
                        lambda league, season, page: seen.append(page) or pages[page])
    assert len(api_football.fetch_league_season(SA, "2324")) == 3
    assert seen == [1, 2, 3]


# --- matching / augment -----------------------------------------------------------


def _af(rows):
    """API-Football frame from (af_id, full, short, minutes, tackles) tuples."""
    frame = pd.DataFrame(rows, columns=["af_id", "player", "short_name", "minutes", "af_tackles"])
    frame["league"] = SA
    frame["season"] = "2324"
    frame["team"] = "T"
    frame["born"] = 1997.0
    for column in api_football.STAT_COLUMNS:
        if column != "af_tackles":
            frame[column] = 10.0
    frame["af_duels"] = 40.0
    return frame


def test_full_name_subset_then_mononym_on_display_name():
    fb = _fb([("Lautaro Martínez", "Inter", 2700, "1997"),
              ("Vitinha", "Genoa", 1800, "2000")])
    af = _af([("1", "Lautaro Javier Martínez", "L. Martínez", 2710, 30.0),
              ("2", "Vítor Machado Ferreira", "Vitinha", 1805, 20.0)])
    mapping = api_football.match_to_fbref(fb, af).set_index("player")
    assert mapping.loc["Lautaro Martínez", "af_id"] == "1"
    assert mapping.loc["Vitinha", "af_id"] == "2"
    assert mapping.loc["Vitinha", "name_rule"].startswith("short_")


def test_minutes_disagreement_blocks_a_match():
    fb = _fb([("Lautaro Martínez", "Inter", 2700, "1997")])
    af = _af([("1", "Lautaro Javier Martínez", "L. Martínez", 900, 30.0)])
    assert api_football.match_to_fbref(fb, af).empty


def test_augment_rescales_to_row_minutes_and_adds_duel_rate():
    # Split season inside the league: two FBref rows, one API-Football row.
    fb = _fb([("Lautaro Martínez", "Inter", 1800, "1997"),
              ("Lautaro Martínez", "Roma", 900, "1997")])
    af = _af([("1", "Lautaro Javier Martínez", "L. Martínez", 2700, 30.0)])
    out = api_football.augment(fb, af, verbose=False).set_index("team")
    # 30 tackles over 2700' = 1 per 90; rows carry it in proportion.
    assert out.loc["Inter", "af_tackles"] == pytest.approx(20.0)
    assert out.loc["Roma", "af_tackles"] == pytest.approx(10.0)
    assert out.loc["Inter", "af_duels_won_pct"] == pytest.approx(10.0 / 40.0)


def test_augment_without_data_adds_nan_columns(monkeypatch):
    monkeypatch.setattr(api_football, "load", lambda: None)
    fb = _fb([("A", "T", 900, "1997")])
    out = api_football.augment(fb, verbose=False)
    for column in api_football.AUGMENT_COLUMNS + api_football.RATE_COLUMNS:
        assert out[column].isna().all()
