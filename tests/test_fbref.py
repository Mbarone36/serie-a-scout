"""Tests for the `fbref` source-selection facade.

`config.FBREF_SOURCE == "dump"` (the default) must still serve seasons past
the dump's coverage (2010-2023) once they've been scraped live and cached to
disk elsewhere -- e.g. the 2023-24+ seasons a still-running scrape is
producing concurrently with these tests. These tests isolate that entirely
from the real dump/cache on disk via `tmp_path` and `monkeypatch`, and never
perform real I/O beyond `tmp_path`, pyreadr, or a network call.
"""

import pandas as pd
import pytest

from src import config


@pytest.fixture(autouse=True)
def _isolated_raw_dir(tmp_path, monkeypatch):
    """Every test gets its own empty RAW_DIR, so a real fbref/ or dump/ cache
    on disk from actual project use never leaks into these tests."""
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    from src import fbref
    monkeypatch.setattr(fbref, "_completeness_cache", {})
    return tmp_path


def _forbid(name):
    def _raise(*args, **kwargs):
        raise AssertionError(f"{name} must not be called for this dispatch path")
    return _raise


def test_dump_season_dispatches_to_load_dump_and_never_calls_scrape_fbref(monkeypatch):
    monkeypatch.setattr(config, "FBREF_SOURCE", "dump")

    from src import load_dump, scrape_fbref

    monkeypatch.setattr(load_dump, "available_seasons", lambda league: ["2122", "2223"])
    calls = []
    monkeypatch.setattr(
        load_dump, "player_season_stats",
        lambda league, season: calls.append(("player", league, season)) or pd.DataFrame({"ok": [1]}),
    )
    monkeypatch.setattr(
        load_dump, "team_season_stats",
        lambda league, season: calls.append(("team", league, season)) or pd.DataFrame({"ok": [1]}),
    )
    monkeypatch.setattr(scrape_fbref, "player_season_stats", _forbid("scrape_fbref.player_season_stats"))
    monkeypatch.setattr(scrape_fbref, "team_season_stats", _forbid("scrape_fbref.team_season_stats"))

    from src import fbref
    player_out = fbref.player_season_stats("ITA-Serie A", "2223")
    team_out = fbref.team_season_stats("ITA-Serie A", "2223")

    assert calls == [("player", "ITA-Serie A", "2223"), ("team", "ITA-Serie A", "2223")]
    assert not player_out.empty and not team_out.empty


def test_non_dump_season_reads_scrape_cache_without_live_fetch(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "FBREF_SOURCE", "dump")

    from src import load_dump, scrape_fbref

    # The dump doesn't cover 2324 -- only older seasons.
    monkeypatch.setattr(load_dump, "available_seasons", lambda league: ["2223"])
    monkeypatch.setattr(load_dump, "player_season_stats", _forbid("load_dump.player_season_stats"))
    monkeypatch.setattr(scrape_fbref, "player_season_stats", _forbid("scrape_fbref.player_season_stats (live fetch)"))

    # Simulate a concurrently-running live scrape having already cached
    # 2023-24 Serie A to disk under scrape_fbref's own cache key/path.
    cache_dir = tmp_path / "fbref" / "player"
    cache_dir.mkdir(parents=True)
    expected = pd.DataFrame({"player": ["Someone"], "team": ["Roma"], "minutes": [900]})
    expected.to_parquet(cache_dir / "ITA-Serie_A_2324.parquet")

    from src import fbref
    out = fbref.player_season_stats("ITA-Serie A", "2324")

    pd.testing.assert_frame_equal(out.reset_index(drop=True), expected.reset_index(drop=True))


def test_non_dump_season_without_cache_returns_empty_and_never_calls_scrape_fbref(monkeypatch):
    monkeypatch.setattr(config, "FBREF_SOURCE", "dump")

    from src import load_dump, scrape_fbref

    monkeypatch.setattr(load_dump, "available_seasons", lambda league: [])  # dump has nothing
    monkeypatch.setattr(scrape_fbref, "player_season_stats", _forbid("scrape_fbref.player_season_stats"))
    monkeypatch.setattr(scrape_fbref, "team_season_stats", _forbid("scrape_fbref.team_season_stats"))

    from src import fbref
    out = fbref.player_season_stats("ITA-Serie A", "2526")  # no cache file exists either

    assert isinstance(out, pd.DataFrame)
    assert out.empty


def test_scrape_source_always_delegates_to_scrape_fbref_even_for_a_dump_season(monkeypatch):
    monkeypatch.setattr(config, "FBREF_SOURCE", "scrape")

    from src import load_dump, scrape_fbref

    monkeypatch.setattr(load_dump, "player_season_stats", _forbid("load_dump.player_season_stats"))
    monkeypatch.setattr(load_dump, "available_seasons", _forbid("load_dump.available_seasons"))
    calls = []
    monkeypatch.setattr(
        scrape_fbref, "player_season_stats",
        lambda league, season: calls.append(("player", league, season)) or pd.DataFrame({"ok": [1]}),
    )
    monkeypatch.setattr(
        scrape_fbref, "team_season_stats",
        lambda league, season: calls.append(("team", league, season)) or pd.DataFrame({"ok": [1]}),
    )

    from src import fbref
    fbref.player_season_stats("ITA-Serie A", "2223")  # a season the dump *does* cover
    fbref.team_season_stats("ITA-Serie A", "2223")

    assert calls == [("player", "ITA-Serie A", "2223"), ("team", "ITA-Serie A", "2223")]


def test_available_seasons_unions_dump_and_scrape_cache(monkeypatch, tmp_path):
    from src import load_dump

    monkeypatch.setattr(load_dump, "available_seasons", lambda league: ["1718", "1819", "2223"])

    cache_dir = tmp_path / "fbref" / "player"
    cache_dir.mkdir(parents=True)
    for season in ("2324", "2425"):
        pd.DataFrame({"a": [1]}).to_parquet(cache_dir / f"ITA-Serie_A_{season}.parquet")
    # A different league's cache file must not be counted.
    pd.DataFrame({"a": [1]}).to_parquet(cache_dir / "ENG-Premier_League_2526.parquet")

    from src import fbref
    seasons = fbref.available_seasons("ITA-Serie A")

    assert seasons == sorted({"1718", "1819", "2223", "2324", "2425"})


# --- season_is_complete: a season present isn't necessarily a season fully
# played -- e.g. the historical dump's "2223" export stops at matchday 13 ---


def test_season_is_complete_true_at_a_full_season():
    from src import fbref

    frame = pd.DataFrame({"team_matches": [38, 38, 34]})
    assert fbref.season_is_complete(frame) is True


def test_season_is_complete_false_short_of_the_threshold():
    from src import fbref

    frame = pd.DataFrame({"team_matches": [13, 13, 10]})
    assert fbref.season_is_complete(frame) is False


def test_season_is_complete_false_for_empty_frame():
    from src import fbref

    assert fbref.season_is_complete(pd.DataFrame()) is False


def test_season_is_complete_false_when_column_missing():
    from src import fbref

    frame = pd.DataFrame({"player": ["A"], "minutes": [900]})
    assert fbref.season_is_complete(frame) is False


# --- dump-mode dispatch prefers a complete scraped season over an
# incomplete dump export ----------------------------------------------------


def test_dump_incomplete_season_serves_scrape_cache_when_present(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "FBREF_SOURCE", "dump")

    from src import fbref, load_dump

    monkeypatch.setattr(load_dump, "available_seasons", lambda league: ["2223"])
    incomplete = pd.DataFrame({
        "player": ["A"], "team": ["Napoli"], "minutes": [500], "team_matches": [13],
    })
    monkeypatch.setattr(load_dump, "player_season_stats", lambda league, season: incomplete)

    cache_dir = tmp_path / "fbref" / "player"
    cache_dir.mkdir(parents=True)
    scraped = pd.DataFrame({"player": ["A"], "team": ["Napoli"], "minutes": [3000]})
    scraped.to_parquet(cache_dir / "ITA-Serie_A_2223.parquet")

    out = fbref.player_season_stats("ITA-Serie A", "2223")

    pd.testing.assert_frame_equal(out.reset_index(drop=True), scraped.reset_index(drop=True))


def test_dump_complete_season_serves_dump_without_warning(monkeypatch, caplog):
    monkeypatch.setattr(config, "FBREF_SOURCE", "dump")

    from src import fbref, load_dump

    monkeypatch.setattr(load_dump, "available_seasons", lambda league: ["2223"])
    complete = pd.DataFrame({
        "player": ["A"], "team": ["Napoli"], "minutes": [2800], "team_matches": [38],
    })
    monkeypatch.setattr(load_dump, "player_season_stats", lambda league, season: complete)

    with caplog.at_level("WARNING", logger="src.fbref"):
        out = fbref.player_season_stats("ITA-Serie A", "2223")

    pd.testing.assert_frame_equal(out.reset_index(drop=True), complete.reset_index(drop=True))
    assert not caplog.records


def test_dump_incomplete_season_serves_dump_with_warning_when_no_cache(monkeypatch, caplog):
    monkeypatch.setattr(config, "FBREF_SOURCE", "dump")

    from src import fbref, load_dump

    monkeypatch.setattr(load_dump, "available_seasons", lambda league: ["2223"])
    incomplete = pd.DataFrame({
        "player": ["A"], "team": ["Napoli"], "minutes": [500], "team_matches": [13],
    })
    monkeypatch.setattr(load_dump, "player_season_stats", lambda league, season: incomplete)

    with caplog.at_level("WARNING", logger="src.fbref"):
        out = fbref.player_season_stats("ITA-Serie A", "2223")

    pd.testing.assert_frame_equal(out.reset_index(drop=True), incomplete.reset_index(drop=True))
    assert any("incomplete" in record.message for record in caplog.records)


def test_team_dispatch_derives_completeness_from_player_frame_and_caches_it(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "FBREF_SOURCE", "dump")

    from src import fbref, load_dump

    monkeypatch.setattr(load_dump, "available_seasons", lambda league: ["2223"])
    incomplete_player = pd.DataFrame({
        "player": ["A"], "team": ["Napoli"], "minutes": [500], "team_matches": [13],
    })
    player_calls = []
    monkeypatch.setattr(
        load_dump, "player_season_stats",
        lambda league, season: player_calls.append((league, season)) or incomplete_player,
    )
    dump_team_frame = pd.DataFrame({"team": ["Napoli"], "possession": [50.0]})
    monkeypatch.setattr(load_dump, "team_season_stats", lambda league, season: dump_team_frame)

    cache_dir = tmp_path / "fbref" / "team"
    cache_dir.mkdir(parents=True)
    scraped_team = pd.DataFrame({"team": ["Napoli"], "possession": [55.0]})
    scraped_team.to_parquet(cache_dir / "ITA-Serie_A_2223.parquet")

    # A team-kind call with no prior player-kind call must still consult the
    # player frame to judge completeness (team frames carry no team_matches).
    out = fbref.team_season_stats("ITA-Serie A", "2223")
    pd.testing.assert_frame_equal(out.reset_index(drop=True), scraped_team.reset_index(drop=True))
    assert player_calls == [("ITA-Serie A", "2223")]

    # A second dispatch for the same league-season reuses the cached
    # completeness decision instead of reloading the player table again.
    fbref.team_season_stats("ITA-Serie A", "2223")
    assert player_calls == [("ITA-Serie A", "2223")]


def test_harmonise_renames_live_interceptions_to_dump_name():
    from src import fbref
    live = pd.DataFrame({"player": ["A"], "performance_int": [7]})
    out = fbref.harmonise_player_columns(live)
    assert "performance_int" not in out.columns
    assert out["int"].tolist() == [7]


def test_harmonise_leaves_dump_int_untouched():
    from src import fbref
    dump = pd.DataFrame({"player": ["A"], "int": [3], "performance_int": [9]})
    pd.testing.assert_frame_equal(fbref.harmonise_player_columns(dump), dump)


def test_scrape_cache_frame_gets_int_column(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "FBREF_SOURCE", "dump")
    from src import load_dump
    monkeypatch.setattr(load_dump, "available_seasons", lambda league: [])
    cache_dir = tmp_path / "fbref" / "player"
    cache_dir.mkdir(parents=True)
    pd.DataFrame({"player": ["A"], "team": ["Roma"], "performance_int": [4]}).to_parquet(
        cache_dir / "ITA-Serie_A_2526.parquet")

    from src import fbref
    out = fbref.player_season_stats("ITA-Serie A", "2526")
    assert out["int"].tolist() == [4]


def test_harmonise_renames_live_misc_trio_to_dump_names():
    from src import fbref
    live = pd.DataFrame({"performance_tklw": [5], "performance_crs": [12], "performance_fld": [3]})
    out = fbref.harmonise_player_columns(live)
    assert out[["tklw", "crs", "fld"]].iloc[0].tolist() == [5, 12, 3]
    assert "tackles_tkl" not in out.columns  # tackles won is not tackles attempted
