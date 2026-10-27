import logging
from pathlib import Path

import pandas as pd
import pytest
import requests

from src import pull_tm, scrape_tm

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# search_player / search_club
# ---------------------------------------------------------------------------


def test_search_player_parses_fetch_result(monkeypatch):
    """FINDING (follow-up ruling): the outgoing query text must be the
    normalised/transliterated name, not the raw FBref spelling -- a cached
    search run with the raw accented name for "Filip Đuričić"
    parsed to zero hits, while Transfermarkt's own slug for that player is
    "filip-djuricic". See test_search_player_queries_transliterated_name
    below for that exact case."""
    html = (FIXTURES / "tm_search.html").read_text(encoding="utf-8")
    seen = {}

    def fake_fetch(url, key):
        seen["url"] = url
        seen["key"] = key
        return html

    monkeypatch.setattr(scrape_tm, "fetch", fake_fetch)
    out = pull_tm.search_player("Khvicha Kvaratskhelia")

    assert "query=khvicha+kvaratskhelia" in seen["url"]
    assert seen["key"] == "search/khvicha kvaratskhelia".replace(" ", "_")
    assert list(out.columns) == ["tm_id", "name", "birth_date", "nationality", "slug"]
    assert (out["tm_id"] == 502670).any()


def test_search_player_queries_transliterated_name(monkeypatch):
    """A name with letters that don't survive NFKD accent-stripping on
    their own (diagnosis b: Đ/Ø/Æ/... ) must still be searched as its
    ASCII transliteration, not the raw accented spelling and not the old
    broken (letter-dropped) normalised form -- "Filip Đuričić" is
    queried as "filip+djuricic", matching Transfermarkt's own slug
    "filip-djuricic"."""
    html = "<html></html>"
    seen = {}

    def fake_fetch(url, key):
        seen["url"] = url
        seen["key"] = key
        return html

    monkeypatch.setattr(scrape_tm, "fetch", fake_fetch)
    pull_tm.search_player("Filip Đuričić")

    assert "query=filip+djuricic" in seen["url"]
    assert seen["key"] == "search/filip_djuricic"


def test_search_player_keeps_apostrophe_in_query(monkeypatch):
    from urllib.parse import parse_qs, quote_plus, urlparse

    from src import entities

    seen = []

    def fake_fetch(url, key):
        seen.append((url, key))
        return (FIXTURES / "tm_search.html").read_text(encoding="utf-8")

    monkeypatch.setattr(scrape_tm, "fetch", fake_fetch)
    out = pull_tm.search_player("M'Baye Niang")

    assert len(seen) == 1  # non-empty result: no fallback
    assert not out.empty
    url, key = seen[0]
    assert quote_plus(entities.search_form("M'Baye Niang")) in url
    assert parse_qs(urlparse(url).query)["query"] == ["m'baye niang"]
    assert key == "search/m'baye_niang"


def test_search_player_falls_back_to_normalised_name_only_on_empty(monkeypatch):
    from urllib.parse import parse_qs, urlparse

    seen = []

    def fake_fetch(url, key):
        seen.append((parse_qs(urlparse(url).query)["query"][0], key))
        return "<html></html>"  # parses to zero rows

    monkeypatch.setattr(scrape_tm, "fetch", fake_fetch)
    out = pull_tm.search_player("Jean-Daniel Akpa-Akpro")

    assert out.empty
    assert seen == [
        ("jean-daniel akpa-akpro", "search/jean-daniel_akpa-akpro"),
        ("jean daniel akpa akpro", "search/jean_daniel_akpa_akpro"),
    ]


def test_search_player_no_fallback_when_forms_identical(monkeypatch):
    seen = []
    monkeypatch.setattr(
        scrape_tm, "fetch", lambda url, key: seen.append(key) or "<html></html>"
    )
    pull_tm.search_player("Filip Đuričić")
    assert seen == ["search/filip_djuricic"]


def test_search_club_returns_best_match_id(monkeypatch):
    monkeypatch.setattr(
        scrape_tm,
        "fetch",
        lambda url, key: '<a href="/ssc-napoli/startseite/verein/6195">Napoli</a>',
    )
    assert pull_tm.search_club("Napoli") == 6195


def test_search_club_returns_none_when_unresolved(monkeypatch):
    monkeypatch.setattr(scrape_tm, "fetch", lambda url, key: "<html></html>")
    assert pull_tm.search_club("Nonexistent FC") is None


def test_search_club_skips_first_hit_when_name_mismatched(monkeypatch):
    # FINDING D: the quick search's first hit isn't reliably the right club.
    # "Real Madrid" scores far below the 80 threshold against "Napoli", so
    # search_club must fall through to the second, name-matching hit.
    html = (
        '<a href="/real-madrid/startseite/verein/9999">Real Madrid</a>'
        '<a href="/ssc-napoli/startseite/verein/6195">Napoli</a>'
    )
    monkeypatch.setattr(scrape_tm, "fetch", lambda url, key: html)
    assert pull_tm.search_club("Napoli") == 6195


def test_search_club_returns_none_when_no_hit_meets_threshold(monkeypatch):
    html = '<a href="/real-madrid/startseite/verein/9999">Real Madrid</a>'
    monkeypatch.setattr(scrape_tm, "fetch", lambda url, key: html)
    assert pull_tm.search_club("Napoli") is None


# ---------------------------------------------------------------------------
# _refine_candidate_birth_dates (FINDING A)
# ---------------------------------------------------------------------------


def test_refine_candidate_birth_dates_sets_exact_date_from_profile(monkeypatch):
    profile_html = (FIXTURES / "tm_profile.html").read_text(encoding="utf-8")
    expected_birth_date = scrape_tm.parse_player_identity(profile_html)["birth_date"]
    assert expected_birth_date is not None  # sanity: the fixture actually has one

    fbref_players = pd.DataFrame({
        "player": ["Khvicha Kvaratskhelia"],
        "born": [2001.0],
        "nationality": ["GEO"],
    })
    estimate = pd.Timestamp(year=2001, month=7, day=1)  # age-derived placeholder
    candidates = pd.DataFrame({
        "tm_id": [502670],
        "name": ["Khvicha Kvaratskhelia"],
        "birth_date": [estimate],
        "nationality": ["GEO"],
        "slug": ["khvicha-kvaratskhelia"],
    })

    fetched = []

    def fake_fetch(url, key):
        fetched.append(key)
        assert url == f"{pull_tm.BASE}/khvicha-kvaratskhelia/profil/spieler/502670"
        return profile_html

    monkeypatch.setattr(scrape_tm, "fetch", fake_fetch)
    out = pull_tm._refine_candidate_birth_dates(fbref_players, candidates)

    assert fetched == ["prof/502670"]
    row = out[out["tm_id"] == 502670].iloc[0]
    assert pd.Timestamp(row["birth_date"]).date() == expected_birth_date
    assert bool(row["birth_date_is_estimate"]) is False


def test_refine_candidate_birth_dates_never_fetches_low_score_candidate(monkeypatch):
    # Same nationality as the FBref player, but "Zaza Nemsadze" is nowhere
    # near a fuzzy-name match for "Khvicha Kvaratskhelia" -- must never be
    # profile-fetched, only the genuine lead is.
    fbref_players = pd.DataFrame({
        "player": ["Khvicha Kvaratskhelia"],
        "born": [2001.0],
        "nationality": ["GEO"],
    })
    estimate = pd.Timestamp(year=2001, month=7, day=1)
    candidates = pd.DataFrame({
        "tm_id": [502670, 111111],
        "name": ["Khvicha Kvaratskhelia", "Zaza Nemsadze"],
        "birth_date": [estimate, estimate],
        "nationality": ["GEO", "GEO"],
        "slug": ["khvicha-kvaratskhelia", "zaza-nemsadze"],
    })

    fetched = []

    def fake_fetch(url, key):
        fetched.append(key)
        return (FIXTURES / "tm_profile.html").read_text(encoding="utf-8")

    monkeypatch.setattr(scrape_tm, "fetch", fake_fetch)
    pull_tm._refine_candidate_birth_dates(fbref_players, candidates)

    assert fetched == ["prof/502670"], "a candidate scoring < 88 must never be profile-fetched"


def test_refine_candidate_birth_dates_skips_mismatched_nationality(monkeypatch):
    """A near-but-not-exact name score (< 100) under a nationality mismatch
    must still be rejected -- only an EXACT (100) name score is allowed to
    bypass the nationality gate (path (c), covering the dual-nationality
    pass in `entities.match_players`). The candidate name here is a
    one-letter typo of the FBref name so token_sort_ratio is high (>= 88)
    but not 100."""
    fbref_players = pd.DataFrame({
        "player": ["Khvicha Kvaratskhelia"],
        "born": [2001.0],
        "nationality": ["GEO"],
    })
    estimate = pd.Timestamp(year=2001, month=7, day=1)
    candidates = pd.DataFrame({
        "tm_id": [502670],
        "name": ["Khvicha Kvaratshelia"],  # one-letter typo -> high but not exact score
        "birth_date": [estimate],
        "nationality": ["BRA"],  # nationality mismatch
        "slug": ["khvicha-kvaratskhelia"],
    })

    fetched = []
    monkeypatch.setattr(scrape_tm, "fetch", lambda url, key: fetched.append(key) or "")
    out = pull_tm._refine_candidate_birth_dates(fbref_players, candidates)

    assert fetched == []
    row = out[out["tm_id"] == 502670].iloc[0]
    assert pd.Timestamp(row["birth_date"]) == estimate
    assert bool(row["birth_date_is_estimate"]) is True


def test_refine_candidate_birth_dates_falls_back_when_profile_lacks_date(monkeypatch):
    fbref_players = pd.DataFrame({
        "player": ["Khvicha Kvaratskhelia"],
        "born": [2001.0],
        "nationality": ["GEO"],
    })
    estimate = pd.Timestamp(year=2001, month=7, day=1)
    candidates = pd.DataFrame({
        "tm_id": [502670],
        "name": ["Khvicha Kvaratskhelia"],
        "birth_date": [estimate],
        "nationality": ["GEO"],
        "slug": ["khvicha-kvaratskhelia"],
    })

    monkeypatch.setattr(scrape_tm, "fetch", lambda url, key: "<html></html>")
    out = pull_tm._refine_candidate_birth_dates(fbref_players, candidates)

    row = out[out["tm_id"] == 502670].iloc[0]
    assert pd.Timestamp(row["birth_date"]) == estimate, "keeps the age-derived estimate"
    assert bool(row["birth_date_is_estimate"]) is True


def test_refine_candidate_birth_dates_tolerates_one_failed_profile_fetch(monkeypatch, caplog):
    """A permanent error (e.g. HTTP 403/404) fetching one candidate's profile
    page must not abort refinement for every other candidate: the failing
    candidate keeps its age-derived estimate with birth_date_is_estimate
    True, the other candidate still gets refined to the exact date, and a
    WARNING naming the failing tm_id is logged."""
    profile_html = (FIXTURES / "tm_profile.html").read_text(encoding="utf-8")
    expected_birth_date = scrape_tm.parse_player_identity(profile_html)["birth_date"]

    fbref_players = pd.DataFrame({
        "player": ["Khvicha Kvaratskhelia", "Zaza Nemsadze"],
        "born": [2001.0, 1999.0],
        "nationality": ["GEO", "GEO"],
    })
    estimate = pd.Timestamp(year=2001, month=7, day=1)
    candidates = pd.DataFrame({
        "tm_id": [502670, 111111],
        "name": ["Khvicha Kvaratskhelia", "Zaza Nemsadze"],
        "birth_date": [estimate, estimate],
        "nationality": ["GEO", "GEO"],
        "slug": ["khvicha-kvaratskhelia", "zaza-nemsadze"],
    })

    def fake_fetch(url, key):
        if key == "prof/111111":
            raise requests.HTTPError("403 Client Error")
        assert key == "prof/502670"
        return profile_html

    monkeypatch.setattr(scrape_tm, "fetch", fake_fetch)
    with caplog.at_level(logging.WARNING):
        out = pull_tm._refine_candidate_birth_dates(fbref_players, candidates)

    failed_row = out[out["tm_id"] == 111111].iloc[0]
    assert pd.Timestamp(failed_row["birth_date"]) == estimate
    assert bool(failed_row["birth_date_is_estimate"]) is True

    ok_row = out[out["tm_id"] == 502670].iloc[0]
    assert pd.Timestamp(ok_row["birth_date"]).date() == expected_birth_date
    assert bool(ok_row["birth_date_is_estimate"]) is False

    assert any("111111" in record.message for record in caplog.records)


def test_refine_candidate_birth_dates_fetches_subset_alias_candidate(monkeypatch):
    """FBref bug: `entities.match_players`'s subset-alias path (TM name's
    token set a SUBSET of FBref's, e.g. TM "Dalbert" for FBref "Dalbert
    Henrique") requires an exact birth date to accept the match, but the
    old gate only ever fetched profiles for the plain token_sort_ratio
    path -- so a same-nationality subset alias like this one was never
    profile-fetched and stayed stuck on the estimate forever. The widened
    gate must fetch it (path (b))."""
    profile_html = (FIXTURES / "tm_profile.html").read_text(encoding="utf-8")
    expected_birth_date = scrape_tm.parse_player_identity(profile_html)["birth_date"]

    fbref_players = pd.DataFrame({
        "player": ["Dalbert Henrique"],
        "born": [1993.0],
        "nationality": ["BRA"],
    })
    estimate = pd.Timestamp(year=1993, month=7, day=1)
    candidates = pd.DataFrame({
        "tm_id": [1],
        "name": ["Dalbert"],
        "birth_date": [estimate],
        "nationality": ["BRA"],
        "slug": ["dalbert"],
    })

    fetched = []

    def fake_fetch(url, key):
        fetched.append(key)
        assert url == f"{pull_tm.BASE}/dalbert/profil/spieler/1"
        return profile_html

    monkeypatch.setattr(scrape_tm, "fetch", fake_fetch)
    out = pull_tm._refine_candidate_birth_dates(fbref_players, candidates)

    assert fetched == ["prof/1"]
    row = out[out["tm_id"] == 1].iloc[0]
    assert pd.Timestamp(row["birth_date"]).date() == expected_birth_date
    assert bool(row["birth_date_is_estimate"]) is False


def test_refine_candidate_birth_dates_never_fetches_reverse_subset_direction(monkeypatch):
    """The subset check only ever runs in the TM-shorter direction: a TM
    candidate whose name is LONGER than the FBref name (e.g. TM "Danilo
    Pereira" for FBref mononym "Danilo") must never be profile-fetched via
    the subset path -- reversing it would let an unrelated, longer TM name
    piggyback on a short FBref mononym (see `entities._best_match`)."""
    fbref_players = pd.DataFrame({
        "player": ["Danilo"],
        "born": [1991.0],
        "nationality": ["BRA"],
    })
    estimate = pd.Timestamp(year=1991, month=7, day=1)
    candidates = pd.DataFrame({
        "tm_id": [2],
        "name": ["Danilo Pereira"],
        "birth_date": [estimate],
        "nationality": ["BRA"],
        "slug": ["danilo-pereira"],
    })

    fetched = []
    monkeypatch.setattr(scrape_tm, "fetch", lambda url, key: fetched.append(key) or "")
    out = pull_tm._refine_candidate_birth_dates(fbref_players, candidates)

    assert fetched == []
    row = out[out["tm_id"] == 2].iloc[0]
    assert bool(row["birth_date_is_estimate"]) is True


def test_refine_candidate_birth_dates_fetches_subset_alias_despite_nationality_mismatch(monkeypatch):
    """Gate (d): `_best_match`'s subset scoring (TM token set a SUBSET of
    FBref's) feeds into `entities.match_players`'s dual-nationality pass
    too whenever `token_set_ratio` is a perfect 100 -- that pass drops the
    nationality gate for any overall score of 100, and a subset match's
    score IS that token_set_ratio. So a subset alias like TM "Dalbert" for
    FBref "Dalbert Henrique" must be profile-fetched even when their
    nationalities disagree, not just when they match (gate (b))."""
    profile_html = (FIXTURES / "tm_profile.html").read_text(encoding="utf-8")
    expected_birth_date = scrape_tm.parse_player_identity(profile_html)["birth_date"]

    fbref_players = pd.DataFrame({
        "player": ["Dalbert Henrique"],
        "born": [1993.0],
        "nationality": ["BRA"],
    })
    estimate = pd.Timestamp(year=1993, month=7, day=1)
    candidates = pd.DataFrame({
        "tm_id": [5],
        "name": ["Dalbert"],
        "birth_date": [estimate],
        "nationality": ["POR"],  # nationality mismatch
        "slug": ["dalbert"],
    })

    fetched = []

    def fake_fetch(url, key):
        fetched.append(key)
        assert url == f"{pull_tm.BASE}/dalbert/profil/spieler/5"
        return profile_html

    monkeypatch.setattr(scrape_tm, "fetch", fake_fetch)
    out = pull_tm._refine_candidate_birth_dates(fbref_players, candidates)

    assert fetched == ["prof/5"]
    row = out[out["tm_id"] == 5].iloc[0]
    assert pd.Timestamp(row["birth_date"]).date() == expected_birth_date
    assert bool(row["birth_date_is_estimate"]) is False


def test_refine_candidate_birth_dates_never_fetches_reverse_subset_with_nationality_mismatch(monkeypatch):
    """Gate (d) is direction-locked the same way as (b): a TM name LONGER
    than the FBref name (e.g. TM "Danilo Pereira" for FBref mononym
    "Danilo") must never be profile-fetched via the subset path, even under
    a nationality mismatch and even though token_set_ratio is symmetric and
    would itself read 100 -- reversing the direction would let an
    unrelated, longer TM name piggyback on a short FBref mononym."""
    fbref_players = pd.DataFrame({
        "player": ["Danilo"],
        "born": [1991.0],
        "nationality": ["BRA"],
    })
    estimate = pd.Timestamp(year=1991, month=7, day=1)
    candidates = pd.DataFrame({
        "tm_id": [6],
        "name": ["Danilo Pereira"],
        "birth_date": [estimate],
        "nationality": ["POR"],  # nationality mismatch
        "slug": ["danilo-pereira"],
    })

    fetched = []
    monkeypatch.setattr(scrape_tm, "fetch", lambda url, key: fetched.append(key) or "")
    out = pull_tm._refine_candidate_birth_dates(fbref_players, candidates)

    assert fetched == []
    row = out[out["tm_id"] == 6].iloc[0]
    assert bool(row["birth_date_is_estimate"]) is True


def test_refine_candidate_birth_dates_fetches_exact_name_despite_nationality_mismatch(monkeypatch):
    """The dual-nationality second pass in `entities.match_players` accepts
    a candidate whose normalised name scores EXACTLY 100 even when
    nationality doesn't match (e.g. "Junior Sambia": Transfermarkt lists
    CTA, FBref FRA) -- and, like the other paths, requires an exact birth
    date. Path (c) must fetch this candidate's profile regardless of the
    nationality mismatch."""
    profile_html = (FIXTURES / "tm_profile.html").read_text(encoding="utf-8")
    expected_birth_date = scrape_tm.parse_player_identity(profile_html)["birth_date"]

    fbref_players = pd.DataFrame({
        "player": ["Junior Sambia"],
        "born": [1996.0],
        "nationality": ["FRA"],
    })
    estimate = pd.Timestamp(year=1996, month=7, day=1)
    candidates = pd.DataFrame({
        "tm_id": [3],
        "name": ["Junior Sambia"],
        "birth_date": [estimate],
        "nationality": ["CTA"],  # nationality mismatch, name is exact
        "slug": ["junior-sambia"],
    })

    fetched = []

    def fake_fetch(url, key):
        fetched.append(key)
        assert url == f"{pull_tm.BASE}/junior-sambia/profil/spieler/3"
        return profile_html

    monkeypatch.setattr(scrape_tm, "fetch", fake_fetch)
    out = pull_tm._refine_candidate_birth_dates(fbref_players, candidates)

    assert fetched == ["prof/3"]
    row = out[out["tm_id"] == 3].iloc[0]
    assert pd.Timestamp(row["birth_date"]).date() == expected_birth_date
    assert bool(row["birth_date_is_estimate"]) is False


def test_refine_candidate_birth_dates_never_fetches_non_subset_low_score_candidate(monkeypatch):
    """A candidate that is neither a full-name match (>= 88), nor a
    same-nationality subset alias (>= 88 token_set_ratio), nor an exact
    (100) name match regardless of nationality, must never be
    profile-fetched -- existing low-score rejection behaviour must survive
    the widened gate."""
    fbref_players = pd.DataFrame({
        "player": ["Marco Rossi"],
        "born": [1995.0],
        "nationality": ["ITA"],
    })
    estimate = pd.Timestamp(year=1995, month=7, day=1)
    candidates = pd.DataFrame({
        "tm_id": [4],
        "name": ["Mario Bianchi"],  # unrelated name, same nationality
        "birth_date": [estimate],
        "nationality": ["ITA"],
        "slug": ["mario-bianchi"],
    })

    fetched = []
    monkeypatch.setattr(scrape_tm, "fetch", lambda url, key: fetched.append(key) or "")
    out = pull_tm._refine_candidate_birth_dates(fbref_players, candidates)

    assert fetched == []
    row = out[out["tm_id"] == 4].iloc[0]
    assert bool(row["birth_date_is_estimate"]) is True


def test_refine_candidate_birth_dates_handles_empty_candidates():
    fbref_players = pd.DataFrame({
        "player": ["Nobody"], "born": [2000.0], "nationality": ["ITA"],
    })
    candidates = pd.DataFrame(columns=scrape_tm.SEARCH_COLUMNS)
    out = pull_tm._refine_candidate_birth_dates(fbref_players, candidates)
    assert out.empty
    assert "birth_date_is_estimate" in out.columns


# ---------------------------------------------------------------------------
# pull_club_values
# ---------------------------------------------------------------------------


def test_pull_club_values_records_nan_for_unresolvable_club(tmp_path, monkeypatch):
    monkeypatch.setattr(pull_tm.config, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(pull_tm, "search_club", lambda name: None)
    pull_tm.pull_club_values(["Nonexistent FC"])
    out = pd.read_parquet(tmp_path / "club_values.parquet")
    assert pd.isna(out["squad_value_eur"].iloc[0])


def test_pull_club_values_resolves_and_fetches_squad_value(tmp_path, monkeypatch):
    monkeypatch.setattr(pull_tm.config, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(pull_tm, "search_club", lambda name: 6195)
    monkeypatch.setattr(
        scrape_tm,
        "fetch",
        lambda url, key: '<div class="data-header__market-value-wrapper">€140.00m</div>',
    )
    pull_tm.pull_club_values(["Napoli"])
    out = pd.read_parquet(tmp_path / "club_values.parquet")
    assert out["squad_value_eur"].iloc[0] == 140e6
    assert out["team"].iloc[0] == "Napoli"


# ---------------------------------------------------------------------------
# _pull_one
# ---------------------------------------------------------------------------


def test_pull_one_fetches_injuries_value_and_transfer_json(monkeypatch):
    calls = []

    def fake_fetch(url, key):
        calls.append(key)
        if key.startswith("inj/"):
            return (FIXTURES / "tm_injuries.html").read_text(encoding="utf-8")
        if key.startswith("mv/"):
            return (FIXTURES / "tm_value_graph.json").read_text(encoding="utf-8")
        if key.startswith("tr/"):
            return (FIXTURES / "tm_transfers.json").read_text(encoding="utf-8")
        raise AssertionError(f"unexpected fetch key {key!r}")

    monkeypatch.setattr(scrape_tm, "fetch", fake_fetch)
    result = pull_tm._pull_one(502670, "Khvicha Kvaratskhelia")

    assert calls == ["inj/502670", "mv/502670", "tr/502670"]
    assert set(result) == {"injuries", "values", "fees"}
    assert list(result["injuries"].columns) == scrape_tm.INJURY_COLUMNS
    assert list(result["values"].columns) == scrape_tm.MV_COLUMNS
    assert list(result["fees"].columns) == scrape_tm.TRANSFER_COLUMNS
    assert not result["injuries"].empty
    assert not result["values"].empty
    assert not result["fees"].empty


def test_pull_one_uses_ceapi_url_templates(monkeypatch):
    urls = []
    monkeypatch.setattr(
        scrape_tm,
        "fetch",
        lambda url, key: urls.append(url) or "{}",
    )
    pull_tm._pull_one(999, "Some Player")
    assert urls[1] == scrape_tm.MV_URL.format(id=999)
    assert urls[2] == scrape_tm.TRANSFERS_URL.format(id=999)


# ---------------------------------------------------------------------------
# pull_players resumability
# ---------------------------------------------------------------------------


def test_pull_players_skips_already_pulled_players(tmp_path, monkeypatch):
    """Adapted from the task brief's `test_build_skips_players_already_pulled`:
    the stub frames here carry one real row each (rather than being wholly
    empty) so that "already pulled" can actually be observed from disk, the
    way `pull_players` determines it (via the `tm_pulled.csv` manifest --
    FINDING C)."""
    monkeypatch.setattr(pull_tm.config, "PROCESSED_DIR", tmp_path)
    calls = []

    def fake_pull_one(tm_id, name):
        calls.append(tm_id)
        return {
            "injuries": pd.DataFrame([{
                "season": "2223", "injury": "Hamstring strain",
                "days_missed": 10, "games_missed": 2,
            }]),
            "values": pd.DataFrame([{"date": pd.Timestamp("2023-01-01"), "value_eur": 5e6}]),
            "fees": pd.DataFrame([{
                "season": "2223", "from_club": "X", "to_club": "Y", "fee_eur": 1e6,
                "is_loan": False, "transfer_date": pd.Timestamp("2022-08-01"),
            }]),
        }

    monkeypatch.setattr(pull_tm, "_pull_one", fake_pull_one)
    matches = pd.DataFrame({"fbref_player": ["A"], "tm_id": [1], "score": [100]})

    pull_tm.pull_players(matches)
    pull_tm.pull_players(matches)

    assert calls == [1], "second run must reuse cached parquet, not refetch"
    out = pd.read_parquet(tmp_path / "injuries.parquet")
    assert list(out["player"]) == ["A"]
    manifest = pd.read_csv(tmp_path / "tm_pulled.csv")
    assert list(manifest["player"]) == ["A"]


def test_pull_players_manifest_survives_zero_injury_player(tmp_path, monkeypatch):
    """FINDING C: resumability keyed on `injuries.parquet["player"]` meant a
    player whose injuries page legitimately parses to zero rows was never
    detectable as "already pulled" and got re-pulled -- and his value/fee
    rows re-appended -- on every run. Three `pull_players` calls for a
    player with an empty injuries frame must fetch him exactly once, list
    him in the manifest, and never duplicate his value/fee rows."""
    monkeypatch.setattr(pull_tm.config, "PROCESSED_DIR", tmp_path)
    calls = []

    def fake_pull_one(tm_id, name):
        calls.append(tm_id)
        return {
            "injuries": pd.DataFrame(columns=scrape_tm.INJURY_COLUMNS),
            "values": pd.DataFrame([{"date": pd.Timestamp("2023-01-01"), "value_eur": 5e6}]),
            "fees": pd.DataFrame([{
                "season": "2223", "from_club": "X", "to_club": "Y", "fee_eur": 1e6,
                "is_loan": False, "transfer_date": pd.Timestamp("2022-08-01"),
            }]),
        }

    monkeypatch.setattr(pull_tm, "_pull_one", fake_pull_one)
    matches = pd.DataFrame({"fbref_player": ["ZeroInjuryGuy"], "tm_id": [7], "score": [100]})

    pull_tm.pull_players(matches)
    pull_tm.pull_players(matches)
    pull_tm.pull_players(matches)

    assert calls == [7], "a zero-injury player must still be pulled exactly once"

    injuries = pd.read_parquet(tmp_path / "injuries.parquet")
    values = pd.read_parquet(tmp_path / "market_values.parquet")
    fees = pd.read_parquet(tmp_path / "fees.parquet")
    assert injuries.empty
    assert len(values) == 1, "no duplicate market-value rows across three runs"
    assert len(fees) == 1, "no duplicate fee rows across three runs"

    manifest = pd.read_csv(tmp_path / "tm_pulled.csv")
    assert list(manifest["player"]) == ["ZeroInjuryGuy"]


def test_pull_players_writes_well_formed_empty_parquets_when_nothing_to_pull(tmp_path, monkeypatch):
    monkeypatch.setattr(pull_tm.config, "PROCESSED_DIR", tmp_path)
    matches = pd.DataFrame(columns=["fbref_player", "tm_id", "score"])

    pull_tm.pull_players(matches)

    injuries = pd.read_parquet(tmp_path / "injuries.parquet")
    values = pd.read_parquet(tmp_path / "market_values.parquet")
    fees = pd.read_parquet(tmp_path / "fees.parquet")
    assert list(injuries.columns) == scrape_tm.INJURY_COLUMNS + ["player"]
    assert list(values.columns) == scrape_tm.MV_COLUMNS + ["player"]
    assert list(fees.columns) == scrape_tm.TRANSFER_COLUMNS + ["player"]
    assert injuries.empty and values.empty and fees.empty


# ---------------------------------------------------------------------------
# build() scoping
# ---------------------------------------------------------------------------


def test_build_scopes_to_arrival_players(tmp_path, monkeypatch):
    monkeypatch.setattr(pull_tm.config, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(pull_tm.config, "MANUAL_DIR", tmp_path)

    fbref_identities = pd.DataFrame({
        "player": ["Arrival One", "Not An Arrival"],
        "born": [1998.0, 1990.0],
        "nationality": ["ITA", "ITA"],
    })
    transfers = pd.DataFrame({
        "player": ["Arrival One"],
        "to_team": ["Napoli"],
        "from_team": ["Udinese"],
    })

    def fake_read_parquet(path):
        name = Path(path).name
        if name == "fbref_identities.parquet":
            return fbref_identities
        if name == "transfers.parquet":
            return transfers
        raise FileNotFoundError(path)

    monkeypatch.setattr(pull_tm.pd, "read_parquet", fake_read_parquet)

    searched = []
    monkeypatch.setattr(
        pull_tm,
        "search_player",
        lambda name: searched.append(name) or pd.DataFrame(columns=scrape_tm.SEARCH_COLUMNS),
    )
    pulled_matches = []
    monkeypatch.setattr(pull_tm, "pull_players", lambda matches: pulled_matches.append(matches))
    club_calls = []
    monkeypatch.setattr(pull_tm, "pull_club_values", lambda names: club_calls.append(list(names)))

    pull_tm.build()

    assert searched == ["Arrival One"], "must only search players present in transfers.parquet"
    assert club_calls == [["Napoli", "Udinese"]]
    assert (tmp_path / "unmatched_review.csv").exists()
