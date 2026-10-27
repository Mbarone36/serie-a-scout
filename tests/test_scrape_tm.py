import datetime
from pathlib import Path

import pandas as pd
import pytest
import requests

from src import scrape_tm

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def injuries_html() -> str:
    return (FIXTURES / "tm_injuries.html").read_text(encoding="utf-8")


@pytest.fixture
def profile_html() -> str:
    return (FIXTURES / "tm_profile.html").read_text(encoding="utf-8")


@pytest.fixture
def value_graph_json() -> str:
    return (FIXTURES / "tm_value_graph.json").read_text(encoding="utf-8")


@pytest.fixture
def transfers_json() -> str:
    return (FIXTURES / "tm_transfers.json").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# parse_injuries
# ---------------------------------------------------------------------------


def test_parse_injuries_returns_expected_columns(injuries_html):
    out = scrape_tm.parse_injuries(injuries_html)
    assert list(out.columns) == ["season", "injury", "days_missed", "games_missed"]


def test_parse_injuries_yields_numeric_days(injuries_html):
    out = scrape_tm.parse_injuries(injuries_html)
    assert pd.api.types.is_integer_dtype(out["days_missed"])
    assert (out["days_missed"] >= 0).all()


def test_parse_injuries_normalises_season_format(injuries_html):
    out = scrape_tm.parse_injuries(injuries_html)
    assert not out.empty
    assert out["season"].str.fullmatch(r"\d{4}").all()


def test_parse_injuries_has_rows_from_fixture(injuries_html):
    out = scrape_tm.parse_injuries(injuries_html)
    assert len(out) == 4
    first = out.iloc[0]
    assert first["season"] == "2526"
    assert first["injury"] == "Ankle sprain"
    assert first["days_missed"] == 8
    assert first["games_missed"] == 1


def test_parse_injuries_on_player_with_no_injuries():
    html = "<html><body><div>No injuries recorded</div></body></html>"
    out = scrape_tm.parse_injuries(html)
    assert out.empty
    assert list(out.columns) == ["season", "injury", "days_missed", "games_missed"]


def test_days_to_int_handles_transfermarkt_formatting():
    assert scrape_tm._days_to_int("42 days") == 42
    assert scrape_tm._days_to_int("-") == 0
    assert scrape_tm._days_to_int("") == 0


def test_normalise_season():
    assert scrape_tm._normalise_season("21/22") == "2122"
    assert scrape_tm._normalise_season("") == ""


# ---------------------------------------------------------------------------
# parse_player_identity
# ---------------------------------------------------------------------------


def test_parse_player_identity(profile_html):
    ident = scrape_tm.parse_player_identity(profile_html)
    assert set(ident) == {"name", "birth_date", "nationality"}
    assert ident["birth_date"] is not None
    assert ident["name"] == "Khvicha Kvaratskhelia"
    assert ident["birth_date"] == datetime.date(2001, 2, 12)
    assert ident["nationality"] == "Georgia"


# ---------------------------------------------------------------------------
# parse_market_value_history
# ---------------------------------------------------------------------------


def test_parse_market_value_history_is_chronological(value_graph_json):
    out = scrape_tm.parse_market_value_history(value_graph_json)
    assert list(out.columns) == ["date", "value_eur"]
    assert not out.empty
    assert out["date"].is_monotonic_increasing
    assert (out["value_eur"] > 0).all()


def test_parse_market_value_history_row_count(value_graph_json):
    out = scrape_tm.parse_market_value_history(value_graph_json)
    assert len(out) == 5


def test_parse_market_value_history_invalid_json():
    out = scrape_tm.parse_market_value_history("not json")
    assert out.empty
    assert list(out.columns) == ["date", "value_eur"]


def test_parse_market_value_history_empty_list_json():
    out = scrape_tm.parse_market_value_history('{"list": []}')
    assert out.empty


# ---------------------------------------------------------------------------
# parse_transfer_history
# ---------------------------------------------------------------------------


def test_parse_transfer_history_returns_expected_columns(transfers_json):
    out = scrape_tm.parse_transfer_history(transfers_json)
    assert list(out.columns) == [
        "season", "from_club", "to_club", "fee_eur", "is_loan", "transfer_date",
    ]


def test_parse_transfer_history_row_count_and_dtype(transfers_json):
    out = scrape_tm.parse_transfer_history(transfers_json)
    assert len(out) == 3
    assert pd.api.types.is_float_dtype(out["fee_eur"])
    assert pd.api.types.is_bool_dtype(out["is_loan"])


def test_parse_transfer_history_psg_move(transfers_json):
    out = scrape_tm.parse_transfer_history(transfers_json)
    row = out[(out["season"] == "2425") & (out["to_club"] == "PSG")].iloc[0]
    assert row["from_club"] == "Napoli"
    assert row["fee_eur"] == 80e6


def test_parse_transfer_history_includes_transfer_date(transfers_json):
    # dateUnformatted in the fixture is already ISO ("2025-01-17").
    out = scrape_tm.parse_transfer_history(transfers_json)
    assert pd.api.types.is_datetime64_any_dtype(out["transfer_date"])
    row = out[(out["season"] == "2425") & (out["to_club"] == "PSG")].iloc[0]
    assert row["transfer_date"] == pd.Timestamp("2025-01-17")


def test_parse_transfer_history_invalid_json():
    out = scrape_tm.parse_transfer_history("not json")
    assert out.empty
    assert list(out.columns) == [
        "season", "from_club", "to_club", "fee_eur", "is_loan", "transfer_date",
    ]


def test_parse_transfer_history_marks_loan_rows(transfers_json):
    # Fixture has two loan-flavoured rows: "End of loan" (18/19, Loko Moscow
    # -> Rustavi) and "Loan fee: <br/><i>€100k</i>" (18/19, Rustavi -> Loko
    # Moscow) -- everything else is a plain fee, a free-ish '-', or a real sale.
    out = scrape_tm.parse_transfer_history(transfers_json)
    end_of_loan = out[(out["season"] == "1819") & (out["to_club"] == "Rustavi")].iloc[0]
    assert end_of_loan["is_loan"]
    assert end_of_loan["fee_eur"] == 0.0

    loan_with_fee = out[(out["season"] == "1819") & (out["to_club"] == "Loko Moscow")].iloc[0]
    assert loan_with_fee["is_loan"]
    assert loan_with_fee["fee_eur"] == 100e3

    psg_sale = out[(out["season"] == "2425") & (out["to_club"] == "PSG")].iloc[0]
    assert not psg_sale["is_loan"]


def test_is_loan_synthetic_loan_fee_row():
    assert scrape_tm._is_loan("Loan fee: €2.00m")
    assert scrape_tm._fee_to_eur("Loan fee: €2.00m") == 2e6


def test_fee_to_eur_cases():
    assert scrape_tm._fee_to_eur("€80.00m") == 80e6
    assert scrape_tm._fee_to_eur("€450k") == 450e3
    assert scrape_tm._fee_to_eur("free transfer") == 0.0
    assert scrape_tm._fee_to_eur("loan transfer") == 0.0
    assert scrape_tm._fee_to_eur("End of loan") == 0.0
    assert scrape_tm._fee_to_eur("-") != scrape_tm._fee_to_eur("-")  # NaN
    assert scrape_tm._fee_to_eur("?") != scrape_tm._fee_to_eur("?")  # NaN


# ---------------------------------------------------------------------------
# parse_club_squad_value
# ---------------------------------------------------------------------------


def test_parse_club_squad_value_against_profile_fixture(profile_html):
    # No club-page fixture exists; the profile fixture shares the same
    # `.data-header__market-value-wrapper` markup so it's used as a proxy.
    value = scrape_tm.parse_club_squad_value(profile_html)
    assert value == 140e6


@pytest.mark.parametrize(
    "text,expected",
    [
        ("€ 140.00 m Last update: 01/06/2026", 140e6),
        ("€1.20bn", 1.2e9),
        ("€450k", 450e3),
        ("€500.50m", 500.5e6),
    ],
)
def test_parse_club_squad_value_synthetic(text, expected):
    html = f'<div class="data-header__market-value-wrapper">{text}</div>'
    assert scrape_tm.parse_club_squad_value(html) == expected


def test_parse_club_squad_value_missing_node_returns_nan():
    value = scrape_tm.parse_club_squad_value("<html><body></body></html>")
    assert value != value  # NaN


# ---------------------------------------------------------------------------
# module-level URL templates
# ---------------------------------------------------------------------------


def test_url_templates_format():
    assert scrape_tm.MV_URL.format(id=502670) == (
        "https://www.transfermarkt.com/ceapi/marketValueDevelopment/graph/502670"
    )
    assert scrape_tm.TRANSFERS_URL.format(id=502670) == (
        "https://www.transfermarkt.com/ceapi/transferHistory/list/502670"
    )


# ---------------------------------------------------------------------------
# parse_search_results
# ---------------------------------------------------------------------------


@pytest.fixture
def search_html() -> str:
    return (FIXTURES / "tm_search.html").read_text(encoding="utf-8")


def test_parse_search_results_returns_candidate_identities(search_html):
    out = scrape_tm.parse_search_results(search_html)
    assert list(out.columns) == ["tm_id", "name", "birth_date", "nationality", "slug"]
    assert len(out) >= 1
    assert out["tm_id"].notna().all()


def test_parse_search_results_on_empty_page():
    out = scrape_tm.parse_search_results("<html><body>no hits</body></html>")
    assert out.empty
    assert list(out.columns) == ["tm_id", "name", "birth_date", "nationality", "slug"]


def test_parse_search_results_finds_khvicha_among_namesakes(search_html, monkeypatch):
    # The fixture (query "kvaratskhelia") holds three different players sharing
    # that surname; Khvicha Kvaratskhelia is tm_id 502670. The page only
    # exposes age (not a birth date) and a full country name (not a 3-letter
    # code), so both are reconstructed -- pin `_today` for a deterministic
    # birth year out of the fixture's static "25".
    monkeypatch.setattr(scrape_tm, "_today", lambda: datetime.date(2026, 9, 1))
    out = scrape_tm.parse_search_results(search_html)
    assert len(out) == 3

    row = out[out["tm_id"] == 502670].iloc[0]
    assert row["name"] == "Khvicha Kvaratskhelia"
    assert row["nationality"] == "GEO"
    assert pd.Timestamp(row["birth_date"]).year == 2001
    assert row["slug"] == "khvicha-kvaratskhelia"


def test_country_to_code_known_and_fallback():
    assert scrape_tm._country_to_code("Georgia") == "GEO"
    assert scrape_tm._country_to_code("Ivory Coast") == "CIV"
    assert scrape_tm._country_to_code("") == ""
    assert scrape_tm._country_to_code("Wakanda") == "WAK"


def test_country_to_code_transfermarkt_spellings():
    """FINDING (diagnosis a): Transfermarkt's own flag-title spellings for
    Türkiye, Cote d'Ivoire and The Gambia previously fell through to the
    first-3-letters fallback (TRK/COT/THE), which don't match FBref's
    trigraphs (TUR/CIV/GAM) and so silently failed the nationality gate in
    `entities.match_players`."""
    assert scrape_tm._country_to_code("Türkiye") == "TUR"
    assert scrape_tm._country_to_code("Cote d'Ivoire") == "CIV"
    assert scrape_tm._country_to_code("The Gambia") == "GAM"


def test_birth_date_from_age_uses_reference_year(monkeypatch):
    monkeypatch.setattr(scrape_tm, "_today", lambda: datetime.date(2026, 9, 1))
    birth_date = scrape_tm._birth_date_from_age("25")
    assert pd.Timestamp(birth_date).year == 2001
    assert pd.isna(scrape_tm._birth_date_from_age(""))


# ---------------------------------------------------------------------------
# parse_club_search_results
# ---------------------------------------------------------------------------


def test_parse_club_search_results_returns_ids_best_first():
    # FINDING D: parse_club_search_results now returns (club_id, label)
    # pairs so search_club can verify a hit is actually a plausible name
    # match before trusting it, instead of blindly taking the first hit.
    html = ('<a href="/ssc-napoli/startseite/verein/6195">Napoli</a>'
            '<a href="/us-lecce/startseite/verein/1005">Lecce</a>')
    assert scrape_tm.parse_club_search_results(html) == [
        (6195, "Napoli"), (1005, "Lecce"),
    ]


def test_parse_club_search_results_on_empty_page():
    assert scrape_tm.parse_club_search_results("<html></html>") == []


def test_parse_club_search_results_deduplicates():
    html = ('<a href="/ssc-napoli/startseite/verein/6195">Napoli</a>'
            '<a href="/ssc-napoli/spielplan/verein/6195">Napoli fixtures</a>')
    assert scrape_tm.parse_club_search_results(html) == [(6195, "Napoli")]


def test_parse_club_search_results_falls_back_to_slug_when_link_has_no_text():
    html = '<a href="/ssc-napoli/startseite/verein/6195"><img alt="Napoli"></a>'
    assert scrape_tm.parse_club_search_results(html) == [(6195, "ssc napoli")]


# ---------------------------------------------------------------------------
# fetch() retry/backoff on transient failures
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, text="ok", status_code=200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} error", response=self)


def test_fetch_retries_on_read_timeout_then_succeeds(monkeypatch, tmp_path):
    monkeypatch.setattr(scrape_tm.config, "RAW_DIR", tmp_path)
    waits = []
    monkeypatch.setattr(scrape_tm._limiter, "wait", lambda: waits.append(1))
    sleeps = []
    monkeypatch.setattr(scrape_tm.time, "sleep", lambda s: sleeps.append(s))

    timeouts = []

    def fake_get(url, headers=None, timeout=None):
        timeouts.append(timeout)
        if len(timeouts) < 3:
            raise scrape_tm.requests.exceptions.ReadTimeout("read timed out")
        return _FakeResponse(text="ok")

    monkeypatch.setattr(scrape_tm.requests, "get", fake_get)
    out = scrape_tm.fetch("https://example.com/x", key="retry/ok")

    assert out == "ok"
    assert timeouts == [30, 60, 90]
    assert sleeps == [10, 20]
    assert len(waits) == 3, "rate limiter must be armed before EVERY attempt"
    assert (tmp_path / "tm" / "retry" / "ok.html").exists()


def test_fetch_retries_on_5xx_then_succeeds(monkeypatch, tmp_path):
    monkeypatch.setattr(scrape_tm.config, "RAW_DIR", tmp_path)
    monkeypatch.setattr(scrape_tm._limiter, "wait", lambda: None)
    monkeypatch.setattr(scrape_tm.time, "sleep", lambda s: None)

    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(timeout)
        if len(calls) < 3:
            return _FakeResponse(status_code=503)
        return _FakeResponse(text="ok")

    monkeypatch.setattr(scrape_tm.requests, "get", fake_get)
    out = scrape_tm.fetch("https://example.com/y", key="retry/5xx")

    assert out == "ok"
    assert calls == [30, 60, 90]


def test_fetch_raises_after_three_failures_and_does_not_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(scrape_tm.config, "RAW_DIR", tmp_path)
    monkeypatch.setattr(scrape_tm._limiter, "wait", lambda: None)
    sleeps = []
    monkeypatch.setattr(scrape_tm.time, "sleep", lambda s: sleeps.append(s))

    def fake_get(url, headers=None, timeout=None):
        raise scrape_tm.requests.exceptions.ReadTimeout("read timed out")

    monkeypatch.setattr(scrape_tm.requests, "get", fake_get)

    with pytest.raises(scrape_tm.requests.exceptions.ReadTimeout):
        scrape_tm.fetch("https://example.com/z", key="retry/fail")

    assert sleeps == [10, 20], "two backoff sleeps between three attempts"
    assert not (tmp_path / "tm" / "retry" / "fail.html").exists(), "cache written only on success"


def test_fetch_does_not_retry_on_4xx(monkeypatch, tmp_path):
    monkeypatch.setattr(scrape_tm.config, "RAW_DIR", tmp_path)
    monkeypatch.setattr(scrape_tm._limiter, "wait", lambda: None)
    sleeps = []
    monkeypatch.setattr(scrape_tm.time, "sleep", lambda s: sleeps.append(s))

    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(timeout)
        return _FakeResponse(status_code=404)

    monkeypatch.setattr(scrape_tm.requests, "get", fake_get)

    with pytest.raises(scrape_tm.requests.exceptions.HTTPError):
        scrape_tm.fetch("https://example.com/w", key="retry/404")

    assert calls == [30], "a 4xx must not be retried"
    assert sleeps == []
