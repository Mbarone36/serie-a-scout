import pandas as pd
import pytest
from src import scrape_fbref


def test_stat_types_cover_required_tables():
    for needed in ("standard", "shooting", "passing", "possession", "defense", "misc", "playing_time"):
        assert needed in scrape_fbref.STAT_TYPES


def test_merge_stat_tables_joins_on_player_identity():
    standard = pd.DataFrame({
        "player": ["A", "B"], "team": ["X", "Y"], "pos": ["FW", "MF"], "minutes": [900, 800],
    })
    shooting = pd.DataFrame({
        "player": ["A", "B"], "team": ["X", "Y"], "npxg": [5.0, 1.0],
    })
    out = scrape_fbref.merge_stat_tables({"standard": standard, "shooting": shooting})
    assert list(out.columns) == ["player", "team", "pos", "minutes", "npxg"]
    assert out.loc[out.player == "A", "npxg"].item() == 5.0


def test_merge_stat_tables_tolerates_duplicated_column_within_a_table():
    # The live "playing_time" table flattens two sub-headers to the same
    # name; merge_stat_tables must dedupe each table's columns before
    # joining rather than choking on (or fanning out from) the duplicate.
    standard = pd.DataFrame({
        "player": ["A"], "team": ["X"], "pos": ["FW"], "playing_time_mp": [30],
    })
    playing_time = pd.DataFrame(
        [["A", "X", 2700, 2700]],
        columns=["player", "team", "playing_time_min", "playing_time_min"],
    )
    out = scrape_fbref.merge_stat_tables({"standard": standard, "playing_time": playing_time})
    assert list(out.columns).count("playing_time_min") == 1
    assert len(out) == 1
    assert out["playing_time_min"].iloc[0] == 2700


def test_merge_stat_tables_keeps_all_players_from_standard():
    standard = pd.DataFrame({"player": ["A", "B"], "team": ["X", "Y"], "pos": ["FW", "MF"], "minutes": [900, 10]})
    shooting = pd.DataFrame({"player": ["A"], "team": ["X"], "npxg": [5.0]})
    out = scrape_fbref.merge_stat_tables({"standard": standard, "shooting": shooting})
    assert len(out) == 2
    assert pd.isna(out.loc[out.player == "B", "npxg"].item())


def test_normalise_identity_extracts_three_letter_nationality():
    frame = pd.DataFrame({"player": ["A"], "nation": ["it ITA"], "team": ["X"],
                          "season": ["2223"], "playing_time_mp": [30],
                          "pos": ["FW"], "born": [1999], "age": [24], "minutes": [900]})
    out = scrape_fbref.normalise_identity(frame)
    assert out["nationality"].iloc[0] == "ITA"


def test_normalise_identity_derives_team_matches_from_squad_maximum():
    frame = pd.DataFrame({"player": ["A", "B"], "team": ["X", "X"],
                          "season": ["2223", "2223"], "playing_time_mp": [30, 38],
                          "pos": ["FW", "MF"], "born": [1999, 2000],
                          "age": [24, 23], "minutes": [900, 800]})
    out = scrape_fbref.normalise_identity(frame)
    assert set(out["team_matches"]) == {38}


def test_check_required_columns_reports_missing_names():
    frame = pd.DataFrame({"player": ["A"], "team": ["X"]})
    assert scrape_fbref.check_required_columns(frame) == [
        "pos", "born", "age", "minutes", "playing_time_mp",
    ]


def test_normalise_identity_raises_on_missing_minutes():
    frame = pd.DataFrame({
        "player": ["A"], "team": ["X"], "pos": ["FW"], "born": [1999],
        "age": [24], "playing_time_mp": [30], "season": ["2223"],
    })
    with pytest.raises(ValueError, match="minutes"):
        scrape_fbref.normalise_identity(frame)


def test_normalise_identity_passes_with_complete_frame():
    frame = pd.DataFrame({
        "player": ["A"], "team": ["X"], "pos": ["FW"], "born": [1999],
        "age": [24], "minutes": [900], "playing_time_mp": [30],
        "nation": ["it ITA"], "season": ["2223"],
    })
    out = scrape_fbref.normalise_identity(frame)
    assert scrape_fbref.check_required_columns(out) == []


def test_normalise_identity_derives_minutes_from_live_playing_time_min():
    # FBref's live pages (seasons "2324"+) carry no `minutes` column at all —
    # the closest equivalent is `playing_time_min`. This is the live-shaped
    # frame: no `minutes`, and none of the advanced tables' columns either.
    frame = pd.DataFrame({
        "player": ["A"], "team": ["X"], "pos": ["FW"], "born": [1999],
        "age": [24], "playing_time_mp": [30], "playing_time_min": [2700],
        "nation": ["it ITA"], "season": ["2324"],
    })
    out = scrape_fbref.normalise_identity(frame)
    assert scrape_fbref.check_required_columns(out) == []
    assert out["minutes"].iloc[0] == 2700
    assert out["nationality"].iloc[0] == "ITA"
    assert out["team_matches"].iloc[0] == 30


def test_normalise_identity_dedupes_duplicate_named_columns():
    # The live "playing_time" table flattens two sub-headers to the same
    # name; a duplicate `playing_time_min` must not reach the required-
    # columns check as two same-named columns, and the surviving one (the
    # first) is what `minutes` gets derived from.
    frame = pd.DataFrame(
        [["A", "X", "FW", 1999, 24, 30, 2700, 2700, "it ITA", "2324"]],
        columns=["player", "team", "pos", "born", "age", "playing_time_mp",
                 "playing_time_min", "playing_time_min", "nation", "season"],
    )
    assert frame.columns.duplicated().any()  # sanity: the fixture really is duplicated

    out = scrape_fbref.normalise_identity(frame)
    assert list(out.columns).count("playing_time_min") == 1
    assert out["minutes"].iloc[0] == 2700


def test_flatten_multiindex_columns():
    df = pd.DataFrame(
        [[1, 2]],
        columns=pd.MultiIndex.from_tuples([("Standard", "Gls"), ("Expected", "npxG")]),
    )
    out = scrape_fbref.flatten_columns(df)
    assert list(out.columns) == ["standard_gls", "expected_npxg"]


def test_flatten_multiindex_columns_translates_percent_before_squash():
    # "SoT" and "SoT%" must NOT collapse to the same name — '%' -> '_pct'
    # has to happen before the non-alphanumeric squash, or the shooting
    # table's SoT and SoT% columns silently collide into one "standard_sot"
    # column (dropping one of the two).
    df = pd.DataFrame(
        [[1, 2]],
        columns=pd.MultiIndex.from_tuples([("Standard", "SoT"), ("Standard", "SoT%")]),
    )
    out = scrape_fbref.flatten_columns(df)
    assert list(out.columns) == ["standard_sot", "standard_sot_pct"]
    assert len(set(out.columns)) == 2


def test_normalise_team_columns_maps_poss_to_possession():
    # FBref's live team pages flatten possession to "poss", but the club
    # style vector (src.clubs.STYLE_COLUMNS) expects the historical dump's
    # name "possession".
    frame = pd.DataFrame([["Milan", 55.2]], columns=["team", "poss"])
    out = scrape_fbref.normalise_team_columns(frame)
    assert "possession" in out.columns
    assert "poss" not in out.columns
    assert out["possession"].iloc[0] == 55.2


def test_normalise_team_columns_leaves_existing_possession_unchanged():
    frame = pd.DataFrame([["Milan", 55.2, 40.0]], columns=["team", "possession", "poss"])
    out = scrape_fbref.normalise_team_columns(frame)
    assert out["possession"].iloc[0] == 55.2


def test_normalise_team_columns_dedupes_duplicate_named_columns():
    frame = pd.DataFrame(
        [["Milan", 55.2, 999.0]],
        columns=["team", "possession", "possession"],
    )
    assert frame.columns.duplicated().any()  # sanity: the fixture really is duplicated

    out = scrape_fbref.normalise_team_columns(frame)
    assert list(out.columns).count("possession") == 1
    assert out["possession"].iloc[0] == 55.2


def test_patch_league_aliases_adds_bundesliga_without_overwriting_existing_keys(monkeypatch):
    fake_dict = {"Fußball-Bundesliga": "GER-Bundesliga", "Serie A": "SOME-OTHER-VALUE"}
    monkeypatch.setattr(scrape_fbref.sd.fbref, "BIG_FIVE_DICT", fake_dict)

    scrape_fbref._patch_league_aliases()

    assert fake_dict["Bundesliga"] == "GER-Bundesliga"
    # setdefault: pre-existing keys are left untouched, not overwritten.
    assert fake_dict["Serie A"] == "SOME-OTHER-VALUE"
    assert fake_dict["Fußball-Bundesliga"] == "GER-Bundesliga"


def test_split_league_selects_requested_league_rows():
    frame = pd.DataFrame({
        "league": ["ITA-Serie A", "ENG-Premier League", "ITA-Serie A"],
        "player": ["A", "B", "C"],
    })
    out = scrape_fbref.split_league(frame, "ITA-Serie A")
    assert list(out["player"]) == ["A", "C"]


def test_split_league_raises_when_no_league_column():
    frame = pd.DataFrame({"player": ["A"], "team": ["X"]})
    with pytest.raises(ValueError, match="league"):
        scrape_fbref.split_league(frame, "ITA-Serie A")


def test_split_league_returns_empty_when_no_rows_match():
    frame = pd.DataFrame({"league": ["ENG-Premier League"], "player": ["B"]})
    out = scrape_fbref.split_league(frame, "ITA-Serie A")
    assert out.empty
    assert list(out.columns) == list(frame.columns)


def test_split_league_warns_on_nan_league_rows_and_leaves_them_out(caplog):
    # A NaN league means an FBref competition name soccerdata's BIG_FIVE_DICT
    # doesn't map (e.g. a rename split_league hasn't seen an alias for yet).
    # It must not be guessed at — only matching rows come back, but a
    # WARNING naming the count makes a future rename visible.
    frame = pd.DataFrame({
        "league": ["ITA-Serie A", None, None],
        "player": ["A", "B", "C"],
    })
    with caplog.at_level("WARNING", logger="src.scrape_fbref"):
        out = scrape_fbref.split_league(frame, "ITA-Serie A")

    assert list(out["player"]) == ["A"]
    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("2" in msg for msg in warnings)


def test_player_season_stats_records_and_logs_skipped_tables(monkeypatch, caplog):
    class FakeClient:
        def read_player_season_stats(self, stat_type):
            if stat_type == "standard":
                return pd.DataFrame({
                    "player": ["A"], "team": ["X"], "pos": ["FW"], "born": [1999],
                    "age": [24], "minutes": [900], "playing_time_mp": [30],
                    "league": ["ITA-Serie A"],
                })
            raise TypeError(f"Invalid argument: stat_type should be in [...], got {stat_type!r}")

    monkeypatch.setattr(scrape_fbref, "_fbref", lambda season: FakeClient())
    monkeypatch.setattr(scrape_fbref.cache, "cached_frame", lambda key, builder: builder())
    monkeypatch.setattr(scrape_fbref.config, "REQUEST_DELAY_SECONDS", 0)

    with caplog.at_level("WARNING", logger="src.scrape_fbref"):
        out = scrape_fbref.player_season_stats("ITA-Serie A", "2223")

    expected_skipped = [s for s in scrape_fbref.STAT_TYPES if s != "standard"]
    assert out.attrs["skipped_tables"] == expected_skipped
    assert "passing" in out.attrs["skipped_tables"]
    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("passing" in msg and "ITA-Serie A" in msg and "2223" in msg for msg in warnings)


def test_human_captcha_fbref_pauses_for_human_and_never_auto_solves(monkeypatch):
    calls = []
    monkeypatch.setattr("builtins.input", lambda: calls.append("input"))

    class ExplodingDriver:
        def uc_gui_handle_captcha(self):
            raise AssertionError("must not auto-solve the CAPTCHA")

        def uc_gui_handle_cf(self):
            raise AssertionError("must not auto-solve the CAPTCHA")

    instance = scrape_fbref.HumanCaptchaFBref.__new__(scrape_fbref.HumanCaptchaFBref)
    instance._driver = ExplodingDriver()

    instance.solve_captcha()

    assert calls == ["input"]


class _FakeClientConfig:
    def __init__(self):
        self.timeout = None


class _FakeCommandExecutorWithClientConfig:
    def __init__(self):
        self._client_config = _FakeClientConfig()


class _FakeCommandExecutorWithSetTimeoutOnly:
    def __init__(self):
        self.set_timeout_calls = []

    def set_timeout(self, seconds):
        self.set_timeout_calls.append(seconds)


class _FakeDriver:
    def __init__(self, command_executor):
        self.page_load_timeout = None
        self.script_timeout = None
        self.cdp_calls = []
        self.quit_called = False
        self.command_executor = command_executor

    def set_page_load_timeout(self, seconds):
        self.page_load_timeout = seconds

    def set_script_timeout(self, seconds):
        self.script_timeout = seconds

    def execute_cdp_cmd(self, cmd, params):
        self.cdp_calls.append((cmd, params))

    def quit(self):
        self.quit_called = True


class _FakeSbDriverFactory:
    """Records the kwargs `sb.Driver(...)` was called with and returns a fake driver."""

    def __init__(self, command_executor):
        self.kwargs = None
        self._command_executor = command_executor

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return _FakeDriver(self._command_executor)


def _make_instance(headless=False):
    instance = scrape_fbref.HumanCaptchaFBref.__new__(scrape_fbref.HumanCaptchaFBref)
    instance.headless = headless
    instance.path_to_browser = None
    instance.proxy = lambda: None
    instance.headers = {"User-Agent": "x"}
    return instance


def test_init_webdriver_configures_lighter_pages_and_all_timeouts(monkeypatch):
    fake_executor = _FakeCommandExecutorWithClientConfig()
    fake_factory = _FakeSbDriverFactory(fake_executor)

    import seleniumbase as sb
    monkeypatch.setattr(sb, "Driver", fake_factory)

    instance = _make_instance()
    old_driver = _FakeDriver(fake_executor)
    instance._driver = old_driver

    result = instance._init_webdriver()

    assert old_driver.quit_called is True

    assert fake_factory.kwargs["uc"] is True
    assert fake_factory.kwargs["ad_block_on"] is True
    assert fake_factory.kwargs["block_images"] is True
    assert fake_factory.kwargs["page_load_strategy"] == "eager"
    assert fake_factory.kwargs["headless"] is False
    assert fake_factory.kwargs["binary_location"] is None
    assert fake_factory.kwargs["host_resolver_rules"] is None
    assert fake_factory.kwargs["proxy"] is None

    assert result.page_load_timeout == scrape_fbref.PAGE_LOAD_TIMEOUT_SECONDS == 90
    assert result.script_timeout == scrape_fbref.SCRIPT_TIMEOUT_SECONDS == 30
    assert fake_executor._client_config.timeout == scrape_fbref.COMMAND_TIMEOUT_SECONDS == 120

    assert result.cdp_calls == [
        ("Network.setExtraHTTPHeaders", {"headers": {"User-Agent": "x"}})
    ]


def test_init_webdriver_falls_back_to_set_timeout_when_no_client_config(monkeypatch):
    fake_executor = _FakeCommandExecutorWithSetTimeoutOnly()
    fake_factory = _FakeSbDriverFactory(fake_executor)

    import seleniumbase as sb
    monkeypatch.setattr(sb, "Driver", fake_factory)

    instance = _make_instance()

    instance._init_webdriver()

    assert fake_executor.set_timeout_calls == [scrape_fbref.COMMAND_TIMEOUT_SECONDS]


def test_apply_command_timeout_prefers_client_config_path():
    executor_holder = _FakeCommandExecutorWithClientConfig()

    class _Driver:
        command_executor = executor_holder

    path = scrape_fbref._apply_command_timeout(_Driver(), 42)

    assert path == "_client_config.timeout"
    assert executor_holder._client_config.timeout == 42


def test_apply_command_timeout_falls_back_to_set_timeout():
    executor_holder = _FakeCommandExecutorWithSetTimeoutOnly()

    class _Driver:
        command_executor = executor_holder

    path = scrape_fbref._apply_command_timeout(_Driver(), 42)

    assert path == "set_timeout"
    assert executor_holder.set_timeout_calls == [42]
