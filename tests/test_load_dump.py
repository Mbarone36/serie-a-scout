import pandas as pd

from src import config, load_dump, scrape_fbref


FIXTURE = "tests/fixtures/dump_standard_sample.csv"


def _sample() -> pd.DataFrame:
    return pd.read_csv(FIXTURE)


def test_end_year_to_season_formats_two_digit_years():
    assert load_dump.end_year_to_season(2023) == "2223"
    assert load_dump.end_year_to_season(2018) == "1718"


def test_comp_to_league_keys_and_values():
    assert set(load_dump.COMP_TO_LEAGUE.keys()) == {
        "Serie A", "Premier League", "La Liga", "Bundesliga", "Ligue 1",
    }
    assert set(load_dump.COMP_TO_LEAGUE.values()) <= set(config.CANDIDATE_LEAGUES)


def test_rename_columns_maps_standard_table_to_snake_case():
    out = load_dump.rename_columns(_sample(), "standard")
    for col in (
        "player", "team", "minutes", "playing_time_mp",
        "expected_npxg", "expected_xag", "born", "nation", "pos", "age",
    ):
        assert col in out.columns, col
    assert "Squad" not in out.columns
    assert "Min_Playing" not in out.columns


def test_rename_columns_parses_age_to_numeric_years():
    # The fixture has both formats for real: "years-days" for 2022-23
    # rows ("25-303") and plain whole years for the 2017-18 rows ("18").
    out = load_dump.rename_columns(_sample(), "standard")
    assert pd.api.types.is_numeric_dtype(out["age"])

    by_player = out.set_index("player")["age"]
    assert by_player["Jeremie Boga"] == 25.0       # "25-303" -> 25.0
    assert by_player["Ademola Lookman"] == 25.0    # "25-013" -> 25.0
    assert by_player["Musa Barrow"] == 18.0        # "18" -> 18.0


def test_parse_age_handles_dash_plain_and_blank_values():
    assert load_dump._parse_age("25-303") == 25.0
    assert load_dump._parse_age("18") == 18.0
    assert load_dump._parse_age("") != load_dump._parse_age("")  # NaN != NaN
    assert pd.isna(load_dump._parse_age(""))
    assert pd.isna(load_dump._parse_age(None))
    assert pd.isna(load_dump._parse_age(float("nan")))


def test_rename_columns_does_not_reparse_team_age():
    # Team `Age` (average squad age, e.g. 25.6) is already numeric and must
    # be left alone -- it is only "Player" tables whose `Age` is the
    # worldfootballR "years-days" text format.
    team_frame = pd.DataFrame({
        "Season_End_Year": [2023], "Squad": ["Atalanta"], "Comp": ["Serie A"],
        "Team_or_Opponent": ["team"], "Num_Players": [24], "Age": [26.7],
        "Poss": [55.2],
    })
    out = load_dump.rename_columns(team_frame, "standard")
    assert out["age"].iloc[0] == 26.7


def test_rename_columns_maps_standard_basics_to_feature_onfield_names():
    # Gls/Ast must land on the same names src/features_onfield.py's
    # COUNTING_COLUMNS uses for the live-scrape era, so a training frame
    # built from the dump and one built from the live scraper are
    # interchangeable for these two columns.
    frame = pd.DataFrame({
        "Player": ["A"], "Squad": ["Roma"], "Gls": [10.0], "Ast": [5.0],
    })
    out = load_dump.rename_columns(frame, "standard")
    assert out["performance_gls"].iloc[0] == 10.0
    assert out["performance_ast"].iloc[0] == 5.0
    assert "Gls" not in out.columns
    assert "Ast" not in out.columns


def test_rename_columns_maps_shooting_basics_to_feature_onfield_names():
    frame = pd.DataFrame({
        "Player": ["A"], "Squad": ["Roma"],
        "Sh_Standard": [50.0], "SoT_Standard": [20.0], "SoT_percent_Standard": [40.0],
    })
    out = load_dump.rename_columns(frame, "shooting")
    assert out["standard_sh"].iloc[0] == 50.0
    assert out["standard_sot"].iloc[0] == 20.0
    assert out["standard_sot_pct"].iloc[0] == 40.0


def test_rename_columns_drops_duplicate_identity_from_non_standard_tables():
    frame = _sample()[["Season_End_Year", "Squad", "Comp", "Player", "Nation", "Pos", "Age", "Born", "CrdY"]]
    out = load_dump.rename_columns(frame, "misc")
    assert "nation" not in out.columns
    assert "pos" not in out.columns
    assert "age" not in out.columns
    assert "born" not in out.columns
    assert "player" in out.columns
    assert "team" in out.columns


def test_dedupe_before_merge_prevents_fanout_and_keeps_higher_minutes():
    """Regression test for the fan-out bug found during self-review.

    The dump has a handful of genuine Player+Squad name collisions (two
    different real people sharing a name at the same club/season, e.g.
    Adrian Lopez at La Coruna in 2009-10). Every stat table carries the
    collision, not just `standard`. Deduping only `standard` before the
    left-joins in `merge_stat_tables` left `shooting` (etc.) with 2 rows
    for the shared key, so the join fanned that 1 deduped `standard` row
    out to 2 rows again. This asserts that running `_dedupe_on_identity`
    on every table before merging -- what `_season_stats.build()` does --
    collapses the pair to one row (the higher-minutes one) with no fan-out.
    """
    identity_cols = ["league", "season", "player", "team"]
    standard = pd.DataFrame({
        "league": ["ITA-Serie A"] * 3,
        "season": ["2223"] * 3,
        "player": ["Dup Player", "Dup Player", "Solo Player"],
        "team": ["Roma", "Roma", "Roma"],
        "minutes": [2044.0, 270.0, 900.0],
        "pos": ["FW", "FW", "MF"],
    })
    shooting = pd.DataFrame({
        "league": ["ITA-Serie A"] * 3,
        "season": ["2223"] * 3,
        "player": ["Dup Player", "Dup Player", "Solo Player"],
        "team": ["Roma", "Roma", "Roma"],
        "npxg": [5.0, 1.0, 3.0],
    })

    tables = {
        "standard": load_dump._dedupe_on_identity(standard, identity_cols),
        "shooting": load_dump._dedupe_on_identity(shooting, identity_cols),
    }
    merged = scrape_fbref.merge_stat_tables(tables)

    dup_rows = merged[merged["player"] == "Dup Player"]
    assert len(dup_rows) == 1  # (a) exactly one output row for the duplicated identity
    assert dup_rows["minutes"].iloc[0] == 2044.0  # (b) the higher-minutes row survives
    assert dup_rows["npxg"].iloc[0] == 5.0  # its shooting stats travel with it, not the discarded row's

    distinct_identities = merged[identity_cols].drop_duplicates()
    assert len(merged) == len(distinct_identities)  # (c) no fan-out: 1 row per identity
    assert len(merged) == 2


def test_dedupe_before_merge_would_fail_without_deduping_every_table():
    """Sanity check that the regression test above is actually exercising the
    bug: deduping only `standard` (the original, broken behaviour) still
    fans out on a `shooting` table that keeps its duplicate rows.
    """
    identity_cols = ["league", "season", "player", "team"]
    standard = pd.DataFrame({
        "league": ["ITA-Serie A"] * 2,
        "season": ["2223"] * 2,
        "player": ["Dup Player", "Dup Player"],
        "team": ["Roma", "Roma"],
        "minutes": [2044.0, 270.0],
    })
    shooting_not_deduped = pd.DataFrame({
        "league": ["ITA-Serie A"] * 2,
        "season": ["2223"] * 2,
        "player": ["Dup Player", "Dup Player"],
        "team": ["Roma", "Roma"],
        "npxg": [5.0, 1.0],
    })

    tables = {
        "standard": load_dump._dedupe_on_identity(standard, identity_cols),
        "shooting": shooting_not_deduped,
    }
    merged = scrape_fbref.merge_stat_tables(tables)
    assert len(merged) == 2  # fan-out: the bug this task's fix eliminates


def test_select_league_season_filters_and_returns_empty_when_absent():
    frame = pd.DataFrame({
        "league": ["ITA-Serie A", "ITA-Serie A", "ENG-Premier League"],
        "season": ["2223", "2122", "2223"],
        "player": ["A", "B", "C"],
    })
    out = load_dump.select_league_season(frame, "ITA-Serie A", "2223")
    assert list(out["player"]) == ["A"]

    absent = load_dump.select_league_season(frame, "ITA-Serie A", "9900")
    assert absent.empty
    assert list(absent.columns) == list(frame.columns)


# `fbref.py` facade dispatch tests (dump vs. scrape-cache-only vs. live scrape)
# live in tests/test_fbref.py, isolated from the real dump/cache on disk via
# tmp_path -- the two that used to live here only worked by relying on the
# real dump directory and are superseded there.
