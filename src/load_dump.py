"""Loader for the pre-exported FBref Big-5 dump (worldfootballR .rds files).

Exposes the same `player_season_stats(league, season)` / `team_season_stats`
contract as `scrape_fbref`, so `fbref.py` can dispatch to either source
without callers noticing the difference.

IMPORTANT: this module must not import `scrape_fbref` at module scope. That
module does `import soccerdata as sd` unconditionally, and this loader (plus
its tests) must work without ever touching soccerdata's network machinery.
The two functions from `scrape_fbref` we reuse (`merge_stat_tables`,
`normalise_identity`) are imported lazily, inside the functions that need
real data.
"""

import functools
import re

import pandas as pd
import pyreadr

from src import cache, config

# The 11 worldfootballR stat tables exported per {player, team} x Big-5.
TABLE_NAMES = (
    "standard", "shooting", "passing", "passing_types", "gca", "defense",
    "possession", "playing_time", "misc", "keepers", "keepers_adv",
)

# `Comp` values in the dump -> the soccerdata-style strings the rest of the
# project uses (config.CANDIDATE_LEAGUES).
COMP_TO_LEAGUE: dict[str, str] = {
    "Serie A": "ITA-Serie A",
    "Premier League": "ENG-Premier League",
    "La Liga": "ESP-La Liga",
    "Bundesliga": "GER-Bundesliga",
    "Ligue 1": "FRA-Ligue 1",
}

assert set(COMP_TO_LEAGUE.values()) <= set(config.CANDIDATE_LEAGUES)

# worldfootballR names -> the snake_case names the rest of the project uses.
# Only entries that need a mapping DIFFERENT from generic snake-casing are
# listed here; everything else falls through `_snake()` unchanged (and
# "Player" -> "player", "Nation" -> "nation", "Pos" -> "pos", "Age" -> "age",
# "Born" -> "born", "Int" -> "int" already come out right that way).
_EXPLICIT_RENAMES = {
    "Squad": "team",
    "MP_Playing": "playing_time_mp",
    "Min_Playing": "minutes",
    "npxG_Expected": "expected_npxg",
    "xAG_Expected": "expected_xag",
    "CrdY": "performance_crdy",
    "Fls": "performance_fls",
    # standard — basics that live on in FBref's live-page schema too
    # (see src/features_onfield.py), so the dump must name them identically.
    "Gls": "performance_gls",
    "Ast": "performance_ast",
    # shooting — same basics-parity reasoning as above.
    "Sh_Standard": "standard_sh",
    "SoT_Standard": "standard_sot",
    "SoT_percent_Standard": "standard_sot_pct",
    # passing
    "Prog": "progression_prgp",
    # possession
    "Prog_Carries": "progression_prgc",
    "Succ_Dribbles": "possession_take_ons_succ",
    "Att Pen_Touches": "possession_touches_att_pen",
    "Mis_Carries": "possession_carries_mis",
    "Dis_Carries": "possession_carries_dis",
    # defense
    "Tkl_Tackles": "tackles_tkl",
    "Att 3rd_Tackles": "tackles_att_3rd",
    # misc
    "Won_percent_Aerial": "aerial_duels_won_pct",
    # team-only
    "Poss": "possession",
    "Att_Total": "total_passes_att",
    "Crs_Pass": "pass_types_crs",
}

# Columns that duplicate the `standard` table's identity and must be dropped
# from every other table before merging, so they don't collide.
_DUPLICATE_IDENTITY_COLS = {"nation", "pos", "age", "born", "url"}


def _snake(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()


def end_year_to_season(end_year: int) -> str:
    """2023 -> '2223' (season ending in `end_year`)."""
    end_year = int(end_year)
    start_year = end_year - 1
    return f"{start_year % 100:02d}{end_year % 100:02d}"


def _parse_age(value) -> float:
    """worldfootballR's `Age` is text: plain whole years for older seasons
    ("18"), "years-days" for recent ones ("25-303"). Both mean whole years
    for our purposes, so take the part before any "-" and coerce to float;
    blank/missing/unparseable values become NaN rather than raising, since
    a bad age shouldn't crash the whole load.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return float("nan")
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none"):
        return float("nan")
    years_part = text.split("-", 1)[0]
    try:
        return float(years_part)
    except ValueError:
        return float("nan")


def rename_columns(frame: pd.DataFrame, table: str) -> pd.DataFrame:
    """worldfootballR column names -> snake_case names used across the project."""
    is_player_table = "Player" in frame.columns
    out = frame.copy()
    new_cols = [_EXPLICIT_RENAMES.get(c, _snake(c)) for c in out.columns]
    out.columns = new_cols
    if table != "standard":
        drop = [c for c in _DUPLICATE_IDENTITY_COLS if c in out.columns]
        out = out.drop(columns=drop)
    elif "url" in out.columns:
        out = out.drop(columns=["url"])
    # Player tables' `Age` is "years-days" text in recent seasons ("25-303")
    # and plain digit text in older ones ("18") -- both need to become a
    # numeric year count for downstream percentile/threshold use. Team
    # tables' `Age` (average squad age, e.g. 25.6) is already numeric and is
    # left untouched.
    if is_player_table and "age" in out.columns:
        out["age"] = out["age"].map(_parse_age)
    return out


def select_league_season(frame: pd.DataFrame, league: str, season: str) -> pd.DataFrame:
    """Rows for one league-season; an empty frame (same columns) if none match."""
    if "league" not in frame.columns or "season" not in frame.columns:
        return frame.iloc[0:0].copy()
    mask = (frame["league"] == league) & (frame["season"] == season)
    return frame.loc[mask].copy()


def _dump_path(kind: str, table: str) -> "object":
    return config.DUMP_DIR / f"big5_{kind}_{table}.rds"


def _read_rds(path) -> pd.DataFrame:
    result = pyreadr.read_r(str(path))
    return next(iter(result.values()))


@functools.lru_cache(maxsize=None)
def _load_full_table(kind: str, table: str):
    """Read+prepare one raw dump table (all leagues/seasons). Cached in-process."""
    path = _dump_path(kind, table)
    if not path.exists():
        return None
    raw = _read_rds(path)
    if kind == "team":
        raw = raw[raw["Team_or_Opponent"] == "team"].copy()
    else:
        raw = raw.copy()
    raw["league"] = raw["Comp"].map(COMP_TO_LEAGUE)
    raw["season"] = raw["Season_End_Year"].map(end_year_to_season)
    return rename_columns(raw, table)


def _dedupe_on_identity(frame: pd.DataFrame, identity_cols: list[str]) -> pd.DataFrame:
    """Collapse duplicate identity keys to one row.

    A handful of Player+Squad name collisions exist in the raw dump (two
    distinct real people sharing a name at the same club in the same
    season). Left as-is, a duplicated key in ANY table being merged
    multiplies rows during `merge_stat_tables`'s joins (2x2x2 -> 8 rows for
    one player). We collapse duplicates in every table we merge, not just
    `standard`: where a `minutes` column is available (the `standard` and
    `keepers` tables) we keep the higher-minutes row, and elsewhere we keep
    the first row encountered, which is enough to guarantee one row per
    identity in the final merged frame.
    """
    if frame.empty:
        return frame
    if not frame.duplicated(subset=identity_cols).any():
        return frame
    if "minutes" in frame.columns:
        frame = frame.sort_values("minutes", ascending=False)
    return frame.drop_duplicates(subset=identity_cols, keep="first")


def _season_stats(kind: str, league: str, season: str) -> pd.DataFrame:
    identity_cols = ["league", "season", "team"] if kind == "team" else ["league", "season", "player", "team"]

    def build() -> pd.DataFrame:
        tables: dict[str, pd.DataFrame] = {}
        for table in TABLE_NAMES:
            full = _load_full_table(kind, table)
            if full is None:
                continue
            sub = select_league_season(full, league, season)
            if table == "standard" and sub.empty:
                return pd.DataFrame()
            tables[table] = _dedupe_on_identity(sub, identity_cols)
        if "standard" not in tables or tables["standard"].empty:
            return pd.DataFrame()

        from src import scrape_fbref  # lazy: avoid importing soccerdata eagerly

        merged = scrape_fbref.merge_stat_tables(tables)
        if kind == "team":
            # Match scrape_fbref.team_season_stats exactly: it never calls
            # normalise_identity for team frames (no player/pos/born/nation
            # columns exist at team grain). Doing so here would trip the
            # REQUIRED_COLUMNS check normalise_identity now enforces (added
            # in 3594c0a) and raise ValueError on every team pull.
            return merged
        return scrape_fbref.normalise_identity(merged)

    key = f"dump/{kind}/{league.replace(' ', '_')}_{season}"
    return cache.cached_frame(key, build)


def player_season_stats(league: str, season: str) -> pd.DataFrame:
    """All player stat tables for one league-season, merged into one frame."""
    return _season_stats("player", league, season)


def team_season_stats(league: str, season: str) -> pd.DataFrame:
    """Team-level stat tables for one league-season."""
    return _season_stats("team", league, season)


def available_seasons(league: str) -> list[str]:
    """Seasons present in the dump for `league`, derived from the standard table."""
    full = _load_full_table("player", "standard")
    if full is None:
        return []
    seasons = full.loc[full["league"] == league, "season"].unique().tolist()
    return sorted(seasons)
