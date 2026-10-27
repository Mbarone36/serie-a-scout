"""Source-agnostic facade for FBref player/team season stats.

Every later task should import from here, never from `load_dump` or
`scrape_fbref` directly, so the data source (offline dump vs. live scrape)
is a config switch (`config.FBREF_SOURCE`) rather than a code change.

Dispatch rules for `player_season_stats` / `team_season_stats`:

1. `config.FBREF_SOURCE == "scrape"`: always delegate to `scrape_fbref`,
   which may perform a live fetch. Unchanged from a plain pass-through.
2. `config.FBREF_SOURCE == "dump"` (the default): if `load_dump` has that
   league-season, serve it from the dump. Otherwise -- e.g. a season past
   the dump's 2023 cutoff -- serve it from the *scraper's on-disk cache
   only*: if a parquet already exists at the scraper's own cache path for
   that league-season (written by a `scrape_fbref.player_season_stats`/
   `team_season_stats` call running elsewhere, possibly concurrently),
   read it directly; otherwise return an empty frame. This path NEVER
   imports `scrape_fbref` or constructs a live client -- it is a plain
   `pandas.read_parquet` of a file scrape_fbref's own `cache.cached_frame`
   already wrote, so it can't race with, or trigger, a live pull.

Both backends are otherwise imported lazily, inside the functions that need
them, so that importing this module never triggers `scrape_fbref`'s
module-level `import soccerdata` (and its network-adjacent side effects)
when the configured source is "dump" -- which is what the test suite (and,
by default, every later task) uses.
"""

import logging
from pathlib import Path

import pandas as pd

from src import config

logger = logging.getLogger(__name__)

# Per-(league, season) cache of (is_complete, max team_matches), populated by
# `_dump_season_completeness` the first time either a player- or team-kind
# dispatch needs it for that league-season, so a team-kind dispatch never
# has to reload the dump's player table just to answer a question the
# player-kind dispatch already answered (or will answer) for the same
# league-season within this process.
_completeness_cache: dict[tuple[str, str], tuple[bool, float]] = {}


def season_is_complete(frame: pd.DataFrame) -> bool:
    """Whether a player-stats `frame` covers a full Big-5 season.

    True iff `frame` carries a `team_matches` column (added by
    `scrape_fbref.normalise_identity` for both the live scraper and the
    dump loader) whose max value across rows is at least
    `config.MIN_COMPLETE_SEASON_MATCHES`. False for an empty frame or one
    missing the column entirely -- there is nothing to judge completeness
    from, so it is treated as not (yet) complete rather than guessed at.
    """
    if frame is None or frame.empty or "team_matches" not in frame.columns:
        return False
    value = frame["team_matches"].max()
    return bool(pd.notna(value) and value >= config.MIN_COMPLETE_SEASON_MATCHES)


def _dump_season_completeness(
    league: str, season: str, player_frame: pd.DataFrame | None
) -> tuple[bool, float]:
    """(is_complete, max team_matches) for the dump's `league`/`season`
    PLAYER frame, cached per (league, season) within the process.

    Team frames carry no `team_matches` column of their own (team grain has
    no per-team match count to max over), so completeness for team-kind
    dispatches is always decided from the corresponding player frame. When
    the caller already has that player frame in hand (the player-kind
    dispatch path), pass it as `player_frame` to avoid loading it twice;
    otherwise it is loaded here (a plain dump read, never a live fetch).
    """
    key = (league, season)
    if key in _completeness_cache:
        return _completeness_cache[key]
    if player_frame is None:
        from src import load_dump
        player_frame = load_dump.player_season_stats(league, season)
    if player_frame is not None and not player_frame.empty and "team_matches" in player_frame.columns:
        max_matches = player_frame["team_matches"].max()
    else:
        max_matches = float("nan")
    result = (season_is_complete(player_frame), max_matches)
    _completeness_cache[key] = result
    return result


def _scrape_cache_path(kind: str, league: str, season: str) -> Path:
    """The exact on-disk path `scrape_fbref.{player,team}_season_stats` cache to.

    Mirrors `scrape_fbref`'s own cache key (`f"fbref/{kind}/{league...}_{season}"`)
    verbatim -- kept here rather than imported so this module never needs to
    import `scrape_fbref` just to read a path template.
    """
    key = f"fbref/{kind}/{league.replace(' ', '_')}_{season}"
    return config.RAW_DIR / f"{key}.parquet"


def _read_scrape_cache(kind: str, league: str, season: str) -> pd.DataFrame | None:
    """The scraper's cached frame for one league-season, or None if absent.

    Pure file read -- never triggers a live fetch, never imports
    `scrape_fbref` or `soccerdata`.
    """
    path = _scrape_cache_path(kind, league, season)
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _scrape_cached_seasons(kind: str, league: str) -> list[str]:
    """Seasons for `league` that already have a scraper cache file on disk."""
    directory = config.RAW_DIR / "fbref" / kind
    if not directory.exists():
        return []
    prefix = f"{league.replace(' ', '_')}_"
    return [p.stem[len(prefix):] for p in directory.glob(f"{prefix}*.parquet")]


def _dispatch(kind: str, league: str, season: str) -> pd.DataFrame:
    if config.FBREF_SOURCE == "scrape":
        from src import scrape_fbref
        fn = scrape_fbref.player_season_stats if kind == "player" else scrape_fbref.team_season_stats
        return fn(league, season)

    from src import load_dump

    if season in load_dump.available_seasons(league):
        fn = load_dump.player_season_stats if kind == "player" else load_dump.team_season_stats
        dump_frame = fn(league, season)
        complete, max_matches = _dump_season_completeness(
            league, season, dump_frame if kind == "player" else None,
        )
        if not complete:
            cached = _read_scrape_cache(kind, league, season)
            if cached is not None:
                logger.info(
                    "serving %s %s from scrape cache: dump season incomplete "
                    "(max matches %s)", league, season, max_matches,
                )
                return cached
            logger.warning(
                "%s %s dump season is incomplete (max matches %s) and no scrape "
                "cache exists yet -- serving the incomplete dump frame",
                league, season, max_matches,
            )
        return dump_frame

    cached = _read_scrape_cache(kind, league, season)
    if cached is not None:
        return cached
    return pd.DataFrame()


# Live-scrape name -> dump name, for stats both eras report. FBref's live
# pages (seasons "2324"+) drop the "defense"/"passing" tables but keep these
# in "misc", where they flatten to `performance_*`. Each pair was verified on
# the 2022-23 overlap: equal-minutes rows agree exactly or within Opta's 1-2
# revision noise. `performance_tklw` is tackles WON, so it maps to the dump's
# misc `tklw`, never to `tackles_tkl` (tackles attempted).
_LIVE_TO_DUMP = {
    "performance_int": "int",
    "performance_tklw": "tklw",
    "performance_crs": "crs",
    "performance_fld": "fld",
}


def harmonise_player_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Give live-scrape player frames the dump's names for shared stats.

    Renames only where the dump name is absent, so a dump frame (which may
    carry both) passes through untouched and these features don't
    neutral-fill to 0.5 for every live-era row.
    """
    renames = {live: dump for live, dump in _LIVE_TO_DUMP.items()
               if live in frame.columns and dump not in frame.columns}
    return frame.rename(columns=renames) if renames else frame


def player_season_stats(league: str, season: str) -> pd.DataFrame:
    return harmonise_player_columns(_dispatch("player", league, season))


def team_season_stats(league: str, season: str) -> pd.DataFrame:
    return _dispatch("team", league, season)


def available_seasons(league: str) -> list[str]:
    """Every season `league` can be served for: dump seasons plus whatever
    the scraper has already cached to disk (elsewhere, possibly concurrently
    with this process), sorted.
    """
    from src import load_dump

    dump_seasons = set(load_dump.available_seasons(league))
    scrape_seasons = set(_scrape_cached_seasons("player", league))
    return sorted(dump_seasons | scrape_seasons)
