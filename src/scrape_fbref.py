"""FBref player/team season stat pulls.

Run `pull_all()` (or `python -m src.scrape_fbref`) in a terminal you are
watching — it pauses and waits on stdin whenever FBref serves a CAPTCHA.

A live run was once observed hung with Python at 0% CPU blocked on its
socket to chromedriver, chromedriver itself at 0% CPU, and Chrome's renderer
idle with the FBref page already fully loaded. The culprit: Selenium's
remote client has NO command timeout by default
(`RemoteConnection._timeout = socket.getdefaulttimeout()`, i.e. None), so a
chromedriver command that never returns (soccerdata calls `driver.page_source`
on FBref pages that can run ~8 MB) blocks forever — `driver.set_page_load_timeout`
only bounds `driver.get()`, not every other command. `HumanCaptchaFBref`
below applies a client-side command timeout (`COMMAND_TIMEOUT_SECONDS`, 120s)
on top of the page-load and script timeouts, so a hung command now raises
instead of hanging. soccerdata's `_download_and_save` retry loop already
catches exceptions, retries up to 5 times, and re-creates the driver via
`_init_webdriver` on each attempt, so the raised timeout self-heals into a
retry instead of blocking the run forever. The driver is also built with
ad-blocking, image-loading disabled, and the "eager" page-load strategy
(fires once the DOM is parsed, without waiting on every subresource) so the
pages it has to wait on are far lighter to begin with.
"""

import logging
import re

import pandas as pd
import soccerdata as sd

from src import cache, config

logger = logging.getLogger(__name__)


PAGE_LOAD_TIMEOUT_SECONDS = 90
SCRIPT_TIMEOUT_SECONDS = 30
COMMAND_TIMEOUT_SECONDS = 120


def _apply_command_timeout(driver, seconds: int) -> str:
    """Bound every Selenium remote-connection command to `seconds`.

    Selenium's remote client has no command timeout by default, so a
    chromedriver command that never returns hangs forever. Selenium 4.9+
    exposes the timeout via `command_executor._client_config.timeout`
    (a `ClientConfig` attribute); older selenium only offers the deprecated
    `command_executor.set_timeout(seconds)` classmethod. Try the modern path
    first and fall back to the deprecated one, returning which was used so
    the caller can log it.
    """
    executor = driver.command_executor
    client_config = getattr(executor, "_client_config", None)
    if client_config is not None and hasattr(client_config, "timeout"):
        client_config.timeout = seconds
        return "_client_config.timeout"
    executor.set_timeout(seconds)
    return "set_timeout"


class HumanCaptchaFBref(sd.FBref):
    """soccerdata's FBref reader, with automatic CAPTCHA solving replaced by a pause.

    When FBref serves a challenge the run stops, the human solves it in the
    visible Chrome window, and the scrape resumes. The software never attempts
    the challenge itself.

    `_init_webdriver` is a full re-implementation of soccerdata's own
    version (soccerdata/_common.py), not a call to `super()._init_webdriver()`,
    because the parent method builds the `sb.Driver` without ad-block,
    without disabling images, without an "eager" page-load strategy, and
    without any command timeout — leaving a heavy FBref page free to hang
    `driver.page_source` (or any other command) indefinitely. See the module
    docstring for the failure this was observed to cause and why a raised
    TimeoutException here is safe: soccerdata's `_download_and_save` retry
    loop catches it, retries up to 5 times, and re-creates the driver via
    this method on each attempt.
    """

    def _init_webdriver(self) -> "sb.Driver":  # noqa: F821 - sb is imported lazily below
        """Start the Selenium driver: soccerdata's own setup, plus lighter
        pages and bounded timeouts.

        Mirrors soccerdata's `_init_webdriver` (soccerdata/_common.py:576-596)
        step for step — quitting any existing driver, computing `proxy_str`/
        `resolver_rules` the same way, building `sb.Driver` with the same
        core arguments, and applying `self.headers` via
        `execute_cdp_cmd("Network.setExtraHTTPHeaders", ...)` — then adds:
        `ad_block_on=True` and `block_images=True` (FBref's ad-heavy pages
        are enormous; this makes them far lighter to load), and
        `page_load_strategy="eager"` (returns from `driver.get()` once the
        DOM is parsed rather than waiting on every subresource). After
        creation it applies a page-load timeout, a script timeout, and a
        client-side command timeout (see `_apply_command_timeout`) so a
        command that would otherwise hang forever raises instead.
        """
        import seleniumbase as sb

        if hasattr(self, "_driver"):
            self._driver.quit()

        proxy_str = self.proxy()
        resolver_rules = None
        if proxy_str is not None:
            resolver_rules = "MAP * ~NOTFOUND , EXCLUDE 127.0.0.1"

        driver = sb.Driver(
            uc=True,
            headless=self.headless,
            binary_location=self.path_to_browser,
            host_resolver_rules=resolver_rules,
            proxy=proxy_str,
            ad_block_on=True,
            block_images=True,
            page_load_strategy="eager",
        )

        if self.headers:
            driver.execute_cdp_cmd("Network.setExtraHTTPHeaders", {"headers": self.headers})

        driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT_SECONDS)
        driver.set_script_timeout(SCRIPT_TIMEOUT_SECONDS)
        command_timeout_path = _apply_command_timeout(driver, COMMAND_TIMEOUT_SECONDS)
        logger.info(
            "FBref webdriver limits applied: page-load=%ss, script=%ss, command=%ss (via %s), "
            "ad_block_on=True, block_images=True, page_load_strategy=eager.",
            PAGE_LOAD_TIMEOUT_SECONDS, SCRIPT_TIMEOUT_SECONDS, COMMAND_TIMEOUT_SECONDS, command_timeout_path,
        )
        return driver

    def solve_captcha(self) -> None:
        print("\a\n>>> CAPTCHA on FBref. Solve it in the Chrome window, then press Enter here. <<<")
        input()


# Verified directly against the installed soccerdata 1.9.1 source
# (site-packages/soccerdata/fbref.py: the `player_stats` whitelist at
# line 536-546 inside `read_player_season_stats`, and the `team_stats`
# whitelist at line 267-277 inside `read_team_season_stats` — both local
# variables, not part of the public API, so they are duplicated here rather
# than imported). If a newer soccerdata release supports more stat_type
# values, update these two tuples; see task-3-report.md for how this was
# confirmed and what it means when it isn't true.
SUPPORTED_PLAYER_STAT_TYPES = ("standard", "keeper", "shooting", "playing_time", "misc")
SUPPORTED_TEAM_STAT_TYPES = ("standard", "keeper", "shooting", "playing_time", "misc")

# STAT_TYPES documents which FBref stat categories this model *wants*.
# "passing", "passing_types", "goal_shot_creation", "defense" and
# "possession" are NOT in SUPPORTED_PLAYER_STAT_TYPES for soccerdata 1.9.1 —
# calling read_player_season_stats with any of them raises TypeError before
# any request is made. player_season_stats() below catches that, logs a
# WARNING naming the league/season/stat_type, and records it in
# `frame.attrs["skipped_tables"]` — it is never silent. pull_all() also
# prints which of these are unsupported before a multi-hour run starts.
STAT_TYPES = (
    "standard", "shooting", "passing", "passing_types",
    "goal_shot_creation", "defense", "possession", "playing_time", "misc",
)

REQUIRED_COLUMNS = ("player", "team", "pos", "born", "age", "minutes", "playing_time_mp")

_JOIN_KEYS = ["player", "team"]


def check_required_columns(frame: pd.DataFrame) -> list[str]:
    """Names from REQUIRED_COLUMNS missing from `frame`, in REQUIRED_COLUMNS order."""
    return [c for c in REQUIRED_COLUMNS if c not in frame.columns]


def split_league(frame: pd.DataFrame, league: str) -> pd.DataFrame:
    """Select one Big-5 league's rows out of a "Big 5 European Leagues Combined" read.

    soccerdata's FBref reader, when constructed with
    leagues="Big 5 European Leagues Combined", already maps each row's raw
    FBref "Comp" value to the canonical league string (via its own internal
    BIG_FIVE_DICT) into a `league` column and drops "Comp" entirely —
    confirmed by reading the installed soccerdata 1.9.1 source directly
    (fbref.py read_player_season_stats:576-585, read_team_season_stats:318-325).
    So after `reset_index()` + `flatten_columns()` there is nothing to parse
    or strip: this is a plain filter on an already-canonical `league` column.

    Raises ValueError (naming the columns present) if no `league` column
    exists at all — that means flatten_columns/reset_index produced something
    unexpected and silently returning everything would hide it. A `league`
    column that exists but has no matching rows is a legitimate outcome (that
    league has no data for this pull) and returns an empty frame, not an error.
    """
    if "league" not in frame.columns:
        raise ValueError(
            f"split_league: no 'league' column found; columns are {list(frame.columns)}"
        )
    nan_count = int(frame["league"].isna().sum())
    if nan_count:
        logger.warning(
            "split_league: %d row(s) have a NaN league (an FBref competition name not "
            "mapped by soccerdata's BIG_FIVE_DICT — see _patch_league_aliases) and are "
            "left out, not guessed at; requested league=%r",
            nan_count, league,
        )
    return frame.loc[frame["league"] == league].copy()


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse FBref's two-level column headers into snake_case names."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    flat = []
    for upper, lower in df.columns:
        parts = [p for p in (upper, lower) if p and not str(p).startswith("Unnamed")]
        name = "_".join(parts) if parts else "unnamed"
        # Translate '%' to '_pct' BEFORE the squash below, so e.g. "SoT%"
        # becomes "standard_sot_pct" rather than colliding with the
        # unadorned "SoT" column (both would otherwise squash to
        # "standard_sot", silently dropping one of the two via a duplicate
        # column name).
        name = name.replace("%", "_pct")
        name = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
        flat.append(name)
    out = df.copy()
    out.columns = flat
    return out


def _dedupe_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate-named columns, keeping the first occurrence.

    FBref's live "playing_time" table flattens two distinct sub-headers to
    the same name (`playing_time_min`), so a table can carry a real
    duplicate-named column before it ever reaches a merge. Left alone,
    selecting/joining on a duplicated name multiplies columns (or rows, via
    a duplicated join key) instead of raising, so this is applied before
    every merge below.
    """
    return frame.loc[:, ~frame.columns.duplicated(keep="first")]


def normalise_team_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map live FBref team columns to the names the club style vector expects.

    FBref's live team pages (seasons "2324"+) flatten possession to `poss`,
    but `src.clubs.STYLE_COLUMNS` expects the historical dump's name
    `possession`. Rename `poss` -> `possession` when `possession` isn't
    already present, then dedupe any duplicate-named columns this — or an
    earlier merge — may have produced.
    """
    frame = frame.copy()
    if "poss" in frame.columns and "possession" not in frame.columns:
        frame = frame.rename(columns={"poss": "possession"})
    return _dedupe_columns(frame)


def merge_stat_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Left-join every stat table onto `standard`, keeping all its players."""
    merged = _dedupe_columns(tables["standard"]).copy()
    for name, table in tables.items():
        if name == "standard":
            continue
        table = _dedupe_columns(table)
        keys = [k for k in _JOIN_KEYS if k in table.columns and k in merged.columns]
        new_cols = [c for c in table.columns if c not in merged.columns or c in keys]
        merged = merged.merge(table[new_cols], on=keys, how="left")
    return merged


def normalise_identity(frame: pd.DataFrame) -> pd.DataFrame:
    """Give downstream modules the column names they expect.

    FBref calls the nationality column `nation` and never reports how many
    fixtures a club played. Both are required by the adaptation bundle (Task 10)
    and the minutes target (Task 12), so they are derived once, here.

    Raises ValueError if a non-empty frame is missing any of REQUIRED_COLUMNS —
    silently passing through a frame with the wrong schema is what let a
    flattened-column mismatch go unnoticed until much later tasks.

    FBref's live pages (seasons "2324"+) no longer carry a `minutes` column
    at all — the closest equivalent is `playing_time_min` (from the
    "standard" or "playing_time" tables). When `minutes` is absent but
    `playing_time_min` is present, it is used to derive `minutes` so the
    REQUIRED_COLUMNS check below (and everything downstream that expects
    `minutes`) keeps working on the live schema too.
    """
    out = frame.copy()
    # Dedupe first: a duplicate-named column (e.g. the live "playing_time"
    # table's doubled `playing_time_min`) must not reach the required-columns
    # check or the derivations below as two columns sharing one name.
    out = _dedupe_columns(out)
    if "minutes" not in out.columns and "playing_time_min" in out.columns:
        out["minutes"] = out["playing_time_min"]
    if "nation" in out.columns and "nationality" not in out.columns:
        out["nationality"] = out["nation"].astype(str).str[-3:].str.upper()
    if "playing_time_mp" in out.columns:
        out["team_matches"] = out.groupby(["team", "season"])["playing_time_mp"].transform("max")
    if not out.empty:
        missing = check_required_columns(out)
        if missing:
            raise ValueError(f"normalise_identity: frame is missing required columns: {missing}")
    return out


_BIG_5_COMBINED = "Big 5 European Leagues Combined"


def _patch_league_aliases() -> None:
    """Add missing raw-FBref-name aliases to soccerdata's Big-5 league map.

    soccerdata's FBref reader maps each row's raw "Comp" value (and the
    seasons index) to a canonical league id via its own internal
    `sd.fbref.BIG_FIVE_DICT`. FBref's live combined-competition pages now
    label the German league plainly "Bundesliga", but BIG_FIVE_DICT (as
    installed, soccerdata 1.9.1) only has the historical key
    "Fußball-Bundesliga" — so those rows map to NaN `league` and
    split_league(frame, "GER-Bundesliga") silently returns nothing (verified:
    507 NaN-league rows in the 2023-24 standard table). `setdefault` each
    alias so an existing key, or a future soccerdata release's own fix, is
    never overwritten.
    """
    aliases = {
        "Bundesliga": "GER-Bundesliga",
        "Fußball-Bundesliga": "GER-Bundesliga",
        "Premier League": "ENG-Premier League",
        "La Liga": "ESP-La Liga",
        "Serie A": "ITA-Serie A",
        "Ligue 1": "FRA-Ligue 1",
    }
    for raw_name, canonical in aliases.items():
        sd.fbref.BIG_FIVE_DICT.setdefault(raw_name, canonical)


def _fbref(season: str):
    """A combined-competition FBref reader for one season, covering all Big-5 leagues.

    Constructing one per individual league (e.g. leagues="ITA-Serie A")
    raised `ValueError: Invalid league 'ITA-Serie A'` for some seasons against
    the installed soccerdata 1.9.1 — the combined competition is the reliable
    path, so every caller reads it and filters with split_league().
    """
    _patch_league_aliases()
    return HumanCaptchaFBref(
        leagues=_BIG_5_COMBINED,
        seasons=season,
        data_dir=config.RAW_DIR / "soccerdata",
        headless=False,
    )


def player_season_stats(league: str, season: str) -> pd.DataFrame:
    """All player stat tables for one league-season, merged into one frame."""

    def build() -> pd.DataFrame:
        limiter = cache.RateLimiter(config.REQUEST_DELAY_SECONDS)
        client = _fbref(season)
        tables = {}
        skipped: list[str] = []
        for stat_type in STAT_TYPES:
            limiter.wait()
            try:
                raw = client.read_player_season_stats(stat_type=stat_type)
            except Exception as exc:  # noqa: BLE001 — unsupported/missing table, logged below
                skipped.append(stat_type)
                logger.warning(
                    "Skipping FBref player stat table %r for %s %s (unsupported by this "
                    "soccerdata version, or missing for this league-season): %s",
                    stat_type, league, season, exc,
                )
                continue
            flat = flatten_columns(raw.reset_index())
            tables[stat_type] = split_league(flat, league)
        if "standard" not in tables:
            empty = pd.DataFrame()
            empty.attrs["skipped_tables"] = skipped
            return empty
        merged = merge_stat_tables(tables)
        merged["league"] = league
        merged["season"] = season
        result = normalise_identity(merged)
        result.attrs["skipped_tables"] = skipped
        return result

    return cache.cached_frame(f"fbref/player/{league.replace(' ', '_')}_{season}", build)


def team_season_stats(league: str, season: str) -> pd.DataFrame:
    """Team-level stat tables for one league-season, used for club style vectors."""

    def build() -> pd.DataFrame:
        limiter = cache.RateLimiter(config.REQUEST_DELAY_SECONDS)
        client = _fbref(season)
        tables = {}
        skipped: list[str] = []
        # "passing"/"possession"/"defense" are not in SUPPORTED_TEAM_STAT_TYPES
        # for soccerdata 1.9.1; this loop already only asks for supported
        # types, so a skip here means a table missing for this league-season,
        # not an unsupported stat_type — still logged and recorded either way.
        for stat_type in ("standard", "shooting", "playing_time", "misc"):
            limiter.wait()
            try:
                raw = client.read_team_season_stats(stat_type=stat_type)
            except Exception as exc:  # noqa: BLE001
                skipped.append(stat_type)
                logger.warning(
                    "Skipping FBref team stat table %r for %s %s: %s",
                    stat_type, league, season, exc,
                )
                continue
            flat = flatten_columns(raw.reset_index())
            flat = flat.rename(columns={"squad": "team"})
            tables[stat_type] = split_league(flat, league)
        if "standard" not in tables:
            empty = pd.DataFrame()
            empty.attrs["skipped_tables"] = skipped
            return empty
        merged = tables["standard"]
        for name, table in tables.items():
            if name == "standard":
                continue
            new_cols = [c for c in table.columns if c not in merged.columns or c == "team"]
            merged = merged.merge(table[new_cols], on="team", how="left")
        merged["league"] = league
        merged["season"] = season
        merged = normalise_team_columns(merged)
        merged.attrs["skipped_tables"] = skipped
        return merged

    return cache.cached_frame(f"fbref/team/{league.replace(' ', '_')}_{season}", build)


def pull_all() -> None:
    """Fetch every league-season. Resumable: cached pulls are skipped."""
    print("Run this in a terminal you are watching; it pauses for you when FBref serves a CAPTCHA.")
    unsupported = [s for s in STAT_TYPES if s not in SUPPORTED_PLAYER_STAT_TYPES]
    if unsupported:
        print(
            f"WARNING: this soccerdata version does not support player stat_type(s) "
            f"{unsupported} — those tables will be skipped for every league-season pulled below."
        )
    for league in config.CANDIDATE_LEAGUES:
        for season in config.SEASONS:
            print(f"player {league} {season}")
            player_season_stats(league, season)
            if league == config.SERIE_A:
                print(f"team   {league} {season}")
                team_season_stats(league, season)


if __name__ == "__main__":
    pull_all()
