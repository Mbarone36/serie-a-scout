"""API-Football source: defensive / dribbling / duel stats for ALL seasons.

FBref's live pages (seasons "2324"+) dropped the defense and possession
tables, so tackles, take-ons and aerials only exist for the dump era.
Mixing the two cohorts degrades the model (see `src/understat.py`), so, like
Understat for xG, API-Football is used as ONE source for every season in
scope: tackles, blocks, interceptions, duels, dribbles and key passes.

Fetching uses plain `requests` through the project's own disk cache and a
rate limiter against the official v3 API (``/players?league=&season=&page=``,
20 players per page), authenticated with the user's own key from the
``API_FOOTBALL_KEY`` environment variable. Every page is cached on disk, so a
pull interrupted by the daily request quota resumes where it stopped.

API-Football and FBref share no identifiers, so matching reuses
`understat.match_to_fbref` (normalised name AND minutes agreement, one to
one) — first on the full registered name ("Lautaro Javier Martínez"), then,
for whoever is still unmatched, on the display name, which is how mononyms
("Vitinha") are listed. `augment` writes each matched player's per-90 rates
back onto every FBref row exactly as `understat.augment` does.
"""

import json
import logging
import os

import numpy as np
import pandas as pd
import requests

from src import cache, config, understat

logger = logging.getLogger(__name__)

BASE = "https://v3.football.api-sports.io"
PLAYERS_URL = BASE + "/players"
KEY_ENV = "API_FOOTBALL_KEY"
_TIMEOUT = 60

LEAGUE_IDS = {
    "ITA-Serie A": 135,
    "ENG-Premier League": 39,
    "ESP-La Liga": 140,
    "GER-Bundesliga": 78,
    "FRA-Ligue 1": 61,
}

PROCESSED_FILE = "api_football.parquet"

# (statistics block, field) -> our column. Season totals, may be null.
_FIELD_MAP = {
    ("tackles", "total"): "af_tackles",
    ("tackles", "blocks"): "af_blocks",
    ("tackles", "interceptions"): "af_interceptions",
    ("duels", "total"): "af_duels",
    ("duels", "won"): "af_duels_won",
    ("dribbles", "attempts"): "af_dribbles_att",
    ("dribbles", "success"): "af_dribbles_succ",
    ("dribbles", "past"): "af_dribbled_past",
    ("passes", "key"): "af_key_passes",
    ("passes", "total"): "af_passes",
}

STAT_COLUMNS = list(_FIELD_MAP.values())
FRAME_COLUMNS = ["league", "season", "af_id", "player", "short_name", "team",
                 "born", "minutes", *STAT_COLUMNS]

# Minutes-proportional totals written onto FBref rows (per_90 recovers the
# rate). `af_duels` / `af_dribbles_att` are kept only to derive the rate
# below: volume of attempts is not itself a quality signal.
# `af_dribbled_past` is parsed but not used: the API returns it null for
# every Big-5 player in every season 2017-18..2025-26.
# `af_passes` is not a player feature; `clubs.add_profile_columns` sums it
# into each club's passing volume.
AUGMENT_COLUMNS = ["af_tackles", "af_blocks", "af_interceptions", "af_duels_won",
                   "af_dribbles_succ", "af_key_passes", "af_passes"]
# Shares, written unscaled onto every row of the player-season.
RATE_COLUMNS = ["af_duels_won_pct"]

_limiter = cache.RateLimiter(config.API_FOOTBALL_DELAY_SECONDS)


class QuotaExceeded(RuntimeError):
    """The account's request quota is used up; stop the whole pull."""


class PlanRestricted(RuntimeError):
    """The account's plan does not cover this league-season."""


def season_to_year(season: str) -> int:
    """'2324' -> 2023 (API-Football keys a season by its starting year)."""
    return 2000 + int(str(season)[:2])


def _api_key() -> str:
    key = os.environ.get(KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(
            f"{KEY_ENV} is not set. Sign up at api-football.com, then "
            f"`export {KEY_ENV}=<your key>` in the shell that runs the pull.")
    return key


def check_payload(payload) -> None:
    """Raise on an error payload so it is never cached.

    API-Football reports quota/plan/auth problems as HTTP 200 with a
    non-empty `errors` object (or list), not as an HTTP error status.
    """
    if not isinstance(payload, dict) or "response" not in payload:
        raise ValueError(f"unexpected JSON shape: {str(payload)[:120]!r}")
    errors = payload.get("errors") or {}
    if not errors:
        return
    text = json.dumps(errors)
    lowered = text.lower()
    if "limit" in lowered or "requests" in lowered:
        raise QuotaExceeded(text)
    if "plan" in lowered or "season" in lowered:
        raise PlanRestricted(text)
    raise ValueError(f"API-Football error: {text}")


def fetch_page(league: str, season: str, page: int) -> str:
    """Raw JSON text for one page of one league-season, disk-cached."""
    league_id = LEAGUE_IDS[league]
    year = season_to_year(season)

    def build() -> str:
        _limiter.wait()
        response = requests.get(
            PLAYERS_URL,
            params={"league": league_id, "season": year, "page": page},
            headers={"x-apisports-key": _api_key()},
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        text = response.text
        try:
            payload = json.loads(text)
        except ValueError as exc:
            raise ValueError(f"non-JSON response: {text[:120]!r}") from exc
        check_payload(payload)
        return text

    return cache.cached_text(f"api_football/{league_id}_{year}_p{page}", build)


def fetch_league_season(league: str, season: str) -> list[str]:
    """Every page's JSON text for one league-season."""
    first = fetch_page(league, season, 1)
    total = int((json.loads(first).get("paging") or {}).get("total") or 1)
    return [first] + [fetch_page(league, season, p) for p in range(2, total + 1)]


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _born(birth_date) -> float:
    try:
        return float(str(birth_date)[:4])
    except (TypeError, ValueError):
        return np.nan


def parse_pages(pages: list[str], league: str, season: str) -> pd.DataFrame:
    """One row per API-Football player in the league-season.

    A player's `statistics` holds one entry per club (and possibly other
    competitions); only entries for this league are kept, and a mid-season
    move within the league is summed into one row, as Understat reports it.
    """
    league_id = LEAGUE_IDS[league]
    rows = []
    for text in pages:
        for item in json.loads(text).get("response", []):
            player = item.get("player") or {}
            full = " ".join(p for p in (player.get("firstname"), player.get("lastname")) if p)
            for stats in item.get("statistics") or []:
                if (stats.get("league") or {}).get("id") != league_id:
                    continue
                row = {
                    "league": league, "season": season,
                    "af_id": str(player.get("id", "")),
                    "player": (full or player.get("name") or "").strip(),
                    "short_name": str(player.get("name") or "").strip(),
                    "team": str((stats.get("team") or {}).get("name") or "").strip(),
                    "born": _born((player.get("birth") or {}).get("date")),
                    "minutes": np.nan_to_num(_num((stats.get("games") or {}).get("minutes"))),
                }
                for (block, field), column in _FIELD_MAP.items():
                    row[column] = _num((stats.get(block) or {}).get(field))
                rows.append(row)
    frame = pd.DataFrame(rows, columns=FRAME_COLUMNS)
    if frame.empty:
        return frame
    first = {c: "first" for c in ("player", "short_name", "born")}
    sums = {c: lambda s: s.sum(min_count=1) for c in ["minutes", *STAT_COLUMNS]}
    frame = (frame.groupby(["league", "season", "af_id"], as_index=False, sort=False)
             .agg({**first, "team": lambda s: "/".join(dict.fromkeys(s)), **sums}))
    frame["minutes"] = frame["minutes"].astype(int)
    return frame[FRAME_COLUMNS]


def pull_all(leagues=None, seasons=None) -> pd.DataFrame:
    """Fetch + parse every league-season and write `api_football.parquet`.

    Resumable: pages are cached, so a rerun only fetches what is missing.
    A plan-restricted or failing league-season is reported and skipped; an
    exhausted quota stops the pull (with what was gathered so far written),
    since every further request would fail too.
    """
    leagues = list(leagues if leagues is not None else config.CANDIDATE_LEAGUES)
    seasons = list(seasons if seasons is not None else config.SEASONS)
    frames, failures = [], []
    for league in leagues:
        for season in seasons:
            try:
                frame = parse_pages(fetch_league_season(league, season), league, season)
            except QuotaExceeded as exc:
                print(f"STOPPED: API-Football quota exhausted ({exc}); rerun later to resume")
                failures.append((league, season, repr(exc)))
                break
            except Exception as exc:  # report and move on
                logger.warning("api-football %s %s failed: %r", league, season, exc)
                print(f"FAILED api-football {league} {season}: {exc!r}")
                failures.append((league, season, repr(exc)))
                continue
            print(f"api-football {league} {season}: {len(frame)} players")
            frames.append(frame)
        else:
            continue
        break
    out = (pd.concat(frames, ignore_index=True) if frames
           else pd.DataFrame(columns=FRAME_COLUMNS))
    out.to_parquet(config.PROCESSED_DIR / PROCESSED_FILE)
    out.attrs["failures"] = failures
    return out


def load() -> pd.DataFrame | None:
    """The processed API-Football frame, or None (with a WARNING) when absent."""
    path = config.PROCESSED_DIR / PROCESSED_FILE
    if not path.exists():
        print(f"WARNING: {PROCESSED_FILE} not found — API-Football columns "
              "will be NaN (features neutral-filled)")
        return None
    return pd.read_parquet(path)


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------


def _as_understat(af: pd.DataFrame, name_column: str) -> pd.DataFrame:
    """The columns `understat.match_to_fbref` reads, under its names."""
    return af[["league", "season", "af_id", name_column, "minutes"]].rename(
        columns={"af_id": "us_id", name_column: "player"})


def match_to_fbref(fbref_seasons: pd.DataFrame, af: pd.DataFrame) -> pd.DataFrame:
    """One-to-one FBref player-season <-> API-Football row mapping.

    Pass 1 matches on the full registered name; pass 2 retries the players
    and API-Football rows pass 1 left unmatched, on the display name.
    Returns columns league, season, player, af_id, af_player, name_rule.
    """
    columns = ["league", "season", "player", "af_id", "af_player", "name_rule"]
    if fbref_seasons.empty or af is None or af.empty:
        return pd.DataFrame(columns=columns)

    first = understat.match_to_fbref(fbref_seasons, _as_understat(af, "player"))
    taken_fb = set(zip(first["league"], first["season"], first["player"]))
    taken_af = set(zip(first["league"], first["season"], first["us_id"]))
    fb_rest = fbref_seasons[[k not in taken_fb for k in zip(
        fbref_seasons["league"], fbref_seasons["season"], fbref_seasons["player"])]]
    af_rest = af[[k not in taken_af for k in zip(af["league"], af["season"], af["af_id"])]]
    second = understat.match_to_fbref(fb_rest, _as_understat(af_rest, "short_name"))
    second = second.assign(name_rule="short_" + second["name_rule"].astype(str))

    both = pd.concat([first, second], ignore_index=True)
    return both.rename(columns={"us_id": "af_id", "us_player": "af_player"})[columns]


# --------------------------------------------------------------------------
# Augmenting FBref rows
# --------------------------------------------------------------------------


def augment(seasons: pd.DataFrame, af: pd.DataFrame | None = None,
            verbose: bool = True) -> pd.DataFrame:
    """Add `AUGMENT_COLUMNS` (minutes-proportional) and `RATE_COLUMNS` to
    every FBref row; unmatched rows are NaN. Same arithmetic as
    `understat.augment`: per-90 on API-Football's own minutes, rescaled by
    each FBref row's minutes. `af=None` loads `api_football.parquet`.
    """
    added = AUGMENT_COLUMNS + RATE_COLUMNS
    out = seasons.drop(columns=[c for c in added if c in seasons.columns])
    if af is None:
        af = load()
    if af is None or af.empty or out.empty:
        for column in added:
            out[column] = np.nan
        return out

    mapping = match_to_fbref(out, af)
    rates = mapping[["league", "season", "player", "af_id"]].merge(
        af[["league", "season", "af_id", "minutes", "af_duels", *AUGMENT_COLUMNS]],
        on=["league", "season", "af_id"], how="left",
    )
    af_minutes = rates["minutes"].astype(float).replace(0, np.nan)
    rates["af_duels_won_pct"] = (rates["af_duels_won"]
                                 / rates["af_duels"].astype(float).replace(0, np.nan))
    for column in AUGMENT_COLUMNS:
        rates[column] = rates[column] / af_minutes * 90.0
    rates = rates.drop(columns=["af_id", "minutes", "af_duels"])

    index = out.index
    merged = out.reset_index(drop=True).merge(
        rates, on=["league", "season", "player"], how="left", validate="many_to_one",
    )
    row_minutes = merged["minutes"].astype(float)
    for column in AUGMENT_COLUMNS:
        merged[column] = merged[column] * row_minutes / 90.0
    merged.index = index

    if verbose:
        report = understat.match_rates(out, mapping.rename(columns={"af_id": "us_id"}))
        if len(report):
            total = report["matched"].sum() / report["eligible"].sum()
            print(f"api-football match overall: {report['matched'].sum()}/"
                  f"{report['eligible'].sum()} ({total:.1%})")
    return merged


if __name__ == "__main__":
    result = pull_all()
    print(f"{len(result)} player-seasons written to {config.PROCESSED_DIR / PROCESSED_FILE}")
    for failure in result.attrs.get("failures", []):
        print("failure:", failure)
