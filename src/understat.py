"""Understat xG source: a consistent expected-goals signal across ALL seasons.

FBref removed its advanced (xG / progression / possession) tables from the
live pages, so seasons "2223"+ carry basics only, while the historical dump
has xG for earlier seasons. Mixing the two cohorts degrades the model badly.
Understat publishes npxG / xA / xGChain / xGBuildup for the Big 5 back to
2014-15, so it is used as one xG source for every season.

Fetching uses plain `requests` through the project's own disk cache and
rate limiter (same pattern as `scrape_tm.fetch`) against Understat's own
front-end JSON endpoint ``/getLeagueData/<slug>/<start_year>``. Nothing here
impersonates a browser.

Understat and FBref share no identifiers, so `match_to_fbref` pairs rows
within each (league, season) by normalised name AND minutes agreement, one
to one. `augment` then writes each matched player's Understat per-90 rates
back onto every FBref row (scaled by that row's own minutes) so split-season
rows get proportional values and `normalize.per_90` recovers the rate.
"""

import html
import json
import logging

import numpy as np
import pandas as pd
import requests
from rapidfuzz import fuzz

from src import cache, config, entities

logger = logging.getLogger(__name__)

BASE = "https://understat.com"
LEAGUE_URL = BASE + "/getLeagueData/{slug}/{year}"
HEADERS = {
    "User-Agent": config.RESEARCH_USER_AGENT,
    "X-Requested-With": "XMLHttpRequest",
}
_TIMEOUT = 60

LEAGUE_SLUGS = {
    "ITA-Serie A": "Serie_A",
    "ENG-Premier League": "EPL",
    "ESP-La Liga": "La_liga",
    "GER-Bundesliga": "Bundesliga",
    "FRA-Ligue 1": "Ligue_1",
}

PROCESSED_FILE = "understat.parquet"

# Understat JSON field -> our column. All arrive as strings.
_FIELD_MAP = {
    "npxG": "us_npxg",
    "xA": "us_xa",
    "xGChain": "us_xgchain",
    "xGBuildup": "us_xgbuildup",
    "key_passes": "us_key_passes",
    "shots": "us_shots",
    "games": "us_games",
}

PLAYER_COLUMNS = ["player", "team", "minutes", *_FIELD_MAP.values()]
FRAME_COLUMNS = ["league", "season", "us_id", *PLAYER_COLUMNS]

# The columns `augment` writes onto FBref rows (minutes-proportional totals).
# `us_games` is kept in the Understat frame / mapping diagnostics only: an
# appearance count does not scale with a split row's minutes.
AUGMENT_COLUMNS = ["us_npxg", "us_xa", "us_xgchain", "us_xgbuildup",
                   "us_key_passes", "us_shots"]

NAME_FUZZ_THRESHOLD = 88
MINUTES_TOLERANCE_ABS = 90
MINUTES_TOLERANCE_REL = 0.10
FUZZY_LAST_TOKEN_RATIO = 90
MONONYM_MINUTES_ABS = 45
MONONYM_MINUTES_REL = 0.05

# Candidate precedence: an exact name beats a subset beats a fuzzy match,
# whatever their scores (a loose fuzzy hit must never outrank a real subset).
_RULE_RANK = {"exact": 0, "subset": 1, "fuzzy": 2}

MAPPING_COLUMNS = ["league", "season", "player", "us_id", "us_player",
                   "fb_minutes", "us_minutes", "minutes_diff", "name_score",
                   "name_rule"]

_limiter = cache.RateLimiter(config.REQUEST_DELAY_SECONDS)


def season_to_start_year(season: str) -> int:
    """'2324' -> 2023 (Understat keys a season by its starting year)."""
    return 2000 + int(str(season)[:2])


def fetch_league_season(league: str, season: str) -> str:
    """Raw JSON text for one league-season, disk-cached and rate-limited.

    A non-2xx response or a body that is not the expected JSON object raises,
    so `cache.cached_text` never persists an error page to disk.
    """
    slug = LEAGUE_SLUGS[league]
    year = season_to_start_year(season)
    url = LEAGUE_URL.format(slug=slug, year=year)

    def build() -> str:
        _limiter.wait()
        response = requests.get(url, headers=HEADERS, timeout=_TIMEOUT)
        response.raise_for_status()
        text = response.text
        try:
            payload = json.loads(text)
        except ValueError as exc:
            raise ValueError(f"non-JSON response from {url}: {text[:120]!r}") from exc
        if not isinstance(payload, dict) or "players" not in payload:
            raise ValueError(f"unexpected JSON shape from {url}: keys={list(payload)[:5]}")
        return text

    return cache.cached_text(f"understat/{slug}_{year}", build)


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def parse_players(json_text: str, league: str, season: str) -> pd.DataFrame:
    """One row per Understat player in the league-season (season totals)."""
    payload = json.loads(json_text)
    rows = []
    for item in payload.get("players", []):
        row = {
            "league": league,
            "season": season,
            "us_id": str(item.get("id", "")),
            "player": html.unescape(str(item.get("player_name", ""))).strip(),
            "team": html.unescape(str(item.get("team_title", ""))).strip(),
            "minutes": int(np.nan_to_num(_num(item.get("time")))),
        }
        for field, column in _FIELD_MAP.items():
            row[column] = _num(item.get(field))
        rows.append(row)
    frame = pd.DataFrame(rows, columns=FRAME_COLUMNS)
    frame["minutes"] = frame["minutes"].astype(int)
    for column in _FIELD_MAP.values():
        frame[column] = frame[column].astype(float)
    return frame


def pull_all(leagues=None, seasons=None) -> pd.DataFrame:
    """Fetch + parse every league-season and write `understat.parquet`.

    Resumable: each league-season's raw JSON is in the text cache, so a rerun
    only fetches what is missing. A failing league-season is logged, reported
    in the returned frame's `attrs["failures"]`, and skipped (not retried).
    """
    leagues = list(leagues if leagues is not None else config.CANDIDATE_LEAGUES)
    seasons = list(seasons if seasons is not None else config.SEASONS)
    frames, failures = [], []
    for league in leagues:
        for season in seasons:
            try:
                text = fetch_league_season(league, season)
                frame = parse_players(text, league, season)
            except Exception as exc:  # report and move on; never hammer the site
                logger.warning("understat %s %s failed: %r", league, season, exc)
                print(f"FAILED understat {league} {season}: {exc!r}")
                failures.append((league, season, repr(exc)))
                continue
            print(f"understat {league} {season}: {len(frame)} players")
            frames.append(frame)
    out = (pd.concat(frames, ignore_index=True) if frames
           else pd.DataFrame(columns=FRAME_COLUMNS))
    out.to_parquet(config.PROCESSED_DIR / PROCESSED_FILE)
    out.attrs["failures"] = failures
    return out


def load() -> pd.DataFrame | None:
    """The processed Understat frame, or None (with a WARNING) when absent."""
    path = config.PROCESSED_DIR / PROCESSED_FILE
    if not path.exists():
        print(f"WARNING: {PROCESSED_FILE} not found — Understat xG columns "
              "will be NaN (features neutral-filled, labels fall back to FBref xG)")
        return None
    return pd.read_parquet(path)


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------


def _name_verdict(fb_key: str, us_key: str) -> tuple[float, str, bool] | None:
    """(score, rule, mononym) when the two normalised names agree, else None.

    - exact: identical normalised names.
    - subset: one token set contains the other AND either the shorter name is
      a single token (a mononym such as "Kepa"; `mononym` is True so the
      caller tightens the minutes gate) or the shorter name's LAST token is
      among the shared tokens ("luis alberto" in "luis alberto romero
      alconchel"). This stops a shared first name alone from matching.
    - fuzzy: `token_sort_ratio` >= NAME_FUZZ_THRESHOLD AND the last tokens of
      the two names are equal or `fuzz.ratio` >= FUZZY_LAST_TOKEN_RATIO, so
      "aboubacar ali" never fuzzy-matches "aboubacar lo" (different surnames).
    """
    if not fb_key or not us_key:
        return None
    if fb_key == us_key:
        return 100.0, "exact", False
    score = float(fuzz.token_sort_ratio(fb_key, us_key))
    fb_list, us_list = fb_key.split(), us_key.split()
    fb_tokens, us_tokens = set(fb_list), set(us_list)
    if fb_tokens <= us_tokens or us_tokens <= fb_tokens:
        shorter = fb_list if len(fb_tokens) <= len(us_tokens) else us_list
        shared = fb_tokens & us_tokens
        if len(shorter) == 1:
            return score, "subset", True
        if shorter[-1] in shared:
            return score, "subset", False
    if score >= NAME_FUZZ_THRESHOLD:
        last_fb, last_us = fb_list[-1], us_list[-1]
        if last_fb == last_us or fuzz.ratio(last_fb, last_us) >= FUZZY_LAST_TOKEN_RATIO:
            return score, "fuzzy", False
    return None


def _minutes_agree(a: float, b: float, mononym: bool = False) -> bool:
    if mononym:
        tolerance = max(MONONYM_MINUTES_ABS, MONONYM_MINUTES_REL * max(a, b))
    else:
        tolerance = max(MINUTES_TOLERANCE_ABS, MINUTES_TOLERANCE_REL * max(a, b))
    return abs(a - b) <= tolerance


def ambiguous_namesakes(fbref_seasons: pd.DataFrame) -> set[tuple[str, str, str]]:
    """(league, season, player) names that cannot be safely aggregated.

    FBref rows are summed per player NAME within a (league, season). A name
    on more than one team is either one player who moved mid-season (safe to
    sum) or two different people sharing a name (a 20-minute cameo plus a
    2000-minute starter would sum to the starter's Understat minutes and the
    merge would attach the starter's rate to the cameo row). They are told
    apart by `born` (all rows non-null: one value = same person, several =
    different people; non-null on only some rows = ambiguous), then, only
    when `born` is NaN on EVERY row, by `nationality` the same way. With
    neither decisive the name is ambiguous. Both "different people" and "unknown"
    are returned: neither gets Understat data.
    """
    needed = {"league", "season", "player", "team"}
    if fbref_seasons.empty or not needed <= set(fbref_seasons.columns):
        return set()
    teams = fbref_seasons.groupby(["league", "season", "player"], dropna=False)["team"].nunique()
    multi = teams[teams > 1].index
    if len(multi) == 0:
        return set()
    sub = fbref_seasons.set_index(["league", "season", "player"]).loc[multi]
    ambiguous = set()
    for key, group in sub.groupby(level=[0, 1, 2], dropna=False):
        verdict = None  # True = same person, False = different people
        for column in ("born", "nationality"):
            if column not in group.columns:
                continue
            values = group[column]
            distinct = values.dropna().astype(str).nunique()
            if distinct > 1:
                verdict = False
                break
            if distinct == 1:
                # Present on every row: one value = same person. Present on
                # only some rows: undecidable, and nationality must not
                # override a partially known `born`.
                verdict = bool(values.notna().all())
                break
        if verdict is not True:
            ambiguous.add(key)
    return ambiguous


def match_to_fbref(fbref_seasons: pd.DataFrame, us: pd.DataFrame) -> pd.DataFrame:
    """One-to-one FBref player-season <-> Understat row mapping.

    Within each (league, season), FBref minutes are summed per player (a
    split season sums its club rows; Understat reports a mid-season mover
    within one league as a single row). A pair is a candidate when the
    normalised names agree (see `_name_verdict`) AND minutes agree within
    max(90, 10% of the larger) (max(45, 5%) for a mononym subset match).
    Candidates are taken greedily by rule (exact > subset > fuzzy), then best
    name score, then closest minutes, never reusing either side, so no
    Understat row is assigned to two FBref players (nor vice versa).

    Names shared by different (or undeterminable) FBref people within one
    league-season are skipped entirely (see `ambiguous_namesakes`).
    """
    if fbref_seasons.empty or us is None or us.empty:
        return pd.DataFrame(columns=MAPPING_COLUMNS)

    fb = (fbref_seasons.groupby(["league", "season", "player"], dropna=False)["minutes"]
          .sum().reset_index().rename(columns={"minutes": "fb_minutes"}))
    skipped = ambiguous_namesakes(fbref_seasons)
    if skipped:
        fb = fb[[k not in skipped for k in zip(fb["league"], fb["season"], fb["player"])]].copy()
        if fb.empty:
            return pd.DataFrame(columns=MAPPING_COLUMNS)
    fb["key"] = fb["player"].map(entities.normalise_name)
    us = us.copy()
    us["key"] = us["player"].map(entities.normalise_name)

    rows = []
    us_groups = {k: g for k, g in us.groupby(["league", "season"], sort=False)}
    for (league, season), fb_group in fb.groupby(["league", "season"], sort=False):
        us_group = us_groups.get((league, season))
        if us_group is None or us_group.empty:
            continue
        fb_min = fb_group["fb_minutes"].to_numpy(dtype=float)
        us_min = us_group["minutes"].to_numpy(dtype=float)
        # Minutes gate first (cheap, vectorised) to prune the name scoring.
        diff = np.abs(fb_min[:, None] - us_min[None, :])
        tol = np.maximum(MINUTES_TOLERANCE_ABS,
                         MINUTES_TOLERANCE_REL * np.maximum(fb_min[:, None], us_min[None, :]))
        fb_idx, us_idx = np.nonzero(diff <= tol)

        fb_keys = fb_group["key"].to_numpy()
        us_keys = us_group["key"].to_numpy()
        candidates = []
        for i, j in zip(fb_idx, us_idx):
            verdict = _name_verdict(fb_keys[i], us_keys[j])
            if verdict is None:
                continue
            score, rule, mononym = verdict
            if mononym and not _minutes_agree(fb_min[i], us_min[j], mononym=True):
                continue
            candidates.append((_RULE_RANK[rule], -score, diff[i, j], i, j, score, rule))
        candidates.sort(key=lambda c: c[:5])

        taken_fb, taken_us = set(), set()
        fb_players = fb_group["player"].to_numpy()
        us_ids = us_group["us_id"].to_numpy()
        us_players = us_group["player"].to_numpy()
        for _, _, d, i, j, score, rule in candidates:
            if i in taken_fb or j in taken_us:
                continue
            taken_fb.add(i)
            taken_us.add(j)
            rows.append({
                "league": league, "season": season, "player": fb_players[i],
                "us_id": us_ids[j], "us_player": us_players[j],
                "fb_minutes": float(fb_min[i]), "us_minutes": float(us_min[j]),
                "minutes_diff": float(d), "name_score": score, "name_rule": rule,
            })
    return pd.DataFrame(rows, columns=MAPPING_COLUMNS)


def match_rates(fbref_seasons: pd.DataFrame, mapping: pd.DataFrame,
                min_minutes: int = config.MIN_MINUTES_PRIOR_SEASON) -> pd.DataFrame:
    """Per (league, season): matched / FBref players with >= `min_minutes`.

    `ambiguous` counts the eligible players whose name is shared by different
    (or undeterminable) FBref people and was therefore skipped by the match.
    """
    skipped = ambiguous_namesakes(fbref_seasons)
    totals = (fbref_seasons.groupby(["league", "season", "player"], dropna=False)["minutes"]
              .sum().reset_index())
    eligible = totals[totals["minutes"] >= min_minutes]
    matched_keys = set(zip(mapping["league"], mapping["season"], mapping["player"]))
    keys = list(zip(eligible["league"], eligible["season"], eligible["player"]))
    eligible = eligible.assign(
        matched=[k in matched_keys for k in keys],
        ambiguous=[k in skipped for k in keys],
    )
    out = (eligible.groupby(["league", "season"])[["matched", "ambiguous"]].sum()
           .join(eligible.groupby(["league", "season"]).size().rename("eligible"))
           .reset_index())
    out = out[["league", "season", "matched", "eligible", "ambiguous"]]
    out["rate"] = out["matched"] / out["eligible"]
    return out


# --------------------------------------------------------------------------
# Augmenting FBref rows
# --------------------------------------------------------------------------


def augment(seasons: pd.DataFrame, us: pd.DataFrame | None = None,
            verbose: bool = True) -> pd.DataFrame:
    """Add `AUGMENT_COLUMNS` to every FBref row.

    Each matched player's Understat season totals become per-90 rates on
    Understat's OWN minutes (total / us_minutes * 90), and each FBref row
    gets `rate * row_minutes / 90`. So a split-season player's club rows
    carry values proportional to their minutes, and `normalize.per_90` on
    any row recovers exactly the Understat rate. Unmatched rows are NaN.

    `us=None` loads `understat.parquet`; when that file is missing a WARNING
    is printed and the columns are added as all-NaN.
    """
    out = seasons.drop(columns=[c for c in AUGMENT_COLUMNS if c in seasons.columns])
    if us is None:
        us = load()
    if us is None or us.empty or out.empty:
        for column in AUGMENT_COLUMNS:
            out[column] = np.nan
        return out

    mapping = match_to_fbref(out, us)
    rates = mapping[["league", "season", "player", "us_id"]].merge(
        us[["league", "season", "us_id", "minutes", *AUGMENT_COLUMNS]],
        on=["league", "season", "us_id"], how="left",
    )
    us_minutes = rates["minutes"].astype(float).replace(0, np.nan)
    for column in AUGMENT_COLUMNS:
        rates[column] = rates[column] / us_minutes * 90.0
    rates = rates.drop(columns=["us_id", "minutes"])

    index = out.index
    merged = out.reset_index(drop=True).merge(
        rates, on=["league", "season", "player"], how="left", validate="many_to_one",
    )
    row_minutes = merged["minutes"].astype(float)
    for column in AUGMENT_COLUMNS:
        merged[column] = merged[column] * row_minutes / 90.0
    merged.index = index

    if verbose:
        report = match_rates(out, mapping)
        for row in report.itertuples(index=False):
            print(f"understat match {row.league} {row.season}: "
                  f"{row.matched}/{row.eligible} ({row.rate:.1%}), "
                  f"{row.ambiguous} ambiguous namesakes")
        if len(report):
            total = report["matched"].sum() / report["eligible"].sum()
            print(f"understat match overall: {report['matched'].sum()}/"
                  f"{report['eligible'].sum()} ({total:.1%}), "
                  f"{report['ambiguous'].sum()} ambiguous namesakes")
    return merged


# --------------------------------------------------------------------------
# Team-level stats (club style profile)
# --------------------------------------------------------------------------

TEAM_COLUMNS = ["league", "season", "us_team", "matches", "team_npxg",
                "team_deep", "team_press", "team_xpts"]


def parse_teams(json_text: str, league: str, season: str) -> pd.DataFrame:
    """One row per Understat team: per-match npxG, deep completions and
    expected points, plus pressing intensity = opponent passes faced per
    defensive action, inverted (1 / PPDA) so that higher = presses more."""
    payload = json.loads(json_text)
    rows = []
    for team in (payload.get("teams") or {}).values():
        history = team.get("history") or []
        if not history:
            continue
        ppda_att = sum(_num((m.get("ppda") or {}).get("att")) for m in history)
        ppda_def = sum(_num((m.get("ppda") or {}).get("def")) for m in history)
        n = len(history)
        rows.append({
            "league": league, "season": season,
            "us_team": html.unescape(str(team.get("title", ""))).strip(),
            "matches": n,
            "team_npxg": sum(_num(m.get("npxG")) for m in history) / n,
            "team_deep": sum(_num(m.get("deep")) for m in history) / n,
            "team_press": ppda_def / ppda_att if ppda_att else np.nan,
            "team_xpts": sum(_num(m.get("xpts")) for m in history) / n,
        })
    return pd.DataFrame(rows, columns=TEAM_COLUMNS)


def team_frame(leagues=None, seasons=None) -> pd.DataFrame:
    """Team stats for every league-season, read from the same disk cache
    `pull_all` fills (a missing league-season is fetched once)."""
    leagues = list(leagues if leagues is not None else config.CANDIDATE_LEAGUES)
    seasons = list(seasons if seasons is not None else config.SEASONS)
    frames = []
    for league in leagues:
        for season in seasons:
            try:
                frames.append(parse_teams(fetch_league_season(league, season), league, season))
            except Exception as exc:  # report and move on, as pull_all does
                print(f"FAILED understat teams {league} {season}: {exc!r}")
    return (pd.concat(frames, ignore_index=True) if frames
            else pd.DataFrame(columns=TEAM_COLUMNS))


def team_name_map(fbref_seasons: pd.DataFrame, us: pd.DataFrame) -> pd.DataFrame:
    """(league, season, us_team) -> FBref `team`, learned from the players.

    The two sources name clubs differently ("AC Milan" / "Milan"), but their
    players are already matched one to one. Each Understat team takes the
    FBref team holding most of its matched players' minutes. Understat rows
    for mid-season movers ("Bologna,Torino") are skipped as voters.
    """
    columns = ["league", "season", "us_team", "team"]
    mapping = match_to_fbref(fbref_seasons, us)
    if mapping.empty:
        return pd.DataFrame(columns=columns)
    single = us[~us["team"].astype(str).str.contains(",")]
    votes = (mapping[["league", "season", "player", "us_id"]]
             .merge(single[["league", "season", "us_id", "team"]]
                    .rename(columns={"team": "us_team"}),
                    on=["league", "season", "us_id"])
             .merge(fbref_seasons[["league", "season", "player", "team", "minutes"]],
                    on=["league", "season", "player"]))
    weight = (votes.groupby(["league", "season", "us_team", "team"])["minutes"].sum()
              .reset_index().sort_values("minutes", ascending=False))
    return weight.drop_duplicates(["league", "season", "us_team"])[columns].reset_index(drop=True)
