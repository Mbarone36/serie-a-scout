import numpy as np
import pandas as pd

from src import normalize

# Every dimension is sourced from data published for EVERY season in scope,
# so a 2025-26 club is profiled exactly like a 2017-18 one. (FBref's live
# team pages dropped the passing/defense tables from 2022-23 on, which left
# the previous FBref-only profile neutral-filled on 6 of 7 dimensions.)
STYLE_COLUMNS = (
    "possession",    # share of the ball (FBref team table)
    "team_passes",   # passing volume per match (API-Football, summed over players)
    "team_deep",     # completions near the opponent's box per match (Understat)
    "team_crs",      # crosses per match (FBref misc, summed over players)
    "team_press",    # pressing intensity, 1 / PPDA (Understat)
    "team_npxg",     # attacking quality, npxG per match (Understat)
)

VECTOR_COLUMNS = [f"{c}_pct" for c in STYLE_COLUMNS]

# Player-row columns summed into a per-match team rate: (source, output).
_PLAYER_SUMS = {"af_passes": "team_passes", "crs": "team_crs"}
_UNDERSTAT_TEAM = ["team_npxg", "team_deep", "team_press", "team_xpts"]


def _team_rates_from_players(seasons: pd.DataFrame) -> pd.DataFrame:
    """Per (league, team, season), each summed stat per match: total over
    the rows that report it / their minutes * 990 (11 players x 90'), so a
    few unmatched players lower coverage rather than the rate itself."""
    out = None
    for source, target in _PLAYER_SUMS.items():
        if source not in seasons.columns:
            continue
        rows = seasons[seasons[source].notna() & (seasons["minutes"] > 0)]
        rate = (rows.groupby(["league", "team", "season"])[[source, "minutes"]].sum()
                .assign(**{target: lambda g, s=source: g[s] / g["minutes"] * 990.0})
                [[target]])
        out = rate if out is None else out.join(rate, how="outer")
    if out is None:
        return pd.DataFrame(columns=["league", "team", "season"])
    return out.reset_index()


def add_profile_columns(team_stats: pd.DataFrame, seasons: pd.DataFrame,
                        us_teams: pd.DataFrame | None = None,
                        name_map: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach the STYLE_COLUMNS not in FBref's team table (plus `team_xpts`)
    to `team_stats`, keyed by FBref team name and season.

    `us_teams` is `understat.team_frame()`, `name_map` is
    `understat.team_name_map()`; either missing leaves those columns NaN
    (neutral-filled by `build_style_vectors`).
    """
    added = [c for c in STYLE_COLUMNS if c != "possession"] + ["team_xpts"]
    out = team_stats.drop(columns=[c for c in added if c in team_stats.columns])
    keys = ["league", "team", "season"]

    out = out.merge(_team_rates_from_players(seasons), on=keys, how="left")
    if us_teams is not None and name_map is not None and not us_teams.empty:
        named = (us_teams.merge(name_map, on=["league", "season", "us_team"])
                 [keys + _UNDERSTAT_TEAM].drop_duplicates(keys))
        out = out.merge(named, on=keys, how="left")
    for column in added:
        if column not in out.columns:
            out[column] = np.nan
    return out


def build_style_vectors(team_stats: pd.DataFrame) -> pd.DataFrame:
    """Percentile-rank each club against its own league-season."""
    frame = team_stats.copy()
    for column in STYLE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame["position_group"] = "team"
    ranked = normalize.percentile_within(frame, list(STYLE_COLUMNS))
    ranked[VECTOR_COLUMNS] = ranked[VECTOR_COLUMNS].fillna(0.5)
    return ranked[["team", "season", *VECTOR_COLUMNS]].reset_index(drop=True)


def vector_for(styles: pd.DataFrame, team: str, season: str) -> np.ndarray:
    match = styles[(styles["team"] == team) & (styles["season"] == season)]
    if match.empty:
        raise KeyError(f"no style vector for {team} in {season}")
    return match.iloc[0][VECTOR_COLUMNS].to_numpy(dtype=np.float32)
