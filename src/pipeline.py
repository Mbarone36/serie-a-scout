"""End-to-end dataset assembly: one row per (player, arrival, pre-move season).

`build()` is the only entry point that touches disk. Every other function
here is pure and network-free so the test suite can exercise the join logic
directly.
"""

import pandas as pd

from src import (clubs, config, features_offfield, features_onfield, labels,
                  fbref, transfers, understat, api_football)

PLAYER_FEATURE_COLUMNS = features_onfield.FEATURE_COLUMNS + [
    column for columns in features_offfield.BUNDLES.values() for column in columns
]
CLUB_FEATURE_COLUMNS = clubs.VECTOR_COLUMNS

_ARRIVAL_META = ["arrival_season", "to_team", "from_team", "from_league"]

# AMENDMENT: seasons "2324"+ come from a scraper that cannot reach FBref's
# passing/possession/defense tables, so several on-field features on those
# rows are neutral 0.5 fills rather than measurements. This must be reported
# alongside the row/player counts, not silently absorbed into the fill.
MIXED_ERA_CUTOFF = "2324"

# Columns Transfermarkt parquets must carry even when the bulk pull hasn't
# run yet and the files don't exist on disk. Keeping these here (rather than
# inline in `build()`) is what makes `_load_tm` unit-testable in isolation.
_TM_SCHEMAS = {
    "injuries.parquet": (["player", "season", "injury", "days_missed", "games_missed"], "injury"),
    "market_values.parquet": (["player", "date", "value_eur"], "value"),
    "club_values.parquet": (["team", "squad_value_eur"], "retention"),
    "fees.parquet": (["player", "season", "from_club", "to_club", "fee_eur", "is_loan"], "retention"),
}


def expand_per_season(arrivals: pd.DataFrame, seasons: pd.DataFrame) -> pd.DataFrame:
    """One row per (player, arrival, pre-move season) -- guaranteed unique.

    This is the spec's 'expanded per-player' cohort. Rows within a player are
    correlated, which is why every split is grouped by player.

    CONTROLLER RULING (fix round 1): a pre-move season can be split across
    two clubs in the source data (e.g. a mid-season move within the same
    fbref-labelled season -- Ounas 2021: 308 min at Cagliari, 1199 min at
    Crotone). Left alone this yields two rows sharing the same
    (player, arrival_season, season) key. We keep only the club-row with the
    MOST minutes in that season -- its per-90 percentiles rest on the larger
    sample -- and record how many clubs shared the season in `season_clubs`
    (1 normally, 2+ for a split) so the information isn't silently lost.
    Ties on minutes break deterministically on team name (ascending).
    """
    rows = []
    for _, arrival in arrivals.iterrows():
        cutoff = int(arrival["arrival_season"][:2])
        history = seasons[
            (seasons["player"] == arrival["player"])
            & (seasons["season"].astype(str).str[:2].astype(int) < cutoff)
        ]
        for season, group in history.groupby("season", sort=False):
            group = group.sort_values(
                ["minutes", "team"], ascending=[False, True], kind="stable"
            )
            season_row = group.iloc[0]
            rows.append({**season_row.to_dict(),
                         **{k: arrival[k] for k in _ARRIVAL_META},
                         "season_clubs": int(len(group))})
    if not rows:
        return pd.DataFrame(columns=[*seasons.columns, *_ARRIVAL_META, "season_clubs"])
    return pd.DataFrame(rows).reset_index(drop=True)


def _assert_unique_keys(frame: pd.DataFrame, keys: list[str]) -> None:
    """Raise ValueError naming a sample offending key if `keys` isn't unique.

    `assemble` calls this right after `expand_per_season` as a guard, not a
    fix: `expand_per_season` is the one place that is supposed to enforce
    "one row per (player, arrival_season, season)". If a duplicate still
    reaches `assemble`, that's a bug upstream (or a caller bypassing
    `expand_per_season` entirely) and joining the bundles on to it would
    silently fan out rows, so we fail loudly instead.
    """
    dup_mask = frame.duplicated(subset=keys, keep=False)
    if dup_mask.any():
        sample = frame.loc[dup_mask, keys].iloc[0].to_dict()
        raise ValueError(
            f"duplicate key in expanded frame at {keys}: {sample!r} "
            f"({int(dup_mask.sum())} rows affected)"
        )


def _load_tm(name: str, columns: list[str], bundle: str | None = None) -> pd.DataFrame:
    """Read a Transfermarkt processed parquet, tolerating its absence.

    The bulk Transfermarkt pull runs later in the plan; until then these
    files may not exist. Rather than crash, `build()` substitutes an empty
    frame with the right columns and prints a loud warning naming which
    downstream bundle will fall back to median/neutral fills as a result.
    """
    if bundle is None:
        bundle = _TM_SCHEMAS.get(name, (None, "affected"))[1]
    path = config.PROCESSED_DIR / name
    if not path.exists():
        print(f"WARNING: {name} not found — {bundle} features will be median-filled")
        return pd.DataFrame(columns=columns)
    return pd.read_parquet(path)


def mixed_era_report(frame: pd.DataFrame) -> dict:
    """Count/share of rows from the post-2324 scrape, and which on-field
    feature columns were entirely absent (not merely sparse) for them.

    Must run on a frame that still carries the raw on-field counting/rate
    columns un-filled (i.e. before `assemble`'s neutral-fill step erases the
    signal that a column was ever absent) -- both `expand_per_season`'s
    output and `assemble`'s final frame qualify, since neither fillna touches
    those raw columns, only their derived `_pct`/`_p90_pct` counterparts.
    """
    total = len(frame)
    if total == 0 or "season" not in frame.columns:
        return {"count": 0, "total": total, "share": 0.0, "absent_columns": []}

    mask = frame["season"].astype(str) >= MIXED_ERA_CUTOFF
    count = int(mask.sum())
    share = count / total if total else 0.0

    absent: list[str] = []
    if count:
        subset = frame[mask]
        for column in features_onfield.COUNTING_COLUMNS:
            if column not in subset.columns or subset[column].isna().all():
                absent.append(f"{column}_p90_pct")
        for column in features_onfield.RATE_COLUMNS:
            if column not in subset.columns or subset[column].isna().all():
                absent.append(f"{column}_pct")

    return {"count": count, "total": total, "share": share, "absent_columns": absent}


def labelable_arrivals(arrivals: pd.DataFrame, seasons: pd.DataFrame) -> pd.DataFrame:
    """Arrivals whose full label window is already observable in `seasons`.

    `labels.build` scores a player's outcome over
    `labels.window_seasons(arrival_season)` -- the arrival season plus the
    next `config.LABEL_WINDOW_SEASONS - 1` seasons. When the LAST window
    season has no Serie A rows loaded yet, there is no end-of-window row to
    observe: not because the player actually left with no trace, but because
    that season of data simply hasn't arrived yet. `labels.build` cannot
    tell the two apart -- it scores the former the same as the latter, via
    `unobserved_exit_target` (0.0 retention) plus an `output` ranked on a
    truncated window. That is a genuine "not yet knowable" case and belongs
    in the unlabelled prediction set, not in training rows scored as if the
    player had vanished.

    Keeps only arrivals where EVERY season in the window is not just present
    but COMPLETE in `seasons` -- `fbref.season_is_complete` on that season's
    Serie A rows (see its docstring: a season present but exported/scraped
    mid-year, like the historical dump's partial "2223" matchday-13 cutoff,
    is no more observable than one that hasn't arrived at all -- a window
    scored on a third of a season is not a real outcome). Pure and
    network-free so it's directly unit-testable.
    """
    if arrivals.empty:
        return arrivals
    serie_a = seasons[seasons["league"] == config.SERIE_A]
    complete_seasons = {
        season: fbref.season_is_complete(group)
        for season, group in serie_a.groupby("season", sort=False)
    }
    mask = arrivals["arrival_season"].apply(
        lambda arrival_season: all(
            complete_seasons.get(window_season, False)
            for window_season in labels.window_seasons(arrival_season)
        )
    )
    return arrivals[mask].reset_index(drop=True)


def assemble(arrivals, seasons, team_stats, squad_nationalities,
             injuries, values, club_values, fees) -> pd.DataFrame:
    total_arrivals = len(arrivals)
    arrivals = labelable_arrivals(arrivals, seasons)
    excluded = total_arrivals - len(arrivals)
    latest_labelable = arrivals["arrival_season"].max() if len(arrivals) else None
    print(f"{excluded} arrival(s) excluded: label window not yet observable "
          f"(latest labelable arrival_season: {latest_labelable})")

    on_field = features_onfield.build(seasons)
    expanded = expand_per_season(arrivals, on_field)
    if expanded.empty:
        return expanded

    _assert_unique_keys(expanded, ["player", "arrival_season", "season"])

    # `labels.build` outputs a `minutes` TARGET column (post-move Serie A
    # minutes share). `expand_per_season` already carries a `minutes` column
    # of its own -- the raw minutes played in that pre-move season row. Left
    # alone the merge below would silently suffix both to `minutes_x`/
    # `minutes_y`, and the final `dropna(subset=config.TARGETS)` would raise
    # a KeyError (or worse, silently keep NaN targets) because "minutes" no
    # longer exists. Rename the raw per-season figure out of the way first.
    if "minutes" in expanded.columns:
        expanded = expanded.rename(columns={"minutes": "season_minutes"})

    key = ["player", "arrival_season"]
    bundles = [
        features_offfield.build_adaptation(arrivals, seasons, squad_nationalities),
        features_offfield.build_trajectory(arrivals, seasons),
        features_offfield.build_injury(arrivals, injuries),
        features_offfield.build_availability(arrivals, seasons, injuries),
        features_offfield.build_value(arrivals, values),
        labels.build(arrivals, seasons, club_values, fees),
    ]
    for bundle in bundles:
        expanded = expanded.merge(bundle, on=key, how="left")

    styles = clubs.build_style_vectors(team_stats)
    expanded = expanded.merge(
        styles.rename(columns={"team": "to_team", "season": "arrival_season"}),
        on=["to_team", "arrival_season"], how="left",
    )

    # Median-fill first (uses real signal wherever any exists), then a
    # neutral 0.5 fallback for a column with ZERO signal across the whole
    # frame (e.g. value_at_move/value_trend_2y today, since
    # market_values.parquet doesn't exist yet -- its own median is NaN, and
    # fillna(NaN) is a no-op, which would otherwise leave the column 100%
    # NaN all the way into training).
    expanded[PLAYER_FEATURE_COLUMNS] = (
        expanded[PLAYER_FEATURE_COLUMNS]
        .fillna(expanded[PLAYER_FEATURE_COLUMNS].median())
        .fillna(0.5)
    )
    expanded[CLUB_FEATURE_COLUMNS] = expanded[CLUB_FEATURE_COLUMNS].fillna(0.5)
    return expanded.dropna(subset=list(config.TARGETS)).reset_index(drop=True)


def build_candidates(season: str) -> pd.DataFrame:
    """Every player with a real `season`, scored as a hypothetical arrival.

    `model_frame.parquet` only ever covers players who actually transferred
    into Serie A. The CLI needs to score ANY player with a recent season --
    so this builds the same feature vector for every player-season with
    enough minutes, treating it as a hypothetical arrival in
    `labels.next_season(season)`. There is no real destination club yet
    (`to_team` is left blank), so there is no club vector and no targets --
    the caller supplies whichever club/preset vector it wants to score
    against.

    Eligibility and the synthetic arrival's `from_team`/`from_league`/
    `nationality` mirror `transfers.detect_arrivals`: minutes are summed
    across any clubs a player split `season` between, and the club with the
    most minutes that season is the attributed source club.

    Refuses (`ValueError`) a `season` whose Serie A player frame is not yet
    complete (see `fbref.season_is_complete`) -- a partial pre-move season,
    like the historical dump's matchday-13 "2223" cutoff, would make every
    candidate row it produces rest on a fraction of a real season.
    """
    serie_a_season = fbref.player_season_stats(config.SERIE_A, season)
    if not fbref.season_is_complete(serie_a_season):
        raise ValueError(
            f"build_candidates: {config.SERIE_A} season {season!r} is not "
            f"complete (max team matches below {config.MIN_COMPLETE_SEASON_MATCHES}); "
            "refusing to build candidate rows off a partial pre-move season"
        )
    arrival_season = labels.next_season(season)
    output_columns = ["player", "team", "league", "season", "age",
                       "position_group", "nationality", *PLAYER_FEATURE_COLUMNS]

    season_rows = pd.concat(
        [fbref.player_season_stats(league, season) for league in config.CANDIDATE_LEAGUES],
        ignore_index=True,
    )

    def _empty_result() -> pd.DataFrame:
        result = pd.DataFrame(columns=output_columns)
        result.to_parquet(config.PROCESSED_DIR / "candidates.parquet")
        print(f"{len(result)} rows, 0 players with no TM data")
        return result

    if season_rows.empty:
        return _empty_result()

    totals = season_rows.groupby("player")["minutes"].sum()
    eligible = totals[totals >= config.MIN_MINUTES_PRIOR_SEASON].index
    eligible_rows = season_rows[season_rows["player"].isin(eligible)]
    if eligible_rows.empty:
        return _empty_result()

    # The most-played club that season attributes from_team/from_league/
    # nationality; `prior_minutes` is the SUM across every club that season
    # (a player split between two clubs still clears the bar on his total).
    sources = (
        eligible_rows.sort_values("minutes", ascending=False)
        .drop_duplicates(subset=["player"], keep="first")
        .set_index("player")
    )
    candidates = pd.DataFrame({
        "player": sources.index,
        "arrival_season": arrival_season,
        "to_team": "",
        "from_team": sources["team"].values,
        "from_league": sources["league"].values,
        "prior_minutes": sources.index.map(totals).values,
        "nationality": sources["nationality"].values if "nationality" in sources.columns else "",
        "arrival_clubs": 1,
    }).reset_index(drop=True)

    # Full history up to and including `season`, across the candidate
    # leagues -- not just `season` itself -- so the on-field trend and
    # trajectory bundles have prior seasons to compute a trend over.
    cutoff_prefix = int(season[:2])
    history_seasons = [s for s in config.SEASONS if int(s[:2]) <= cutoff_prefix]
    seasons = pd.concat(
        [fbref.player_season_stats(league, s)
         for league in config.CANDIDATE_LEAGUES for s in history_seasons],
        ignore_index=True,
    )
    if seasons.empty:
        return _empty_result()
    # Understat xG (all seasons) -- see `understat.augment`; NaN when absent.
    seasons = understat.augment(seasons)
    # API-Football defensive/dribbling/duel stats (all seasons) -- see
    # `api_football.augment`; NaN when absent.
    seasons = api_football.augment(seasons)

    on_field = features_onfield.build(seasons)
    # Reuses expand_per_season purely for its per-season split-club dedup
    # (keep the higher-minutes club, record season_clubs) -- then narrows
    # the usual multi-season history down to just the requested `season`.
    expanded = expand_per_season(candidates, on_field)
    if expanded.empty:
        return _empty_result()
    _assert_unique_keys(expanded, ["player", "arrival_season", "season"])
    expanded = expanded[expanded["season"] == season].reset_index(drop=True)
    if expanded.empty:
        return _empty_result()

    # No destination club yet, so no squad to count compatriots against.
    squad_nationalities = pd.DataFrame(columns=["team", "season", "nationality", "n"])
    injuries = _load_tm("injuries.parquet", *_TM_SCHEMAS["injuries.parquet"])
    values = _load_tm("market_values.parquet", *_TM_SCHEMAS["market_values.parquet"])

    key = ["player", "arrival_season"]
    bundles = [
        features_offfield.build_adaptation(candidates, seasons, squad_nationalities),
        features_offfield.build_trajectory(candidates, seasons),
        features_offfield.build_injury(candidates, injuries),
        features_offfield.build_availability(candidates, seasons, injuries),
        features_offfield.build_value(candidates, values),
    ]
    for bundle in bundles:
        expanded = expanded.merge(bundle, on=key, how="left")

    # See `assemble` for why a plain median fillna isn't enough on its own.
    expanded[PLAYER_FEATURE_COLUMNS] = (
        expanded[PLAYER_FEATURE_COLUMNS]
        .fillna(expanded[PLAYER_FEATURE_COLUMNS].median())
        .fillna(0.5)
    )

    tm_players = set()
    if not injuries.empty:
        tm_players |= set(injuries["player"])
    if not values.empty:
        tm_players |= set(values["player"])
    no_tm_count = int((~expanded["player"].isin(tm_players)).sum())

    result = expanded[output_columns].reset_index(drop=True)
    result.to_parquet(config.PROCESSED_DIR / "candidates.parquet")
    print(f"{len(result)} rows, {no_tm_count} players with no TM data")
    return result


def build() -> pd.DataFrame:
    seasons = pd.concat(
        [fbref.player_season_stats(league, season)
         for league in config.CANDIDATE_LEAGUES for season in config.SEASONS],
        ignore_index=True,
    )
    # Understat xG (all seasons) -- see `understat.augment`; NaN when absent.
    seasons = understat.augment(seasons)
    # API-Football defensive/dribbling/duel stats (all seasons) -- see
    # `api_football.augment`; NaN when absent.
    seasons = api_football.augment(seasons)
    team_stats = pd.concat(
        [fbref.team_season_stats(config.SERIE_A, season) for season in config.SEASONS],
        ignore_index=True,
    )
    # Club style profile from all-season sources -- see `clubs.STYLE_COLUMNS`.
    us_players = understat.load()
    name_map = (understat.team_name_map(seasons, us_players)
                if us_players is not None else None)
    team_stats = clubs.add_profile_columns(
        team_stats, seasons, understat.team_frame([config.SERIE_A]), name_map)
    arrivals = transfers.detect_arrivals(seasons)

    squad_nationalities = (
        seasons.groupby(["team", "season", "nationality"]).size()
        .reset_index(name="n")
    )
    injuries = _load_tm("injuries.parquet", *_TM_SCHEMAS["injuries.parquet"])
    values = _load_tm("market_values.parquet", *_TM_SCHEMAS["market_values.parquet"])
    club_values = _load_tm("club_values.parquet", *_TM_SCHEMAS["club_values.parquet"])
    fees = _load_tm("fees.parquet", *_TM_SCHEMAS["fees.parquet"])

    frame = assemble(arrivals, seasons, team_stats, squad_nationalities,
                      injuries, values, club_values, fees)
    frame.to_parquet(config.PROCESSED_DIR / "model_frame.parquet")
    # query.py --club reads this to build a real club's style vector.
    team_stats.to_parquet(config.PROCESSED_DIR / "clubs.parquet")

    n_players = frame["player"].nunique() if "player" in frame.columns else 0
    print(f"{len(frame)} rows, {n_players} distinct players")

    report = mixed_era_report(frame)
    print(f"mixed-era rows (season >= '{MIXED_ERA_CUTOFF}'): {report['count']} "
          f"of {report['total']} ({report['share']:.1%})")
    if report["absent_columns"]:
        print("on-field feature columns entirely absent (0.5-filled) for "
              "mixed-era rows: " + ", ".join(report["absent_columns"]))

    return frame


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2 and sys.argv[1] == "candidates":
        if len(sys.argv) < 3:
            raise SystemExit("usage: python -m src.pipeline candidates <season>")
        build_candidates(sys.argv[2])
    else:
        build()
