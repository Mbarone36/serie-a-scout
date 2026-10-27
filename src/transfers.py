import pandas as pd

from src import config

ARRIVAL_COLUMNS = [
    "player", "arrival_season", "to_team", "from_team", "from_league",
    "prior_minutes", "arrival_clubs", "nationality",
]


def previous_season(season: str) -> str:
    """'2223' -> '2122'."""
    start = int(season[:2])
    return f"{start - 1:02d}{start:02d}"


def _current_born(group: pd.DataFrame):
    """The player's `born` for this arrival-season group, or None if unknown.

    Used to disambiguate same-name players when matching against prior
    seasons. If the arrival-season rows carry no non-null `born` at all we
    fall back to name-only matching entirely.
    """
    if "born" not in group.columns:
        return None
    born_values = group["born"].dropna().unique()
    return born_values[0] if len(born_values) > 0 else None


def _prior_rows_for(all_seasons: pd.DataFrame, player: str, prior_season: str, born) -> pd.DataFrame:
    """Prior-season rows for `player`, filtered by `born` when both sides know it.

    Rule: require `born` equality whenever BOTH the current-season identity
    and the candidate prior row have a non-null `born`; otherwise fall back
    to name-only matching (so callers -- and the existing test suite -- that
    never supply a `born` column keep working unchanged).
    """
    candidates = all_seasons[
        (all_seasons["player"] == player) & (all_seasons["season"] == prior_season)
    ]
    if born is not None and pd.notna(born) and "born" in candidates.columns:
        candidates = candidates[candidates["born"].isna() | (candidates["born"] == born)]
    return candidates


def detect_arrivals(all_seasons: pd.DataFrame) -> pd.DataFrame:
    """Serie A arrivals, found by diffing squad membership season over season."""
    serie_a = all_seasons[all_seasons["league"] == config.SERIE_A]
    rows = []
    for (player, season), group in serie_a.groupby(["player", "season"], sort=False):
        prior = previous_season(season)
        born = _current_born(group)

        prior_rows = _prior_rows_for(all_seasons, player, prior, born)
        if prior_rows.empty:
            continue  # no pre-move season on record

        # A player can hold more than one Serie A club in his arrival season
        # (a mid-season move). The clubs he already had last season (any
        # league) are NOT arrivals and must not block the club(s) that
        # actually are new -- e.g. Vlahovic staying registered at Fiorentina
        # while moving mid-season to Juventus: Fiorentina must not swallow
        # the Juventus arrival just because it also appears this season.
        current_serie_a_clubs = set(group["team"].unique())
        prior_clubs = set(prior_rows["team"].unique())
        new_clubs = current_serie_a_clubs - prior_clubs
        if not new_clubs:
            continue  # already at every one of these clubs last season

        # Among the genuinely new clubs, the one with the most minutes is
        # the deterministic `to_team`; how many new clubs there are is
        # recorded as `arrival_clubs` (1 in the ordinary case).
        team_minutes = (
            group[group["team"].isin(new_clubs)]
            .groupby("team")["minutes"].sum()
            .sort_values(ascending=False)
        )
        to_team = team_minutes.index[0]
        arrival_clubs = int(team_minutes.shape[0])

        # Judge the 450-minute floor against the player's whole prior
        # season (he may have split it across multiple clubs), but keep
        # attributing from_team/from_league to wherever he played the most.
        prior_minutes_total = float(prior_rows["minutes"].sum())
        if prior_minutes_total < config.MIN_MINUTES_PRIOR_SEASON:
            continue
        source = prior_rows.sort_values("minutes", ascending=False).iloc[0]

        rows.append({
            "player": player,
            "arrival_season": season,
            "to_team": to_team,
            "from_team": source["team"],
            "from_league": source["league"],
            "prior_minutes": prior_minutes_total,
            "arrival_clubs": arrival_clubs,
            # Required by the adaptation bundle. Without it that whole bundle
            # silently evaluates to zero instead of failing loudly.
            "nationality": source.get("nationality", ""),
        })
    return pd.DataFrame(rows, columns=ARRIVAL_COLUMNS).drop_duplicates(
        subset=["player", "arrival_season"]
    ).reset_index(drop=True)


def derive_identities(combined: pd.DataFrame) -> pd.DataFrame:
    """Distinct (player, born, nationality) identities for Transfermarkt matching.

    Deliberately does NOT collapse by `player` name alone: the fbref dump
    contains multiple distinct real people sharing a name (e.g. two
    different "Danny Ward"s), each with their own `born`/`nationality`.
    Collapsing on name would silently conflate them into one identity.
    """
    return (
        combined[["player", "born", "nationality"]]
        .dropna(subset=["player"])
        .drop_duplicates(subset=["player", "born", "nationality"])
        .reset_index(drop=True)
    )


def build() -> pd.DataFrame:
    from src import fbref

    frames = [
        fbref.player_season_stats(league, season)
        for league in config.CANDIDATE_LEAGUES
        for season in config.SEASONS
    ]
    combined = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    arrivals = detect_arrivals(combined)
    arrivals.to_parquet(config.PROCESSED_DIR / "transfers.parquet")

    # Task 6b matches these against Transfermarkt.
    identities = derive_identities(combined)
    identities.to_parquet(config.PROCESSED_DIR / "fbref_identities.parquet")
    return arrivals


if __name__ == "__main__":
    print(build().head(30).to_string(index=False))
