import pandas as pd

from src import config, fbref

REQUIRED_ADVANCED_COLUMNS = (
    "expected_npxg",
    "progression_prgp",
    "possession_take_ons_succ",
)


def assess(frame: pd.DataFrame) -> dict:
    """Fraction of required advanced columns populated for a league-season."""
    if frame.empty:
        return {"coverage": 0.0, "n_players": 0, "missing_columns": list(REQUIRED_ADVANCED_COLUMNS)}
    missing = [c for c in REQUIRED_ADVANCED_COLUMNS if c not in frame.columns]
    present = [c for c in REQUIRED_ADVANCED_COLUMNS if c in frame.columns]
    filled = sum(frame[c].notna().mean() for c in present)
    coverage_score = filled / len(REQUIRED_ADVANCED_COLUMNS)
    return {
        "coverage": float(coverage_score),
        "n_players": int(len(frame)),
        "missing_columns": missing,
    }


def build_report(leagues=None, seasons=None) -> pd.DataFrame:
    leagues = leagues or config.CANDIDATE_LEAGUES
    seasons = seasons or config.SEASONS
    rows = []
    for league in leagues:
        for season in seasons:
            frame = fbref.player_season_stats(league, season)
            rows.append({"league": league, "season": season, **assess(frame)})
    return pd.DataFrame(rows)


def trusted_leagues(report: pd.DataFrame, min_coverage: float = 0.9) -> list[str]:
    """Leagues whose median season coverage clears the threshold."""
    if report.empty:
        return []
    medians = report.groupby("league")["coverage"].median()
    return sorted(medians[medians >= min_coverage].index.tolist())


if __name__ == "__main__":
    report = build_report()
    report.to_csv(config.PROCESSED_DIR / "coverage_report.csv", index=False)
    print(report.to_string(index=False))
    print("\ntrusted:", trusted_leagues(report))
