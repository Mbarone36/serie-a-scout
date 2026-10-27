import numpy as np
import pandas as pd

from src import transfers

_ROMANCE = ["ITA", "ESP", "ARG", "BRA", "FRA", "POR", "ROU", "URU", "COL", "CHI",
            "MEX", "PER", "ECU", "PAR", "VEN", "BEL", "SUI", "MDA", "CPV", "ANG"]
_GERMANIC = ["GER", "NED", "ENG", "SCO", "WAL", "IRL", "DEN", "SWE", "NOR", "ISL",
             "AUT", "USA", "CAN", "AUS", "NZL", "RSA"]
_SLAVIC = ["POL", "SRB", "CRO", "SVN", "SVK", "CZE", "RUS", "UKR", "BUL", "BIH",
           "MKD", "MNE", "BLR"]

LANGUAGE_FAMILY: dict[str, str] = (
    {n: "romance" for n in _ROMANCE}
    | {n: "germanic" for n in _GERMANIC}
    | {n: "slavic" for n in _SLAVIC}
)

BUNDLES = {
    "adaptation": ["first_move_abroad", "n_countries", "language_romance",
                   "compatriots_in_squad", "prior_seasons_on_record"],
    "trajectory": ["clubs_per_year", "bounced_between_clubs", "loan_count",
                   "lost_and_regained_place", "career_minutes"],
    "injury": ["injury_days_prior", "injury_spells_prior", "injury_longest_absence"],
    "availability": ["appearance_share", "fixtures_missed_total", "fixtures_missed_while_fit"],
    "value": ["value_at_move", "value_trend_2y"],
}


def _prior(seasons: pd.DataFrame, player: str, arrival_season: str) -> pd.DataFrame:
    """Every recorded season strictly before the move. The leakage boundary.

    `season` is coerced to string before slicing so an int64 season column
    read back from parquet (e.g. 2122) does not crash `.str[:2]`.
    """
    cutoff = int(arrival_season[:2])
    rows = seasons[seasons["player"] == player]
    return rows[rows["season"].astype(str).str[:2].astype(int) < cutoff]


def build_adaptation(arrivals, seasons, squad_nationalities) -> pd.DataFrame:
    """Adaptation-context bundle.

    `compatriots_in_squad` subtracts the arriving player himself: the squad-
    nationality table is built from the same season rows that contain the
    arriving player, so his own row is counted in `n` and must be removed
    (floored at 0) to avoid inflating the count by one.
    """
    rows = []
    for _, arrival in arrivals.iterrows():
        history = _prior(seasons, arrival["player"], arrival["arrival_season"])
        countries = history["league"].str[:3].unique()
        nat = arrival.get("nationality", "")
        compatriots = squad_nationalities[
            (squad_nationalities["team"] == arrival["to_team"])
            & (squad_nationalities["season"] == arrival["arrival_season"])
            & (squad_nationalities["nationality"] == nat)
        ]["n"].sum()
        rows.append({
            "player": arrival["player"],
            "arrival_season": arrival["arrival_season"],
            "first_move_abroad": float(len(countries) <= 1),
            "n_countries": float(max(len(countries), 1)),
            "language_romance": float(LANGUAGE_FAMILY.get(nat, "other") == "romance"),
            "compatriots_in_squad": float(max(compatriots - 1, 0)),
            "prior_seasons_on_record": float(len(history)),
        })
    return pd.DataFrame(rows)


def build_trajectory(arrivals, seasons) -> pd.DataFrame:
    """Career-trajectory bundle.

    `bounced_between_clubs` flags a player with at least one loan-shaped
    round trip (`loans > 0`) who has also played for more than two distinct
    clubs prior to the move (`len(set(clubs)) > 2`) — it is not a claim
    about competition levels, only about club-hopping.
    """
    rows = []
    for _, arrival in arrivals.iterrows():
        history = _prior(seasons, arrival["player"], arrival["arrival_season"]).sort_values("season")
        clubs = history["team"].tolist()
        n_seasons = max(len(history), 1)
        loans = sum(
            1 for i in range(1, len(clubs) - 1)
            if clubs[i] != clubs[i - 1] and clubs[i + 1] == clubs[i - 1]
        )
        minutes = history["minutes"].tolist()
        regained = 0.0
        for i in range(1, len(minutes) - 1):
            if minutes[i] < 0.6 * minutes[i - 1] and minutes[i + 1] > minutes[i - 1]:
                regained = 1.0
        rows.append({
            "player": arrival["player"],
            "arrival_season": arrival["arrival_season"],
            "clubs_per_year": len(set(clubs)) / n_seasons,
            "bounced_between_clubs": float(loans > 0 and len(set(clubs)) > 2),
            "loan_count": float(loans),
            "lost_and_regained_place": regained,
            "career_minutes": float(history["minutes"].sum()),
        })
    return pd.DataFrame(rows)


def _players_covered_by(injuries: pd.DataFrame) -> set:
    """Players the injury source has ANY row for, in any season.

    The distinction this set draws is the whole of finding #4. "Transfermarkt
    lists this player and records no absences before the move" is a measurement
    of durability worth zero days. "Transfermarkt has never heard of this
    player" is missing data. Both used to come out as 0.0, so an unscraped
    player was indistinguishable from an ever-fit one — and the spec's own
    Known Limitations section says the opposite in as many words: "Absence of
    injury data is not evidence of durability."

    Absent players get NaN, which the pipeline's median fill then handles as
    missing data should be handled. An entirely empty injuries frame therefore
    yields NaN for every player, which is correct: nothing was measured.
    """
    if injuries.empty or "player" not in injuries.columns:
        return set()
    return set(injuries["player"])


def build_injury(arrivals, injuries) -> pd.DataFrame:
    """Injury-durability bundle, truncated at the transfer date.

    NaN (not 0.0) for a player the injury source has no rows for at all — see
    `_players_covered_by`.
    """
    covered = _players_covered_by(injuries)
    rows = []
    for _, arrival in arrivals.iterrows():
        if arrival["player"] not in covered:
            rows.append({
                "player": arrival["player"],
                "arrival_season": arrival["arrival_season"],
                "injury_days_prior": np.nan,
                "injury_spells_prior": np.nan,
                "injury_longest_absence": np.nan,
            })
            continue
        prior = _prior(injuries, arrival["player"], arrival["arrival_season"])
        rows.append({
            "player": arrival["player"],
            "arrival_season": arrival["arrival_season"],
            "injury_days_prior": float(prior["days_missed"].sum()),
            "injury_spells_prior": float(len(prior)),
            "injury_longest_absence": float(prior["days_missed"].max()) if len(prior) else 0.0,
        })
    return pd.DataFrame(rows)


def build_availability(arrivals, seasons, injuries) -> pd.DataFrame:
    """Availability bundle.

    `fixtures_missed_total` is `available - played`, a plain total across
    all prior seasons — it is not a longest-streak/run statistic.

    `.sum()` on `playing_time_mp` and `team_matches` relies on pandas'
    default `skipna=True` behaviour: a NaN in either column (e.g. a season
    row with an unknown `team_matches`) is skipped rather than propagating
    and zeroing the feature.

    `fixtures_missed_while_fit` is the cross of two sources and is only
    meaningful when both are present. For a player the injury source has no
    rows for at all it is NaN, not a number (finding #4): without injury
    records every missed fixture would be attributed to being out of favour,
    which is exactly the fragile-vs-frozen-out confusion this feature exists to
    resolve. `fixtures_missed_total` comes from FBref alone and is unaffected.
    """
    covered = _players_covered_by(injuries)
    rows = []
    for _, arrival in arrivals.iterrows():
        history = _prior(seasons, arrival["player"], arrival["arrival_season"])
        played = history["playing_time_mp"].sum() if "playing_time_mp" in history else 0.0
        available = history["team_matches"].sum() if "team_matches" in history else 0.0
        missed_total = float(available - played)
        if arrival["player"] in covered:
            injury_rows = _prior(injuries, arrival["player"], arrival["arrival_season"])
            missed_injured = float(injury_rows["games_missed"].sum())
            while_fit = max(missed_total - missed_injured, 0.0)
        else:
            while_fit = np.nan
        rows.append({
            "player": arrival["player"],
            "arrival_season": arrival["arrival_season"],
            "appearance_share": float(played / available) if available else np.nan,
            "fixtures_missed_total": missed_total,
            "fixtures_missed_while_fit": while_fit,
        })
    return pd.DataFrame(rows)


def build_value(arrivals, values) -> pd.DataFrame:
    """Value as of the move, plus two-year growth ratio.

    Cutoff is 1 July of the season-start year -- the summer window. For
    winter-window arrivals this excludes Aug-Dec valuations that precede
    the move (stale, not leaky). Replace with the real transfer date when
    available.
    """
    rows = []
    for _, arrival in arrivals.iterrows():
        cutoff = pd.Timestamp(year=2000 + int(arrival["arrival_season"][:2]), month=7, day=1)
        history = values[(values["player"] == arrival["player"]) & (values["date"] < cutoff)]
        history = history.sort_values("date")
        if history.empty:
            rows.append({
                "player": arrival["player"], "arrival_season": arrival["arrival_season"],
                "value_at_move": np.nan, "value_trend_2y": np.nan,
            })
            continue
        latest = history.iloc[-1]["value_eur"]
        two_years_ago = history[history["date"] >= cutoff - pd.Timedelta(days=730)]
        baseline = two_years_ago.iloc[0]["value_eur"] if len(two_years_ago) else latest
        rows.append({
            "player": arrival["player"],
            "arrival_season": arrival["arrival_season"],
            "value_at_move": float(latest),
            "value_trend_2y": float(latest / baseline) if baseline else 1.0,
        })
    return pd.DataFrame(rows)
