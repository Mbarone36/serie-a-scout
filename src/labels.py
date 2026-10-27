import numpy as np
import pandas as pd

from src import config

MINUTES_PER_MATCH = 90.0
# Rank groups smaller than this get a neutral 0.5 percentile: a singleton would
# otherwise rank 1.0 (and a pair 0.5/1.0) against nobody meaningful.
MIN_RANK_GROUP = 5


def next_season(season: str) -> str:
    end = int(season[2:])
    return f"{end:02d}{end + 1:02d}"


def window_seasons(arrival_season: str) -> list[str]:
    seasons = [arrival_season]
    for _ in range(config.LABEL_WINDOW_SEASONS - 1):
        seasons.append(next_season(seasons[-1]))
    return seasons


def serie_a_minutes(seasons: pd.DataFrame, player: str, window: list[str]) -> float:
    rows = seasons[
        (seasons["player"] == player)
        & (seasons["season"].isin(window))
        & (seasons["league"] == config.SERIE_A)
    ]
    return float(rows["minutes"].sum())


def minutes_target(minutes_played: float, team_matches: float) -> float:
    available = team_matches * MINUTES_PER_MATCH
    if available <= 0:
        return 0.0
    return float(min(minutes_played / available, 1.0))


def _column_sum(frame: pd.DataFrame, column: str) -> float:
    """Sum a column, treating a missing column as contributing 0.0."""
    if column not in frame.columns:
        return 0.0
    return float(frame[column].sum())


def _basis_sources(basis: str) -> list[tuple[str, list[str]]]:
    """Ordered (source, columns) alternatives for `basis`.

    "xg" prefers Understat (`us_npxg + us_xa`, added by `understat.augment`
    and available for every season) and falls back to FBref's
    `expected_npxg + expected_xag` (dump era only). The other bases have a
    single source, recorded as "".
    """
    if basis == "xg":
        return [("understat", ["us_npxg", "us_xa"]),
                ("fbref", ["expected_npxg", "expected_xag"])]
    if basis == "chain":
        return [("", ["us_xgchain"])]
    return [("", ["performance_gls", "performance_ast"])]


# Bases measured relative to the player's team: his share of the attacking
# output his team produced while he was on the pitch.
TEAM_RELATIVE_BASES = {"xg", "chain"}


def team_attack_rates(seasons: pd.DataFrame) -> pd.Series:
    """Team npxG per 90 (i.e. per match), keyed by (league, team, season):
    the team's Understat npxG summed over its players' rows that carry it,
    divided by those rows' minutes * 1/11. A team with no xG rows is absent;
    `_team_rate_for` falls back to the league-season average for it."""
    if not {"us_npxg", "team", "league", "season"} <= set(seasons.columns):
        return pd.Series(dtype=float)
    rows = seasons[seasons["us_npxg"].notna() & (seasons["minutes"] > 0)]
    sums = rows.groupby(["league", "team", "season"])[["us_npxg", "minutes"]].sum()
    return (sums["us_npxg"] / sums["minutes"] * 990.0).rename("team_npxg_p90")


def _team_rate_for(played: pd.DataFrame, rates: pd.Series) -> np.ndarray:
    """Each window row's team attack rate; unknown teams get their
    league-season average, then the overall average, then 1.0 (no rates at
    all), so a missing team never drops a player's evidence."""
    if rates.empty or "team" not in played.columns:
        return np.ones(len(played))
    keys = list(zip(played["league"], played["team"], played["season"]))
    by_league = rates.groupby(level=["league", "season"]).mean()
    out = []
    for league, team, season in keys:
        value = rates.get((league, team, season))
        if value is None or not np.isfinite(value) or value <= 0:
            value = by_league.get((league, season), rates.mean())
        out.append(float(value))
    return np.array(out)


def _masked_rate(played: pd.DataFrame, columns: list[str],
                 team_rates: pd.Series | None = None) -> tuple[float, float, int] | None:
    cols = [c for c in columns if c in played.columns]
    if not cols:
        return None
    mask = played[cols].notna().any(axis=1)
    if not mask.any():
        return None
    subset = played.loc[mask]
    masked_minutes = float(subset["minutes"].sum())
    if masked_minutes <= 0:
        return None
    total = sum(_column_sum(subset, c) for c in cols)
    if team_rates is None:
        return total / masked_minutes * 90.0, masked_minutes, int(mask.sum())
    # Share of team output: the player's total over what his team produced
    # in the minutes he played (each spell at its own club's rate).
    team_output = float((subset["minutes"].to_numpy() / 90.0
                         * _team_rate_for(subset, team_rates)).sum())
    if team_output <= 0:
        return None
    return total / team_output, masked_minutes, int(mask.sum())


def _basis_rate(played: pd.DataFrame, basis: str,
                team_rates: pd.Series | None = None) -> tuple[float, float, int, str] | None:
    """Per-90 rate for `basis`, computed only from the window rows that carry it.

    Finding (commit d48ca6e review): the basis was chosen from whether its
    columns had ANY real value over the window, but the numerator (NaN-skipping
    sum) and the denominator (`minutes.sum()` over the WHOLE window) were not on
    the same rows. A 2022-23 arrival whose 2223 row has xG but whose 2324 row
    comes from the no-xG live scraper got a numerator from ONE season divided by
    minutes from TWO seasons -- silently halving his rate against peers whose xG
    covers the full window. A label built from a partial window is a genuine
    limitation (less evidence went into it), not a bias to hide by pretending
    the window is complete.

    Returns `(rate, masked_minutes, rows_used, source)` using only the rows
    where a `basis` column is non-NaN, or None when that basis has no usable
    evidence at all (columns absent/all-NaN, or the rows that do carry them
    logged zero minutes) so the caller can fall through to the next basis in
    the chain. When a basis has several sources (see `_basis_sources`) every
    source's masked minutes are computed and the source covering MORE of the
    window wins (the earlier-listed source, Understat, wins ties) -- so
    Understat covering half a window does not beat FBref covering all of it.
    Sources are never summed together, so a window is measured on one xG
    model only.
    """
    best = None
    relative = team_rates if basis in TEAM_RELATIVE_BASES else None
    for source, columns in _basis_sources(basis):
        result = _masked_rate(played, columns, relative)
        if result is None:
            continue
        if best is None or result[1] > best[1]:
            best = (*result, source)
    return best


def post_output_percentile(seasons: pd.DataFrame, arrivals: pd.DataFrame,
                           relative_to_team: bool = True) -> pd.DataFrame:
    """Position-adjusted Serie A production over the label window.

    Each arrival's production is measured on whichever basis its window rows
    actually support, ranked among peers on that SAME basis so scales never
    mix:

    - attackers: "xg" (npxG+xAG per 90) when either column has a real value
      over the window; otherwise "ga" (goals+assists per 90).
    - midfielders/defenders: "chain" (Understat xGChain per 90: the xG of
      every possession the player took part in, build-up included) when it
      has a real value; otherwise "ga".
    With `relative_to_team` (the `output_share` target), "xg" and "chain"
    are measured as a SHARE OF THE TEAM'S ATTACK: the
    player's total over his team's npxG in the minutes he played (see
    `team_attack_rates`). In raw per-90 terms the destination club's own
    attacking level explained much of the label (a defender at a dominant side
    joins more dangerous moves just by being there); the share asks how
    involved the player was in what his team created. "ga" stays per 90.
    Without it (the `output` target) every basis is a plain per-90 rate.
    - keepers: "keeper", a neutral 0.5 for anyone who played. No source here
      measures keeping in every season, and attacking involvement would rank
      keepers on something that isn't their job. Minutes and retention still
      judge them; a keeper who never played still scores 0.0.

    Every basis must exist in EVERY season. Midfielders and defenders were
    ranked on FBref progressive passes/carries until FBref's live pages
    dropped them (seasons "2223"+ from the scrape), after which they fell
    back to xG: the same performance scored as a hit for a 2020 arrival and
    a bust for a 2023 one. Understat publishes xGChain for every season.
    `performance_gls`/`performance_ast` (goals/assists) remain the terminal
    fallback for players Understat could not be matched to.

    `output_fallback` is True whenever a player's basis is not the primary
    basis for their position group ("xg" for attackers, "chain" for
    midfielders/defenders, "keeper" for keepers).
    `output_basis` names which basis was actually used ("xg" / "ga" /
    "prog" / "none" for a player with no window rows at all), so a training
    frame can tell "ranked on xG" apart from "ranked on goals" instead of
    collapsing both into one boolean.

    The per-90 rate for a basis is computed ONLY from the window rows that
    actually carry that basis's columns (see `_basis_rate`) — a numerator
    from one season is never divided by minutes drawn from a season that
    numerator never saw. `output_rows_used` counts those contributing rows
    and `output_window_share` is their share of the player's total window
    minutes; a share below 1.0 means the label rests on partial evidence,
    which is a real limitation of the data (less of the window observed on
    that basis), not a bias to correct away.

    The "xg" basis takes whichever source (Understat `us_npxg + us_xa`, FBref
    `expected_npxg + expected_xag`) covers more of the window's minutes,
    Understat on ties. `output_xg_source` records which ("understat" /
    "fbref", "" for any non-xg basis). The two are different xG models, so
    percentiles are ranked within (position_group, output_basis,
    output_xg_source) and never across sources.
    """
    from src import normalize

    PRIMARY_BASIS = {"attacker": "xg", "keeper": "keeper"}  # everyone else: "chain"
    team_rates = (team_attack_rates(seasons[seasons["league"] == config.SERIE_A]
                                    if "league" in seasons.columns else seasons)
                  if relative_to_team else None)

    rows = []
    for _, arrival in arrivals.iterrows():
        window = window_seasons(arrival["arrival_season"])
        played = seasons[
            (seasons["player"] == arrival["player"])
            & (seasons["season"].isin(window))
            & (seasons["league"] == config.SERIE_A)
        ]
        minutes = float(played["minutes"].sum()) if len(played) else 0.0
        if minutes <= 0:
            rows.append({"player": arrival["player"],
                         "arrival_season": arrival["arrival_season"],
                         "position_group": "unknown", "raw": np.nan,
                         "output_fallback": False, "output_basis": "none",
                         "output_rows_used": 0, "output_window_share": 0.0,
                         "output_xg_source": ""})
            continue
        # Use the pos from the window row where the player logged the most
        # minutes, not just the first row on disk (row order is incidental
        # and can put a minor cameo at a different position first).
        most_played_row = played.loc[played["minutes"].idxmax()]
        group = normalize.position_group(most_played_row["pos"])

        if group == "keeper":
            rows.append({"player": arrival["player"],
                         "arrival_season": arrival["arrival_season"],
                         "position_group": group, "raw": np.nan,
                         "output_fallback": False, "output_basis": "keeper",
                         "output_rows_used": len(played), "output_window_share": 1.0,
                         "output_xg_source": ""})
            continue
        chain = ["xg", "ga"] if group == "attacker" else ["chain", "ga"]

        basis = None
        rate = None
        rows_used = None
        masked_minutes = None
        source = ""
        for candidate in chain:
            result = _basis_rate(played, candidate, team_rates)
            if result is not None:
                basis = candidate
                rate, masked_minutes, rows_used, source = result
                break

        if basis is None:
            # Terminal fallback: not even "ga" had a masked row with minutes
            # (e.g. goals/assists columns themselves absent). Fall back to the
            # full window rather than crash — this is the pre-existing
            # "missing column contributes 0" behaviour for the last resort.
            basis = "ga"
            total = _column_sum(played, "performance_gls") + _column_sum(played, "performance_ast")
            rate = float(total) / minutes * 90.0
            masked_minutes = minutes
            rows_used = len(played)

        primary = PRIMARY_BASIS.get(group, "chain")
        fallback = basis != primary

        rows.append({"player": arrival["player"],
                     "arrival_season": arrival["arrival_season"],
                     "position_group": group, "raw": rate,
                     "output_fallback": fallback, "output_basis": basis,
                     "output_rows_used": rows_used,
                     "output_window_share": masked_minutes / minutes,
                     "output_xg_source": source if basis == "xg" else ""})

    frame = pd.DataFrame(rows)
    group_cols = ["position_group", "output_basis", "output_xg_source"]
    frame["post_output_pct"] = frame.groupby(group_cols)["raw"].rank(pct=True)
    # Groups with fewer than MIN_RANK_GROUP rankable members: neutral 0.5
    # (mirrors the single-member rule in normalize.percentile_within).
    size = frame.groupby(group_cols)["raw"].transform("count")
    frame.loc[frame["raw"].notna() & (size < MIN_RANK_GROUP), "post_output_pct"] = 0.5
    # No Serie A minutes at all is a real outcome, not missing data: zero output.
    frame["post_output_pct"] = frame["post_output_pct"].fillna(0.0)
    frame.loc[frame["output_basis"] == "keeper", "post_output_pct"] = 0.5
    return frame[["player", "arrival_season", "position_group", "post_output_pct",
                  "output_fallback", "output_basis", "output_rows_used",
                  "output_window_share", "output_xg_source"]]


def output_target(percentile: float) -> float:
    """Clamp the ranked production value from `post_output_percentile`."""
    return 0.0 if pd.isna(percentile) else float(np.clip(percentile, 0.0, 1.0))


def retention_target(
    end_team: str,
    signing_team: str,
    end_league: str,
    fee_in: float | None,
    fee_out: float | None,
    club_value_from: float | None,
    club_value_to: float | None,
    is_loan: bool = False,
) -> float:
    """Four-rung ladder. See the spec's step-up section for the reasoning.

    A profitable sale to a bigger club is a signing that worked — Kvaratskhelia
    leaving Napoli for PSG is the canonical case, and scoring it as failure would
    label the cohort's most celebrated transfer a bust.

    A loan is not a sale: a fee attached to a loan (e.g. "Loan fee: EUR2.00m")
    still isn't the club letting the player go, so `is_loan` is evaluated
    before the sale ladder rather than being read off `fee_eur`. Loaned out
    of Serie A reads the same as any other departure that isn't a real sale
    (0.0); loaned to another Serie A club is still "kept in the league"
    (0.66), same as a permanent move within Serie A.

    Round 1 fix (Finding B): the 1.0 step-up tier requires STRICT profit
    (`fee_out > fee_in`) together with a bigger-or-equal destination club;
    breaking even now only earns the 0.33 lateral tier. When the inbound fee
    is unknown (NaN — common for undisclosed fees), profit can't be judged at
    all, so the ladder falls back to club size alone: bigger-or-equal club
    scores 1.0, smaller club scores 0.33. `pd.notna` (not `is not None`) is
    used throughout so NaN fees are treated as "unknown", not as zero. The
    same applies when the OUTBOUND fee is the unknown one: with no fee_out to
    compare against, profit can't be judged either, so this also falls back
    to club size alone.

    Round 2 fix: `is_loan` must be checked BEFORE the free-exit rule, not
    after. Most real loans carry no disclosed fee, and `_fee_to_eur` maps
    "loan transfer" text to 0.0 — so if the free-exit check ran first, an
    ordinary zero-fee loan within Serie A (e.g. Napoli sending a player to
    Torino on loan) would incorrectly hit "left for free" (0.0) instead of
    "moved within Serie A" (0.66), since the loan branch would only ever be
    reached for the minority of loans that happen to carry a disclosed fee.
    """
    if end_team == signing_team:
        return 1.0

    if is_loan:
        return 0.66 if end_league == config.SERIE_A else 0.0

    left_for_free = fee_out is not None and pd.notna(fee_out) and fee_out <= 0
    if left_for_free:
        return 0.0

    if end_league == config.SERIE_A:
        return 0.66

    bigger = (
        club_value_to is not None
        and club_value_from is not None
        and pd.notna(club_value_to)
        and pd.notna(club_value_from)
        and club_value_to >= club_value_from
    )

    fee_in_known = fee_in is not None and pd.notna(fee_in)
    fee_out_known = fee_out is not None and pd.notna(fee_out)

    if fee_in_known and fee_out_known:
        strict_profit = fee_out > fee_in
        break_even = fee_out == fee_in
        if bigger and strict_profit:
            return 1.0
        if break_even or strict_profit or bigger:
            return 0.33
        return 0.0

    # Inbound fee unknown: profit is unknowable, so decide on club size alone.
    return 1.0 if bigger else 0.33


def unobserved_exit_target(fee_in, fee_out) -> float:
    """Retention for a player with NO row anywhere in the final window season.

    Finding #3. Previously such a player was handed `end_team=""` and dropped
    into `retention_target`, which — with the usual NaN fees and no club value
    for the empty team name — landed on the "profit unknowable, decide on club
    size" fallback and returned 0.33. That made 0.0 UNREACHABLE in practice:
    609 of 1284 rows sat at 0.33 and not one row scored 0.0, so the bottom rung
    of the spec's four-rung ladder was never trained against.

    RULING: vanishing from the data is not a lateral move. A player who does
    not appear anywhere — not at the signing club, not elsewhere in Serie A,
    not in any covered league — has been released, let go on a free, or dropped
    out of the covered leagues entirely. Every one of those is the spec's 0.0
    rung ("released, contract expired unrenewed, sold at a loss").

    The one exception is positive evidence to the contrary: if the fee data
    shows a known profitable sale (`fee_out > fee_in`, both present), the club
    banked money on him and he moved somewhere the covered leagues do not see.
    That is the 0.33 lateral rung, not a failure.

    Note this applies only to a genuinely ABSENT end-of-window row. A player who
    HAS an end row keeps the full ladder — Kvaratskhelia has a PSG row and still
    scores 1.0.
    """
    known = pd.notna(fee_in) and pd.notna(fee_out)
    return 0.33 if known and fee_out > fee_in else 0.0


def _end_of_window_row(seasons: pd.DataFrame, player: str, final_season: str):
    """The final-window-season row where the player logged the most minutes.

    Finding #11: `after.iloc[0]` took whichever club row happened to sit first
    on disk. For a player who split the season between two clubs that picks the
    cameo as often as the real destination, and `end_team` drives both the
    "still at the signing club" check and the club-value step-up comparison.
    Row order is incidental; minutes are not.
    """
    after = seasons[(seasons["player"] == player) & (seasons["season"] == final_season)]
    if not len(after):
        return None
    if "minutes" not in after.columns:
        return after.iloc[0]
    return after.loc[after["minutes"].fillna(0).idxmax()]


def build(arrivals, seasons, club_values, fees) -> pd.DataFrame:
    """The targets (`config.TARGETS`), one row per (player, arrival_season).

    A player with no row at all in the final window season is an UNOBSERVED
    EXIT and is scored by `unobserved_exit_target` rather than being pushed
    through the full ladder on an empty `end_team` — see that function for the
    ruling and why the old behaviour made 0.0 unreachable (finding #3).
    """
    arrivals = arrivals.merge(
        post_output_percentile(seasons, arrivals, relative_to_team=False),
        on=["player", "arrival_season"], how="left",
    ).merge(
        post_output_percentile(seasons, arrivals, relative_to_team=True)
        [["player", "arrival_season", "post_output_pct"]]
        .rename(columns={"post_output_pct": "post_output_share_pct"}),
        on=["player", "arrival_season"], how="left",
    )
    rows = []
    for _, arrival in arrivals.iterrows():
        window = window_seasons(arrival["arrival_season"])
        played = serie_a_minutes(seasons, arrival["player"], window)
        team_rows = seasons[
            (seasons["team"] == arrival["to_team"]) & (seasons["season"].isin(window))
        ]
        raw_max = team_rows["team_matches"].max()
        # `raw_max or 0` would NOT catch a NaN max (NaN is truthy in Python),
        # so guard explicitly to keep NaN out of minutes_target.
        team_matches = float(raw_max if pd.notna(raw_max) else 0) * len(window)

        end_row = _end_of_window_row(seasons, arrival["player"], window[-1])
        end_team = end_row["team"] if end_row is not None else ""
        end_league = end_row["league"] if end_row is not None else ""

        fee_row = fees[
            (fees["player"] == arrival["player"]) & (fees["season"] == arrival["arrival_season"])
        ]
        fee_in = float(fee_row["fee_eur"].iloc[0]) if len(fee_row) else None
        exit_row = fees[
            (fees["player"] == arrival["player"]) & (fees["season"] == next_season(window[-1]))
        ]
        fee_out = float(exit_row["fee_eur"].iloc[0]) if len(exit_row) else None
        is_loan = (
            bool(exit_row["is_loan"].iloc[0])
            if len(exit_row) and "is_loan" in exit_row.columns
            else False
        )

        def club_value(team: str) -> float | None:
            match = club_values[club_values["team"] == team]
            return float(match["squad_value_eur"].iloc[0]) if len(match) else None

        rows.append({
            "player": arrival["player"],
            "arrival_season": arrival["arrival_season"],
            "minutes": minutes_target(played, team_matches),
            "output": output_target(arrival.get("post_output_pct", np.nan)),
            "output_share": output_target(arrival.get("post_output_share_pct", np.nan)),
            "retention": (
                unobserved_exit_target(fee_in, fee_out)
                if end_row is None
                else retention_target(
                    end_team=end_team,
                    signing_team=arrival["to_team"],
                    end_league=end_league,
                    fee_in=fee_in,
                    fee_out=fee_out,
                    club_value_from=club_value(arrival["to_team"]),
                    club_value_to=club_value(end_team),
                    is_loan=is_loan,
                )
            ),
            "output_fallback": bool(arrival.get("output_fallback", False)),
            "output_basis": arrival.get("output_basis", "none"),
            "output_rows_used": int(arrival.get("output_rows_used", 0)),
            "output_window_share": float(arrival.get("output_window_share", 0.0)),
            "output_xg_source": arrival.get("output_xg_source", "") or "",
        })
    return pd.DataFrame(
        rows,
        columns=["player", "arrival_season", *config.TARGETS, "output_fallback", "output_basis",
                 "output_rows_used", "output_window_share", "output_xg_source"],
    )
