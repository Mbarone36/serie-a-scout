"""Transfermarkt bulk-pull orchestration.

For every FBref player who actually appears in `transfers.parquet` (a Serie A
arrival, not the whole ~5.7k FBref universe), resolve a Transfermarkt id via
quick search + `entities.match_players`, then pull that player's injuries,
market-value history and transfer/fee history. Also resolves every club name
seen as an origin or destination in the arrivals to a Transfermarkt squad
value.

`build()` is the only function here that is meant to run against the live
site; every other function is exercised in tests with `scrape_tm.fetch`
monkeypatched to fixture text, so nothing here reaches the network on its
own.
"""

import csv
import logging
import urllib.parse

import pandas as pd
from rapidfuzz import fuzz

from src import config, entities, scrape_tm

logger = logging.getLogger(__name__)

BASE = scrape_tm.BASE

FILES = {
    "injuries": "injuries.parquet",
    "values": "market_values.parquet",
    "fees": "fees.parquet",
}

# FINDING C: resumability manifest. One player name per line, appended after
# each successfully-pulled player -- keying resumability on presence in
# `injuries.parquet["player"]` (the original design) meant a player whose
# injuries page legitimately parses to zero rows was never recorded as
# "done" and got re-pulled (and his value/fee rows re-appended) on every
# run. This file is the single source of truth for "already pulled".
MANIFEST_FILE = "tm_pulled.csv"

# Fuzzy-name threshold above which a search candidate is treated as a
# plausible match worth spending a network request to confirm/refine.
_CANDIDATE_SCORE_THRESHOLD = 88
# Fuzzy-name threshold above which a club quick-search hit is trusted.
_CLUB_SCORE_THRESHOLD = 80

# Columns each output parquet must carry even when no player contributed any
# rows to it (e.g. the very first run, or a run where every match was
# skipped) -- this is what keeps `pull_players` from writing a schema-less
# parquet, or raising `pd.concat` errors, when nothing new came in.
_BASE_COLUMNS = {
    "injuries": scrape_tm.INJURY_COLUMNS + ["player"],
    "values": scrape_tm.MV_COLUMNS + ["player"],
    "fees": scrape_tm.TRANSFER_COLUMNS + ["player"],
}


def search_player(name: str) -> pd.DataFrame:
    """Quick-search Transfermarkt for a player name; candidate identities.

    The first query is `entities.search_form(name)`: transliterated,
    accent-stripped and lowercased but with apostrophes and hyphens KEPT --
    Transfermarkt returns zero hits for "m baye niang" but resolves
    "m'baye niang". If that parses to zero rows, a second search runs with
    `entities.normalise_name(name)` (punctuation replaced by spaces), under
    its own cache key.
    """
    def _search(text: str) -> pd.DataFrame:
        html = scrape_tm.fetch(
            f"{BASE}/schnellsuche/ergebnis/schnellsuche?query={urllib.parse.quote_plus(text)}",
            key=f"search/{text.replace(' ', '_')}",
        )
        return scrape_tm.parse_search_results(html)

    first = entities.search_form(name)
    result = _search(first)
    if result.empty:
        fallback = entities.normalise_name(name)
        if fallback != first:
            result = _search(fallback)
    return result


def _refine_candidate_birth_dates(
    fbref_players: pd.DataFrame, candidates: pd.DataFrame
) -> pd.DataFrame:
    """Replace an age-derived estimated `birth_date` with the exact one from
    a plausible candidate's Transfermarkt profile page.

    FINDING A: `scrape_tm._birth_date_from_age` sets year = today.year - age,
    which lands a year too high for anyone whose birthday falls later in the
    calendar year than the (unknown) scrape date -- roughly a third of
    players -- silently failing `entities.match_players`'s exact-year gate.
    The gate itself must not be loosened (a false match is worse than an
    unmatched row), so instead this fetches the one page that actually has
    an exact date -- but ONLY for candidates that already look like a
    plausible match under one of `entities.match_players`'s three
    acceptance paths, each of which requires an exact (non-estimated) date
    before it will accept a subset/dual-nationality match:

    (a) nationality matches and `token_sort_ratio` >= 88 (the main-pass
        full-name match);
    (b) nationality matches and the TM name's token set is a SUBSET of the
        FBref name's token set (TM shorter or equal -- e.g. "Dalbert" for
        "Dalbert Henrique") with `token_set_ratio` >= 88 (the main pass's
        subset/alias scoring -- see `entities._best_match`; only this
        direction, never the reverse, for the same false-join reason
        documented there);
    (c) `token_sort_ratio` == 100 regardless of nationality (the
        dual-nationality second pass, which drops the nationality gate but
        only ever accepts a perfect name score);
    (d) the TM name's token set is a SUBSET of the FBref name's token set
        (same direction as (b)) with `token_set_ratio` == 100, regardless
        of nationality -- `_best_match`'s subset scoring feeds into the
        dual-nationality pass too (a perfect token_set_ratio is itself an
        overall score of 100, which is exactly what that pass accepts), so
        a subset alias with a nationality mismatch (e.g. TM "Dalbert" vs
        FBref "Dalbert Henrique" on different passports) can be accepted
        there even though (b) alone would never fetch it without a
        nationality match.

    A candidate that doesn't clear any of these is never profile-fetched,
    keeping the extra network cost bounded by real leads, not the whole
    quick-search result set. When a profile fetch doesn't yield a birth
    date, the original age-derived estimate is kept and
    `birth_date_is_estimate` stays True.
    """
    candidates = candidates.copy()
    if "birth_date_is_estimate" not in candidates.columns:
        candidates["birth_date_is_estimate"] = True
    if candidates.empty:
        return candidates

    to_fetch: set[int] = set()
    for _, player in fbref_players.iterrows():
        player_key = entities.normalise_name(player["player"])
        player_tokens = set(player_key.split())
        player_nat = str(player.get("nationality", "")).upper()[:3]
        for idx, cand in candidates.iterrows():
            cand_key = entities.normalise_name(cand["name"])
            nat_match = str(cand.get("nationality", "")) == player_nat
            sort_score = fuzz.token_sort_ratio(player_key, cand_key)

            if sort_score == 100:
                to_fetch.add(idx)
                continue

            cand_tokens = set(cand_key.split())
            is_subset = bool(cand_tokens) and cand_tokens.issubset(player_tokens)
            set_score = fuzz.token_set_ratio(player_key, cand_key) if is_subset else None

            if is_subset and set_score == 100:
                to_fetch.add(idx)
                continue
            if nat_match and sort_score >= _CANDIDATE_SCORE_THRESHOLD:
                to_fetch.add(idx)
                continue
            if nat_match and is_subset and set_score >= _CANDIDATE_SCORE_THRESHOLD:
                to_fetch.add(idx)

    for idx in sorted(to_fetch):
        row = candidates.loc[idx]
        tm_id = int(row["tm_id"])
        try:
            html = scrape_tm.fetch(
                f"{BASE}/{row['slug']}/profil/spieler/{tm_id}",
                key=f"prof/{tm_id}",
            )
            identity = scrape_tm.parse_player_identity(html)
        except Exception as exc:
            # A permanent error (e.g. HTTP 403/404) on one candidate's
            # profile page must not abort refinement for every other
            # candidate -- fall back to the age-derived estimate for this
            # one and keep going.
            logger.warning(
                "profile fetch failed for tm_id=%s, keeping age-derived "
                "birth_date estimate: %s", tm_id, exc,
            )
            continue
        birth_date = identity.get("birth_date")
        if birth_date:
            candidates.at[idx, "birth_date"] = pd.Timestamp(birth_date)
            candidates.at[idx, "birth_date_is_estimate"] = False

    return candidates


def _pull_one(tm_id: int, name: str) -> dict:
    """Injuries, market-value history and transfer/fee history for one player.

    Market value and transfer history are the `ceapi` JSON endpoints, not
    the HTML profile page -- the profile page is never fetched here.
    """
    slug = entities.normalise_name(name).replace(" ", "-")
    injuries_html = scrape_tm.fetch(
        f"{BASE}/{slug}/verletzungen/spieler/{tm_id}", f"inj/{tm_id}"
    )
    mv_json = scrape_tm.fetch(scrape_tm.MV_URL.format(id=tm_id), f"mv/{tm_id}")
    tr_json = scrape_tm.fetch(scrape_tm.TRANSFERS_URL.format(id=tm_id), f"tr/{tm_id}")
    return {
        "injuries": scrape_tm.parse_injuries(injuries_html),
        "values": scrape_tm.parse_market_value_history(mv_json),
        "fees": scrape_tm.parse_transfer_history(tr_json),
    }


def _existing(kind: str) -> pd.DataFrame:
    path = config.PROCESSED_DIR / FILES[kind]
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _manifest_path():
    return config.PROCESSED_DIR / MANIFEST_FILE


def _load_manifest() -> set:
    """Players already successfully pulled, per `tm_pulled.csv`."""
    path = _manifest_path()
    if not path.exists():
        return set()
    manifest = pd.read_csv(path)
    if "player" not in manifest.columns:
        return set()
    return set(manifest["player"])


def _append_manifest(player: str) -> None:
    """Record one more successfully-pulled player, one line at a time so a
    crash mid-run loses at most the in-flight player, never a completed one."""
    path = _manifest_path()
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(["player"])
        writer.writerow([player])


def _append_output(kind: str, frame: pd.DataFrame) -> None:
    """Append one player's rows for `kind` to its parquet file, deduplicated.

    Written per player (read-concat-dedupe-write) rather than batched at the
    end of `pull_players`, so a crash after this call but before the next
    player's `_append_manifest` never leaves the manifest claiming a player
    whose rows aren't actually on disk.
    """
    path = config.PROCESSED_DIR / FILES[kind]
    existing = _existing(kind)
    combined = pd.concat([existing, frame], ignore_index=True) if len(existing) else frame
    if combined.columns.empty:
        combined = pd.DataFrame(columns=_BASE_COLUMNS[kind])
    combined = combined.drop_duplicates().reset_index(drop=True)
    combined.to_parquet(path)


def pull_players(matches: pd.DataFrame) -> None:
    """Fetch every matched player. Resumable via a manifest of players
    already pulled (`tm_pulled.csv`), NOT via presence in `injuries.parquet`
    -- a player whose injuries page legitimately parses to zero rows can
    never be detected as "already pulled" that way, so he'd be re-pulled
    (and his value/fee rows re-appended) on every run.

    FINDING C: the manifest is appended, and each kind's output parquet is
    written with duplicates dropped, after EACH player -- a crash mid-run
    loses at most the player in flight, and never produces duplicate rows
    in market_values/fees for a player pulled more than once.
    """
    done = _load_manifest()

    for _, row in matches.iterrows():
        if row["fbref_player"] in done:
            continue
        pulled = _pull_one(int(row["tm_id"]), row["fbref_player"])
        for kind, frame in pulled.items():
            frame = frame.copy()
            frame["player"] = row["fbref_player"]
            _append_output(kind, frame)
        _append_manifest(row["fbref_player"])
        done.add(row["fbref_player"])

    # Ensure well-formed (correctly-columned) empty parquet files exist even
    # when nothing new was pulled and no file existed yet.
    for kind in FILES:
        path = config.PROCESSED_DIR / FILES[kind]
        if not path.exists():
            pd.DataFrame(columns=_BASE_COLUMNS[kind]).to_parquet(path)


def search_club(name: str) -> int | None:
    """Resolve a club name to a Transfermarkt club id via the quick search.

    FINDING D: the quick search's first hit isn't reliably the right club
    (a reserve/youth side, or an unrelated club that happens to rank first)
    -- so a hit is only trusted if its label is a plausible fuzzy match
    (rapidfuzz `token_sort_ratio` >= 80) for the query name; hits are
    checked in the order returned and the first one to clear the bar wins.
    If none does, the club is treated as unresolved (NaN squad value)
    rather than risking a false match, which is worse than a missing row.
    """
    html = scrape_tm.fetch(
        f"{BASE}/schnellsuche/ergebnis/schnellsuche?query={name.replace(' ', '+')}",
        key=f"clubsearch/{entities.normalise_name(name).replace(' ', '_')}",
    )
    hits = scrape_tm.parse_club_search_results(html)
    query_key = entities.normalise_name(name)
    for club_id, label in hits:
        score = fuzz.token_sort_ratio(query_key, entities.normalise_name(label))
        if score >= _CLUB_SCORE_THRESHOLD:
            return int(club_id)
    return None


def pull_club_values(team_names) -> None:
    """Squad market value per club, keyed by the team names the transfers
    table uses.

    The step-up check in Task 12 compares the selling club's squad value
    with the destination's. Without this file that comparison cannot run
    and the retention ladder silently collapses to three rungs -- which
    would score a profitable sale to a bigger club as a failure, exactly
    what the spec forbids. Clubs that can't be resolved on Transfermarkt
    get NaN rather than being dropped, so downstream joins don't silently
    lose rows.
    """
    rows = []
    for name in team_names:
        club_id = search_club(name)
        if club_id is None:
            rows.append({"team": name, "squad_value_eur": float("nan")})
            continue
        html = scrape_tm.fetch(
            f"{BASE}/x/startseite/verein/{club_id}", f"club/{club_id}"
        )
        rows.append({
            "team": name,
            "squad_value_eur": scrape_tm.parse_club_squad_value(html),
        })
    pd.DataFrame(rows).to_parquet(config.PROCESSED_DIR / "club_values.parquet")


def build() -> None:
    """Resolve identities and pull injuries/values/fees/club-values.

    CONTROLLER RULING: scoped to players who actually appear in
    `transfers.parquet` (the arrival set), not every FBref identity ever
    seen -- matching and pulling for the full ~5.7k player universe would
    be enormously wasteful when only arrivals feed the model.
    """
    fbref_players = pd.read_parquet(config.PROCESSED_DIR / "fbref_identities.parquet")
    transfers_frame = pd.read_parquet(config.PROCESSED_DIR / "transfers.parquet")

    arrival_players = set(transfers_frame["player"].unique())
    in_scope = fbref_players[fbref_players["player"].isin(arrival_players)].reset_index(drop=True)
    print(
        f"{len(in_scope)} of {len(fbref_players)} FBref identities are in scope "
        f"({len(arrival_players)} unique arrival players in transfers.parquet)"
    )

    candidate_frames = [search_player(name) for name in in_scope["player"].unique()]
    if candidate_frames:
        candidates = pd.concat(candidate_frames, ignore_index=True)
        if not candidates.empty:
            candidates = candidates.drop_duplicates(subset=["tm_id"])
    else:
        candidates = pd.DataFrame(columns=scrape_tm.SEARCH_COLUMNS)

    # FINDING A: refine plausible candidates' estimated birth_date to an
    # exact one from their profile page before the exact-year match gate.
    candidates = _refine_candidate_birth_dates(in_scope, candidates)

    matches, unmatched = entities.match_players(in_scope, candidates)
    matches, unmatched = entities.apply_manual_overrides(matches, unmatched)
    unmatched.to_csv(config.MANUAL_DIR / "unmatched_review.csv", index=False)
    print(
        f"matched {len(matches)}, unmatched {len(unmatched)} "
        f"(review data/manual/unmatched_review.csv)"
    )

    pull_players(matches)

    club_names = sorted(set(transfers_frame["to_team"]) | set(transfers_frame["from_team"]))
    pull_club_values(club_names)


if __name__ == "__main__":
    build()
