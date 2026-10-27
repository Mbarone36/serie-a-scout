import re
import unicodedata

import pandas as pd
from rapidfuzz import fuzz

from src import config

MANUAL_MAP = "id_map.csv"

# Letters that don't decompose under NFKD (so the accent-stripping pass
# below can't recover them) and would otherwise be silently dropped by the
# `[^a-zA-Z\s]` filter -- e.g. "Filip Đuričić" -> "filip uricic" (the "Đ"
# vanishes entirely). Applied BEFORE the NFKD step so downstream accent
# stripping runs on the transliterated ASCII form. Transfermarkt's own
# slugs follow the same convention (e.g. "filip-djuricic").
_TRANSLITERATIONS = {
    "Đ": "dj", "đ": "dj",
    "Ø": "o", "ø": "o",
    "Æ": "ae", "æ": "ae",
    "Ł": "l", "ł": "l",
    "ß": "ss",
    "Œ": "oe", "œ": "oe",
    "Þ": "th", "þ": "th",
    "Ð": "d", "ð": "d",
    "ı": "i", "İ": "i",
}


def _ascii_fold(name: str) -> str:
    """Transliterate the non-decomposing letters, then strip accents (NFKD)."""
    text = str(name)
    for char, replacement in _TRANSLITERATIONS.items():
        text = text.replace(char, replacement)
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def search_form(name: str) -> str:
    """Transfermarkt quick-search form of a name.

    Same transliteration/accent-stripping/lowercasing/whitespace collapse as
    `normalise_name`, but apostrophes and hyphens are KEPT: Transfermarkt's
    search returns zero hits for "m baye niang" or "jean daniel akpa akpro"
    while "m'baye niang" and "jean-daniel akpa-akpro" resolve.
    """
    return re.sub(r"\s+", " ", _ascii_fold(name)).strip().lower()


def normalise_name(name: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    ascii_only = _ascii_fold(name)
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", ascii_only)
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def _best_match(key: str, pool_keys: list[str], threshold: int):
    """Best (index, score, subset_reliant) in `pool_keys` for `key`.

    Ordinarily scores each candidate with `token_sort_ratio`. For a
    candidate whose normalised tokens are a SUBSET of `key`'s tokens (the
    TM name is shorter than or equal to the FBref name -- an alias like
    TM's "Dalbert" for FBref's "Dalbert Henrique", or TM's "Frank Anguissa"
    for FBref's "Andre-Frank Zambo Anguissa"), also considers
    `token_set_ratio` and takes the max of the two.

    This direction matters: token_set_ratio is symmetric, so applying it
    without the subset check lets a SHORT FBref mononym match a LONGER,
    unrelated TM name too -- e.g. `token_set_ratio("danilo", "danilo
    pereira") == 100`, which with a coincident birth year and nationality
    (plausible for two different Brazilians) would be a silent false join
    that the duplicate-claim rule can't catch (only one FBref row ever
    claims that TM id). Restricting token_set_ratio to the TM-shorter
    direction keeps the alias cases it's meant for while refusing to
    inflate the opposite, dangerous direction; token_sort_ratio alone
    still rejects genuinely different same-length names ("Marco Rossi" vs
    "Mario Rossi Junior").

    `subset_reliant` is True when the returned score only cleared
    `threshold` because of the token_set_ratio side of that comparison
    (i.e. token_sort_ratio alone was below threshold) -- callers use this
    to require an exact (non-estimated) birth date before accepting such a
    match, since a subset/alias match is inherently weaker evidence than a
    full-name match.
    """
    best_idx = None
    best_score = -1.0
    best_subset_reliant = False
    key_tokens = set(key.split())
    for idx, other in enumerate(pool_keys):
        sort_score = fuzz.token_sort_ratio(key, other)
        other_tokens = set(other.split())
        subset_reliant = False
        score = sort_score
        if other_tokens and other_tokens.issubset(key_tokens):
            set_score = fuzz.token_set_ratio(key, other)
            if set_score > sort_score:
                score = set_score
                subset_reliant = sort_score < threshold <= set_score
        if score > best_score:
            best_score = score
            best_idx = idx
            best_subset_reliant = subset_reliant
    if best_idx is None:
        return None
    return best_idx, best_score, best_subset_reliant


def _same_person(key_a: str, born_a, key_b: str, born_b) -> bool:
    """Two FBref identities that are one person: identical, non-null birth
    year AND normalised names whose token sets are identical (accent or word
    order variants: "martin erlic" / "martin erlic", "robert lewandowski" /
    "lewandowski robert") or where one token set is a strict subset of the
    other (an added middle name: "leo ostigard" / "leo skiri ostigard").
    There is deliberately no fuzzy-similarity threshold: any two claimants of
    one tm_id already scored >= 88 against the same TM name, so a fuzzy test
    would group different people ("mario rossi" / "mario rossini"). Nationality
    is deliberately not compared (a dual national can appear under each)."""
    if pd.isna(born_a) or pd.isna(born_b) or born_a != born_b:
        return False
    tokens_a, tokens_b = set(key_a.split()), set(key_b.split())
    if not tokens_a or not tokens_b:
        return False
    return tokens_a <= tokens_b or tokens_b <= tokens_a


def match_players(fbref: pd.DataFrame, tm: pd.DataFrame, threshold: int = 88):
    """Match on fuzzy name, gated by birth YEAR and nationality.

    FBref reports only a birth year (`born`), so an exact-date gate can never
    match. Year plus nationality is the strongest gate the sources actually
    support. It stays a gate rather than a tiebreak: footballers share names
    constantly, and a wrong join silently attaches another player's injury
    history to this one.

    Name similarity is `token_sort_ratio`, except for a TM candidate whose
    name is a subset/alias of the FBref name (TM shorter or equal), where
    `token_set_ratio` is also considered -- see `_best_match` for why the
    direction matters. A match that only cleared `threshold` via that
    subset scoring is flagged `subset_match=True` and is only accepted
    when the TM candidate's birth date is exact: if `tm` carries a
    `birth_date_is_estimate` column, a True value on that row rejects the
    match (an alias-strength name match isn't enough evidence on top of an
    already-uncertain birth date); a missing column is treated as exact.

    A narrow second pass then covers dual-nationality mismatches (e.g.
    Transfermarkt records a player's second passport, FBref the first --
    "Grégoire Defrel" is MTQ on FBref, FRA on Transfermarkt): any FBref
    player still unmatched after the first pass is checked, year-gated
    only (no nationality gate), against every TM candidate; a candidate
    whose name scores EXACTLY 100 is accepted and the row is flagged via
    the `nationality_mismatch` column (False for every first-pass row).
    This is deliberately much narrower than the first pass -- dropping the
    nationality gate is only safe when the name match is perfect, otherwise
    a wrong-nationality near-match could silently steal another player's id.
    The same subset_match/exact-birth-date gate applies here too.

    When multiple FBref players claim the same Transfermarkt id, claimants
    that are the same person (identical non-null born AND normalised names
    with identical token sets or one a strict subset of the other -- no fuzzy
    threshold; see `_same_person`) are grouped: all group members keep the
    match and get `same_person_group=True`. Between different people the
    match is only kept if the best group's score beats every other by at
    least 5 points; otherwise all claimants of that id are treated as
    unmatched (sent to review). This rule applies across both passes.
    """
    left = fbref.copy()
    right = tm.copy()
    left["_key"] = left["player"].map(normalise_name)
    right["_key"] = right["name"].map(normalise_name)
    left["_year"] = pd.to_numeric(left["born"], errors="coerce")
    right["_year"] = pd.to_datetime(right["birth_date"], errors="coerce").dt.year
    left["_nat"] = left["nationality"].astype(str).str.upper().str[:3]
    right["_nat"] = right["nationality"].astype(str).str.upper().str[:3]

    columns = [
        "fbref_player", "tm_id", "score", "nationality_mismatch", "subset_match",
        "same_person_group",
    ]
    has_estimate_col = "birth_date_is_estimate" in right.columns
    rows = []
    matched_players = set()
    for _, candidate in left.iterrows():
        pool = right[
            (right["_year"] == candidate["_year"]) & (right["_nat"] == candidate["_nat"])
        ]
        if pool.empty:
            continue
        best = _best_match(candidate["_key"], pool["_key"].tolist(), threshold)
        if best is None or best[1] < threshold:
            continue
        idx, score, subset_reliant = best
        tm_row = pool.iloc[idx]
        if subset_reliant and has_estimate_col and bool(tm_row["birth_date_is_estimate"]):
            continue
        rows.append({
            "fbref_player": candidate["player"],
            "tm_id": tm_row["tm_id"],
            "score": score,
            "nationality_mismatch": False,
            "subset_match": subset_reliant,
            "_born": candidate["_year"],
            "same_person_group": False,
        })
        matched_players.add(candidate["player"])

    # Second pass: dual-nationality mismatches. Year-gated only, and only
    # for FBref players the first pass didn't already claim.
    for _, candidate in left.iterrows():
        if candidate["player"] in matched_players:
            continue
        pool = right[right["_year"] == candidate["_year"]]
        if pool.empty:
            continue
        best = _best_match(candidate["_key"], pool["_key"].tolist(), threshold)
        if best is None or best[1] != 100:
            continue
        idx, score, subset_reliant = best
        tm_row = pool.iloc[idx]
        if subset_reliant and has_estimate_col and bool(tm_row["birth_date_is_estimate"]):
            continue
        rows.append({
            "fbref_player": candidate["player"],
            "tm_id": tm_row["tm_id"],
            "score": 100,
            "nationality_mismatch": True,
            "subset_match": subset_reliant,
            "_born": candidate["_year"],
            "same_person_group": False,
        })
        matched_players.add(candidate["player"])

    matches = pd.DataFrame(rows, columns=columns + ["_born"])

    # Detect duplicate tm_id claims and resolve conflicts. Claimants that are
    # evidently the same person (one person, two FBref identities -- accent
    # variants, an added middle name, a second nationality) are grouped
    # first: every member keeps the match and is flagged `same_person_group`.
    # The gap rule only arbitrates between DIFFERENT people.
    duplicates = matches[matches["tm_id"].duplicated(keep=False)]
    drop_index = set()
    same_person_index = set()
    for tm_id, group in duplicates.groupby("tm_id"):
        members = list(group.index)
        keys = {i: normalise_name(matches.at[i, "fbref_player"]) for i in members}
        parent = {i: i for i in members}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for a_pos, a in enumerate(members):
            for b in members[a_pos + 1:]:
                if _same_person(
                    keys[a], matches.at[a, "_born"], keys[b], matches.at[b, "_born"]
                ):
                    parent[find(a)] = find(b)

        clusters = {}
        for i in members:
            clusters.setdefault(find(i), []).append(i)
        cluster_list = sorted(
            clusters.values(),
            key=lambda idxs: max(matches.at[i, "score"] for i in idxs),
            reverse=True,
        )
        top_scores = [max(matches.at[i, "score"] for i in idxs) for idxs in cluster_list]
        if len(cluster_list) == 1 or (top_scores[0] - top_scores[1]) >= 5:
            winners, losers = cluster_list[0], [i for c in cluster_list[1:] for i in c]
        else:
            winners, losers = [], members
        if len(winners) > 1:
            same_person_index.update(winners)
        for i in losers:
            drop_index.add(i)
            matched_players.discard(matches.at[i, "fbref_player"])

    matches.loc[list(same_person_index), "same_person_group"] = True
    matches = matches.drop(index=list(drop_index))
    matches = matches[columns].reset_index(drop=True)

    unmatched = fbref[~fbref["player"].isin(matched_players)].copy()
    return matches, unmatched


def load_manual_map() -> pd.DataFrame:
    path = config.MANUAL_DIR / MANUAL_MAP
    if not path.exists():
        return pd.DataFrame(columns=["fbref_player", "tm_id"])
    return pd.read_csv(path)


def apply_manual_overrides(matches: pd.DataFrame, unmatched: pd.DataFrame):
    """Fold hand-reviewed pairings in and drop them from the unmatched list."""
    manual = load_manual_map()
    if manual.empty:
        return matches, unmatched
    manual = manual.assign(score=100, nationality_mismatch=False, subset_match=False)
    combined = pd.concat([matches, manual], ignore_index=True)
    combined = combined.drop_duplicates(subset=["fbref_player"], keep="last").reset_index(drop=True)
    remaining = unmatched[~unmatched["player"].isin(manual["fbref_player"])].reset_index(drop=True)
    return combined, remaining
