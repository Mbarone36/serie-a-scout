import pandas as pd
import pytest
from src import entities


def test_normalise_name_strips_accents_and_case():
    assert entities.normalise_name("Khvicha Kvaratskhelia") == "khvicha kvaratskhelia"
    assert entities.normalise_name("Ciro Immobile") == "ciro immobile"
    assert entities.normalise_name("Nicolò Barella") == "nicolo barella"
    assert entities.normalise_name("Zlatan Ibrahimović") == "zlatan ibrahimovic"


def test_normalise_name_collapses_whitespace_and_punctuation():
    assert entities.normalise_name("  De  Ligt, Matthijs ") == "de ligt matthijs"


def test_normalise_name_transliterates_non_decomposing_letters():
    """FINDING (diagnosis b): NFKD only strips accents off letters that
    decompose. Đ/Ø/Æ/Ł/ß/Œ/Þ/Ð don't, so the old `[^a-zA-Z\\s]` filter just
    dropped them -- "Filip Đuričić" became "filip uricic", breaking both the
    fuzzy-match key and (via `pull_tm.search_player`) the search query
    itself. A transliteration pass before NFKD fixes this; Transfermarkt's
    own slug for this player is "filip-djuricic", confirming "dj" is the
    right mapping for Đ (not just dropping it or mapping to a bare "d")."""
    assert entities.normalise_name("Filip Đuričić") == "filip djuricic"
    assert entities.normalise_name("Simon Kjær") == "simon kjaer"
    assert entities.normalise_name("Leo Skiri Østigård") == "leo skiri ostigard"
    assert entities.normalise_name("Æ æ") == "ae ae"
    assert entities.normalise_name("Łukasz") == "lukasz"
    assert entities.normalise_name("Weiß") == "weiss"
    assert entities.normalise_name("Œuvre") == "oeuvre"
    assert entities.normalise_name("Þór Ðór") == "thor dor"


def _pair():
    # FBref reports birth YEAR only (`born`), never a full date.
    fbref = pd.DataFrame({
        "player": ["Nicolo Barella", "Khvicha Kvaratskhelia"],
        "born": [1997, 2001],
        "nationality": ["ITA", "GEO"],
    })
    tm = pd.DataFrame({
        "tm_id": [1, 2],
        "name": ["Nicolò Barella", "Khvicha Kvaratskhelia"],
        "birth_date": ["1997-02-07", "2001-02-12"],
        "nationality": ["ITA", "GEO"],
    })
    return fbref, tm


def test_match_players_pairs_on_name_and_birth_date():
    fbref, tm = _pair()
    matches, unmatched = entities.match_players(fbref, tm)
    assert len(matches) == 2
    assert unmatched.empty
    assert matches.loc[matches.fbref_player == "Nicolo Barella", "tm_id"].item() == 1


def test_match_players_rejects_same_name_different_birth_year():
    fbref = pd.DataFrame({"player": ["Luis Silva"], "born": [1990], "nationality": ["BRA"]})
    tm = pd.DataFrame({"tm_id": [9], "name": ["Luis Silva"], "birth_date": ["1998-05-05"], "nationality": ["BRA"]})
    matches, unmatched = entities.match_players(fbref, tm)
    assert matches.empty
    assert list(unmatched["player"]) == ["Luis Silva"]


def test_match_players_rejects_same_year_different_nationality_when_name_not_exact():
    """The nationality gate is only dropped by the narrow dual-nationality
    second pass (diagnosis d), and only for a normalised-name score of
    EXACTLY 100 -- see test_match_players_dual_nationality_second_pass for
    that case. A high-but-imperfect name score ("Luis Silva" vs "Luiz
    Silva" = 90, above the 88 threshold but not 100) with a matching year
    and a different nationality must still not match: the first pass fails
    the nationality gate, and the second pass's exact-100 requirement isn't
    met either."""
    fbref = pd.DataFrame({"player": ["Luis Silva"], "born": [1990], "nationality": ["BRA"]})
    tm = pd.DataFrame({"tm_id": [9], "name": ["Luiz Silva"], "birth_date": ["1990-05-05"], "nationality": ["POR"]})
    matches, unmatched = entities.match_players(fbref, tm)
    assert matches.empty
    assert list(unmatched["player"]) == ["Luis Silva"]


def test_match_players_reports_unmatched():
    fbref, tm = _pair()
    fbref = pd.concat([fbref, pd.DataFrame({
        "player": ["Unknown Guy"], "born": [1995], "nationality": ["ARG"],
    })], ignore_index=True)
    matches, unmatched = entities.match_players(fbref, tm)
    assert list(unmatched["player"]) == ["Unknown Guy"]


def test_manual_overrides_take_precedence(tmp_path, monkeypatch):
    monkeypatch.setattr(entities.config, "MANUAL_DIR", tmp_path)
    (tmp_path / "id_map.csv").write_text("fbref_player,tm_id\nUnknown Guy,42\n")
    matches = pd.DataFrame(columns=["fbref_player", "tm_id", "score"])
    unmatched = pd.DataFrame({"player": ["Unknown Guy"]})
    matches, unmatched = entities.apply_manual_overrides(matches, unmatched)
    assert matches.loc[0, "tm_id"] == 42
    assert unmatched.empty


def _two_person_tm(name_a, name_b):
    """One TM id, two name variants (as if the id were reachable by two aliases)."""
    return pd.DataFrame({
        "tm_id": [1, 1],
        "name": [name_a, name_b],
        "birth_date": ["1990-05-05", "1990-05-05"],
        "nationality": ["ITA", "ITA"],
    })


def _score(a, b):
    from rapidfuzz import fuzz
    a, b = entities.normalise_name(a), entities.normalise_name(b)
    return max(fuzz.token_sort_ratio(a, b), fuzz.token_set_ratio(a, b))


def test_match_players_duplicate_tm_id_clear_winner():
    """Two FBref players claim the same TM record; the highest score wins by
    >= 5. "Mario Rossini" is not a token subset of "Mario Rossi", so the two
    are different people (same born notwithstanding) and go to the gap rule."""
    fbref = pd.DataFrame({
        "player": ["Mario Rossi", "Mario Rossini"],
        "born": [1990, 1990],
        "nationality": ["ITA", "ITA"],
    })
    tm = pd.DataFrame({
        "tm_id": [1],
        "name": ["Mario Rossi"],
        "birth_date": ["1990-05-05"],
        "nationality": ["ITA"],
    })
    assert _score("Mario Rossi", "Mario Rossi") == 100
    assert 91 < _score("Mario Rossini", "Mario Rossi") < 92  # gap > 5
    matches, unmatched = entities.match_players(fbref, tm)
    assert len(matches) == 1
    assert matches.iloc[0]["fbref_player"] == "Mario Rossi"
    assert not matches.iloc[0]["same_person_group"]
    assert list(unmatched["player"]) == ["Mario Rossini"]


def test_match_players_duplicate_tm_id_tie_close():
    """Two different people (differing, non-subset token sets) claim the same
    TM record with scores within 5 -> neither matches."""
    fbref = pd.DataFrame({
        "player": ["Luiz Silva", "Luis Silvo"],
        "born": [1990, 1990],
        "nationality": ["ITA", "ITA"],
    })
    tm = pd.DataFrame({
        "tm_id": [1],
        "name": ["Luis Silva"],
        "birth_date": ["1990-05-05"],
        "nationality": ["ITA"],
    })
    assert _score("Luiz Silva", "Luis Silva") == 90.0
    assert _score("Luis Silvo", "Luis Silva") == 90.0  # gap 0 < 5
    matches, unmatched = entities.match_players(fbref, tm)
    assert matches.empty, f"Expected no matches, got {len(matches)}"
    assert set(unmatched["player"]) == {"Luiz Silva", "Luis Silvo"}


def test_same_person_rule_is_token_identity_or_subset_only():
    sp = entities._same_person
    assert sp("martin erlic", 1998, "martin erlic", 1998)
    assert sp("robert lewandowski", 1988, "lewandowski robert", 1988)
    assert sp("leo ostigard", 1999, "leo skiri ostigard", 1999)
    assert not sp("mario rossi", 1990, "mario rossini", 1990)  # fuzzy 91.7 is not enough
    assert not sp("martin erlic", 1998, "martin erlic", 1997)
    assert not sp("martin erlic", float("nan"), "martin erlic", float("nan"))


def test_match_players_same_born_similar_names_not_grouped():
    """Same-year "Mario Rossi" / "Mario Rossini" are different people: not
    grouped, resolved by the gap rule (Mario Rossi wins by >= 5)."""
    fbref = pd.DataFrame({
        "player": ["Mario Rossi", "Mario Rossini"],
        "born": [1990, 1990],
        "nationality": ["ITA", "ITA"],
    })
    tm = pd.DataFrame({
        "tm_id": [1], "name": ["Mario Rossi"],
        "birth_date": ["1990-05-05"], "nationality": ["ITA"],
    })
    matches, _ = entities.match_players(fbref, tm)
    assert list(matches["fbref_player"]) == ["Mario Rossi"]
    assert not matches["same_person_group"].any()


@pytest.mark.parametrize(
    "names, nats, born, tm_name",
    [
        (("Martin Erlic", "Martin Erlić"), ("CRO", "CRO"), 1998, "Martin Erlić"),
        (("Leo Østigård", "Leo Skiri Østigård"), ("NOR", "NOR"), 1999, "Leo Østigård"),
        (("Obite N'Dicka", "Obite N'Dicka"), ("CIV", "FRA"), 1999, "Obite N'Dicka"),
    ],
)
def test_match_players_same_person_two_fbref_identities_both_match(names, nats, born, tm_name):
    """One person, two FBref identities (keyed on player, born, nationality)
    claiming the same TM id both keep the match and are flagged."""
    fbref = pd.DataFrame({
        "player": list(names),
        "born": [born, born],
        "nationality": list(nats),
    })
    # TM carries the person once per nationality passport so the
    # nationality-gated pass can reach it for each identity.
    tm = pd.DataFrame({
        "tm_id": [7, 7],
        "name": [tm_name, tm_name],
        "birth_date": [f"{born}-03-03", f"{born}-03-03"],
        "nationality": list(nats),
    })
    matches, unmatched = entities.match_players(fbref, tm)
    assert len(matches) == 2, matches
    assert (matches["tm_id"] == 7).all()
    assert matches["same_person_group"].all()
    assert unmatched.empty


def test_match_players_same_born_dissimilar_names_still_use_gap_rule():
    fbref = pd.DataFrame({
        "player": ["Danilo", "Marco Rossi"],
        "born": [1990, 1990],
        "nationality": ["ITA", "ITA"],
    })
    tm = _two_person_tm("Danilo", "Marco Rossi")
    matches, _ = entities.match_players(fbref, tm)
    assert matches.empty  # tie -> gap rule blocks both; not grouped as one person
    assert "same_person_group" in matches.columns


def test_match_players_non_duplicate_rows_are_not_flagged_same_person():
    fbref, tm = _pair()
    matches, _ = entities.match_players(fbref, tm)
    assert list(matches["same_person_group"]) == [False, False]


def test_search_form_keeps_apostrophes_and_hyphens():
    assert entities.search_form("M'Baye Niang") == "m'baye niang"
    assert entities.search_form("Jean-Daniel Akpa-Akpro") == "jean-daniel akpa-akpro"
    assert entities.search_form("  Filip  Đuričić ") == "filip djuricic"
    assert entities.search_form("Leo Østigård") == "leo ostigard"
    assert entities.normalise_name("M'Baye Niang") == "m baye niang"


def test_match_players_null_born_never_matches():
    """Null `born` on FBref side → never matches even when name and nationality agree."""
    fbref = pd.DataFrame({
        "player": ["Mario Rossi"],
        "born": [None],  # null value
        "nationality": ["ITA"],
    })
    tm = pd.DataFrame({
        "tm_id": [1],
        "name": ["Mario Rossi"],
        "birth_date": ["1990-05-05"],
        "nationality": ["ITA"],
    })
    matches, unmatched = entities.match_players(fbref, tm)
    assert matches.empty
    assert list(unmatched["player"]) == ["Mario Rossi"]


def test_match_players_threshold_boundary():
    """Default threshold 88: score below 88 does not match; at or above 88 does."""
    from rapidfuzz import fuzz

    # Test pair scoring BELOW threshold under BOTH scorers (score is now
    # max(token_sort_ratio, token_set_ratio) -- see
    # test_match_players_subset_alias_names_match_via_token_set below, which
    # is exactly why a subset name like the old "Pedri Gonzalez" /
    # "Pedri Gonzalez Lopez" pair can no longer serve as a below-threshold
    # example: token_set_ratio scores that pair 100.
    # Measured: "Marco Rossi" vs "Mario Rossi Junior" sort=69.0 set=69.0
    fbref_below = pd.DataFrame({
        "player": ["Marco Rossi"],
        "born": [2002],
        "nationality": ["ESP"],
    })
    tm_below = pd.DataFrame({
        "tm_id": [1],
        "name": ["Mario Rossi Junior"],
        "birth_date": ["2002-11-25"],
        "nationality": ["ESP"],
    })

    key_a = entities.normalise_name("Marco Rossi")
    key_b = entities.normalise_name("Mario Rossi Junior")
    score_below = max(fuzz.token_sort_ratio(key_a, key_b), fuzz.token_set_ratio(key_a, key_b))
    assert score_below < 88, f"Expected score below 88, got {score_below}"

    matches_below, unmatched_below = entities.match_players(fbref_below, tm_below)  # default threshold=88
    assert matches_below.empty, f"Expected no match below threshold, got {len(matches_below)}"
    assert len(unmatched_below) == 1

    # Test pair scoring AT/ABOVE threshold (88-100)
    # Measured: "Declan Rice" vs "Declan Rice Jr" = 88.0
    fbref_above = pd.DataFrame({
        "player": ["Declan Rice"],
        "born": [1999],
        "nationality": ["ENG"],
    })
    tm_above = pd.DataFrame({
        "tm_id": [2],
        "name": ["Declan Rice Jr"],
        "birth_date": ["1999-01-14"],
        "nationality": ["ENG"],
    })

    score_above = fuzz.token_sort_ratio(
        entities.normalise_name("Declan Rice"),
        entities.normalise_name("Declan Rice Jr")
    )
    assert 88 <= score_above <= 100, f"Expected score in [88, 100], got {score_above}"

    matches_above, unmatched_above = entities.match_players(fbref_above, tm_above)  # default threshold=88
    assert len(matches_above) == 1, f"Expected match at/above threshold, got {len(matches_above)}"
    assert unmatched_above.empty


# ---------------------------------------------------------------------------
# FINDING (diagnosis c): subset/alias names via token_set_ratio
# ---------------------------------------------------------------------------


def test_match_players_subset_alias_names_match_via_token_set():
    """Subset/alias names score below the 88 threshold under
    token_sort_ratio alone ("Dalbert Henrique" vs TM "Dalbert" = 60.9;
    "Andre-Frank Zambo Anguissa" vs TM "Frank Anguissa" = 70). In both
    cases the TM name's tokens are a SUBSET of the FBref name's tokens (TM
    is the shorter alias), which is the only direction `_best_match` also
    considers token_set_ratio for -- and token_set_ratio scores both pairs
    100. The year+nationality gate is unchanged. Neither TM row here
    carries a `birth_date_is_estimate` column, so the subset-match ->
    exact-birth-date gate treats them as exact and both matches are
    accepted, flagged `subset_match=True`."""
    fbref = pd.DataFrame({
        "player": ["Dalbert Henrique", "Andre-Frank Zambo Anguissa"],
        "born": [1993, 1995],
        "nationality": ["BRA", "CMR"],
    })
    tm = pd.DataFrame({
        "tm_id": [11, 12],
        "name": ["Dalbert", "Frank Anguissa"],
        "birth_date": ["1993-08-08", "1995-11-16"],
        "nationality": ["BRA", "CMR"],
    })
    matches, unmatched = entities.match_players(fbref, tm)
    assert unmatched.empty
    assert set(matches["fbref_player"]) == {"Dalbert Henrique", "Andre-Frank Zambo Anguissa"}
    assert matches.loc[matches.fbref_player == "Dalbert Henrique", "tm_id"].item() == 11
    assert matches.loc[
        matches.fbref_player == "Andre-Frank Zambo Anguissa", "tm_id"
    ].item() == 12
    assert (matches["nationality_mismatch"] == False).all()  # noqa: E712
    assert (matches["subset_match"] == True).all()  # noqa: E712


def test_match_players_genuinely_different_name_still_no_match():
    """A pair that merely shares a surname, with no subset/alias
    relationship, must still fail to match under the combined scorer --
    the fix must not turn matching into pure surname matching."""
    fbref = pd.DataFrame({
        "player": ["Marco Rossi"], "born": [1990], "nationality": ["ITA"],
    })
    tm = pd.DataFrame({
        "tm_id": [1], "name": ["Mario Rossi Junior"],
        "birth_date": ["1990-05-05"], "nationality": ["ITA"],
    })
    matches, unmatched = entities.match_players(fbref, tm)
    assert matches.empty
    assert list(unmatched["player"]) == ["Marco Rossi"]


# ---------------------------------------------------------------------------
# FINDING (diagnosis d): dual-nationality second pass
# ---------------------------------------------------------------------------


def test_match_players_dual_nationality_second_pass():
    """A player recorded under a different nationality on each source
    ("Grégoire Defrel" is MTQ on FBref, FRA on Transfermarkt) still matches
    via the narrow second pass: exact (100) normalised-name score plus a
    matching birth year, nationality gate dropped. The row is flagged via
    `nationality_mismatch`."""
    fbref = pd.DataFrame({
        "player": ["Grégoire Defrel"], "born": [1991], "nationality": ["MTQ"],
    })
    tm = pd.DataFrame({
        "tm_id": [21], "name": ["Grégoire Defrel"],
        "birth_date": ["1991-06-17"], "nationality": ["FRA"],
    })
    matches, unmatched = entities.match_players(fbref, tm)
    assert unmatched.empty
    assert len(matches) == 1
    row = matches.iloc[0]
    assert row["tm_id"] == 21
    assert row["score"] == 100
    assert row["nationality_mismatch"] == True  # noqa: E712


def test_match_players_dual_nationality_pass_still_gates_on_year():
    """A 100-name match with a DIFFERENT birth year must not match, even
    under the nationality-dropped second pass."""
    fbref = pd.DataFrame({
        "player": ["Grégoire Defrel"], "born": [1991], "nationality": ["MTQ"],
    })
    tm = pd.DataFrame({
        "tm_id": [21], "name": ["Grégoire Defrel"],
        "birth_date": ["1985-06-17"], "nationality": ["FRA"],
    })
    matches, unmatched = entities.match_players(fbref, tm)
    assert matches.empty
    assert list(unmatched["player"]) == ["Grégoire Defrel"]


def test_match_players_dual_nationality_pass_does_not_run_when_already_matched():
    """A player already matched on the first pass (correct nationality) is
    not reconsidered by the second pass."""
    fbref, tm = _pair()
    matches, unmatched = entities.match_players(fbref, tm)
    assert unmatched.empty
    assert (matches["nationality_mismatch"] == False).all()  # noqa: E712


# ---------------------------------------------------------------------------
# FINDING (round 2 re-review): token_set_ratio direction must be
# TM-shorter-or-equal only, and a subset-reliant match needs an exact
# birth date.
# ---------------------------------------------------------------------------


def test_match_players_mononym_does_not_match_longer_unrelated_name():
    """A short FBref mononym must not match a longer, unrelated TM name via
    token_set_ratio -- `token_set_ratio("danilo", "danilo pereira") == 100`,
    which with a coincident year+nationality (plausible for two different
    Brazilians) would be a silent false join the duplicate-claim rule can't
    catch (only one FBref row ever claims this TM id). Here the TM name's
    tokens are a SUPERSET of the FBref name's tokens -- the wrong direction
    for `_best_match` to apply token_set_ratio -- so only token_sort_ratio
    (60.0) applies, which is below the 88 threshold."""
    fbref = pd.DataFrame({
        "player": ["Danilo"], "born": [1991], "nationality": ["BRA"],
    })
    tm = pd.DataFrame({
        "tm_id": [31], "name": ["Danilo Pereira"],
        "birth_date": ["1991-09-09"], "nationality": ["BRA"],
    })
    matches, unmatched = entities.match_players(fbref, tm)
    assert matches.empty
    assert list(unmatched["player"]) == ["Danilo"]


def test_match_players_subset_match_requires_exact_birth_date():
    """A subset/alias match ("Dalbert Henrique" vs TM "Dalbert") is only
    accepted when the TM candidate's birth date is exact. When `tm` carries
    a `birth_date_is_estimate` column and that row is True (an
    age-derived, not exact, birth date -- see
    `pull_tm._refine_candidate_birth_dates`), the match is rejected even
    though the year and nationality still gate-match and the name clears
    threshold via token_set_ratio."""
    fbref = pd.DataFrame({
        "player": ["Dalbert Henrique"], "born": [1993], "nationality": ["BRA"],
    })
    tm_exact = pd.DataFrame({
        "tm_id": [11], "name": ["Dalbert"], "birth_date": ["1993-08-08"],
        "nationality": ["BRA"], "birth_date_is_estimate": [False],
    })
    matches, unmatched = entities.match_players(fbref, tm_exact)
    assert len(matches) == 1
    assert matches.iloc[0]["tm_id"] == 11
    assert matches.iloc[0]["subset_match"] == True  # noqa: E712
    assert unmatched.empty

    tm_estimated = pd.DataFrame({
        "tm_id": [11], "name": ["Dalbert"], "birth_date": ["1993-08-08"],
        "nationality": ["BRA"], "birth_date_is_estimate": [True],
    })
    matches2, unmatched2 = entities.match_players(fbref, tm_estimated)
    assert matches2.empty
    assert list(unmatched2["player"]) == ["Dalbert Henrique"]


def test_match_players_anguissa_alias_still_matches():
    """FBref "Andre-Frank Zambo Anguissa" vs TM's shorter alias "Frank
    Anguissa" still matches -- TM's tokens are a subset of FBref's, the
    correct direction for the token_set_ratio boost."""
    fbref = pd.DataFrame({
        "player": ["Andre-Frank Zambo Anguissa"], "born": [1995], "nationality": ["CMR"],
    })
    tm = pd.DataFrame({
        "tm_id": [12], "name": ["Frank Anguissa"],
        "birth_date": ["1995-11-16"], "nationality": ["CMR"],
    })
    matches, unmatched = entities.match_players(fbref, tm)
    assert unmatched.empty
    assert len(matches) == 1
    assert matches.iloc[0]["tm_id"] == 12
    assert matches.iloc[0]["subset_match"] == True  # noqa: E712


def test_normalise_name_folds_turkish_dotless_i():
    assert entities.normalise_name("Kenan Yıldız") == "kenan yildiz"
    assert entities.normalise_name("İlkay Gündoğan") == "ilkay gundogan"
