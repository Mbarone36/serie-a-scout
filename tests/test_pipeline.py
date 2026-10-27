import pandas as pd
import pytest

from src import config, pipeline


def _arrivals():
    return pd.DataFrame({
        "player": ["A"], "arrival_season": ["2223"], "to_team": ["Napoli"],
        "from_team": ["Porto"], "from_league": ["POR-Liga"], "prior_minutes": [2000.0],
    })


def _seasons():
    return pd.DataFrame({
        "player": ["A", "A", "A"],
        "team": ["Porto", "Porto", "Napoli"],
        "league": ["POR-Liga", "POR-Liga", "ITA-Serie A"],
        "season": ["2021", "2122", "2223"],
        "minutes": [1800, 2000, 2200],
    })


def test_expand_emits_one_row_per_pre_move_season():
    out = pipeline.expand_per_season(_arrivals(), _seasons())
    assert len(out) == 2
    assert sorted(out["season"]) == ["2021", "2122"]


def test_expand_never_includes_the_arrival_season_itself():
    out = pipeline.expand_per_season(_arrivals(), _seasons())
    assert "2223" not in out["season"].tolist()


def test_expand_keeps_arrival_metadata_on_every_row():
    out = pipeline.expand_per_season(_arrivals(), _seasons())
    assert set(out["to_team"]) == {"Napoli"}
    assert set(out["arrival_season"]) == {"2223"}


def test_expand_drops_arrivals_with_no_pre_move_seasons():
    seasons = _seasons()[_seasons()["season"] == "2223"]
    assert pipeline.expand_per_season(_arrivals(), seasons).empty


def test_assemble_produces_no_duplicate_rows():
    frame = pipeline.expand_per_season(_arrivals(), _seasons())
    assert not frame.duplicated(subset=["player", "arrival_season", "season"]).any()


# --- fix round 1: collapse a season split across two clubs --------------


def _split_season_arrivals():
    return pd.DataFrame({
        "player": ["Ounas"], "arrival_season": ["2223"], "to_team": ["Napoli"],
        "from_team": ["Cagliari"], "from_league": ["ITA-Serie A"], "prior_minutes": [1507.0],
    })


def _split_season_seasons():
    # Ounas played 2021 split across two clubs (Cagliari then Crotone) before
    # arriving at Napoli in 2223 -- both rows share season "2021".
    return pd.DataFrame({
        "player": ["Ounas", "Ounas", "Ounas"],
        "team": ["Cagliari", "Crotone", "Napoli"],
        "league": ["ITA-Serie A", "ITA-Serie A", "ITA-Serie A"],
        "season": ["2021", "2021", "2223"],
        "minutes": [308, 1199, 2000],
        "pos": ["MF", "MF", "MF"],
    })


def test_expand_collapses_a_split_season_to_the_higher_minutes_club():
    out = pipeline.expand_per_season(_split_season_arrivals(), _split_season_seasons())
    assert len(out) == 1
    row = out.iloc[0]
    assert row["team"] == "Crotone"
    assert row["minutes"] == 1199
    assert row["season_clubs"] == 2


def test_expand_records_season_clubs_as_one_for_a_normal_season():
    out = pipeline.expand_per_season(_arrivals(), _seasons())
    assert set(out["season_clubs"]) == {1}


def test_expand_split_season_tie_breaks_on_team_name():
    arrivals = _split_season_arrivals()
    seasons = _split_season_seasons()
    seasons.loc[seasons["team"] == "Crotone", "minutes"] = 308  # now tied
    out = pipeline.expand_per_season(arrivals, seasons)
    assert len(out) == 1
    # Tie broken alphabetically ascending: "Cagliari" < "Crotone".
    assert out.iloc[0]["team"] == "Cagliari"


def test_assert_unique_keys_raises_naming_a_sample_key():
    frame = pd.DataFrame({
        "player": ["A", "A"], "arrival_season": ["2223", "2223"],
        "season": ["2021", "2021"], "team": ["X", "Y"],
    })
    with pytest.raises(ValueError, match="A"):
        pipeline._assert_unique_keys(frame, ["player", "arrival_season", "season"])


def test_assert_unique_keys_passes_silently_when_unique():
    frame = pd.DataFrame({
        "player": ["A", "B"], "arrival_season": ["2223", "2223"], "season": ["2021", "2021"],
    })
    pipeline._assert_unique_keys(frame, ["player", "arrival_season", "season"])  # no raise


def test_assemble_raises_if_expand_per_season_is_bypassed_with_duplicates(monkeypatch):
    """`assemble`'s own guard must fire even if a caller (or a future bug)
    hands it a frame that skipped `expand_per_season`'s de-duplication."""
    duplicated = pd.DataFrame({
        "player": ["Ounas", "Ounas"],
        "arrival_season": ["2223", "2223"],
        "season": ["2021", "2021"],
        "team": ["Cagliari", "Crotone"],
        "to_team": ["Napoli", "Napoli"],
        "from_team": ["Cagliari", "Cagliari"],
        "from_league": ["ITA-Serie A", "ITA-Serie A"],
        "minutes": [308, 1199],
    })
    monkeypatch.setattr(pipeline, "expand_per_season", lambda arrivals, seasons: duplicated)
    with pytest.raises(ValueError, match="Ounas"):
        pipeline.assemble(
            _split_season_arrivals(), _split_season_seasons(), pd.DataFrame(),
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
        )


# --- controller ruling: tolerate missing Transfermarkt parquets ---------


def test_load_tm_returns_empty_frame_with_columns_when_file_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    columns = ["player", "season", "injury", "days_missed", "games_missed"]
    out = pipeline._load_tm("injuries.parquet", columns, "injury")
    assert out.empty
    assert list(out.columns) == columns


def test_load_tm_prints_a_warning_naming_the_file_and_bundle(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    pipeline._load_tm("market_values.parquet", ["player", "date", "value_eur"], "value")
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "market_values.parquet" in captured.out
    assert "value" in captured.out


def test_load_tm_reads_the_file_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    columns = ["team", "squad_value_eur"]
    frame = pd.DataFrame({"team": ["Napoli"], "squad_value_eur": [500e6]})
    frame.to_parquet(tmp_path / "club_values.parquet")
    out = pipeline._load_tm("club_values.parquet", columns, "retention")
    assert len(out) == 1
    assert out.iloc[0]["team"] == "Napoli"


# --- amendment: mixed-era reporting -------------------------------------


def _mixed_era_frame():
    # One pre-2324 row with real on-field data, one 2324+ row where some
    # source's columns never made it into the frame at all.
    return pd.DataFrame({
        "player": ["A", "B"],
        "season": ["2223", "2324"],
        "af_tackles": [45.0, float("nan")],
        "tklw": [10.0, float("nan")],
        "af_duels_won_pct": [0.55, float("nan")],
    })


def test_mixed_era_report_counts_and_shares_post_2324_rows():
    report = pipeline.mixed_era_report(_mixed_era_frame())
    assert report["count"] == 1
    assert report["total"] == 2
    assert report["share"] == pytest.approx(0.5)


def test_mixed_era_report_lists_entirely_absent_columns_for_post_2324_rows():
    report = pipeline.mixed_era_report(_mixed_era_frame())
    assert "af_tackles_p90_pct" in report["absent_columns"]
    assert "tklw_p90_pct" in report["absent_columns"]
    assert "af_duels_won_pct_pct" in report["absent_columns"]


def test_mixed_era_report_is_empty_when_no_post_2324_rows():
    frame = _mixed_era_frame()
    frame = frame[frame["season"] == "2223"]
    report = pipeline.mixed_era_report(frame)
    assert report["count"] == 0
    assert report["absent_columns"] == []


def test_mixed_era_report_handles_empty_frame():
    report = pipeline.mixed_era_report(pd.DataFrame())
    assert report == {"count": 0, "total": 0, "share": 0.0, "absent_columns": []}


# --- assemble: end-to-end join smoke test -------------------------------


def _full_arrivals():
    return pd.DataFrame({
        "player": ["A"], "arrival_season": ["2223"], "to_team": ["Napoli"],
        "from_team": ["Porto"], "from_league": ["POR-Liga"], "prior_minutes": [3800.0],
        "nationality": ["POR"],
    })


def _full_seasons():
    return pd.DataFrame({
        "player": ["A", "A", "A", "A"],
        "team": ["Porto", "Porto", "Napoli", "Napoli"],
        "league": ["POR-Liga", "POR-Liga", "ITA-Serie A", "ITA-Serie A"],
        "season": ["2021", "2122", "2223", "2324"],
        "minutes": [1800, 2000, 2200, 1500],
        "pos": ["MF", "MF", "MF", "MF"],
        "nationality": ["POR", "POR", "POR", "POR"],
        "team_matches": [30, 32, 38, 38],
        "playing_time_mp": [20, 22, 30, 20],
        "progression_prgp": [20.0, 25.0, 50.0, 30.0],
        "progression_prgc": [5.0, 6.0, 10.0, 7.0],
        "expected_npxg": [1.0, 1.2, 2.0, 1.5],
        "expected_xag": [1.0, 1.3, 3.0, 1.0],
    })


def _full_team_stats():
    return pd.DataFrame({
        "team": ["Napoli"],
        "league": ["ITA-Serie A"],
        "season": ["2223"],
        "possession": [55.0],
        "team_passes": [500.0],
        "team_deep": [8.0],
        "team_crs": [15.0],
        "team_press": [0.09],
        "team_npxg": [2.0],
    })


def test_assemble_joins_everything_without_losing_or_duplicating_rows():
    arrivals = _full_arrivals()
    seasons = _full_seasons()
    team_stats = _full_team_stats()
    squad_nationalities = (
        seasons.groupby(["team", "season", "nationality"]).size().reset_index(name="n")
    )
    injuries = pd.DataFrame(columns=["player", "season", "injury", "days_missed", "games_missed"])
    values = pd.DataFrame(columns=["player", "date", "value_eur"])
    club_values = pd.DataFrame({"team": ["Napoli"], "squad_value_eur": [500e6]})
    fees = pd.DataFrame(columns=["player", "season", "from_club", "to_club", "fee_eur", "is_loan"])

    frame = pipeline.assemble(arrivals, seasons, team_stats, squad_nationalities,
                               injuries, values, club_values, fees)

    # Two pre-move seasons (2021, 2122) for the sole arrival -> two rows.
    assert len(frame) == 2
    assert not frame.duplicated(subset=["player", "arrival_season", "season"]).any()
    for column in config.TARGETS:
        assert column in frame.columns
        assert frame[column].notna().all()
    for column in pipeline.PLAYER_FEATURE_COLUMNS:
        assert column in frame.columns
    for column in pipeline.CLUB_FEATURE_COLUMNS:
        assert column in frame.columns
        assert frame[column].notna().all()


# --- labelable_arrivals: exclude arrivals whose window isn't observable --


def test_labelable_arrivals_excludes_arrival_in_last_available_season():
    # arrival_season "2324" needs window ["2324", "2425"], but "2425" has
    # no Serie A rows loaded yet -> the window is not observable.
    arrivals = pd.DataFrame({"player": ["X"], "arrival_season": ["2324"]})
    seasons = pd.DataFrame({
        "player": ["X"], "team": ["Napoli"], "league": ["ITA-Serie A"],
        "season": ["2324"], "minutes": [2000], "team_matches": [38],
    })
    out = pipeline.labelable_arrivals(arrivals, seasons)
    assert out.empty


def test_labelable_arrivals_keeps_arrival_two_seasons_before_last_available():
    # arrival_season "2223" needs window ["2223", "2324"], and both seasons
    # have COMPLETE Serie A rows -> the window is fully observable.
    arrivals = pd.DataFrame({"player": ["X"], "arrival_season": ["2223"]})
    seasons = pd.DataFrame({
        "player": ["X", "X"], "team": ["Napoli", "Napoli"],
        "league": ["ITA-Serie A", "ITA-Serie A"],
        "season": ["2223", "2324"], "minutes": [2000, 2000],
        "team_matches": [38, 38],
    })
    out = pipeline.labelable_arrivals(arrivals, seasons)
    assert list(out["arrival_season"]) == ["2223"]


def test_labelable_arrivals_excludes_a_window_season_present_but_incomplete():
    # arrival_season "2223" needs window ["2223", "2324"]; "2324" is present
    # but only exported through matchday 13 -- not a real observed season.
    arrivals = pd.DataFrame({"player": ["X"], "arrival_season": ["2223"]})
    seasons = pd.DataFrame({
        "player": ["X", "X"], "team": ["Napoli", "Napoli"],
        "league": ["ITA-Serie A", "ITA-Serie A"],
        "season": ["2223", "2324"], "minutes": [2000, 400],
        "team_matches": [38, 13],
    })
    out = pipeline.labelable_arrivals(arrivals, seasons)
    assert out.empty


def test_labelable_arrivals_exclusion_count_is_correct():
    # A's window (2223, 2324) is fully observed and complete; B's window
    # (2324, 2425) is not (2425 hasn't arrived) -- only B should be excluded.
    arrivals = pd.DataFrame({
        "player": ["A", "B"], "arrival_season": ["2223", "2324"],
    })
    seasons = pd.DataFrame({
        "player": ["A", "A", "B"], "team": ["Napoli", "Napoli", "Napoli"],
        "league": ["ITA-Serie A", "ITA-Serie A", "ITA-Serie A"],
        "season": ["2223", "2324", "2324"], "minutes": [2000, 2000, 2000],
        "team_matches": [38, 38, 38],
    })
    out = pipeline.labelable_arrivals(arrivals, seasons)
    assert list(out["player"]) == ["A"]
    assert len(arrivals) - len(out) == 1


def test_assemble_prints_exclusion_count_and_latest_labelable_season(capsys):
    arrivals = pd.DataFrame({
        "player": ["A", "B"], "arrival_season": ["2223", "2324"],
        "to_team": ["Napoli", "Napoli"], "from_team": ["Porto", "Porto"],
        "from_league": ["POR-Liga", "POR-Liga"], "prior_minutes": [2000.0, 2000.0],
    })
    seasons = pd.DataFrame({
        "player": ["A", "A", "B"], "team": ["Napoli", "Napoli", "Napoli"],
        "league": ["ITA-Serie A", "ITA-Serie A", "ITA-Serie A"],
        "season": ["2223", "2324", "2324"], "minutes": [2000, 2000, 2000],
        "pos": ["MF", "MF", "MF"], "team_matches": [38, 38, 38],
    })
    pipeline.assemble(
        arrivals, seasons, pd.DataFrame(), pd.DataFrame(),
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
    )
    out = capsys.readouterr().out
    assert "1 arrival(s) excluded" in out
    assert "latest labelable arrival_season: 2223" in out


def test_assemble_excludes_unlabelable_arrival_from_model_frame():
    """Excluded arrivals (window not yet observable) must not reach the
    final model_frame -- even though they would otherwise pass through the
    old buggy path (0.0 retention on an empty end-of-window row)."""
    arrivals = pd.DataFrame({
        "player": ["A", "B"], "arrival_season": ["2223", "2324"],
        "to_team": ["Napoli", "Napoli"], "from_team": ["Porto", "Porto"],
        "from_league": ["POR-Liga", "POR-Liga"], "prior_minutes": [2000.0, 2000.0],
        "nationality": ["POR", "POR"],
    })
    seasons = pd.DataFrame({
        "player": ["A", "A", "A", "B", "B"],
        "team": ["Porto", "Napoli", "Napoli", "Porto", "Napoli"],
        "league": ["POR-Liga", "ITA-Serie A", "ITA-Serie A", "POR-Liga", "ITA-Serie A"],
        "season": ["2122", "2223", "2324", "2223", "2324"],
        "minutes": [2000, 2200, 1500, 2000, 1500],
        "pos": ["MF", "MF", "MF", "MF", "MF"],
        "nationality": ["POR", "POR", "POR", "POR", "POR"],
        "team_matches": [32, 38, 38, 38, 38],
        "playing_time_mp": [22, 30, 20, 30, 20],
        "progression_prgp": [20.0, 50.0, 30.0, 50.0, 30.0],
        "progression_prgc": [5.0, 10.0, 7.0, 10.0, 7.0],
        "expected_npxg": [1.0, 2.0, 1.5, 2.0, 1.5],
        "expected_xag": [1.0, 3.0, 1.0, 3.0, 1.0],
    })
    team_stats = pd.DataFrame({
        "team": ["Napoli", "Napoli"],
        "league": ["ITA-Serie A", "ITA-Serie A"],
        "season": ["2223", "2324"],
        "possession": [55.0, 55.0],
        "team_passes": [500.0, 500.0],
        "team_deep": [8.0, 8.0],
        "team_crs": [15.0, 15.0],
        "team_press": [0.09, 0.09],
        "team_npxg": [2.0, 2.0],
    })
    squad_nationalities = (
        seasons.groupby(["team", "season", "nationality"]).size().reset_index(name="n")
    )
    injuries = pd.DataFrame(columns=["player", "season", "injury", "days_missed", "games_missed"])
    values = pd.DataFrame(columns=["player", "date", "value_eur"])
    club_values = pd.DataFrame({"team": ["Napoli"], "squad_value_eur": [500e6]})
    fees = pd.DataFrame(columns=["player", "season", "from_club", "to_club", "fee_eur", "is_loan"])

    frame = pipeline.assemble(arrivals, seasons, team_stats, squad_nationalities,
                               injuries, values, club_values, fees)

    assert "B" not in set(frame["player"])
    assert "2324" not in set(frame["arrival_season"])
    assert set(frame["player"]) == {"A"}


# --- build_candidates: score any player with a recent season -------------


def _candidate_season_data(team_matches=38):
    # A: eligible (2200 min in "2223", plus a prior "2122" season so the
    #    trend/trajectory bundles have something to compute over).
    # B: below MIN_MINUTES_PRIOR_SEASON in "2223" -> must be excluded.
    return pd.DataFrame({
        "player": ["A", "A", "B"],
        "team": ["Napoli", "Napoli", "Roma"],
        "league": ["ITA-Serie A", "ITA-Serie A", "ITA-Serie A"],
        "season": ["2122", "2223", "2223"],
        "minutes": [2000, 2200, 100],
        "pos": ["MF", "MF", "MF"],
        "nationality": ["ITA", "ITA", "ITA"],
        "age": [23, 24, 20],
        "team_matches": [team_matches, team_matches, team_matches],
    })


def _fake_player_season_stats(data):
    def fn(league, season):
        if league == "ITA-Serie A" and season in set(data["season"]):
            return data[data["season"] == season].copy()
        return pd.DataFrame()
    return fn


def _patch_candidates_fbref(monkeypatch, data=None):
    data = _candidate_season_data() if data is None else data
    monkeypatch.setattr(pipeline.fbref, "player_season_stats", _fake_player_season_stats(data))
    monkeypatch.setattr(
        pipeline, "_load_tm",
        lambda name, columns, bundle=None: pd.DataFrame(columns=columns),
    )
    return data


def test_build_candidates_one_row_per_player_for_requested_season_only(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    _patch_candidates_fbref(monkeypatch)
    out = pipeline.build_candidates("2223")
    assert sorted(out["player"]) == ["A"]
    assert set(out["season"]) == {"2223"}


def test_build_candidates_features_present_and_nan_free(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    _patch_candidates_fbref(monkeypatch)
    out = pipeline.build_candidates("2223")
    for column in pipeline.PLAYER_FEATURE_COLUMNS:
        assert column in out.columns
    assert not out[pipeline.PLAYER_FEATURE_COLUMNS].isna().any().any()


def test_build_candidates_excludes_players_below_minutes_floor(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    _patch_candidates_fbref(monkeypatch)
    out = pipeline.build_candidates("2223")
    assert "B" not in set(out["player"])


def test_build_candidates_has_no_target_columns(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    _patch_candidates_fbref(monkeypatch)
    out = pipeline.build_candidates("2223")
    for column in config.TARGETS:
        assert column not in out.columns


def test_build_candidates_raises_on_incomplete_serie_a_season(monkeypatch, tmp_path):
    # A dump/scrape season exported mid-year (e.g. matchday 13) is too weak
    # a pre-move sample to build a candidate row from -- build_candidates
    # must refuse it outright rather than silently score a partial season.
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    _patch_candidates_fbref(monkeypatch, data=_candidate_season_data(team_matches=13))
    with pytest.raises(ValueError, match="not complete"):
        pipeline.build_candidates("2223")
