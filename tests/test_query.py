import numpy as np
import pandas as pd
import pytest
import query
from src import config, features_onfield


def test_parse_weights_reads_the_documented_format():
    assert query.parse_weights("minutes=0.5,output=0.2,output_share=0.1,retention=0.2") == {
        "minutes": 0.5, "output": 0.2, "output_share": 0.1, "retention": 0.2,
    }


def test_parse_weights_defaults_when_empty():
    assert query.parse_weights("") == {"minutes": 0.40, "output": 0.175,
                                       "output_share": 0.175, "retention": 0.25}


def test_parse_weights_rejects_unknown_target():
    with pytest.raises(ValueError, match="unknown target"):
        query.parse_weights("vibes=1.0")


def test_parse_weights_rejects_weights_that_do_not_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        query.parse_weights("minutes=0.5,output=0.9,retention=0.2")


def test_combine_applies_weights_in_target_order():
    predictions = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    out = query.combine(predictions, {"minutes": 0.5, "output": 0.2, "output_share": 0.1,
                                      "retention": 0.2})
    assert out.tolist() == pytest.approx([0.5, 0.2, 0.1])


def test_bucket_thresholds():
    assert query.bucket(0.80) == "great"
    assert query.bucket(0.50) == "ok"
    assert query.bucket(0.20) == "bust"


def test_standout_features_returns_largest_positive_deviations():
    row = pd.Series({"a": 0.9, "b": 0.5, "c": 0.1})
    median = pd.Series({"a": 0.5, "b": 0.5, "c": 0.5})
    assert query.standout_features(row, median, ["a", "b", "c"], top_n=1) == ["a"]


def test_main_rejects_unknown_preset(capsys):
    code = query.main(["--preset", "tiki-taka"])
    assert code != 0
    assert "unknown preset" in capsys.readouterr().err.lower()


def test_main_requires_either_preset_or_club(capsys):
    code = query.main([])
    assert code != 0


def test_main_reports_missing_model_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(query, "MODEL_PATH", tmp_path / "ensemble.pt")
    code = query.main(["--preset", "high-press-possession"])
    assert code != 0
    assert "no trained model" in capsys.readouterr().err.lower()


def test_load_candidates_uses_candidates_parquet_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    candidates = pd.DataFrame({
        "player": ["Alice", "Bob"],
        "team": ["Milan", "Roma"],
        "league": ["Serie A", "Serie A"],
        "season": ["2425", "2425"],
        "age": [22, 25],
        "position_group": ["attacker", "midfielder"],
        "nationality": ["ITA", "BRA"],
    })
    candidates.to_parquet(tmp_path / "candidates.parquet")

    out = query.load_candidates()

    assert out["player"].tolist() == ["Alice", "Bob"]
    assert out["player"].is_unique


def test_load_candidates_falls_back_to_latest_season_per_player(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    model_frame = pd.DataFrame({
        "player": ["Alice", "Alice", "Bob"],
        "team": ["Milan", "Napoli", "Roma"],
        "season": ["2223", "2324", "2425"],
        "age": [21, 22, 25],
        "position_group": ["attacker", "attacker", "midfielder"],
    })
    model_frame.to_parquet(tmp_path / "model_frame.parquet")

    out = query.load_candidates()

    assert out["player"].is_unique
    alice = out[out["player"] == "Alice"].iloc[0]
    assert alice["team"] == "Napoli"  # the later, 2324 row, not 2223
    err = capsys.readouterr().err
    assert "candidates.parquet not found" in err


def test_standout_features_ignores_off_field_scale():
    onfield_columns = features_onfield.FEATURE_COLUMNS
    row = pd.Series({c: 0.5 for c in onfield_columns})
    row[onfield_columns[0]] = 0.95
    row["value_at_move"] = 50_000_000.0
    median = pd.Series({c: 0.5 for c in onfield_columns})
    median["value_at_move"] = 5_000_000.0

    result = query.standout_features(row, median, onfield_columns, top_n=3)

    assert "value_at_move" not in result
    assert onfield_columns[0] in result


def test_format_row_labels_last_club_and_appends_league():
    row = pd.Series({"player": "Alice", "team": "Milan", "league": "Serie A", "age": 22})
    text = query.format_row(1, row, [0.5] * 4, [0.1] * 4, 0.5, "unclustered", [])
    assert "last club: Milan (Serie A)" in text


def test_format_row_omits_league_when_absent():
    row = pd.Series({"player": "Alice", "team": "Milan", "age": 22})
    text = query.format_row(1, row, [0.5] * 4, [0.1] * 4, 0.5, "unclustered", [])
    club_line = next(line for line in text.splitlines() if "last club:" in line)
    assert club_line.strip() == "last club: Milan"


class _RecordingNet:
    """Stands in for a trained net so a test can see exactly what main() feeds
    the model — the whole point of finding #1 is what reaches the first layer."""

    def __init__(self):
        self.seen = []

    def __call__(self, player_x, club_x):
        import torch

        self.seen.append((player_x.numpy().copy(), club_x.numpy().copy()))
        return torch.zeros(len(player_x), len(config.TARGETS))


def _tiny_scalers(player_columns, club_columns):
    from sklearn.preprocessing import StandardScaler

    player_scaler = StandardScaler().fit(
        np.array([[0.0] * len(player_columns), [10.0] * len(player_columns)]))
    club_scaler = StandardScaler().fit(
        np.array([[0.0] * len(club_columns), [1.0] * len(club_columns)]))
    return player_scaler, club_scaler


def test_load_ensemble_round_trips_the_scalers(tmp_path, monkeypatch):
    """The scalers ride with the weights, or the CLI silently feeds raw
    features to a net trained on standardised ones (finding #1)."""
    import torch
    from src import model, train

    player_columns, club_columns = ["f1", "f2"], ["c1"]
    player_scaler, club_scaler = _tiny_scalers(player_columns, club_columns)
    nets = [model.TwoTowerNet(len(player_columns), len(club_columns))]
    path = tmp_path / "ensemble.pt"
    train.save_ensemble(nets, player_columns, club_columns,
                        player_scaler, club_scaler, path=path)
    monkeypatch.setattr(query, "MODEL_PATH", path)

    ensemble = query.load_ensemble()

    assert ensemble.player_columns == player_columns
    assert ensemble.club_columns == club_columns
    np.testing.assert_allclose(ensemble.player_scaler.mean_, player_scaler.mean_)
    np.testing.assert_allclose(ensemble.club_scaler.mean_, club_scaler.mean_)
    assert len(ensemble.nets) == 1
    assert not ensemble.nets[0].training


def test_load_ensemble_refuses_a_payload_without_scalers(tmp_path, monkeypatch):
    """An ensemble saved before input scaling would produce confidently wrong
    numbers rather than an obvious failure. Refuse it by name."""
    import torch
    from src import model

    path = tmp_path / "ensemble.pt"
    torch.save({"states": [model.TwoTowerNet(2, 1).state_dict()],
                "n_player": 2, "n_club": 1}, path)
    monkeypatch.setattr(query, "MODEL_PATH", path)

    with pytest.raises(ValueError, match="predates input scaling"):
        query.load_ensemble()


def _stub_main(monkeypatch, frame, player_columns, club_columns):
    """Wire main() up to a recording ensemble and no archetype models."""
    player_scaler, club_scaler = _tiny_scalers(player_columns, club_columns)
    recorder = _RecordingNet()
    monkeypatch.setattr(query, "load_candidates", lambda: frame)
    monkeypatch.setattr(query, "load_ensemble", lambda path=None: query.Ensemble(
        [recorder], player_scaler, club_scaler, player_columns, club_columns))
    monkeypatch.setattr(query.archetypes, "load_all", lambda: ({}, []))
    return recorder, player_scaler, club_scaler


def _candidate_frame(player_columns):
    frame = pd.DataFrame({
        "player": ["Alice", "Bob"],
        "team": ["Milan", "Roma"],
        "age": [22, 25],
        "position_group": ["attacker", "midfielder"],
    })
    for column in player_columns:
        frame[column] = [1.0, 7.0]
    for column in features_onfield.FEATURE_COLUMNS:
        frame[column] = 0.5
    return frame


def test_main_scales_player_and_club_inputs_before_inference(monkeypatch, capsys):
    """Finding #1: presets are hand-authored [0, 1] points and real club
    vectors are percentiles, but the net was trained on standardised inputs —
    both halves must go through their scaler."""
    from src import clubs

    player_columns = ["f1", "f2"]
    club_columns = list(clubs.VECTOR_COLUMNS)
    frame = _candidate_frame(player_columns)
    recorder, player_scaler, club_scaler = _stub_main(
        monkeypatch, frame, player_columns, club_columns)

    assert query.main(["--preset", "high-press-possession"]) == 0

    seen_player, seen_club = recorder.seen[0]
    expected_player = player_scaler.transform(
        frame[player_columns].to_numpy(dtype=np.float64))
    expected_club = club_scaler.transform(
        np.tile(query.presets.get("high-press-possession"), (len(frame), 1)).astype(np.float64))
    np.testing.assert_allclose(seen_player, expected_player, rtol=1e-5)
    np.testing.assert_allclose(seen_club, expected_club, rtol=1e-5)


def _clubs_frame():
    return pd.DataFrame({
        "team": ["Napoli", "Napoli"],
        "season": ["2223", "2324"],
        "league": ["ITA-Serie A", "ITA-Serie A"],
        **{c: [0.4, 0.9] for c in clubs_style_columns()},
    })


def clubs_style_columns():
    from src import clubs

    return list(clubs.STYLE_COLUMNS)


def test_main_club_defaults_to_the_latest_season_in_clubs_parquet(
        tmp_path, monkeypatch, capsys):
    """Finding #5: the old default was config.SEASONS[-1] ("2526"), a season
    with no completed team-stat table — so every --club query died on
    "no style vector"."""
    from src import clubs

    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    _clubs_frame().to_parquet(tmp_path / "clubs.parquet")

    player_columns = ["f1", "f2"]
    club_columns = list(clubs.VECTOR_COLUMNS)
    frame = _candidate_frame(player_columns)
    recorder, _, _ = _stub_main(monkeypatch, frame, player_columns, club_columns)

    assert query.main(["--club", "Napoli"]) == 0

    out = capsys.readouterr().out
    assert "season 2324" in out
    assert len(recorder.seen) == 1


def test_latest_club_season_rejects_an_empty_styles_frame():
    with pytest.raises(ValueError, match="no rows"):
        query.latest_club_season(pd.DataFrame(columns=["team", "season"]))


def test_standout_features_excludes_age():
    """Age is a model input but not a scouting insight, and it would otherwise
    crowd an actual metric out of a three-slot display (finding #7)."""
    columns = ["age_pct", "expected_npxg_p90_pct", "tackles_tkl_p90_pct"]
    row = pd.Series({"age_pct": 0.99, "expected_npxg_p90_pct": 0.8,
                     "tackles_tkl_p90_pct": 0.6})
    median = pd.Series({c: 0.5 for c in columns})

    result = query.standout_features(row, median, columns, top_n=2)

    assert "age_pct" not in result
    assert result == ["expected_npxg_p90_pct", "tackles_tkl_p90_pct"]


def test_resolve_model_path_default_is_the_main_ensemble():
    assert query.resolve_model_path(None) == query.MODEL_PATH


def test_resolve_model_path_tag_maps_to_tagged_ensemble(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    assert query.resolve_model_path("advanced") == tmp_path / "models" / "ensemble_advanced.pt"


def test_resolve_model_path_treats_slash_or_pt_suffix_as_a_path(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    assert query.resolve_model_path("models/custom.bin") == tmp_path / "models" / "custom.bin"
    assert query.resolve_model_path("foo.pt") == tmp_path / "foo.pt"
    absolute = tmp_path / "elsewhere" / "m.pt"
    assert query.resolve_model_path(str(absolute)) == absolute


def test_main_missing_tagged_model_names_the_path(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    code = query.main(["--preset", "high-press-possession", "--model", "advanced"])
    assert code != 0
    err = capsys.readouterr().err
    assert str(tmp_path / "models" / "ensemble_advanced.pt") in err


def test_main_default_model_path_is_unchanged(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(query, "MODEL_PATH", tmp_path / "ensemble.pt")
    code = query.main(["--preset", "high-press-possession"])
    assert code != 0
    assert str(tmp_path / "ensemble.pt") in capsys.readouterr().err


def test_main_prints_the_model_in_use_and_passes_the_path(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    player_columns = ["f1", "f2"]
    from src import clubs
    frame = _candidate_frame(player_columns)
    _stub_main(monkeypatch, frame, player_columns, list(clubs.VECTOR_COLUMNS))
    assert query.main(["--preset", "high-press-possession", "--model", "advanced"]) == 0
    assert f"using model: {tmp_path / 'models' / 'ensemble_advanced.pt'}" in capsys.readouterr().out


def test_main_exits_cleanly_when_candidate_frame_lacks_model_columns(monkeypatch, capsys):
    from src import clubs
    frame = _candidate_frame(["f1"])
    _stub_main(monkeypatch, frame, ["f1", "f2"], list(clubs.VECTOR_COLUMNS))
    code = query.main(["--preset", "high-press-possession"])
    assert code != 0
    assert "f2" in capsys.readouterr().err


# --- discipline penalty -----------------------------------------------------

def _discipline_row(crdy, fls):
    return pd.Series({"performance_crdy_p90_pct": crdy, "performance_fls_p90_pct": fls})


def test_discipline_columns_are_real_feature_columns():
    for column in query.DISCIPLINE_COLUMNS:
        assert column in features_onfield.FEATURE_COLUMNS


def test_discipline_penalty_is_zero_at_or_below_the_median():
    assert query.discipline_penalty(_discipline_row(0.5, 0.5), 0.05) == 0.0
    assert query.discipline_penalty(_discipline_row(0.1, 0.3), 0.05) == 0.0


def test_discipline_penalty_is_full_weight_at_the_top_percentile():
    assert query.discipline_penalty(_discipline_row(1.0, 1.0), 0.05) == pytest.approx(0.05)


def test_discipline_penalty_is_linear_in_between():
    # mean 0.75 -> halfway between median and top -> half the weight
    assert query.discipline_penalty(_discipline_row(0.7, 0.8), 0.05) == pytest.approx(0.025)
    assert query.discipline_penalty(_discipline_row(0.75, 0.75), 0.2) == pytest.approx(0.1)


def test_discipline_penalty_averages_only_available_columns():
    row = pd.Series({"performance_crdy_p90_pct": 1.0, "performance_fls_p90_pct": np.nan})
    assert query.discipline_penalty(row, 0.05) == pytest.approx(0.05)


def test_discipline_penalty_is_zero_for_nan_or_missing_columns():
    assert query.discipline_penalty(_discipline_row(np.nan, np.nan), 0.05) == 0.0
    assert query.discipline_penalty(pd.Series({"age": 22}), 0.05) == 0.0


def test_format_row_shows_the_penalty_only_when_material():
    row = pd.Series({"player": "Alice", "team": "Milan", "age": 22})
    shown = query.format_row(1, row, [0.5] * 4, [0.1] * 4, 0.5, "unclustered", [], penalty=0.03)
    assert "(discipline −0.03)" in shown
    hidden = query.format_row(1, row, [0.5] * 4, [0.1] * 4, 0.5, "unclustered", [], penalty=0.004)
    assert "discipline" not in hidden


def test_standout_features_never_include_cards_or_fouls():
    columns = list(features_onfield.FEATURE_COLUMNS)
    row = pd.Series({c: 0.5 for c in columns})
    for column in query.DISCIPLINE_COLUMNS:
        row[column] = 0.99
    row["us_npxg_p90_pct"] = 0.7
    median = pd.Series({c: 0.5 for c in columns})

    result = query.standout_features(row, median, columns, top_n=3)

    assert not set(result) & set(query.DISCIPLINE_COLUMNS)
    assert result[0] == "us_npxg_p90_pct"


class _FixedNet:
    """Returns the given per-player predictions in frame order."""

    def __init__(self, predictions):
        self.predictions = predictions

    def __call__(self, player_x, club_x):
        import torch

        return torch.tensor(self.predictions, dtype=torch.float32)


def _ranking(monkeypatch, capsys, extra_args):
    """Bob leads Alice by 0.02 on raw score but is among the most carded."""
    from src import clubs

    player_columns = ["f1", "f2"]
    frame = _candidate_frame(player_columns)
    frame["position_group"] = "attacker"
    frame.loc[1, "performance_crdy_p90_pct"] = 1.0
    frame.loc[1, "performance_fls_p90_pct"] = 1.0
    _stub_main(monkeypatch, frame, player_columns, list(clubs.VECTOR_COLUMNS))
    net = _FixedNet([[0.50] * 4, [0.52] * 4])
    monkeypatch.setattr(query, "load_ensemble", lambda path=None: query.Ensemble(
        [net], *_tiny_scalers(player_columns, list(clubs.VECTOR_COLUMNS)),
        player_columns, list(clubs.VECTOR_COLUMNS)))
    assert query.main(["--preset", "high-press-possession", *extra_args]) == 0
    out = capsys.readouterr().out
    return [line for line in out.splitlines() if line.strip()[:2] in ("1.", "2.")], out


def test_penalty_reorders_a_high_card_player_who_led_by_less_than_it(monkeypatch, capsys):
    lines, out = _ranking(monkeypatch, capsys, [])
    assert "Alice" in lines[0] and "Bob" in lines[1]
    assert "(discipline −0.05)" in lines[1]
    assert "discipline" not in lines[0]


def test_zero_penalty_reproduces_the_unpenalised_ranking(monkeypatch, capsys):
    lines, out = _ranking(monkeypatch, capsys, ["--discipline-penalty", "0"])
    assert "Bob" in lines[0] and "Alice" in lines[1]
    assert "discipline" not in out


@pytest.mark.parametrize("value", ["-0.1", "0.51", "1", "nan"])
def test_invalid_discipline_penalty_is_rejected_cleanly(value, capsys):
    code = query.main(["--preset", "high-press-possession", "--discipline-penalty", value])
    assert code == 2
    assert "--discipline-penalty" in capsys.readouterr().err


def test_score_is_clamped_to_the_unit_interval(monkeypatch, capsys):
    from src import clubs

    player_columns = ["f1", "f2"]
    frame = _candidate_frame(player_columns)
    frame["position_group"] = "attacker"
    frame[list(query.DISCIPLINE_COLUMNS)] = 1.0
    _stub_main(monkeypatch, frame, player_columns, list(clubs.VECTOR_COLUMNS))
    # _RecordingNet predicts zeros -> raw score 0, penalty must not push below 0
    assert query.main(["--preset", "high-press-possession", "--discipline-penalty", "0.5"]) == 0
    out = capsys.readouterr().out
    assert "score 0.00" in out
    assert "score -" not in out


def test_load_ensemble_rejects_a_model_trained_on_other_targets(tmp_path):
    import torch

    path = tmp_path / "old.pt"
    torch.save({"states": [], "n_player": 1, "n_club": 1, "player_scaler": None,
                "club_scaler": None, "player_columns": [], "club_columns": []}, path)
    with pytest.raises(ValueError, match="Retrain"):
        query.load_ensemble(path)
