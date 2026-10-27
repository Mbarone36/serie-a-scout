import numpy as np
import pandas as pd
import pytest
import torch
from src import train


def _frame(n=80, seed=0):
    rng = np.random.default_rng(seed)
    f1 = rng.random(n)
    return pd.DataFrame({
        "player": [f"p{i}" for i in range(n)],
        "f1": f1,
        "f2": rng.random(n),
        "club_a_pct": rng.random(n),
        "minutes": np.clip(f1 * 0.9 + rng.random(n) * 0.1, 0, 1),
        "output": np.clip(f1 * 0.6 + rng.random(n) * 0.4, 0, 1),
        "output_share": np.clip(f1 * 0.6 + rng.random(n) * 0.4, 0, 1),
        "retention": rng.choice([0.0, 0.33, 0.66, 1.0], n),
    })


PLAYER_COLS = ["f1", "f2"]
CLUB_COLS = ["club_a_pct"]


def test_train_one_returns_a_module_in_eval_mode():
    frame = _frame()
    net = train.train_one(frame[:60], frame[60:], PLAYER_COLS, CLUB_COLS, seed=0, max_epochs=20)
    assert isinstance(net, torch.nn.Module)
    assert not net.training


def test_predict_returns_mean_and_spread_per_target():
    frame = _frame()
    nets = [train.train_one(frame[:60], frame[60:], PLAYER_COLS, CLUB_COLS, s, max_epochs=10)
            for s in range(2)]
    mean, std = train.predict(nets, frame[60:], PLAYER_COLS, CLUB_COLS)
    assert mean.shape == (20, 4)
    assert std.shape == (20, 4)
    assert (std >= 0).all()


def test_training_reduces_loss_on_a_learnable_signal():
    frame = _frame(n=200)
    net = train.train_one(frame[:150], frame[150:], PLAYER_COLS, CLUB_COLS, seed=0, max_epochs=200)
    mean, _ = train.predict([net], frame[150:], PLAYER_COLS, CLUB_COLS)
    from src import baselines
    assert baselines.spearman(frame[150:]["minutes"].to_numpy(), mean[:, 0]) > 0.4


def test_cross_validate_reports_every_target():
    result = train.cross_validate(_frame(), PLAYER_COLS, CLUB_COLS, n_seeds=2, max_epochs=10)
    assert set(result) == {"minutes", "output", "output_share", "retention"}
    assert set(result["minutes"]) == {"spearman", "mae"}


def test_save_ensemble_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(train.config, "ROOT", tmp_path)
    frame = _frame()
    nets = [train.train_one(frame[:60], frame[60:], PLAYER_COLS, CLUB_COLS, 0, max_epochs=5)]
    player_scaler, club_scaler = train.fit_fold_scalers(frame, PLAYER_COLS, CLUB_COLS)
    train.save_ensemble(nets, PLAYER_COLS, CLUB_COLS, player_scaler, club_scaler)
    payload = torch.load(tmp_path / "models" / "ensemble.pt", weights_only=False)
    assert payload["n_player"] == 2
    assert payload["n_club"] == 1
    assert len(payload["states"]) == 1


def test_save_ensemble_persists_the_scalers_and_their_columns(tmp_path, monkeypatch):
    """Weights trained on standardised inputs are meaningless without the
    scalers that standardised them, and query.py cannot refit them — it never
    sees the training rows (finding #1)."""
    monkeypatch.setattr(train.config, "ROOT", tmp_path)
    frame = _frame()
    nets = [train.train_one(frame[:60], frame[60:], PLAYER_COLS, CLUB_COLS, 0, max_epochs=5)]
    player_scaler, club_scaler = train.fit_fold_scalers(frame, PLAYER_COLS, CLUB_COLS)
    train.save_ensemble(nets, PLAYER_COLS, CLUB_COLS, player_scaler, club_scaler)

    payload = torch.load(tmp_path / "models" / "ensemble.pt", weights_only=False)
    assert payload["player_columns"] == PLAYER_COLS
    assert payload["club_columns"] == CLUB_COLS
    np.testing.assert_allclose(payload["player_scaler"].mean_, player_scaler.mean_)
    np.testing.assert_allclose(payload["club_scaler"].mean_, club_scaler.mean_)


def _huge_scale_frame(n=120, seed=5):
    """A frame shaped like the real one: [0, 1] percentiles alongside a
    career_minutes-sized column. Unscaled, the first Linear saturates and the
    ReLU-terminated player tower emits all zeros."""
    rng = np.random.default_rng(seed)
    signal = rng.random(n)
    return pd.DataFrame({
        "player": [f"p{i}" for i in range(n)],
        "f1": rng.random(n),
        "career_minutes": signal * 15000.0,
        "club_a_pct": rng.random(n),
        "minutes": np.clip(signal * 0.9 + rng.random(n) * 0.1, 0, 1),
        "output": np.clip(signal * 0.6 + rng.random(n) * 0.4, 0, 1),
        "output_share": np.clip(signal * 0.6 + rng.random(n) * 0.4, 0, 1),
        "retention": rng.choice([0.0, 0.33, 0.66, 1.0], n),
    })


HUGE_PLAYER_COLS = ["f1", "career_minutes"]


def test_player_tower_is_not_dead_on_a_huge_scale_feature():
    """Finding #1's regression test. Trained on scaled inputs the way
    `cross_validate` now does it, the player tower must emit a non-zero
    embedding on the validation rows. In models/ensemble.pt, 8 of 10 seeds
    emitted exactly zero."""
    frame = _huge_scale_frame()
    player_scaler, club_scaler = train.fit_fold_scalers(frame, HUGE_PLAYER_COLS, CLUB_COLS)
    scaled = train.scale_frame(frame, HUGE_PLAYER_COLS, CLUB_COLS,
                                player_scaler, club_scaler)
    train_frame, val_frame = scaled[:90].reset_index(drop=True), scaled[90:].reset_index(drop=True)

    net = train.train_one(train_frame, val_frame, HUGE_PLAYER_COLS, CLUB_COLS,
                           seed=0, max_epochs=60)

    val = train.dataset.TransferDataset(val_frame, HUGE_PLAYER_COLS, CLUB_COLS)
    with torch.no_grad():
        embedding = net.player_tower(torch.from_numpy(val.player_x))
    assert embedding.abs().sum().item() > 0, "player tower emits an all-zero embedding"


def test_cross_validate_learns_from_a_huge_scale_player_feature():
    """End-to-end proof that `cross_validate` scales its inputs: the only
    signal in the frame lives on a career_minutes-sized column, so a run that
    fed raw features to the network could not rank on it at all."""
    result = train.cross_validate(_huge_scale_frame(), HUGE_PLAYER_COLS, CLUB_COLS,
                                   n_seeds=2, max_epochs=80)
    assert result["minutes"]["spearman"] > 0.5


def test_run_ablations_produces_one_row_per_bundle_plus_full():
    frame = _frame()
    rng = np.random.default_rng(1)
    frame["injury_days_prior"] = rng.random(len(frame))
    player_cols = PLAYER_COLS + ["injury_days_prior"]

    out = train.run_ablations(frame, player_cols, CLUB_COLS, n_seeds=1, max_epochs=5)

    assert "full" in out["variant"].tolist()
    no_rows = out[out["variant"] != "full"]
    assert len(no_rows) >= 1
    # a real ablation row must actually drop a feature, never match the full count
    assert not (no_rows["n_features"] == len(player_cols)).any()


def test_run_ablations_skips_bundles_with_no_overlapping_columns():
    """A bundle whose columns never appear in player_columns must not produce
    a spurious no_<bundle> row identical to full — that would just be wasted
    training reporting a fake ablation effect."""
    frame = _frame()
    out = train.run_ablations(frame, PLAYER_COLS, CLUB_COLS, n_seeds=1, max_epochs=5)
    assert out["variant"].tolist() == ["full"]


def _grouped_frame(n_players=20, rows_per_player=3, seed=2):
    rng = np.random.default_rng(seed)
    n = n_players * rows_per_player
    players = np.repeat([f"p{i}" for i in range(n_players)], rows_per_player)
    f1 = rng.random(n)
    return pd.DataFrame({
        "player": players,
        "f1": f1,
        "f2": rng.random(n),
        "club_a_pct": rng.random(n),
        "minutes": np.clip(f1 * 0.9 + rng.random(n) * 0.1, 0, 1),
        "output": np.clip(f1 * 0.6 + rng.random(n) * 0.4, 0, 1),
        "output_share": np.clip(f1 * 0.6 + rng.random(n) * 0.4, 0, 1),
        "retention": rng.choice([0.0, 0.33, 0.66, 1.0], n),
    })


def test_inner_split_is_grouped_by_player_not_by_row():
    frame = _grouped_frame()
    for seed in (0, 1):
        train_frame, val_frame = train._inner_split(frame, seed)
        train_players = set(train_frame["player"])
        val_players = set(val_frame["player"])
        assert train_players.isdisjoint(val_players)
        assert len(train_frame) > 0
        assert len(val_frame) > 0
        # every row of a player lands entirely on one side of the split
        assert set(frame["player"]) == train_players | val_players


def test_inner_split_falls_back_when_too_few_players():
    frame = _grouped_frame(n_players=1, rows_per_player=5)
    train_frame, val_frame = train._inner_split(frame, seed=0)
    assert train_frame.equals(frame)
    assert val_frame.equals(frame)


def test_early_stopping_restores_the_best_validation_state():
    """With patience=1, training stops after the first non-improving epoch.
    The returned net must be restored to the weights that achieved the best
    validation loss seen, not the weights from the final epoch."""
    frame = _frame(n=40)
    net = train.train_one(frame[:30], frame[30:], PLAYER_COLS, CLUB_COLS,
                           seed=0, max_epochs=50, patience=1)

    val = train.dataset.TransferDataset(frame[30:], PLAYER_COLS, CLUB_COLS)
    val_player = torch.from_numpy(val.player_x)
    val_club = torch.from_numpy(val.club_x)
    val_y = torch.from_numpy(val.y)
    with torch.no_grad():
        recomputed_loss = train.model.combined_loss(net(val_player, val_club), val_y).item()

    assert hasattr(net, "best_val_loss")
    assert recomputed_loss == pytest.approx(net.best_val_loss, abs=1e-5)


def test_select_panel_rows_matches_accented_names():
    """`_select_panel_rows` must use the same accent-insensitive matching as
    `dataset.exclude_panel`, so a panel player spelled with an accent in the
    data ("Rafael Leão") is still found via the unaccented panel spelling
    ("Rafael Leao") rather than silently dropped from the report."""
    frame = _frame(n=10)
    frame.loc[0, "player"] = "Rafael Leão"
    panel = ["Rafael Leao", "Someone Not In The Data"]

    held, found, not_found = train._select_panel_rows(frame, panel)

    assert "Rafael Leão" in held["player"].tolist()
    assert "Rafael Leao" in found
    assert "Someone Not In The Data" in not_found


def test_run_ablations_skips_a_constant_bundle_with_a_note(capsys):
    """A bundle that is constant across the frame has no signal to remove, so
    ablating it measures seed noise and reports it as a feature effect
    (finding #6). Skip it, and say so."""
    frame = _frame()
    frame["injury_days_prior"] = 0.0
    frame["injury_spells_prior"] = 0.0
    frame["injury_longest_absence"] = 0.0
    player_cols = PLAYER_COLS + ["injury_days_prior", "injury_spells_prior",
                                  "injury_longest_absence"]

    out = train.run_ablations(frame, player_cols, CLUB_COLS, n_seeds=1, max_epochs=5)

    assert "no_injury" not in out["variant"].tolist()
    assert "skipped: constant in this frame" in capsys.readouterr().out


def test_run_ablations_includes_the_on_field_bundle():
    """The spec lists on-field among the bundles to ablate; only the off-field
    ones were wired up (finding #6)."""
    from src import features_onfield

    frame = _frame()
    rng = np.random.default_rng(3)
    onfield = features_onfield.FEATURE_COLUMNS[:3]
    for column in onfield:
        frame[column] = rng.random(len(frame))
    player_cols = PLAYER_COLS + onfield

    out = train.run_ablations(frame, player_cols, CLUB_COLS, n_seeds=1, max_epochs=5)

    assert "no_onfield" in out["variant"].tolist()
    row = out[out["variant"] == "no_onfield"].iloc[0]
    assert row["n_features"] == len(PLAYER_COLS)


def test_run_ablations_prints_an_empirical_noise_floor(capsys):
    """Without a noise floor an ablation table invites reading a 0.01 spearman
    gap as evidence (finding #6)."""
    train.run_ablations(_frame(), PLAYER_COLS, CLUB_COLS, n_seeds=1, max_epochs=5)
    out = capsys.readouterr().out
    assert "noise floor" in out
    for target in ("minutes", "output", "output_share", "retention"):
        assert target in out


def test_report_runs_baselines_on_identical_inputs_and_a_club_only_control(monkeypatch):
    """Finding #2: the net sees player+club, so the headline baseline must too.
    The club-only block is the control that makes the marginal value of the
    player features explicit."""
    calls = []

    def recorder(frame, feature_columns, model_name):
        calls.append(list(feature_columns))
        return {t: {"spearman": 0.0, "mae": 0.0} for t in train.config.TARGETS}

    monkeypatch.setattr(train.baselines, "evaluate", recorder)
    monkeypatch.setattr(train.baselines, "MODELS", {"ridge": None})
    monkeypatch.setattr(train, "cross_validate", lambda *a, **k: {
        t: {"spearman": 0.0, "mae": 0.0} for t in train.config.TARGETS})
    monkeypatch.setattr(train, "run_ablations", lambda *a, **k: pd.DataFrame([{"variant": "full"}]))

    train.report(_frame(), PLAYER_COLS, CLUB_COLS, panel=[])

    assert calls == [PLAYER_COLS + CLUB_COLS, CLUB_COLS, PLAYER_COLS]


def _season_frame():
    return pd.DataFrame({
        "player": list("abcd"),
        "season": ["2021", "2122", "2223", "2122"],
        "arrival_season": ["2122", "2122", "2223", "2223"],
    })


def test_filter_frame_no_flags_keeps_everything():
    assert len(train.filter_frame(_season_frame())) == 4


def test_filter_frame_max_season():
    out = train.filter_frame(_season_frame(), max_season="2122")
    assert list(out["player"]) == ["a", "b", "d"]


def test_filter_frame_max_arrival():
    out = train.filter_frame(_season_frame(), max_arrival="2122")
    assert list(out["player"]) == ["a", "b"]


def test_filter_frame_both_flags_intersect():
    out = train.filter_frame(_season_frame(), max_season="2122", max_arrival="2122")
    assert list(out["player"]) == ["a", "b"]
    out = train.filter_frame(_season_frame(), max_season="2021", max_arrival="2223")
    assert list(out["player"]) == ["a"]


def _tiny_ensemble():
    frame = _frame()
    nets = [train.train_one(frame[:60], frame[60:], PLAYER_COLS, CLUB_COLS, 0, max_epochs=3)]
    ps, cs = train.fit_fold_scalers(frame, PLAYER_COLS, CLUB_COLS)
    return nets, ps, cs


def test_save_ensemble_honours_an_explicit_path(tmp_path, monkeypatch):
    monkeypatch.setattr(train.config, "ROOT", tmp_path / "root")
    nets, ps, cs = _tiny_ensemble()
    target = tmp_path / "elsewhere" / "sub" / "ensemble_x.pt"
    train.save_ensemble(nets, PLAYER_COLS, CLUB_COLS, ps, cs, path=target)
    payload = torch.load(target, weights_only=False)
    assert payload["n_player"] == 2 and payload["n_club"] == 1
    assert not (tmp_path / "root" / "models" / "ensemble.pt").exists()


def test_save_ensemble_default_path_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(train.config, "ROOT", tmp_path)
    nets, ps, cs = _tiny_ensemble()
    train.save_ensemble(nets, PLAYER_COLS, CLUB_COLS, ps, cs)
    assert (tmp_path / "models" / "ensemble.pt").exists()


def test_report_without_archetypes_skips_fit_all_and_uses_model_path(tmp_path, monkeypatch):
    from src import archetypes, features_onfield

    fit_calls = []
    monkeypatch.setattr(archetypes, "fit_all", lambda *a, **k: fit_calls.append(a))
    monkeypatch.setattr(features_onfield, "FEATURE_COLUMNS", [], raising=False)
    monkeypatch.setattr(train.config, "ENSEMBLE_SEEDS", 1)
    monkeypatch.setattr(train.baselines, "evaluate",
                        lambda *a, **k: {t: {"spearman": 0.0, "mae": 0.0} for t in train.config.TARGETS})
    monkeypatch.setattr(train.baselines, "MODELS", {"ridge": None})
    monkeypatch.setattr(train, "cross_validate", lambda *a, **k: {
        t: {"spearman": 0.0, "mae": 0.0} for t in train.config.TARGETS})
    monkeypatch.setattr(train, "run_ablations", lambda *a, **k: pd.DataFrame([{"variant": "full"}]))
    real_train_one = train.train_one
    monkeypatch.setattr(train, "train_one",
                        lambda *a, **k: real_train_one(*a, **{**k, "max_epochs": 3}))
    out_path = tmp_path / "models" / "ensemble_tag.pt"
    frame = _frame()
    # normalise_name strips digits, so give players letter-only names
    frame["player"] = [f"player {chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(len(frame))]
    train.report(frame, PLAYER_COLS, CLUB_COLS, panel=["player aa"],
                 model_path=out_path, fit_archetypes=False)
    assert out_path.exists()
    assert fit_calls == []


WEIGHTS = {"minutes": 0.40, "output": 0.175, "output_share": 0.175, "retention": 0.25}


def _arrival_frame(n_players=50, seed=0):
    """Two pre-move season rows per arrival, as the model frame has."""
    base = _frame(n=n_players, seed=seed).assign(arrival_season="2223")
    return pd.concat([base, base.assign(f2=base["f2"] * 0.5)], ignore_index=True)


def test_holdout_split_keeps_every_row_of_a_player_on_one_side():
    frame = _arrival_frame()
    train_frame, test_frame = train.holdout_split(frame, test_share=0.2, seed=0)
    assert not set(train_frame["player"]) & set(test_frame["player"])
    assert test_frame["player"].nunique() == 10
    assert len(train_frame) + len(test_frame) == len(frame)


def test_holdout_evaluate_reports_each_target_and_overall_per_arrival():
    summary, held = train.holdout_evaluate(_arrival_frame(), PLAYER_COLS, CLUB_COLS,
                                           WEIGHTS, n_seeds=2, max_epochs=10)
    assert list(summary["target"]) == ["minutes", "output", "output_share", "retention", "overall"]
    assert summary["accuracy"].between(0, 1).all()
    assert len(held) == 10  # one row per held-out arrival, not per season row
    assert summary.attrs["test_players"] == 10


def test_holdout_success_bar_comes_from_training_players_only():
    frame = _arrival_frame()
    train_frame, _ = train.holdout_split(frame, 0.2, 0)
    summary, _ = train.holdout_evaluate(frame, PLAYER_COLS, CLUB_COLS, WEIGHTS,
                                        n_seeds=1, max_epochs=5)
    bar = summary.set_index("target").loc["minutes", "success_bar"]
    assert bar == pytest.approx(train_frame.groupby("player")["minutes"].mean().median())


def test_holdout_calls_a_learnable_signal_better_than_chance():
    summary, _ = train.holdout_evaluate(_arrival_frame(n_players=200), PLAYER_COLS,
                                        CLUB_COLS, WEIGHTS, n_seeds=2, max_epochs=150)
    assert summary.set_index("target").loc["minutes", "accuracy"] > 0.7


def test_score_holdout_grades_any_models_predictions():
    frame = _arrival_frame()
    train_frame, test_frame = train.holdout_split(frame, 0.2, 0)
    perfect = test_frame[["minutes", "output", "output_share", "retention"]].to_numpy()
    summary, _ = train.score_holdout(train_frame, test_frame, perfect, WEIGHTS)
    assert (summary["accuracy"] == 1.0).all()


def test_train_one_accepts_architecture_and_optimiser_settings():
    frame = _frame()
    net = train.train_one(frame[:60], frame[60:], PLAYER_COLS, CLUB_COLS, seed=0,
                          max_epochs=5, hidden=16, embed=8, dropout=0.1,
                          lr=3e-3, weight_decay=0.1)
    assert net.player_tower[0].out_features == 16
