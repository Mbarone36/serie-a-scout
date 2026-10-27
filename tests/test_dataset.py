import numpy as np
import pandas as pd
import pytest
import torch
from src import dataset


def _frame(n=20):
    return pd.DataFrame({
        "player": [f"p{i % 10}" for i in range(n)],
        "arrival_season": ["2223"] * n,
        "f1": np.linspace(0, 1, n),
        "f2": np.linspace(1, 0, n),
        "club_c1_pct": np.linspace(0, 1, n),
        "minutes": np.linspace(0, 1, n),
        "output": np.linspace(0, 1, n),
        "output_share": np.linspace(0, 1, n),
        "retention": [0.0, 0.33, 0.66, 1.0] * (n // 4),
    })


def test_load_panel_flattens_all_tiers(tmp_path):
    path = tmp_path / "panel.yaml"
    path.write_text("great:\n  - A\nok:\n  - B\nbust:\n  - C\n")
    assert dataset.load_panel(path) == ["A", "B", "C"]


def test_exclude_panel_removes_every_row_for_those_players():
    frame = _frame()
    out = dataset.exclude_panel(frame, ["p0", "p1"])
    assert not out["player"].isin(["p0", "p1"]).any()
    assert len(out) < len(frame)


def test_exclude_panel_matches_names_accent_insensitively():
    """Verify that names with accents in FBref data match unaccented panel names."""
    frame = pd.DataFrame({
        "player": ["Rafael Leão", "Lucas Paquetá", "Gonzalo Higuaín"],
        "f1": [0, 1, 2],
        "f2": [2, 1, 0],
        "club_c1_pct": [0, 1, 2],
        "minutes": [0, 1, 2],
        "output": [0, 1, 2],
        "output_share": [0, 1, 2],
        "retention": [0, 1, 0],
    })
    # Panel names without accents should match FBref names with accents
    out = dataset.exclude_panel(frame, ["Rafael Leao", "Lucas Paqueta"])
    assert len(out) == 1
    assert out["player"].iloc[0] == "Gonzalo Higuaín"


def test_panel_matches_returns_found_and_not_found():
    """Verify panel_matches correctly identifies which panel names exist in frame."""
    frame = pd.DataFrame({
        "player": ["Rafael Leão", "Lucas Paquetá"],
        "f1": [0, 1],
    })
    panel = ["Rafael Leao", "Lucas Paqueta", "Nonexistent Player"]
    found, not_found = dataset.panel_matches(frame, panel)
    assert set(found) == {"Rafael Leao", "Lucas Paqueta"}
    assert set(not_found) == {"Nonexistent Player"}


def test_grouped_folds_clamps_n_splits_to_n_distinct_players():
    """Verify grouped_folds clamps n_splits when fewer distinct players exist."""
    # Frame with only 3 distinct players
    frame = pd.DataFrame({
        "player": ["p0", "p1", "p2"] * 4,
        "arrival_season": ["2223"] * 12,
        "f1": np.linspace(0, 1, 12),
        "f2": np.linspace(1, 0, 12),
        "club_c1_pct": np.linspace(0, 1, 12),
        "minutes": np.linspace(0, 1, 12),
        "output": np.linspace(0, 1, 12),
        "output_share": np.linspace(0, 1, 12),
        "retention": [0.0, 0.5, 1.0] * 4,
    })
    # Request 5 folds but only 3 distinct players; should clamp to 3
    folds = dataset.grouped_folds(frame, n_splits=5)
    assert len(folds) == 3


def test_grouped_folds_never_split_a_player_across_train_and_test():
    frame = _frame()
    for train_idx, test_idx in dataset.grouped_folds(frame, n_splits=5):
        train_players = set(frame.iloc[train_idx]["player"])
        test_players = set(frame.iloc[test_idx]["player"])
        assert train_players.isdisjoint(test_players)


def test_grouped_folds_cover_every_row_exactly_once_across_folds():
    frame = _frame()
    seen = np.concatenate([test for _, test in dataset.grouped_folds(frame, n_splits=5)])
    assert sorted(seen.tolist()) == list(range(len(frame)))


def test_dataset_yields_correct_tensor_shapes():
    frame = _frame()
    ds = dataset.TransferDataset(frame, ["f1", "f2"], ["club_c1_pct"])
    player_x, club_x, y = ds[0]
    assert player_x.shape == (2,)
    assert club_x.shape == (1,)
    assert y.shape == (4,)
    assert player_x.dtype == torch.float32


def test_scaler_is_fitted_on_train_only():
    """Scaler fitted on train only preserves out-of-range test values; fit-on-all squashes them.

    This guards against data leakage: fitting a scaler on train+test combined lets test
    statistics inform training.
    """
    train = _frame(20)
    test = _frame(20)
    test["f1"] = test["f1"] * 100

    # Correct: fit on train only
    scaler_train_only = dataset.fit_scaler(train, ["f1", "f2"])
    scaled_train_only = dataset.apply_scaler(test, ["f1", "f2"], scaler_train_only)

    # Wrong: fit on train+test combined
    combined = pd.concat([train, test], ignore_index=True)
    scaler_leaky = dataset.fit_scaler(combined, ["f1", "f2"])
    scaled_leaky = dataset.apply_scaler(test, ["f1", "f2"], scaler_leaky)

    # Train-only scaling should preserve far-out-of-range values
    train_only_max = scaled_train_only["f1"].max()
    leaky_max = scaled_leaky["f1"].max()

    assert train_only_max > 50, f"Expected train_only to preserve large values, got {train_only_max}"
    assert leaky_max < train_only_max, f"Expected leaky to squash values more than train_only"
