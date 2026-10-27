import numpy as np
import pandas as pd
import pytest
from src import archetypes


def _clustered(n=120, seed=0):
    rng = np.random.default_rng(seed)
    half = n // 2
    return pd.DataFrame({
        "player": [f"p{i}" for i in range(n)],
        "position_group": ["attacker"] * n,
        "a": np.concatenate([rng.normal(0.2, 0.05, half), rng.normal(0.8, 0.05, n - half)]),
        "b": np.concatenate([rng.normal(0.8, 0.05, half), rng.normal(0.2, 0.05, n - half)]),
    })


def test_best_k_finds_the_planted_cluster_count():
    assert archetypes.best_k(_clustered(), ["a", "b"], k_range=(2, 5)) == 2


def test_assign_labels_every_row():
    frame = _clustered()
    models = {"attacker": archetypes.fit(frame, ["a", "b"], "attacker", k_range=(2, 4))}
    labels = archetypes.assign(frame, models, ["a", "b"])
    assert len(labels) == len(frame)
    assert labels.notna().all()


def test_assign_separates_the_two_planted_groups():
    frame = _clustered()
    models = {"attacker": archetypes.fit(frame, ["a", "b"], "attacker", k_range=(2, 4))}
    labels = archetypes.assign(frame, models, ["a", "b"])
    assert labels[:60].nunique() == 1
    assert labels[:60].iloc[0] != labels[60:].iloc[0]


def test_describe_names_the_dominant_features():
    frame = _clustered()
    mixture = archetypes.fit(frame, ["a", "b"], "attacker", k_range=(2, 4))
    described = archetypes.describe(mixture, ["a", "b"], top_n=1)
    assert len(described) == mixture.n_components
    assert all(len(d) == 1 for d in described)


def test_fit_all_round_trips_through_disk(tmp_path, monkeypatch):
    from src import config as cfg
    monkeypatch.setattr(cfg, "ROOT", tmp_path)
    frame = _clustered()
    archetypes.fit_all(frame, ["a", "b"])
    models, features = archetypes.load_all()
    assert "attacker" in models
    assert features == ["a", "b"]


def test_load_all_returns_empty_when_never_fitted(tmp_path, monkeypatch):
    from src import config as cfg
    monkeypatch.setattr(cfg, "ROOT", tmp_path)
    models, features = archetypes.load_all()
    assert models == {}
    assert features == []


def test_assign_handles_unknown_position_group():
    frame = _clustered()
    frame.loc[0, "position_group"] = "keeper"
    models = {"attacker": archetypes.fit(frame, ["a", "b"], "attacker", k_range=(2, 4))}
    labels = archetypes.assign(frame, models, ["a", "b"])
    assert labels.iloc[0] == "unclustered"
