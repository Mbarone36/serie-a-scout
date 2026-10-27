import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


ARCHETYPE_NAMES: dict[str, str] = {}


def display_name(label: str) -> str:
    """Return human-friendly name for archetype label, falling back to raw label."""
    return ARCHETYPE_NAMES.get(label, label)


def best_k(frame: pd.DataFrame, feature_columns: list[str], k_range=(4, 7)) -> int:
    x = frame[feature_columns].to_numpy(dtype=np.float64)
    scores = {}
    for k in range(k_range[0], k_range[1] + 1):
        if k >= len(x):
            continue
        labels = GaussianMixture(n_components=k, random_state=0).fit_predict(x)
        if len(set(labels)) < 2:
            continue
        scores[k] = silhouette_score(x, labels)
    return max(scores, key=scores.get) if scores else k_range[0]


def fit(frame, feature_columns, position_group: str, k_range=(4, 7)) -> GaussianMixture:
    subset = frame[frame["position_group"] == position_group]
    k = best_k(subset, feature_columns, k_range)
    x = subset[feature_columns].to_numpy(dtype=np.float64)
    return GaussianMixture(n_components=k, random_state=0).fit(x)


def assign(frame, models: dict, feature_columns: list[str]) -> pd.Series:
    labels = pd.Series("unclustered", index=frame.index, dtype=object)
    for group, mixture in models.items():
        mask = frame["position_group"] == group
        if not mask.any():
            continue
        x = frame.loc[mask, feature_columns].to_numpy(dtype=np.float64)
        labels.loc[mask] = [f"{group}-{c}" for c in mixture.predict(x)]
    return labels


def fit_all(frame, feature_columns: list[str]) -> dict:
    """One mixture per position group, persisted for the CLI."""
    import joblib

    from src import config

    models = {}
    for group in frame["position_group"].unique():
        if group == "unknown" or (frame["position_group"] == group).sum() < 20:
            continue
        models[group] = fit(frame, feature_columns, group)
    path = config.ROOT / "models" / "archetypes.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "features": feature_columns}, path)
    return models


def load_all() -> tuple[dict, list[str]]:
    import joblib

    from src import config

    path = config.ROOT / "models" / "archetypes.joblib"
    if not path.exists():
        return {}, []
    payload = joblib.load(path)
    return payload["models"], payload["features"]


def describe(mixture: GaussianMixture, feature_columns: list[str], top_n: int = 4):
    """Features on which each cluster centre sits furthest above the mean —
    the raw material for naming clusters by hand."""
    overall = mixture.means_.mean(axis=0)
    out = []
    for centre in mixture.means_:
        deltas = centre - overall
        order = np.argsort(-deltas)[:top_n]
        out.append([feature_columns[i] for i in order])
    return out
