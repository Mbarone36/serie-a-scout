import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from src import config, dataset

MODELS = {
    "ridge": lambda: Ridge(alpha=10.0),
    "gbm": lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05),
}


def spearman(y_true, y_pred) -> float:
    """Rank correlation, with a constant prediction reported as 0 rather than NaN."""
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        return 0.0
    value = spearmanr(y_true, y_pred).statistic
    return 0.0 if np.isnan(value) else float(value)


def evaluate(frame: pd.DataFrame, feature_columns: list[str], model_name: str) -> dict:
    """Cross-validated per-target metrics on player-grouped folds."""
    predictions = {t: np.zeros(len(frame)) for t in config.TARGETS}
    for train_idx, test_idx in dataset.grouped_folds(frame):
        train, test = frame.iloc[train_idx], frame.iloc[test_idx]
        scaler = dataset.fit_scaler(train, feature_columns)
        x_train = scaler.transform(train[feature_columns].to_numpy(dtype=np.float64))
        x_test = scaler.transform(test[feature_columns].to_numpy(dtype=np.float64))
        for target in config.TARGETS:
            model = MODELS[model_name]()
            model.fit(x_train, train[target].to_numpy())
            predictions[target][test_idx] = model.predict(x_test)

    return {
        target: {
            "spearman": spearman(frame[target].to_numpy(), predictions[target]),
            "mae": float(np.mean(np.abs(frame[target].to_numpy() - predictions[target]))),
        }
        for target in config.TARGETS
    }
