from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from src import config, entities


def load_panel(path: Path | str) -> list[str]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    names: list[str] = []
    for tier in ("great", "ok", "bust"):
        names.extend(data.get(tier) or [])
    return names


def exclude_panel(frame: pd.DataFrame, panel: list[str]) -> pd.DataFrame:
    """Exclude all rows whose normalised player name is in the normalised panel."""
    norm_panel = set(entities.normalise_name(name) for name in panel)
    mask = frame["player"].map(entities.normalise_name).isin(norm_panel)
    return frame[~mask].reset_index(drop=True)


def panel_matches(frame: pd.DataFrame, panel: list[str]) -> tuple[list[str], list[str]]:
    """Return (panel names found in frame, panel names not found).

    Normalises both sides (lowercases, strips accents/punctuation, collapses whitespace)
    before comparing.
    """
    frame_norm_to_original = {
        entities.normalise_name(name): name
        for name in frame["player"].unique()
    }
    panel_norm_to_original = {
        entities.normalise_name(name): name
        for name in panel
    }

    found = []
    not_found = []
    for norm_name, orig_name in panel_norm_to_original.items():
        if norm_name in frame_norm_to_original:
            found.append(orig_name)
        else:
            not_found.append(orig_name)

    return found, not_found


def grouped_folds(frame: pd.DataFrame, n_splits: int = 5, seed: int = 0):
    """Splits grouped by player. Never split by row — see Global Constraints.

    n_splits is clamped to max(2, min(n_splits, n_distinct_players)): GroupKFold
    requires at least 2 folds and at most as many folds as distinct groups.
    seed is wired to GroupKFold(shuffle=True, random_state=seed) for reproducibility.
    """
    n_distinct_players = frame["player"].nunique()
    clamped_splits = max(2, min(n_splits, n_distinct_players))
    splitter = GroupKFold(n_splits=clamped_splits, shuffle=True, random_state=seed)
    return list(splitter.split(frame, groups=frame["player"]))


def fit_scaler(train_frame: pd.DataFrame, columns: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_frame[columns].to_numpy(dtype=np.float64))
    return scaler


def apply_scaler(frame: pd.DataFrame, columns: list[str], scaler: StandardScaler) -> pd.DataFrame:
    out = frame.copy()
    out[columns] = scaler.transform(out[columns].to_numpy(dtype=np.float64))
    return out


class TransferDataset(Dataset):
    """Torch view over a feature frame.

    Arrays are forced through `np.ascontiguousarray` because `to_numpy()` can
    hand back a non-writable view of pandas' own block memory; `torch.from_numpy`
    then warns on every batch ("The given NumPy array is not writable..."). An
    explicit contiguous float32 copy silences that and guarantees the tensors
    never alias the caller's frame.
    """

    def __init__(self, frame: pd.DataFrame, player_columns: list[str], club_columns: list[str]):
        self.player_x = np.ascontiguousarray(frame[player_columns].to_numpy(), dtype=np.float32)
        self.club_x = np.ascontiguousarray(frame[club_columns].to_numpy(), dtype=np.float32)
        self.y = np.ascontiguousarray(frame[list(config.TARGETS)].to_numpy(), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.player_x[index]),
            torch.from_numpy(self.club_x[index]),
            torch.from_numpy(self.y[index]),
        )
