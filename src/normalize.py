import numpy as np
import pandas as pd

from src import config

DEFAULT_GROUPING = ("league", "season", "position_group")


def position_group(pos: str) -> str:
    """FBref reports 'FW,MF'; the first listed position is the primary one."""
    primary = str(pos).split(",")[0].strip().upper()
    return config.POSITION_GROUPS.get(primary, "unknown")


def per_90(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    minutes = out["minutes"].replace(0, np.nan)
    for column in columns:
        out[f"{column}_p90"] = out[column] / minutes * 90.0
    return out


def percentile_within(
    frame: pd.DataFrame, columns: list[str], by=DEFAULT_GROUPING
) -> pd.DataFrame:
    """Rank each column to [0, 1] within its own group.

    A single-member group yields 0.5 — with nobody to compare against, the
    honest answer is 'average', not 'best in the league'.
    """
    out = frame.copy()
    grouped = out.groupby(list(by), dropna=False)
    for column in columns:
        ranked = grouped[column].rank(pct=True, na_option="keep")
        sizes = grouped[column].transform("count")
        out[f"{column}_pct"] = np.where(sizes <= 1, 0.5, ranked)
        out.loc[out[column].isna(), f"{column}_pct"] = np.nan
    return out
