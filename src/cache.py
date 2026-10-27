import os
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from src import config


class RateLimiter:
    """Blocks until `min_interval` seconds have passed since the last call."""

    def __init__(self, min_interval: float, now=time.monotonic, sleep=time.sleep):
        self.min_interval = min_interval
        self._now = now
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None:
            elapsed = self._now() - self._last
            remaining = self.min_interval - elapsed
            if remaining > 0:
                self._sleep(remaining)
        self._last = self._now()


def _path(key: str, suffix: str) -> Path:
    return config.RAW_DIR / f"{key}{suffix}"


def cached_frame(key: str, builder: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Return a cached DataFrame, building and storing it on first call."""
    path = _path(key, ".parquet")
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            # Cache file is corrupted; rebuild
            pass
    frame = builder()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(path) + ".tmp")
    frame.to_parquet(tmp_path)
    os.replace(tmp_path, path)
    return frame


def cached_text(key: str, builder: Callable[[], str]) -> str:
    """Return cached raw text (HTML), building and storing it on first call."""
    path = _path(key, ".html")
    if path.exists():
        return path.read_text(encoding="utf-8")
    text = builder()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(path) + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)
    return text
