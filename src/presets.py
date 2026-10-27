import numpy as np

from src.clubs import STYLE_COLUMNS

# Ordering matches clubs.STYLE_COLUMNS exactly:
# possession, passing volume, deep completions (penetration), crossing,
# pressing intensity, npxG.
PRESETS: dict[str, np.ndarray] = {
    "high-press-possession": np.array([0.90, 0.90, 0.85, 0.50, 0.95, 0.85], dtype=np.float32),
    "low-block-counter":     np.array([0.15, 0.15, 0.30, 0.35, 0.10, 0.40], dtype=np.float32),
    "direct-vertical":       np.array([0.40, 0.35, 0.70, 0.55, 0.55, 0.65], dtype=np.float32),
    "positional-control":    np.array([0.95, 0.95, 0.80, 0.40, 0.60, 0.75], dtype=np.float32),
    "wide-overload":         np.array([0.60, 0.65, 0.60, 0.95, 0.60, 0.70], dtype=np.float32),
}

assert all(v.shape == (len(STYLE_COLUMNS),) for v in PRESETS.values())


def get(name: str) -> np.ndarray:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
    return PRESETS[name]
