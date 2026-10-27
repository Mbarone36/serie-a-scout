import numpy as np
import pandas as pd

from src import normalize

# Raw counting stats, converted to per-90 before ranking.
#
# Every column here is reported for EVERY season in scope, so dump-era and
# live-era rows are ranked on the same footing. FBref's live pages (seasons
# "2324"+) dropped the expected/progression/possession/defense/aerial
# columns, so those are deliberately NOT features: an 80/20 player holdout
# over 10 splits showed dropping them costs no accuracy, while keeping them
# left every live-era candidate neutral-filled on 10 inputs. The live-era
# misc columns (`int`, `tklw`, `crs`, `fld`) reach their dump names via
# `fbref.harmonise_player_columns`.
#
# The `us_*` columns come from Understat (`src/understat.augment`) and the
# `af_*` columns from API-Football (`src/api_football.augment`); both cover
# every season and are minutes-proportional totals on each row, so per_90
# recovers the source's per-90 rate.
COUNTING_COLUMNS = (
    "int", "performance_crdy", "performance_fls",
    "performance_gls", "performance_ast", "standard_sh", "standard_sot",
    # Tackles won, crosses, fouls drawn: FBref "misc", both eras.
    "tklw", "crs", "fld",
    "us_npxg", "us_xa", "us_xgchain", "us_xgbuildup", "us_key_passes",
    "af_tackles", "af_blocks", "af_interceptions", "af_duels_won",
    "af_dribbles_succ", "af_key_passes",
)

# Already rates or shares; ranked directly.
RATE_COLUMNS = ("af_duels_won_pct",)

FEATURE_COLUMNS = [f"{c}_p90_pct" for c in COUNTING_COLUMNS] + [
    f"{c}_pct" for c in RATE_COLUMNS
] + ["age_pct"]


def build(seasons: pd.DataFrame) -> pd.DataFrame:
    """Percentile-ranked on-field features, one row per player-season."""
    out = seasons.copy()
    out["position_group"] = out["pos"].map(normalize.position_group)

    for column in COUNTING_COLUMNS + RATE_COLUMNS + ("age",):
        if column not in out.columns:
            out[column] = np.nan

    out = normalize.per_90(out, list(COUNTING_COLUMNS))
    to_rank = [f"{c}_p90" for c in COUNTING_COLUMNS] + list(RATE_COLUMNS) + ["age"]
    out = normalize.percentile_within(out, to_rank)

    # A metric a league never reported is not evidence of weakness — centre it.
    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].fillna(0.5)
    return out
