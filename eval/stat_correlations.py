"""How strongly each model input separates transfer hits from busts.

One row per arrival (player, arrival_season), using the inputs from the
season immediately before the move. A HIT beats the median arrival on the
target (for "overall", the query CLI's weighted score). For each input:

- correlation: Spearman rank correlation with hit (1) / bust (0), with a 95%
  interval from a player-level bootstrap (players with several arrivals are
  resampled together, so they are not counted as independent evidence);
- mutual information: scikit-learn's `mutual_info_classif`, which also picks
  up non-monotonic links, against a noise floor from shuffled labels.

Writes eval/stat_correlations.json and the interactive chart
eval/stat_correlations.html (the data inlined into
eval/stat_correlations_template.html).
Run: .venv/bin/python -m eval.stat_correlations
"""

import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif

from src import config, pipeline

warnings.filterwarnings("ignore")

WEIGHTS = {"minutes": 0.40, "output": 0.175, "output_share": 0.175,
           "retention": 0.25}  # query.DEFAULT_WEIGHTS
N_BOOT = 1000
N_SHUFFLE = 30
SEED = 0

LABELS = {
    "int_p90_pct": "Interceptions", "performance_crdy_p90_pct": "Yellow cards",
    "performance_fls_p90_pct": "Fouls committed", "performance_gls_p90_pct": "Goals",
    "performance_ast_p90_pct": "Assists", "standard_sh_p90_pct": "Shots",
    "standard_sot_p90_pct": "Shots on target", "tklw_p90_pct": "Tackles won",
    "crs_p90_pct": "Crosses", "fld_p90_pct": "Fouls drawn",
    "us_npxg_p90_pct": "Non-penalty xG", "us_xa_p90_pct": "Expected assists (xA)",
    "us_xgchain_p90_pct": "xG chain", "us_xgbuildup_p90_pct": "xG buildup",
    "us_key_passes_p90_pct": "Key passes (Understat)", "af_tackles_p90_pct": "Tackles",
    "af_blocks_p90_pct": "Blocks", "af_interceptions_p90_pct": "Interceptions (API-Football)",
    "af_duels_won_p90_pct": "Duels won", "af_dribbles_succ_p90_pct": "Successful dribbles",
    "af_key_passes_p90_pct": "Key passes (API-Football)", "af_duels_won_pct_pct": "Duel win %",
    "age_pct": "Age", "first_move_abroad": "First move abroad",
    "n_countries": "Countries played in", "language_romance": "Speaks a Romance language",
    "compatriots_in_squad": "Compatriots in new squad",
    "prior_seasons_on_record": "Seasons on record", "clubs_per_year": "Clubs per year",
    "bounced_between_clubs": "Bounced between clubs", "loan_count": "Loans",
    "lost_and_regained_place": "Lost and regained place", "career_minutes": "Career minutes",
    "injury_days_prior": "Days injured", "injury_spells_prior": "Injury spells",
    "injury_longest_absence": "Longest injury absence",
    "appearance_share": "Share of matches played", "fixtures_missed_total": "Matches missed",
    "fixtures_missed_while_fit": "Matches missed while fit",
    "value_at_move": "Market value at move", "value_trend_2y": "Market value trend (2y)",
    "possession_pct": "Club: possession", "team_passes_pct": "Club: passes per match",
    "team_deep_pct": "Club: deep completions", "team_crs_pct": "Club: crosses",
    "team_press_pct": "Club: pressing intensity", "team_npxg_pct": "Club: xG per match",
}


def group_of(column: str) -> str:
    if column in pipeline.CLUB_FEATURE_COLUMNS:
        return "Destination club"
    if column.endswith("_p90_pct") or column in ("af_duels_won_pct_pct",):
        return "On-field stats"
    return "Background"


def arrivals_frame() -> pd.DataFrame:
    frame = pd.read_parquet(config.PROCESSED_DIR / "model_frame.parquet")
    last = (frame.sort_values("season")
            .groupby(["player", "arrival_season"], as_index=False).tail(1)
            .reset_index(drop=True))
    last["overall"] = sum(last[t] * w for t, w in WEIGHTS.items())
    return last


def analyse(arrivals: pd.DataFrame, target: str, columns: list[str]) -> list[dict]:
    rng = np.random.default_rng(SEED)
    hit = (arrivals[target] > arrivals[target].median()).astype(int).to_numpy()
    x = arrivals[columns].to_numpy(dtype=float)

    # Player-level bootstrap: draw players, take all of each one's arrivals.
    players = arrivals["player"].to_numpy()
    groups = pd.Series(np.arange(len(arrivals))).groupby(players).apply(list).tolist()
    boot = np.empty((N_BOOT, len(columns)))
    for b in range(N_BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        for j in range(len(columns)):
            boot[b, j] = _spearman(x[idx, j], hit[idx])

    mi = mutual_info_classif(x, hit, random_state=SEED)
    noise = np.array([mutual_info_classif(x, rng.permutation(hit), random_state=SEED)
                      for _ in range(N_SHUFFLE)])
    # Per-stat floor (as the per-stat correlation intervals): what one stat
    # reaches by chance 5% of the time, pooled over every stat's shuffles.
    mi_floor = float(np.percentile(noise, 95))

    rows = []
    for j, column in enumerate(columns):
        low, high = np.percentile(boot[:, j], [2.5, 97.5])
        r = _spearman(x[:, j], hit)
        rows.append({
            "column": column, "label": LABELS.get(column, column), "group": group_of(column),
            "r": round(r, 4), "low": round(float(low), 4), "high": round(float(high), 4),
            "significant": bool(low > 0 or high < 0),
            "mi": round(float(mi[j]), 4), "mi_floor": round(mi_floor, 4),
            "hit_rate_top": round(float(hit[x[:, j] >= np.quantile(x[:, j], 0.75)].mean()), 3),
            "hit_rate_bottom": round(float(hit[x[:, j] <= np.quantile(x[:, j], 0.25)].mean()), 3),
        })
    return rows


def _spearman(a, b) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    value = spearmanr(a, b).statistic
    return 0.0 if np.isnan(value) else float(value)


if __name__ == "__main__":
    arrivals = arrivals_frame()
    columns = pipeline.PLAYER_FEATURE_COLUMNS + pipeline.CLUB_FEATURE_COLUMNS
    out = {"n_arrivals": len(arrivals), "n_players": int(arrivals["player"].nunique()),
           "targets": {}}
    for target in ["overall", *config.TARGETS]:
        out["targets"][target] = analyse(arrivals, target, columns)
        rows = sorted(out["targets"][target], key=lambda r: -abs(r["r"]))
        print(f"\n== {target}: strongest five")
        for r in rows[:5]:
            print(f"  {r['label']:<32} r={r['r']:+.3f} [{r['low']:+.3f}, {r['high']:+.3f}] MI={r['mi']:.3f}")
        print(f"  significant: {sum(r['significant'] for r in rows)}/{len(rows)}; "
              f"MI above noise: {sum(r['mi'] > r['mi_floor'] for r in rows)}")
    eval_dir = config.ROOT / "eval"
    (eval_dir / "stat_correlations.json").write_text(json.dumps(out, indent=1))
    template = (eval_dir / "stat_correlations_template.html").read_text()
    page = eval_dir / "stat_correlations.html"
    page.write_text(template.replace("__DATA__", json.dumps(out)))
    print(f"\nwritten {page}")
