import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch

from src import archetypes, clubs, config, features_onfield, model, presets

# Output's 35% is split evenly between raw production and share of the
# team's attack.
DEFAULT_WEIGHTS = {"minutes": 0.40, "output": 0.175, "output_share": 0.175,
                   "retention": 0.25}
MODEL_PATH = config.ROOT / "models" / "ensemble.pt"

# Yellow-card and foul rates (percentiles within league-season-position) are
# model inputs, but a transparent query-time penalty makes the cost visible in
# the final score. Zero at or below the median, `weight` at the 100th percentile.
DISCIPLINE_COLUMNS = ("performance_crdy_p90_pct", "performance_fls_p90_pct")
DEFAULT_DISCIPLINE_PENALTY = 0.05
MAX_DISCIPLINE_PENALTY = 0.5


def parse_weights(text: str) -> dict[str, float]:
    if not text:
        return dict(DEFAULT_WEIGHTS)
    weights = {}
    for part in text.split(","):
        name, _, value = part.partition("=")
        name = name.strip()
        if name not in config.TARGETS:
            raise ValueError(f"unknown target {name!r}; expected one of {config.TARGETS}")
        weights[name] = float(value)
    for target in config.TARGETS:
        weights.setdefault(target, 0.0)
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"weights must sum to 1, got {total}")
    return weights


def combine(predictions: np.ndarray, weights: dict[str, float]) -> np.ndarray:
    vector = np.array([weights[t] for t in config.TARGETS], dtype=np.float64)
    return predictions @ vector


def discipline_penalty(row, weight: float) -> float:
    """Linear penalty from the mean card/foul percentile; 0 when missing or <= median."""
    values = []
    for column in DISCIPLINE_COLUMNS:
        if column not in row.index:
            continue
        value = row[column]
        if pd.notna(value):
            values.append(float(value))
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return float(weight) * max(0.0, (mean - 0.5) / 0.5)


def bucket(score: float) -> str:
    if score >= 0.66:
        return "great"
    if score >= 0.33:
        return "ok"
    return "bust"


# Age is a real model input, but "he is older than the median" is not a scouting
# insight, and `age_pct` otherwise crowds out an actual performance metric in a
# three-slot display (finding #7). Cards and fouls are excluded too: they are
# already priced in by the discipline penalty and are not a strength to feature.
# Excluded from the display only.
STANDOUT_EXCLUDED = {"age_pct", *DISCIPLINE_COLUMNS}


def standout_features(row, cohort_median, feature_columns, top_n: int = 3) -> list[str]:
    deltas = {c: float(row[c]) - float(cohort_median[c])
              for c in feature_columns if c not in STANDOUT_EXCLUDED}
    return [c for c, _ in sorted(deltas.items(), key=lambda kv: -kv[1])[:top_n]]


def load_candidates() -> pd.DataFrame:
    """One row per player to rank.

    Prefers `data/processed/candidates.parquet` (the full eligible-player
    pool, built by `pipeline.build_candidates`). Falls back to
    `model_frame.parquet` — which is one row per (player, arrival, pre-move
    season), i.e. past arrivals only — deduped to each player's latest
    season so the fallback also yields one row per player.
    """
    candidates_path = config.PROCESSED_DIR / "candidates.parquet"
    if candidates_path.exists():
        return pd.read_parquet(candidates_path)
    frame = pd.read_parquet(config.PROCESSED_DIR / "model_frame.parquet")
    frame = (
        frame.sort_values("season")
        .groupby("player", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    print(
        "note: candidates.parquet not found — ranking past arrivals from "
        "model_frame (latest pre-move season per player); build it with: "
        ".venv/bin/python -m src.pipeline candidates <season>",
        file=sys.stderr,
    )
    return frame


class Ensemble(NamedTuple):
    """A trained ensemble plus everything needed to feed it correctly.

    The nets are trained on standardised inputs (finding #1), so the scalers
    fitted at training time are not optional metadata — feeding raw features to
    these weights produces garbage silently. The column lists come from the
    payload rather than from `pipeline` so a saved model that predates a feature
    change is fed the columns it was actually trained on.
    """

    nets: list
    player_scaler: object
    club_scaler: object
    player_columns: list[str]
    club_columns: list[str]


def resolve_model_path(value: str | None) -> Path:
    """Map `--model` to a file: a path if it has a "/" or ends in ".pt"
    (relative paths resolve from the project root), else a tag meaning
    `models/ensemble_<tag>.pt`. No value means the main ensemble."""
    if not value:
        return MODEL_PATH
    if "/" in value or value.endswith(".pt"):
        path = Path(value)
        return path if path.is_absolute() else config.ROOT / path
    return config.ROOT / "models" / f"ensemble_{value}.pt"


def load_ensemble(path: Path | None = None) -> Ensemble:
    path = Path(path) if path is not None else MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"no trained model at {path}. Run: .venv/bin/python -m src.train"
        )
    payload = torch.load(path, weights_only=False)
    missing = [k for k in ("player_scaler", "club_scaler",
                            "player_columns", "club_columns") if k not in payload]
    if missing:
        raise ValueError(
            f"{path} predates input scaling and is missing {missing}. "
            "Its predictions would be wrong rather than merely stale — retrain "
            "with: .venv/bin/python -m src.train"
        )
    targets = tuple(payload.get("targets", ("minutes", "output", "retention")))
    if targets != tuple(config.TARGETS):
        raise ValueError(
            f"{path} predicts {list(targets)}, but this version scores "
            f"{list(config.TARGETS)}. Retrain with: .venv/bin/python -m src.train"
        )
    nets = []
    for state in payload["states"]:
        net = model.TwoTowerNet(payload["n_player"], payload["n_club"])
        net.load_state_dict(state)
        net.eval()
        nets.append(net)
    return Ensemble(nets, payload["player_scaler"], payload["club_scaler"],
                    list(payload["player_columns"]), list(payload["club_columns"]))


def latest_club_season(styles: pd.DataFrame) -> str:
    """The most recent season with real club style vectors on disk.

    Finding #5: `--season` used to default to `config.SEASONS[-1]` ("2526"),
    which is a season the pipeline pulls player rows for but has no completed
    team-stat table for — so every `--club` query died on "no style vector".
    The default has to come from the data, not the calendar.
    """
    if styles.empty:
        raise ValueError("clubs.parquet has no rows — run: python -m src.pipeline")
    return str(styles["season"].astype(str).max())


def format_row(rank, row, predictions, spread, score, archetype, standouts,
               penalty: float = 0.0) -> str:
    per_head = "  ".join(
        f"{t}={predictions[i]:.2f}±{spread[i]:.2f}" for i, t in enumerate(config.TARGETS)
    )
    club_label = f"last club: {row.get('team', '')}"
    if "league" in row.index and pd.notna(row.get("league")):
        club_label += f" ({row['league']})"
    penalty_note = f" (discipline \u2212{penalty:.2f})" if penalty > 0.005 else ""
    return (
        f"{rank:>3}. {row['player']:<26} "
        f"age {row.get('age', '?'):>2}  score {score:.2f} ({bucket(score)}){penalty_note}\n"
        f"      {club_label}\n"
        f"      {per_head}\n"
        f"      archetype: {archetype}   standout: {', '.join(standouts)}"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Rank Serie A transfer candidates.")
    parser.add_argument("--preset", help=f"one of: {sorted(presets.PRESETS)}")
    parser.add_argument("--club", help="a real Serie A club name")
    parser.add_argument("--season", default=None,
                         help="defaults to the latest season present in clubs.parquet")
    parser.add_argument("--position", help="attacker, midfielder, defender, keeper")
    parser.add_argument("--max-age", type=int)
    parser.add_argument("--weights", default="")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--model", default=None, metavar="NAME_OR_PATH",
                         help="saved ensemble: a tag (advanced -> models/ensemble_advanced.pt) "
                              "or a path (contains '/' or ends in .pt); default: the main ensemble")
    parser.add_argument("--discipline-penalty", type=float, default=DEFAULT_DISCIPLINE_PENALTY,
                         metavar="FLOAT",
                         help="max score penalty for high yellow-card/foul rates "
                              f"(0 disables; 0-{MAX_DISCIPLINE_PENALTY}; default "
                              f"{DEFAULT_DISCIPLINE_PENALTY})")
    args = parser.parse_args(argv)

    if not 0.0 <= args.discipline_penalty <= MAX_DISCIPLINE_PENALTY:
        print(f"error: --discipline-penalty must be between 0 and "
              f"{MAX_DISCIPLINE_PENALTY}, got {args.discipline_penalty}", file=sys.stderr)
        return 2

    if not args.preset and not args.club:
        print("error: one of --preset or --club is required", file=sys.stderr)
        return 2

    try:
        weights = parse_weights(args.weights)
        frame = load_candidates()
        if args.preset:
            club_vector = presets.get(args.preset)
        else:
            styles = clubs.build_style_vectors(
                pd.read_parquet(config.PROCESSED_DIR / "clubs.parquet")
            )
            season = args.season or latest_club_season(styles)
            print(f"using {args.club} style vector for season {season}")
            club_vector = clubs.vector_for(styles, args.club, season)
        model_path = resolve_model_path(args.model)
        ensemble = load_ensemble(model_path)
        absent = (
            [c for c in ensemble.player_columns if c not in frame.columns]
            + [c for c in ensemble.club_columns if c not in clubs.VECTOR_COLUMNS]
        )
        if absent:
            raise ValueError(
                f"{model_path} needs input columns that are not available: "
                f"{absent}. Rebuild the data or use a compatible model."
            )
    except (KeyError, ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"using model: {model_path}")

    if args.position:
        frame = frame[frame["position_group"] == args.position]
    if args.max_age:
        frame = frame[frame["age"] <= args.max_age]
    if frame.empty:
        print("no candidates match those filters", file=sys.stderr)
        return 1

    # The nets were trained on standardised inputs, so raw features (and raw
    # club vectors — presets and real percentile vectors alike) must go through
    # the scalers that were fitted at training time before inference.
    player_x = torch.tensor(
        ensemble.player_scaler.transform(
            frame[ensemble.player_columns].to_numpy(dtype=np.float64)
        ).astype(np.float32)
    )
    club_x = torch.tensor(
        ensemble.club_scaler.transform(
            np.tile(club_vector, (len(frame), 1)).astype(np.float64)
        ).astype(np.float32)
    )
    with torch.no_grad():
        stacked = np.stack([net(player_x, club_x).numpy() for net in ensemble.nets])
    mean, spread = stacked.mean(axis=0), stacked.std(axis=0)

    penalties = np.array([discipline_penalty(frame.iloc[i], args.discipline_penalty)
                          for i in range(len(frame))], dtype=np.float64)
    scores = np.clip(combine(mean, weights) - penalties, 0.0, 1.0)
    order = np.argsort(-scores)[: args.top]
    median = frame[features_onfield.FEATURE_COLUMNS].median()
    models, archetype_features = archetypes.load_all()
    labels = (
        archetypes.assign(frame, models, archetype_features)
        if models else pd.Series("unclustered", index=frame.index)
    )

    for rank, i in enumerate(order, start=1):
        row = frame.iloc[i]
        print(format_row(
            rank, row, mean[i], spread[i], scores[i],
            archetypes.display_name(labels.iloc[i]),
            standout_features(row, median, features_onfield.FEATURE_COLUMNS),
            penalties[i],
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
