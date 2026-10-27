import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from src import (baselines, config, dataset, entities, features_offfield,
                  features_onfield, model)


def fit_fold_scalers(train_frame, player_columns, club_columns):
    """Fit both input scalers on TRAIN rows only, never on val/test rows.

    Finding #1: the towers end in ReLU and the raw feature frame mixes [0, 1]
    percentiles with `career_minutes` (up to ~15,000), `fixtures_missed_total`
    (~166) and `compatriots_in_squad` (~24). Those magnitudes saturate the first
    linear layer, and in the shipped 10-seed ensemble 8 of 10 seeds emitted an
    all-zero `player_tower` embedding — the player half of the two-tower model
    was dead and only the club vector reached the heads. Standardising the
    inputs is the fix; fitting per outer fold on train rows only is what keeps
    it from leaking test statistics into training.
    """
    return (
        dataset.fit_scaler(train_frame, player_columns),
        dataset.fit_scaler(train_frame, club_columns),
    )


def scale_frame(frame, player_columns, club_columns, player_scaler, club_scaler):
    """Apply already-fitted player and club scalers to `frame`."""
    scaled = dataset.apply_scaler(frame, player_columns, player_scaler)
    return dataset.apply_scaler(scaled, club_columns, club_scaler)


def _inner_split(frame: pd.DataFrame, seed: int, val_share: float = 0.15):
    """Split a frame into (train, val) grouped by player, not by row.

    A positional `frame[:cut]` / `frame[cut:]` split can put some of a
    player's rows in train and others in val, so early stopping would
    validate on rows of a player it is also training on — the row-level
    leak the project's grouped-split constraint forbids. `GroupShuffleSplit`
    guarantees the two halves have disjoint player sets.

    Falls back to using the whole frame as both train and val when there
    are fewer than 2 distinct players — there is no way to form two
    disjoint non-empty player groups in that case, and this keeps callers
    (tiny synthetic frames in tests) from crashing.
    """
    players = frame["player"]
    if players.nunique() < 2:
        return frame, frame
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_share, random_state=seed)
    train_idx, val_idx = next(splitter.split(frame, groups=players))
    return (
        frame.iloc[train_idx].reset_index(drop=True),
        frame.iloc[val_idx].reset_index(drop=True),
    )



def _select_panel_rows(frame: pd.DataFrame, panel: list[str]):
    """Rows of `frame` belonging to panel players, matched accent-insensitively.

    `dataset.exclude_panel` already excludes panel players from training using
    `entities.normalise_name` on both sides (so "Rafael Leão" is correctly
    excluded even though the panel file spells it "Rafael Leao"). This mirrors
    that same normalisation for *selecting* the panel rows to display, via
    `dataset.panel_matches`, so a panel player with an accented name is not
    silently dropped from the withheld-panel report. Returns
    (held_rows, found_names, not_found_names) — found/not_found are as spelled
    in `panel`, per `dataset.panel_matches`.
    """
    found, not_found = dataset.panel_matches(frame, panel)
    norm_panel = {entities.normalise_name(name) for name in panel}
    held = frame[frame["player"].map(entities.normalise_name).isin(norm_panel)]
    return held, found, not_found


def train_one(train_frame, val_frame, player_columns, club_columns,
              seed: int, max_epochs: int = 300, patience: int = 25, *,
              hidden: int = 64, embed: int = 32, dropout: float = 0.3,
              lr: float = 1e-3, weight_decay: float = 1e-2):
    """Train a single seed with early stopping on the validation fold."""
    torch.manual_seed(seed)
    net = model.TwoTowerNet(len(player_columns), len(club_columns),
                            hidden=hidden, embed=embed, dropout=dropout)
    optimiser = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    loader = DataLoader(
        dataset.TransferDataset(train_frame, player_columns, club_columns),
        batch_size=32, shuffle=True,
    )
    val = dataset.TransferDataset(val_frame, player_columns, club_columns)
    val_player = torch.from_numpy(val.player_x)
    val_club = torch.from_numpy(val.club_x)
    val_y = torch.from_numpy(val.y)

    best_loss, best_state, stale = float("inf"), None, 0
    for _ in range(max_epochs):
        net.train()
        for player_x, club_x, y in loader:
            optimiser.zero_grad()
            model.combined_loss(net(player_x, club_x), y).backward()
            optimiser.step()

        net.eval()
        with torch.no_grad():
            loss = model.combined_loss(net(val_player, val_club), val_y).item()
        if loss < best_loss - 1e-5:
            best_loss, best_state, stale = loss, copy.deepcopy(net.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.best_val_loss = best_loss
    net.eval()
    return net


def predict(nets, frame, player_columns, club_columns):
    """Ensemble mean and per-seed spread. Spread is the uncertainty estimate."""
    ds = dataset.TransferDataset(frame, player_columns, club_columns)
    player_x = torch.from_numpy(ds.player_x)
    club_x = torch.from_numpy(ds.club_x)
    with torch.no_grad():
        stacked = np.stack([net(player_x, club_x).numpy() for net in nets])
    return stacked.mean(axis=0), stacked.std(axis=0)


def cross_validate(frame, player_columns, club_columns,
                    n_seeds: int = config.ENSEMBLE_SEEDS, max_epochs: int = 300,
                    seed_offset: int = 0) -> dict:
    """Player-grouped cross-validation, pooled out-of-fold predictions.

    Uses the same pooled spearman/mae aggregation as `baselines.evaluate` so
    the network and the baselines are compared on equal footing.

    Both feature blocks are standardised per outer fold (finding #1). The
    scalers are fitted on that fold's TRAIN rows and merely applied to its test
    rows, exactly as `baselines.evaluate` already does — anything else would
    leak test-set statistics into training.

    `seed_offset` shifts the ensemble's seed range, which `run_ablations` uses
    to re-run the full model under different initialisations and report an
    empirical noise floor.
    """
    predictions = np.zeros((len(frame), len(config.TARGETS)))
    for train_idx, test_idx in dataset.grouped_folds(frame):
        train_frame = frame.iloc[train_idx].reset_index(drop=True)
        test_frame = frame.iloc[test_idx].reset_index(drop=True)
        player_scaler, club_scaler = fit_fold_scalers(
            train_frame, player_columns, club_columns)
        train_scaled = scale_frame(train_frame, player_columns, club_columns,
                                    player_scaler, club_scaler)
        test_scaled = scale_frame(test_frame, player_columns, club_columns,
                                   player_scaler, club_scaler)
        nets = [
            train_one(*_inner_split(train_scaled, seed), player_columns,
                      club_columns, seed, max_epochs)
            for seed in range(seed_offset, seed_offset + n_seeds)
        ]
        mean, _ = predict(nets, test_scaled, player_columns, club_columns)
        predictions[test_idx] = mean

    return {
        target: {
            "spearman": baselines.spearman(frame[target].to_numpy(), predictions[:, i]),
            "mae": float(np.mean(np.abs(frame[target].to_numpy() - predictions[:, i]))),
        }
        for i, target in enumerate(config.TARGETS)
    }


def holdout_split(frame: pd.DataFrame, test_share: float = 0.2, seed: int = 0):
    """(train, test) with disjoint PLAYER sets: every row of a held-out
    player — all their arrivals and pre-move seasons — is in test."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_share, random_state=seed)
    train_idx, test_idx = next(splitter.split(frame, groups=frame["player"]))
    return (frame.iloc[train_idx].reset_index(drop=True),
            frame.iloc[test_idx].reset_index(drop=True))


def _per_arrival(frame: pd.DataFrame, predictions: np.ndarray | None = None) -> pd.DataFrame:
    """One row per (player, arrival_season): targets are constant within an
    arrival; its pre-move season rows' predictions are averaged."""
    keys = ["player", "arrival_season"]
    out = frame[keys + list(config.TARGETS)].copy()
    if predictions is not None:
        for i, target in enumerate(config.TARGETS):
            out[f"pred_{target}"] = predictions[:, i]
    return out.groupby(keys, as_index=False, sort=False).mean()


def holdout_evaluate(frame, player_columns, club_columns, weights: dict[str, float], *,
                     test_share: float = 0.2, seed: int = 0,
                     n_seeds: int = config.ENSEMBLE_SEEDS, max_epochs: int = 300,
                     **net_params):
    """Train on ~80% of players, predict the untouched ~20%.

    Scalers are fitted on the training players only; `net_params` are passed
    through to `train_one`. Scoring is `score_holdout`.

    Returns (summary DataFrame, per-arrival predictions DataFrame).
    """
    train_frame, test_frame = holdout_split(frame, test_share, seed)
    player_scaler, club_scaler = fit_fold_scalers(train_frame, player_columns, club_columns)
    train_scaled = scale_frame(train_frame, player_columns, club_columns,
                               player_scaler, club_scaler)
    test_scaled = scale_frame(test_frame, player_columns, club_columns,
                              player_scaler, club_scaler)
    nets = [train_one(*_inner_split(train_scaled, s), player_columns, club_columns,
                      s, max_epochs, **net_params)
            for s in range(n_seeds)]
    mean, _ = predict(nets, test_scaled, player_columns, club_columns)
    return score_holdout(train_frame, test_frame, mean, weights)


def score_holdout(train_frame, test_frame, predictions: np.ndarray,
                  weights: dict[str, float]):
    """Grade any model's row-level `predictions` for `test_frame`.

    A held-out arrival is called a success on a target when its value beats
    the TRAINING arrivals' median (so nothing about the held-out players sets
    the bar); the model's call is correct when its prediction lands on the
    same side. "overall" applies the same rule to the weighted score the
    query CLI ranks by.
    """
    train_arrivals = _per_arrival(train_frame)
    held = _per_arrival(test_frame, predictions)
    vector = np.array([weights[t] for t in config.TARGETS])
    train_arrivals["overall"] = train_arrivals[list(config.TARGETS)].to_numpy() @ vector
    held["overall"] = held[list(config.TARGETS)].to_numpy() @ vector
    held["pred_overall"] = held[[f"pred_{t}" for t in config.TARGETS]].to_numpy() @ vector

    rows = []
    for target in [*config.TARGETS, "overall"]:
        bar = float(train_arrivals[target].median())
        actual = held[target] > bar
        called = held[f"pred_{target}"] > bar
        held[f"{target}_success"] = actual
        held[f"{target}_called"] = called
        majority = bool((train_arrivals[target] > bar).mean() >= 0.5)
        rows.append({
            "target": target, "success_bar": bar,
            "accuracy": float((actual == called).mean()),
            "majority_baseline": float((actual == majority).mean()),
            "spearman": baselines.spearman(held[target].to_numpy(),
                                           held[f"pred_{target}"].to_numpy()),
            "mae": float(np.mean(np.abs(held[target] - held[f"pred_{target}"]))),
        })
    summary = pd.DataFrame(rows)
    summary.attrs.update(train_players=train_frame["player"].nunique(),
                         test_players=test_frame["player"].nunique(),
                         test_arrivals=len(held))
    return summary, held


def ablation_bundles() -> dict[str, list[str]]:
    """Every feature bundle the ablation sweep removes, on-field included.

    Finding #6: the spec lists on-field as one of the bundles to ablate, but
    only the off-field ones were wired up — so the sweep never answered whether
    the ~13 percentile-ranked performance features earn their place.
    """
    return {"onfield": list(features_onfield.FEATURE_COLUMNS),
            **features_offfield.BUNDLES}


def run_ablations(frame, player_columns, club_columns, n_seeds: int = 3,
                   max_epochs: int = 150, noise_seed_offset: int = 100) -> pd.DataFrame:
    """Full model versus each feature bundle removed.

    A bundle whose columns are all constant across `frame` carries no signal to
    remove, so ablating it measures run-to-run noise and nothing else. Those are
    skipped with a printed note rather than reported as a real effect.

    The sweep also prints an empirical noise floor: the full model re-run under
    a different seed offset, reported as |Δ spearman| per target. Any ablation
    gap smaller than that floor is indistinguishable from seed jitter.
    """
    rows = []
    variants = {"full": player_columns}
    for bundle, columns in ablation_bundles().items():
        present = [c for c in columns if c in player_columns]
        remaining = [c for c in player_columns if c not in columns]
        if not present or not remaining:
            continue
        if all(frame[c].nunique(dropna=False) <= 1 for c in present):
            print(f"ablation no_{bundle} skipped: constant in this frame")
            continue
        variants[f"no_{bundle}"] = remaining

    for name, columns in variants.items():
        result = cross_validate(frame, columns, club_columns, n_seeds, max_epochs)
        rows.append({"variant": name, "n_features": len(columns),
                     **{f"{t}_spearman": result[t]["spearman"] for t in config.TARGETS}})

    noise = cross_validate(frame, player_columns, club_columns, n_seeds,
                            max_epochs, seed_offset=noise_seed_offset)
    deltas = "  ".join(
        f"{t}=|{abs(noise[t]['spearman'] - rows[0][f'{t}_spearman']):.3f}|"
        for t in config.TARGETS
    )
    print(f"noise floor (full re-run at seed offset {noise_seed_offset}), "
          f"|Δ spearman| per target: {deltas}")
    print("ablation gaps below this floor are seed jitter, not evidence.")
    return pd.DataFrame(rows)


def save_ensemble(nets, player_columns: list[str], club_columns: list[str],
                   player_scaler, club_scaler, path=None) -> None:
    """Persist in the exact shape `query.load_ensemble` expects.

    The scalers travel with the weights (finding #1). A net trained on
    standardised inputs is meaningless when fed raw ones at inference, and the
    CLI has no way to reconstruct the fitted means/variances from the candidate
    pool — it does not see the training rows. The column lists are persisted
    alongside so the CLI applies each scaler to exactly the columns, in exactly
    the order, it was fitted on.
    """
    path = (config.ROOT / "models" / "ensemble.pt") if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"states": [net.state_dict() for net in nets],
         "n_player": len(player_columns), "n_club": len(club_columns),
         "player_scaler": player_scaler, "club_scaler": club_scaler,
         "player_columns": list(player_columns), "club_columns": list(club_columns),
         "targets": list(config.TARGETS)},
        path,
    )


def report(frame, player_columns, club_columns, panel: list[str], *,
           model_path=None, fit_archetypes: bool = True) -> None:
    """The full evaluation the spec requires, printed in one place."""
    labelled = dataset.exclude_panel(frame, panel)

    # Finding #2: the headline baseline block must see EXACTLY what the net
    # sees. Ridge on player features alone against a net on player+club is not
    # a comparison, it is two different problems. The club-only block is the
    # control that makes the marginal value of the player half explicit: if
    # club-only already matches player+club, the player tower is decoration.
    print("=== baselines: player + club (identical inputs to the network) ===")
    for name in baselines.MODELS:
        print(name, baselines.evaluate(labelled, player_columns + club_columns, name))

    print("\n=== baselines: club only (control — what the destination alone buys) ===")
    for name in baselines.MODELS:
        print(name, baselines.evaluate(labelled, club_columns, name))

    print("\n=== baselines: player only (control — what the player alone buys) ===")
    for name in baselines.MODELS:
        print(name, baselines.evaluate(labelled, player_columns, name))

    print("\n=== network ===")
    net_metrics = cross_validate(labelled, player_columns, club_columns)
    print(net_metrics)

    print("\n=== ablations ===")
    print(run_ablations(labelled, player_columns, club_columns).to_string(index=False))

    print("\n=== withheld panel (qualitative, not a metric) ===")
    held, found, not_found = _select_panel_rows(frame, panel)
    print(f"panel players found: {found}")
    print(f"panel players absent from the data: {not_found}")
    if held.empty:
        print("no panel players present in the dataset — check eval/panel.yaml names")
        return
    # The final ensemble's scalers are fitted on every labelled row (the panel
    # is already excluded from `labelled`), then applied to the withheld panel
    # rows before scoring them — never fitted on the rows being scored.
    player_scaler, club_scaler = fit_fold_scalers(labelled, player_columns, club_columns)
    labelled_scaled = scale_frame(labelled, player_columns, club_columns,
                                   player_scaler, club_scaler)
    held_scaled = scale_frame(held, player_columns, club_columns,
                               player_scaler, club_scaler)
    nets = [train_one(*_inner_split(labelled_scaled, s), player_columns, club_columns, s)
            for s in range(config.ENSEMBLE_SEEDS)]
    mean, std = predict(nets, held_scaled, player_columns, club_columns)
    for i, (_, row) in enumerate(held.iterrows()):
        print(f"{row['player']:<28} "
              + "  ".join(f"{t}={mean[i, j]:.2f}±{std[i, j]:.2f}"
                          for j, t in enumerate(config.TARGETS)))

    save_ensemble(nets, player_columns, club_columns, player_scaler, club_scaler,
                  path=model_path)
    saved_to = (config.ROOT / "models" / "ensemble.pt") if model_path is None else model_path
    print(f"\nensemble saved to {saved_to}")

    if not fit_archetypes:
        return

    try:
        from src import archetypes, features_onfield
    except ModuleNotFoundError as exc:
        print(f"\narchetype fitting skipped — module not available yet ({exc})")
        return

    archetypes.fit_all(frame, features_onfield.FEATURE_COLUMNS)
    print(f"\nensemble and archetypes saved under {config.ROOT / 'models'}")


def filter_frame(frame: pd.DataFrame, max_season=None, max_arrival=None) -> pd.DataFrame:
    """Keep rows whose pre-move `season` / `arrival_season` is <= the bound.

    Seasons are compared as strings ("2122" style codes sort correctly).
    """
    keep = pd.Series(True, index=frame.index)
    if max_season is not None:
        keep &= frame["season"].astype(str) <= str(max_season)
    if max_arrival is not None:
        keep &= frame["arrival_season"].astype(str) <= str(max_arrival)
    return frame[keep]


def _parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-season", help="keep rows with season <= this (e.g. 2122)")
    parser.add_argument("--max-arrival", help="keep rows with arrival_season <= this")
    parser.add_argument("--tag", help="save to models/ensemble_<tag>.pt, skip archetypes")
    parser.add_argument("--holdout", type=float, metavar="SHARE",
                        help="hold out this share of players (e.g. 0.2), train on the "
                             "rest, report success/flop calls on them, and exit")
    parser.add_argument("--holdout-seed", type=int, default=0,
                        help="which random player split --holdout uses (default 0)")
    return parser.parse_args(argv)


def print_holdout(summary: pd.DataFrame, held: pd.DataFrame) -> None:
    print(f"=== holdout: trained on {summary.attrs['train_players']} players, "
          f"tested on {summary.attrs['test_players']} unseen players "
          f"({summary.attrs['test_arrivals']} arrivals) ===")
    print("success = beats the training players' median; accuracy = share of "
          "held-out arrivals whose success/flop the model called right")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    from src import pipeline

    args = _parse_args()
    frame = pd.read_parquet(config.PROCESSED_DIR / "model_frame.parquet")
    total_rows = len(frame)
    total_players = frame["player"].nunique()
    frame = filter_frame(frame, args.max_season, args.max_arrival)
    if args.max_season is not None or args.max_arrival is not None or args.tag:
        print(f"filters: max_season={args.max_season} max_arrival={args.max_arrival}")
        print(f"kept {len(frame)}/{total_rows} rows, "
              f"{frame['player'].nunique()}/{total_players} players")
        if "arrival_season" in frame.columns:
            print(frame["arrival_season"].value_counts().sort_index())
        if "output_basis" in frame.columns:
            print(frame["output_basis"].value_counts())
    if args.holdout is not None:
        from query import DEFAULT_WEIGHTS

        labelled = dataset.exclude_panel(
            frame, dataset.load_panel(config.ROOT / "eval" / "panel.yaml"))
        summary, held = holdout_evaluate(
            labelled, pipeline.PLAYER_FEATURE_COLUMNS, pipeline.CLUB_FEATURE_COLUMNS,
            DEFAULT_WEIGHTS, test_share=args.holdout, seed=args.holdout_seed)
        print_holdout(summary, held)
        out_path = config.PROCESSED_DIR / f"holdout_predictions_seed{args.holdout_seed}.csv"
        held.sort_values("pred_overall", ascending=False).to_csv(out_path, index=False)
        print(f"\nper-player predictions written to {out_path}")
        raise SystemExit(0)
    report(
        frame,
        pipeline.PLAYER_FEATURE_COLUMNS,
        pipeline.CLUB_FEATURE_COLUMNS,
        dataset.load_panel(config.ROOT / "eval" / "panel.yaml"),
        model_path=(config.ROOT / "models" / f"ensemble_{args.tag}.pt") if args.tag else None,
        fit_archetypes=not args.tag,
    )
