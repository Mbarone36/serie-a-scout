# serie-a-scout

A PyTorch model that predicts how a footballer will fare after transferring to
Serie A, conditioned on the tactical style of the club he joins, plus a
command-line tool that ranks transfer candidates for any Serie A club or
tactical preset.

```
$ python query.py --preset high-press-possession --position attacker --max-age 25 --top 3
using model: models/ensemble.pt
  1. Vitinha                    age 25.0  score 0.69 (great) (discipline −0.01)
  2. Jonathan David             age 25.0  score 0.64 (ok)
  3. Rasmus Højlund             age 22.0  score 0.64 (ok)
```

## What it predicts

For every player who moved into Serie A, the model looks at what he did
*before* the move and predicts four outcomes over his first two Serie A
seasons:

| Target | Meaning |
| --- | --- |
| `minutes` | Share of available Serie A minutes played |
| `output` | Position-adjusted production: xG + xA for attackers, progressive and build-up involvement otherwise |
| `output_share` | The player's share of his team's output |
| `retention` | Whether he stayed: still at the club or sold up at a profit (1.0), moved within Serie A (0.66), sold sideways (0.33), released, loaned out or sold at a loss (0.0) |

The four predictions are kept separate. The CLI combines them at query time
with weights you choose (`--weights minutes=0.5,output=0.3,...`), so "will he
play" and "will he produce" are never baked into one label.

## Results

Evaluated on players the model never saw: an 80/20 split by player, so no
player appears in both training and test (`python -m src.train --holdout 0.2`,
seed 0: 122 held-out players, 162 transfers). "Accuracy" is how often the
model correctly calls a transfer above or below the training median.

| Target | Hit/bust accuracy | Majority-guess baseline | Spearman |
| --- | --- | --- | --- |
| Output | **72%** | 59% | 0.64 |
| Retention | 58% | 55% | 0.23 |
| Minutes | 53% | 52% | 0.17 |

Output is the strong result. Retention and minutes are only modestly better
than guessing: they depend on contracts, loans and managers, which the
features barely capture. These figures come from a single split; rerun with
`--holdout-seed 1, 2, …` to see the spread.

Across 5-fold player-grouped cross-validation (Spearman rank correlation):

| Model | Output | Output share | Retention | Minutes |
| --- | --- | --- | --- | --- |
| Network, 10-seed ensemble | 0.65 | **0.33** | 0.24 | 0.18 |
| Ridge regression | **0.66** | 0.32 | 0.19 | 0.18 |
| Gradient boosting | 0.64 | 0.32 | **0.27** | 0.18 |

On output a linear model is as good as the network, and gradient boosting
ranks retention slightly better; the network's clearest advantage is on
output share. Ablations show on-field and xG features drive output and
output share, injury history and market value drive minutes, and
availability and injuries matter most for retention.

## How it works

- **Data.** 672 Serie A arrivals from the Big 5 leagues across eight transfer
  windows (2018-19 to 2025-26). 2,398 labelled training rows from 613 players
  (one row per pre-move season). The current candidate pool holds 1,975
  players.
- **Features (41 per player + 6 club-style dimensions).** On-field production
  (goals, assists, shots, xG/xA, xGChain/xGBuildup, key passes, tackles,
  interceptions, blocks, duels, dribbles, crosses, fouls drawn, discipline,
  age), adaptation context (first move abroad, language, compatriots), career
  trajectory, injury history, availability and market value. Every stat is percentile-ranked within its own league, season and
  position before reaching the model, so a Bundesliga 0.6 xG and a Serie A
  0.6 xG are each judged against their own league.
- **Model.** A conditioned two-tower network: one tower embeds the player,
  one embeds the destination club's style (possession, passing, depth,
  crossing, pressing, chance quality). Their concatenation and elementwise
  product (the "fit" term) feed a shared head with four outputs. Trained as a
  10-seed ensemble; the spread across seeds is reported as uncertainty.
- **Evaluation discipline.** Splits are grouped by player. Features use only
  seasons strictly before the move. A transfer is labelled only once its
  two-season window is complete. Every result is compared against ridge and
  gradient-boosting baselines on identical inputs, plus "club only" and
  "player only" controls.

## Data sources

On-field and club inputs only use stats published for every season in scope
(2017-18 to 2025-26), so recent seasons are measured the same way as older
ones. FBref lost its Opta advanced stats in January 2026, which is why the
advanced columns come from other sources.

- **FBref:** basic player and team tables. 2017-18 to 2021-22 come from the
  [worldfootballR_data](https://github.com/JaseZiv/worldfootballR_data)
  export; later seasons from FBref's current pages.
- **Understat:** player xG, xA, xGChain and xGBuildup, and team npxG, PPDA and
  deep completions.
- **API-Football:** tackles, blocks, interceptions, duels, dribbles, key
  passes and passes. Needs your own API key.
- **Transfermarkt:** injury history, market values, transfer fees and club
  squad values.

No raw data or trained models are included in this repository. Everything is
rebuilt locally from the sources above; please respect each site's terms of
use and rate limits (all scrapers are rate-limited and cache every response).

## Setup

Python 3.12+ (developed on 3.13).

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch pandas numpy scipy scikit-learn pyarrow \
    soccerdata requests beautifulsoup4 lxml rapidfuzz pytest pyyaml joblib pyreadr
.venv/bin/pytest
```

Optionally identify the scrapers with a contact address:

```bash
export SCRAPER_CONTACT='you@example.com'
```

## Building the data

1. **FBref, historical:** download the worldfootballR Big-5 `.rds` files into
   `data/raw/dump/`.
2. **FBref, recent seasons:** FBref's pages sit behind a bot check, so save
   them from your own browser into `data/raw/soccerdata/` under soccerdata's
   cache filenames (`players_Big 5 European Leagues Combined_<season>_<table>.html`),
   then build them from that cache:

   ```bash
   .venv/bin/python -m src.scrape_fbref
   ```

3. **Arrivals:**

   ```bash
   .venv/bin/python -m src.transfers
   ```

4. **Understat:**

   ```bash
   .venv/bin/python -c "from src import understat, config; understat.pull_all(config.CANDIDATE_LEAGUES, config.SEASONS)"
   ```

5. **API-Football** (a paid plan finishes in one run; the free tier allows about 100 requests a day, and the pull resumes where it stopped):

   ```bash
   export API_FOOTBALL_KEY='your-key'
   .venv/bin/python -m src.api_football
   ```

6. **Transfermarkt:** players that can't be matched automatically are listed in
   `data/manual/unmatched_review.csv`; confirmed ids go in `data/manual/id_map.csv`.

   ```bash
   .venv/bin/python -m src.pull_tm
   ```

7. **Assemble the training set and the candidate pool:**

   ```bash
   .venv/bin/python -m src.pipeline
   .venv/bin/python -m src.pipeline candidates 2526
   ```

## Training and evaluating

```bash
.venv/bin/python -m src.train                    # 10-seed ensemble -> models/ensemble.pt
.venv/bin/python -m src.train --holdout 0.2      # hit/bust accuracy on unseen players
.venv/bin/python -m src.train --max-season 2122 --tag advanced   # train on a season subset
.venv/bin/python -m eval.stat_correlations       # per-stat hit/bust chart -> eval/stat_correlations.html
```

## Ranking candidates

```bash
.venv/bin/python query.py --preset high-press-possession --position attacker --top 10
.venv/bin/python query.py --club Atalanta --position midfielder --weights minutes=0.2,output=0.2,retention=0.6
```

- `--preset` or `--club`: a hand-authored tactical style, or a real club's measured style.
- `--weights`: how much each predicted outcome counts (must sum to 1).
- `--discipline-penalty`: a small deduction for high card and foul rates (default 0.05; 0 turns it off).
- `--model advanced`: score with an alternative saved ensemble.

## Project layout

```
src/            data sources, features, labels, model, training
query.py        candidate-ranking CLI
eval/           withheld-player panel and the per-stat correlation chart
tests/          pytest suite (fixtures only; no network access)
docs/           design spec and implementation plan
data/manual/    hand-curated Transfermarkt id map
```

## Limitations

- Retention and minutes are hard to predict from on-pitch data; treat those
  scores as weak signals.
- The labels measure outcomes, which mix player quality with injuries,
  managerial changes and luck.
- Only Big-5 source leagues are covered, so players arriving from other
  leagues (e.g. Kvaratskhelia from Georgia) are outside the model's reach.
- Player matching across sources is automatic (about 98–99% matched); a few
  nickname cases are mapped by hand.
