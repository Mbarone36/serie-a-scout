# Serie A Transfer Success Model — Design

**Date:** 2026-08-29
**Status:** Draft for review (rev 3)

## Problem

Predict how a player joining Serie A will fare there, conditioned on the tactical
identity of the club he joins. The output is a ranked shortlist: given a club
(real or a hand-authored tactical preset), a position, and optional filters,
return the players most likely to work out there.

Two capabilities in one model:

1. **How will he do?** — three separately predicted outcomes, not one score.
2. **Does he fit how we play?** — handled by conditioning the model on a
   destination-club style vector, so tactical presets are queries against the same
   model rather than a separate system.

## Data sources

**FBref** (via the `soccerdata` package, which wraps it with disk caching) for all
on-field performance, squad lists, and team season stats. **Transfermarkt** for
injury history and market values.

FBref was chosen over FotMob as the performance source because it carries
StatsBomb's advanced stats with real historical depth (2017-18 onward) in
season-aggregated tables: progressive passes and carries, take-ons,
press-resistance proxies. Those metrics are what make the style vectors
meaningful. FotMob is stronger on live match data and its own player ratings,
which would leave the model ranking players on goals plus a rating computed by
someone else.

Excluded deliberately: FIFA/sofifa ratings, which are subjective scout opinions
that EA revises upward after a player succeeds, leaking the outcome into the
features.

### Scraping

FBref rate-limits aggressively. One request per ~4 seconds, every response cached
to `data/raw/`, fully resumable. The initial pull takes a few hours and is done
once. Pulled:

- Serie A player season stats, 2017-18 through 2025-26, all six stat tables
  (standard, shooting, passing, possession, defense, misc, playing time)
- The same tables for every league players are sourced from
- Serie A **team** season stats (for club style vectors)
- Squad lists per club per season

From Transfermarkt: per-season injury records (days missed, spell count, longest
absence), player market-value histories, transfer fees, and club squad market
values. Same caching and rate-limiting discipline.

Advanced FBref stats begin in 2017-18 for the Big 5 and later for most secondary
leagues. Coverage of smaller leagues is thin and must be verified per league
before anything depends on it.

## Cohort

Serie A arrivals only, expanded per-player: **each pre-move season a player has on
record is its own training row**, and intra-Serie A transfers count as arrivals.
This yields roughly 900–1,200 rows.

Rows within a player are heavily correlated, so effective sample size is
substantially smaller than the row count — likely a few hundred independent
players. Two consequences, both binding:

- **All splits are grouped by player.** Never split by row. A player's 2019 season
  must never train while his 2021 season tests.
- **Ridge and gradient-boosting baselines are mandatory.** If the neural net does
  not beat them, that is the reported result and the simpler model ships.

### Identifying transfers

FBref publishes no transfer feed. Arrivals are derived by diffing squad lists: a
player in a club's season-S squad who was absent from its season S-1 squad is an
arrival. His pre-move season is wherever he appears in S-1. Transfer detection
therefore stays on a single source with nothing to reconcile.

## Features

### On-field (~40)

Per-90 metrics across the FBref tables: npxG, xA, shots, shot-on-target rate,
progressive passes, progressive carries, successful take-ons, touches in the
attacking penalty area, pass completion by distance band, key passes, crosses,
tackles plus interceptions, aerial win rate, attacking-third defensive actions,
miscontrols and dispossessions (press-resistance proxies), fouls, cards. Plus age
at transfer and minutes played.

Cards and fouls per 90 are retained as a behavioural temperament trace — a record
of what a player did, as opposed to a rating of what someone thought of him.

**Every metric is percentile-ranked within its own (league, season, position
group) before reaching the model.** Raw per-90 figures are not comparable across
leagues — 0.6 npxG/90 in the Eredivisie is not 0.6 npxG/90 in Serie A — and a
model fed raw values learns the league rather than the player. Source league also
carries an explicit strength coefficient.

Because each pre-move season is a row, year-over-year trend features are included
where a player has multiple seasons on record.

### Off-field — adaptation context (~6)

Derived entirely from FBref nationality and squad data:

- First move abroad (boolean)
- Number of countries played in
- Language-family match with Italian (Romance / Germanic / Slavic / other)
- Compatriots already present in the destination squad
- Continent of origin
- Age at first professional season

### Off-field — career trajectory (~5)

Falls out of the squad-diffing pipeline:

- Clubs per year to date
- Prior failed transfer (moved up a level, then back down)
- Loan history count
- Ever lost a starting place and regained it
- Career minutes to date

### Off-field — injury durability (~3)

From Transfermarkt, truncated at the transfer date: days missed per prior season,
number of distinct injury spells, and longest single absence.

Injury history is the most predictive off-field variable in the published
literature — fragile players fail transfers at a markedly higher rate.

### Off-field — availability and standing (~3)

From FBref squad membership versus matches actually played: share of the club's
fixtures a squad-listed player appeared in, and the longest run of consecutive
fixtures missed while squad-listed.

Crucially, crossing this against the injury records yields **fixtures missed while
fit** — a player who was available and still not selected. That is a standing /
out-of-favour signal, and it is a different thing from fragility. Neither source
produces it alone.

### Market value (~2)

Transfermarkt market value at the time of the move, and its trajectory over the
prior two years.

Value is used rather than transfer fee because it exists for every player at every
moment. A fee only exists once a transfer has happened, and the working product
scores players who have not moved. Value also gives the model a handle on the
minutes confound described under Known limitations.

Ablations report the model with and without this bundle: expensive players play
more partly on merit and partly self-fulfillingly, and it is worth knowing how
much of the model's skill rests on it.

### Entity resolution

FBref and Transfermarkt share no identifiers. Matching is by normalised name plus
birth date plus nationality, above a similarity threshold. Unmatched and ambiguous
players are written to a review file; confirmed matches are stored in a persistent
`data/manual/id_map.csv` that is never regenerated automatically.

This needs manual passes and is the fiddliest part of the build.

### Club style vector (~14)

Built from Serie A team season stats for the club and season of arrival:
possession share, average pass length (directness), progressive passing volume,
crossing volume, opponent passes per defensive action (pressing intensity), share
of defensive actions in the attacking third, shots from open play, team npxG,
prior-season league finish.

**Tactical presets are hand-authored points in this same vector space.** Querying
with a preset means feeding the model a synthetic club. Five to start:
`high-press-possession`, `low-block-counter`, `direct-vertical`,
`positional-control`, `wide-overload`.

## Targets

Measured over the **two seasons following arrival**. Two seasons gives slow
adapters a fair runway while keeping every arrival through 2023-24 in the labelled
cohort; 2024-25 and 2025-26 arrivals are unlabelled and form a natural holdout for
sanity-checking.

**Three targets are predicted separately. They are never combined into a single
training label.**

| Head | Target | Range | Loss |
| --- | --- | --- | --- |
| `minutes` | Share of available Serie A minutes played | [0, 1] | Huber |
| `output` | Position-adjusted production, percentile-ranked within position group — npxG+xA for attackers, progressive actions and duel win rate for midfielders and defenders | [0, 1] | Huber |
| `retention` | Four-rung ladder, see below | {0, 0.33, 0.66, 1} | BCE |

Minutes accrued at any Serie A club count toward `minutes`. Minutes outside Serie
A count as zero.

### The retention ladder and the step-up check

| Value | Outcome at the end of the window |
| --- | --- |
| 1.0 | Still at the signing club, **or sold to a club of equal-or-greater stature at a profit** |
| 0.66 | Transferred permanently to another Serie A club |
| 0.33 | Sold out of Serie A at roughly break-even to a smaller but credible club |
| 0.0 | Released, contract expired unrenewed, sold at a loss, or loaned out of Serie A |

A step up is judged by the destination club's total squad market value against the
selling club's at the time of the exit, confirmed by profit on the fee. Free
transfers out and unrenewed contracts score 0.0 regardless of destination.

**Why a step-up exit scores full marks:** Kvaratskhelia left Napoli for PSG in
January 2025. Scoring an exit as failure would label the single most celebrated
signing in the cohort a bust. A profitable sale to a bigger club is a signing that
worked, for both club and player, and the label has to say so.

### Why the targets are not combined

An earlier revision defined success as a single weighted composite (0.40 minutes /
0.35 output / 0.25 retention) and considered learning those weights from data.
That approach is circular: the weights define the label, so optimising them for
prediction accuracy searches for whichever definition of success is *easiest to
predict from pre-transfer data*, not the truest one. The optimum tends toward a
label that is near-constant or that collapses onto a feature already in the
inputs. Reported accuracy improves because the target moved, not because the model
learned anything.

Predicting the three components separately removes the circularity entirely: each
head is trained against a directly observed outcome.

### Weights at query time

Weighting happens at inference, under user control:

```
--weights minutes=0.5,output=0.3,retention=0.2
```

Defaults are 0.40 / 0.35 / 0.25 — a starting position, not a claim about truth.
Different clubs legitimately want different things: a title contender weights
`output`, a mid-table side weights `minutes`.

**Training weights and query weights are unrelated.** The three heads are weighted
equally in the training loss, each against its own observed data. Query weights
only combine already-trained predictions.

Bucketing into great / ok / bust happens at display time, from the weighted
combination.

### Known limitations

- **Minutes share is confounded by cost.** An expensive signing is played through
  poor form in a way a free transfer is not. Market value as a feature gives the
  model a handle on this, but it does not fully dissolve it: value and merit are
  entangled in the world, not only in the data.
- **The targets measure outcomes**, which bundle player quality with injury luck,
  managerial change, and squad circumstance. Unavoidable with outcome-based
  labels, and worth remembering when reading confident-looking rankings.
- **Injury records are inconsistently reported**, especially in smaller leagues,
  where minor absences often go unlogged. Absence of injury data is not evidence
  of durability, and the model cannot tell the difference.

## Leakage constraints

Enforced by construction, not convention:

- No post-arrival performance data in any feature.
- Injury records, availability and market value are truncated at the transfer
  date; only seasons strictly before the move contribute features.
- Fees and destination-club values feed the *label* only (the step-up check),
  never the features. They are post-move by nature, which is correct for a target
  and disqualifying for an input.
- Percentile normalisation is fitted within (league, season) using only that
  season's data, so no future-season information leaks backward.
- Withheld panel players are excluded from every fold, including validation.

## Architecture

Conditioned two-tower network with three heads:

```
player features (~59) ──► player MLP ──► 32-d ─┐                    ┌──► minutes
                                               ├──► concat +        ├──► output
club style vector (14) ──► club MLP   ──► 32-d ─┘    elementwise ───┤
                                                     product         └──► retention
                                                        │
                                                     shared head
```

The elementwise product term carries the player-by-club interactions, which is the
entire reason for conditioning on the club.

Three heads over a shared trunk is multi-task learning, which regularises: the
trunk must find a representation useful for all three outcomes, which is harder to
overfit than any one of them. At this sample size that is a real benefit, not just
tidiness.

Deliberately small: 64/32 hidden units, dropout 0.3, weight decay, early stopping
on a grouped validation fold. Total training loss is the unweighted sum of the
three head losses.

**Ensemble of 10 seeds, averaged.** At this sample size a single model's
predictions move meaningfully between runs. Spread across seeds is reported per
head as an uncertainty estimate; candidates the ensemble disagrees about are
flagged rather than ranked confidently.

## Evaluation

5-fold `GroupKFold`, grouped by player. Reported on every run:

- **Spearman rank correlation per head** — the headline metrics, since the product
  ranks candidates rather than predicting absolute values
- MAE per head; tier accuracy on the default-weighted combination
- **Baselines:** ridge regression and gradient boosting, per head, on identical
  folds
- **Feature-bundle ablations:** full model versus each of on-field, adaptation
  context, career trajectory, injury durability, availability, and market value
  removed. With ~73 features on a few hundred independent players, this
  establishes which bundles earn their place rather than assuming they do. Market
  value is the one to watch: if the model leans on it heavily, much of the
  apparent skill is really "expensive players get picked".
- **Withheld panel** — reported separately as a qualitative check, never mixed
  into headline metrics

### Withheld panel

`eval/panel.yaml` holds hand-picked players across three tiers (great / ok /
bust). **Every row belonging to these players is dropped from training entirely**,
not merely held back at test time. The panel is a smell test, not a measurement —
too small and too selection-biased to support a metric.

Note: Kvaratskhelia's pre-move season at Dinamo Batumi is almost certainly not in
FBref, as the Georgian Erovnuli Liga lacks advanced-stat coverage. His Rubin Kazan
seasons must be verified before he anchors the panel.

## Archetypes

Gaussian mixture over on-field style percentiles within each position group, 4–6
clusters per group, count chosen by silhouette score and named by hand from the
cluster centroids. Purely descriptive: archetypes label candidates in output and
do not feed the model.

## CLI

```
python query.py --preset high-press-possession --position winger --max-age 24 --top 20
python query.py --club atalanta --position striker --weights minutes=0.5,output=0.3,retention=0.2
python query.py --player "Name" --club napoli
```

`--preset` uses a hand-authored style vector; `--club` uses a real Serie A side's
actual vector. Per candidate the output gives: all three predicted values with
ensemble spread, the weighted score, tier, archetype, current club and league,
age, and the features where he most departs from the cohort median.

Showing the three heads separately is the point — "he will play but will not
produce" is more useful than any single number.

Inference is not restricted to players who already transferred: any player in a
covered league with a recent season can be scored against any club or preset
vector. That is the working product.

## Repository layout

```
serie-a-scout/
  data/raw/            cached source pulls, never re-scraped
  data/processed/      players.parquet, clubs.parquet, transfers.parquet
  data/manual/         id_map.csv, reviewed match overrides
  src/scrape.py        FBref + Transfermarkt pulls, cached and resumable
  src/entities.py      cross-source matching and the manual override map
  src/transfers.py     arrival detection by squad-diffing
  src/features.py      player and club vectors, within-league normalisation
  src/labels.py        the three targets, incl. the step-up check
  src/dataset.py       torch Dataset, player-grouped splits
  src/model.py         conditioned two-tower network, three heads
  src/baselines.py     ridge and gradient boosting
  src/archetypes.py    style clustering
  src/presets.py       tactical preset vectors
  src/train.py         training, ensembling, ablations
  query.py             CLI
  eval/panel.yaml      withheld players
```

## Build order

1. Scraping and caching; verify league coverage before anything depends on it
2. Entity resolution and the manual ID map
3. Transfer detection by squad-diffing
4. Feature construction and within-league normalisation
5. Target construction, including the step-up check
6. Baselines first — establishes the bar the network must clear
7. The conditioned network, ensembling, ablations
8. Archetypes
9. CLI

Baselines precede the network deliberately. They are cheap, and if they prove hard
to beat, that reshapes everything after step 6.

## Out of scope for v1

- Web UI
- Any third data source beyond FBref and Transfermarkt
- Leagues without FBref advanced-stat coverage
- Personal-life or psychometric data — it does not exist in usable form
