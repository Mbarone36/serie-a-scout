# Serie A Transfer Success Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PyTorch model that predicts how a player will fare after joining a Serie A club — conditioned on that club's tactical style — and a CLI that ranks transfer candidates for any real club or hand-authored tactical preset.

**Architecture:** A cached two-source scraping layer (FBref for performance, Transfermarkt for injuries and market values) feeds a feature pipeline that percentile-ranks every metric within its own league/season/position group. A conditioned two-tower network embeds the player and the destination club separately, combines them with an elementwise product to capture fit, and predicts three outcomes from three heads over a shared trunk. Weighting of those three outcomes happens at query time, never in the label.

**Tech Stack:** Python 3.13, PyTorch 2.10, pandas 2.3, scikit-learn, soccerdata (FBref wrapper), requests + BeautifulSoup (Transfermarkt), rapidfuzz (entity matching), pyarrow (parquet), pytest.

**Spec:** `docs/superpowers/specs/2026-08-29-serie-a-transfer-success-design.md`

## Global Constraints

Every task's requirements implicitly include this section.

- **Two data sources only:** FBref and Transfermarkt. No third source.
- **All train/test splits are grouped by player.** Never split by row. A player's 2019 season must never train while his 2021 season tests.
- **Every performance metric is percentile-ranked within its own (league, season, position group)** before reaching the model. Raw per-90 values never enter the network.
- **Feature leakage:** injury, availability and market-value features are truncated at the transfer date. Only seasons strictly before the move contribute features.
- **Label-only data:** transfer fees and destination-club values feed the step-up check in the label. They must never appear as model inputs.
- **Baselines are mandatory and built before the network** (Task 11 precedes Task 12). Ridge and gradient boosting are reported on identical folds. If the network does not beat them, that is the reported result.
- **Panel players are excluded from every fold, including validation** — not merely held back at test time.
- **Label window is the two seasons following arrival.**
- **Scraping:** minimum 4 seconds between requests to any host, every response cached to disk, all pulls resumable.
- **No network access in tests.** All scraper tests run against saved HTML/JSON fixtures in `tests/fixtures/`.
- Target ordering is fixed everywhere as `("minutes", "output", "retention")`.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/config.py` | Seasons, league IDs, position groups, paths, target names. No logic. |
| `src/cache.py` | Generic disk-cache decorator and rate limiter shared by both scrapers. |
| `src/scrape_fbref.py` | FBref pulls via soccerdata → raw parquet. |
| `src/scrape_tm.py` | Transfermarkt HTTP client + injury/value/club-value parsers. |
| `src/coverage.py` | League coverage report; gates which leagues the pipeline trusts. |
| `src/entities.py` | Cross-source name matching and the persistent manual ID map. |
| `src/transfers.py` | Arrival detection by squad-diffing. |
| `src/normalize.py` | Percentile ranking within (league, season, position group). |
| `src/features.py` | Player feature assembly across all bundles. |
| `src/clubs.py` | Club style vectors from team season stats. |
| `src/presets.py` | Hand-authored tactical preset vectors. |
| `src/labels.py` | The three targets, including the step-up check. |
| `src/dataset.py` | torch Dataset, player-grouped splits, panel exclusion. |
| `src/baselines.py` | Ridge and gradient boosting, per head. |
| `src/model.py` | Conditioned two-tower network with three heads. |
| `src/train.py` | Training loop, seed ensembling, ablations, metrics reporting. |
| `src/archetypes.py` | Gaussian-mixture style clustering. |
| `query.py` | CLI. |

Split by responsibility, not layer: `scrape_fbref.py` and `scrape_tm.py` are separate because they fail differently and are tested differently. `normalize.py` is separate from `features.py` because the percentile logic is the single most leakage-prone piece of code in the project and deserves its own test file.

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `src/__init__.py`, `src/config.py`, `tests/__init__.py`, `tests/test_config.py`, `.gitignore`

**Interfaces:**
- Consumes: nothing
- Produces: `config.SEASONS: list[str]`, `config.SERIE_A: str`, `config.POSITION_GROUPS: dict[str, str]`, `config.TARGETS: tuple[str, str, str]`, `config.RAW_DIR/PROCESSED_DIR/MANUAL_DIR: pathlib.Path`, `config.LABEL_WINDOW_SEASONS: int`

- [ ] **Step 1: Initialise the repository and virtualenv**

```bash
cd .
git init
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch pandas numpy scipy scikit-learn pyarrow \
    soccerdata requests beautifulsoup4 lxml rapidfuzz pytest
```

If `soccerdata` fails to install on Python 3.13, create the venv with
`python3.12 -m venv .venv` instead and re-run. Record whichever worked in the
README; every later task uses `.venv/bin/python`.

- [ ] **Step 2: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
data/raw/
data/processed/
.pytest_cache/
models/
```

`data/manual/` is deliberately NOT ignored — the reviewed ID map is hand-curated
work that must be version-controlled.

- [ ] **Step 3: Write the failing test**

```python
# tests/test_config.py
from src import config


def test_seasons_cover_advanced_stat_era():
    assert config.SEASONS[0] == "1718"
    assert "2526" in config.SEASONS
    assert len(config.SEASONS) == 9


def test_position_groups_map_fbref_codes():
    assert config.POSITION_GROUPS["FW"] == "attacker"
    assert config.POSITION_GROUPS["DF"] == "defender"
    assert config.POSITION_GROUPS["GK"] == "keeper"


def test_targets_are_ordered_triple():
    assert config.TARGETS == ("minutes", "output", "retention")
```

- [ ] **Step 4: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.config'`

- [ ] **Step 5: Write `src/config.py`**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MANUAL_DIR = DATA_DIR / "manual"

for _d in (RAW_DIR, PROCESSED_DIR, MANUAL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# FBref advanced stats begin in 2017-18 for the Big 5.
SEASONS = ["1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]

SERIE_A = "ITA-Serie A"

# Leagues we attempt to pull. Task 3 narrows this to those with real coverage.
CANDIDATE_LEAGUES = [
    "ITA-Serie A", "ENG-Premier League", "ESP-La Liga",
    "GER-Bundesliga", "FRA-Ligue 1",
]

POSITION_GROUPS = {"FW": "attacker", "MF": "midfielder", "DF": "defender", "GK": "keeper"}

TARGETS = ("minutes", "output", "retention")

LABEL_WINDOW_SEASONS = 2
MIN_MINUTES_PRIOR_SEASON = 450
REQUEST_DELAY_SECONDS = 4.0
ENSEMBLE_SEEDS = 10
```

- [ ] **Step 6: Write `pyproject.toml`**

```toml
[project]
name = "serie-a-scout"
version = "0.1.0"
requires-python = ">=3.12"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore src/ tests/ docs/
git commit -m "feat: project scaffold and configuration"
```

---

### Task 2: Cache and rate-limit layer

**Files:**
- Create: `src/cache.py`, `tests/test_cache.py`

**Interfaces:**
- Consumes: `config.RAW_DIR`, `config.REQUEST_DELAY_SECONDS`
- Produces: `cache.cached_frame(key: str, builder: Callable[[], pd.DataFrame]) -> pd.DataFrame`, `cache.RateLimiter(min_interval: float).wait() -> None`, `cache.cached_text(key: str, builder: Callable[[], str]) -> str`

Both scrapers depend on this. Writing it once is what makes every pull resumable:
a run that dies halfway through resumes from cache rather than re-fetching.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cache.py
import pandas as pd
import pytest
from src import cache


def test_cached_frame_calls_builder_once(tmp_path, monkeypatch):
    monkeypatch.setattr(cache.config, "RAW_DIR", tmp_path)
    calls = []

    def builder():
        calls.append(1)
        return pd.DataFrame({"a": [1, 2]})

    first = cache.cached_frame("thing", builder)
    second = cache.cached_frame("thing", builder)

    assert len(calls) == 1
    pd.testing.assert_frame_equal(first, second)


def test_cached_frame_survives_process_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(cache.config, "RAW_DIR", tmp_path)
    cache.cached_frame("x", lambda: pd.DataFrame({"a": [1]}))

    def explode():
        raise AssertionError("builder must not be called when cache exists")

    out = cache.cached_frame("x", explode)
    assert list(out["a"]) == [1]


def test_rate_limiter_waits_between_calls():
    clock = {"t": 0.0}
    slept = []
    limiter = cache.RateLimiter(
        min_interval=4.0,
        now=lambda: clock["t"],
        sleep=lambda s: (slept.append(s), clock.__setitem__("t", clock["t"] + s)),
    )
    limiter.wait()
    limiter.wait()
    assert slept == [4.0]


def test_rate_limiter_does_not_wait_if_enough_time_passed():
    clock = {"t": 0.0}
    slept = []
    limiter = cache.RateLimiter(
        min_interval=4.0,
        now=lambda: clock["t"],
        sleep=lambda s: slept.append(s),
    )
    limiter.wait()
    clock["t"] = 10.0
    limiter.wait()
    assert slept == []
```

Note the injected `now`/`sleep`: the rate limiter is tested without the test
actually sleeping. A test that sleeps 4 real seconds is a test nobody runs.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.cache'`

- [ ] **Step 3: Write `src/cache.py`**

```python
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
        return pd.read_parquet(path)
    frame = builder()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return frame


def cached_text(key: str, builder: Callable[[], str]) -> str:
    """Return cached raw text (HTML), building and storing it on first call."""
    path = _path(key, ".html")
    if path.exists():
        return path.read_text(encoding="utf-8")
    text = builder()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_cache.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/cache.py tests/test_cache.py
git commit -m "feat: disk cache and rate limiter for scrapers"
```

---

### Task 3: FBref pulls

**Files:**
- Create: `src/scrape_fbref.py`, `tests/test_scrape_fbref.py`

**Interfaces:**
- Consumes: `cache.cached_frame`, `config.SEASONS`, `config.CANDIDATE_LEAGUES`
- Produces: `fbref.player_season_stats(league: str, season: str) -> pd.DataFrame` with columns `player, team, league, season, pos, age, minutes` plus all stat columns; `fbref.team_season_stats(league: str, season: str) -> pd.DataFrame`; `scrape_fbref.STAT_TYPES: tuple[str, ...]`; `scrape_fbref.pull_all() -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scrape_fbref.py
import pandas as pd
import pytest
from src import scrape_fbref


def test_stat_types_cover_required_tables():
    for needed in ("standard", "shooting", "passing", "possession", "defense", "misc", "playing_time"):
        assert needed in scrape_fbref.STAT_TYPES


def test_merge_stat_tables_joins_on_player_identity():
    standard = pd.DataFrame({
        "player": ["A", "B"], "team": ["X", "Y"], "pos": ["FW", "MF"], "minutes": [900, 800],
    })
    shooting = pd.DataFrame({
        "player": ["A", "B"], "team": ["X", "Y"], "npxg": [5.0, 1.0],
    })
    out = scrape_fbref.merge_stat_tables({"standard": standard, "shooting": shooting})
    assert list(out.columns) == ["player", "team", "pos", "minutes", "npxg"]
    assert out.loc[out.player == "A", "npxg"].item() == 5.0


def test_merge_stat_tables_keeps_all_players_from_standard():
    standard = pd.DataFrame({"player": ["A", "B"], "team": ["X", "Y"], "pos": ["FW", "MF"], "minutes": [900, 10]})
    shooting = pd.DataFrame({"player": ["A"], "team": ["X"], "npxg": [5.0]})
    out = scrape_fbref.merge_stat_tables({"standard": standard, "shooting": shooting})
    assert len(out) == 2
    assert pd.isna(out.loc[out.player == "B", "npxg"].item())


def test_normalise_identity_extracts_three_letter_nationality():
    frame = pd.DataFrame({"player": ["A"], "nation": ["it ITA"], "team": ["X"],
                          "season": ["2223"], "playing_time_mp": [30]})
    out = scrape_fbref.normalise_identity(frame)
    assert out["nationality"].iloc[0] == "ITA"


def test_normalise_identity_derives_team_matches_from_squad_maximum():
    frame = pd.DataFrame({"player": ["A", "B"], "team": ["X", "X"],
                          "season": ["2223", "2223"], "playing_time_mp": [30, 38]})
    out = scrape_fbref.normalise_identity(frame)
    assert set(out["team_matches"]) == {38}


def test_flatten_multiindex_columns():
    df = pd.DataFrame(
        [[1, 2]],
        columns=pd.MultiIndex.from_tuples([("Standard", "Gls"), ("Expected", "npxG")]),
    )
    out = scrape_fbref.flatten_columns(df)
    assert list(out.columns) == ["standard_gls", "expected_npxg"]
```

`merge_stat_tables` and `flatten_columns` are pure functions taking DataFrames.
That is deliberate: the network-touching part of this module is three lines and
untestable, so all real logic lives in functions that never see a socket.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_scrape_fbref.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.scrape_fbref'`

- [ ] **Step 3: Write `src/scrape_fbref.py`**

```python
import re

import pandas as pd

from src import cache, config

STAT_TYPES = (
    "standard", "shooting", "passing", "passing_types",
    "goal_shot_creation", "defense", "possession", "playing_time", "misc",
)

_JOIN_KEYS = ["player", "team"]


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse FBref's two-level column headers into snake_case names."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    flat = []
    for upper, lower in df.columns:
        parts = [p for p in (upper, lower) if p and not str(p).startswith("Unnamed")]
        name = "_".join(parts) if parts else "unnamed"
        name = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
        flat.append(name)
    out = df.copy()
    out.columns = flat
    return out


def merge_stat_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Left-join every stat table onto `standard`, keeping all its players."""
    merged = tables["standard"].copy()
    for name, table in tables.items():
        if name == "standard":
            continue
        keys = [k for k in _JOIN_KEYS if k in table.columns and k in merged.columns]
        new_cols = [c for c in table.columns if c not in merged.columns or c in keys]
        merged = merged.merge(table[new_cols], on=keys, how="left")
    return merged


def normalise_identity(frame: pd.DataFrame) -> pd.DataFrame:
    """Give downstream modules the column names they expect.

    FBref calls the nationality column `nation` and never reports how many
    fixtures a club played. Both are required by the adaptation bundle (Task 10)
    and the minutes target (Task 12), so they are derived once, here.
    """
    out = frame.copy()
    if "nation" in out.columns and "nationality" not in out.columns:
        out["nationality"] = out["nation"].astype(str).str[-3:].str.upper()
    if "playing_time_mp" in out.columns:
        out["team_matches"] = out.groupby(["team", "season"])["playing_time_mp"].transform("max")
    return out


def _fbref(league: str, season: str):
    import soccerdata as sd

    return sd.FBref(leagues=league, seasons=season, data_dir=config.RAW_DIR / "soccerdata")


def player_season_stats(league: str, season: str) -> pd.DataFrame:
    """All player stat tables for one league-season, merged into one frame."""

    def build() -> pd.DataFrame:
        limiter = cache.RateLimiter(config.REQUEST_DELAY_SECONDS)
        client = _fbref(league, season)
        tables = {}
        for stat_type in STAT_TYPES:
            limiter.wait()
            try:
                raw = client.read_player_season_stats(stat_type=stat_type)
            except Exception:  # noqa: BLE001 — a missing table is normal for small leagues
                continue
            tables[stat_type] = flatten_columns(raw.reset_index())
        if "standard" not in tables:
            return pd.DataFrame()
        merged = merge_stat_tables(tables)
        merged["league"] = league
        merged["season"] = season
        return normalise_identity(merged)

    return cache.cached_frame(f"fbref/player/{league.replace(' ', '_')}_{season}", build)


def team_season_stats(league: str, season: str) -> pd.DataFrame:
    """Team-level stat tables for one league-season, used for club style vectors."""

    def build() -> pd.DataFrame:
        limiter = cache.RateLimiter(config.REQUEST_DELAY_SECONDS)
        client = _fbref(league, season)
        tables = {}
        for stat_type in ("standard", "passing", "possession", "defense", "misc"):
            limiter.wait()
            try:
                raw = client.read_team_season_stats(stat_type=stat_type)
            except Exception:  # noqa: BLE001
                continue
            flat = flatten_columns(raw.reset_index())
            tables[stat_type] = flat.rename(columns={"squad": "team"})
        if "standard" not in tables:
            return pd.DataFrame()
        merged = tables["standard"]
        for name, table in tables.items():
            if name == "standard":
                continue
            new_cols = [c for c in table.columns if c not in merged.columns or c == "team"]
            merged = merged.merge(table[new_cols], on="team", how="left")
        merged["league"] = league
        merged["season"] = season
        return merged

    return cache.cached_frame(f"fbref/team/{league.replace(' ', '_')}_{season}", build)


def pull_all() -> None:
    """Fetch every league-season. Resumable: cached pulls are skipped."""
    for league in config.CANDIDATE_LEAGUES:
        for season in config.SEASONS:
            print(f"player {league} {season}")
            player_season_stats(league, season)
            if league == config.SERIE_A:
                print(f"team   {league} {season}")
                team_season_stats(league, season)


if __name__ == "__main__":
    pull_all()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_scrape_fbref.py -v`
Expected: 6 passed

- [ ] **Step 5: Start the real pull in the background**

```bash
.venv/bin/python -m src.scrape_fbref
```

This takes a few hours. It is resumable — if it dies, re-run it and cached
league-seasons are skipped. Later tasks can be developed against whatever has
landed so far; only Task 15 needs the full set.

- [ ] **Step 6: Commit**

```bash
git add src/scrape_fbref.py tests/test_scrape_fbref.py
git commit -m "feat: FBref player and team season stat pulls"
```

---

### Task 4: League coverage report

**Files:**
- Create: `src/coverage.py`, `tests/test_coverage.py`

**Interfaces:**
- Consumes: `fbref.player_season_stats`
- Produces: `coverage.REQUIRED_ADVANCED_COLUMNS: tuple[str, ...]`, `coverage.assess(frame: pd.DataFrame) -> dict`, `coverage.build_report(leagues, seasons) -> pd.DataFrame`, `coverage.trusted_leagues(report: pd.DataFrame, min_coverage: float = 0.9) -> list[str]`

The spec requires verifying coverage *before anything depends on it*. A league
whose advanced stats are 60% null will silently poison the percentile
normalisation, and the failure looks like a mediocre model rather than a bug.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coverage.py
import numpy as np
import pandas as pd
from src import coverage


def _frame(npxg_nulls: int, n: int = 10) -> pd.DataFrame:
    npxg = [1.0] * n
    for i in range(npxg_nulls):
        npxg[i] = np.nan
    return pd.DataFrame({
        "player": [f"p{i}" for i in range(n)],
        "expected_npxg": npxg,
        "progression_prgp": [5.0] * n,
        "possession_take_ons_succ": [1.0] * n,
        "minutes": [900] * n,
    })


def test_assess_reports_full_coverage():
    result = coverage.assess(_frame(npxg_nulls=0))
    assert result["coverage"] == 1.0
    assert result["n_players"] == 10


def test_assess_reports_partial_coverage():
    result = coverage.assess(_frame(npxg_nulls=5))
    assert result["coverage"] < 1.0


def test_assess_handles_missing_column_entirely():
    frame = _frame(0).drop(columns=["expected_npxg"])
    result = coverage.assess(frame)
    assert result["coverage"] < 1.0
    assert "expected_npxg" in result["missing_columns"]


def test_trusted_leagues_filters_below_threshold():
    report = pd.DataFrame({
        "league": ["A", "A", "B", "B"],
        "season": ["2223", "2324", "2223", "2324"],
        "coverage": [1.0, 0.95, 0.2, 0.1],
        "n_players": [500, 500, 400, 400],
    })
    assert coverage.trusted_leagues(report, min_coverage=0.9) == ["A"]


def test_trusted_leagues_excludes_empty_report():
    report = pd.DataFrame(columns=["league", "season", "coverage", "n_players"])
    assert coverage.trusted_leagues(report) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_coverage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.coverage'`

- [ ] **Step 3: Write `src/coverage.py`**

```python
import pandas as pd

from src import config, fbref

REQUIRED_ADVANCED_COLUMNS = (
    "expected_npxg",
    "progression_prgp",
    "possession_take_ons_succ",
)


def assess(frame: pd.DataFrame) -> dict:
    """Fraction of required advanced columns populated for a league-season."""
    if frame.empty:
        return {"coverage": 0.0, "n_players": 0, "missing_columns": list(REQUIRED_ADVANCED_COLUMNS)}
    missing = [c for c in REQUIRED_ADVANCED_COLUMNS if c not in frame.columns]
    present = [c for c in REQUIRED_ADVANCED_COLUMNS if c in frame.columns]
    filled = sum(frame[c].notna().mean() for c in present)
    coverage_score = filled / len(REQUIRED_ADVANCED_COLUMNS)
    return {
        "coverage": float(coverage_score),
        "n_players": int(len(frame)),
        "missing_columns": missing,
    }


def build_report(leagues=None, seasons=None) -> pd.DataFrame:
    leagues = leagues or config.CANDIDATE_LEAGUES
    seasons = seasons or config.SEASONS
    rows = []
    for league in leagues:
        for season in seasons:
            frame = fbref.player_season_stats(league, season)
            rows.append({"league": league, "season": season, **assess(frame)})
    return pd.DataFrame(rows)


def trusted_leagues(report: pd.DataFrame, min_coverage: float = 0.9) -> list[str]:
    """Leagues whose median season coverage clears the threshold."""
    if report.empty:
        return []
    medians = report.groupby("league")["coverage"].median()
    return sorted(medians[medians >= min_coverage].index.tolist())


if __name__ == "__main__":
    report = build_report()
    report.to_csv(config.PROCESSED_DIR / "coverage_report.csv", index=False)
    print(report.to_string(index=False))
    print("\ntrusted:", trusted_leagues(report))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_coverage.py -v`
Expected: 5 passed

- [ ] **Step 5: Run the real report once the Task 3 pull has progressed**

```bash
.venv/bin/python -m src.coverage
```

Read the output. Any league not in the `trusted` list must be removed from
`config.CANDIDATE_LEAGUES` before Task 8 — its players cannot be percentile-ranked
meaningfully.

- [ ] **Step 6: Commit**

```bash
git add src/coverage.py tests/test_coverage.py
git commit -m "feat: FBref league coverage assessment"
```

---
### Task 5: Transfermarkt client and parsers

**Files:**
- Create: `src/scrape_tm.py`, `tests/test_scrape_tm.py`, `tests/fixtures/tm_injuries.html`, `tests/fixtures/tm_profile.html`

**Interfaces:**
- Consumes: `cache.cached_text`, `cache.RateLimiter`
- Produces: `scrape_tm.fetch(url: str, key: str) -> str`, `scrape_tm.parse_injuries(html: str) -> pd.DataFrame` with columns `season, injury, days_missed, games_missed`, `scrape_tm.parse_market_value_history(html: str) -> pd.DataFrame` with columns `date, value_eur`, `scrape_tm.parse_club_squad_value(html: str) -> float`, `scrape_tm.parse_player_identity(html: str) -> dict` with keys `name, birth_date, nationality`

Parsers are pure functions over HTML strings. Nothing in this module's test suite
touches the network — that is what makes the tests runnable and what makes a
site layout change produce a clear red test rather than silent nulls.

- [ ] **Step 1: Save real fixture pages by hand**

```bash
mkdir -p tests/fixtures
curl -sA "Mozilla/5.0 (research; contact you@example.com)" \
  "https://www.transfermarkt.com/khvicha-kvaratskhelia/verletzungen/spieler/537398" \
  -o tests/fixtures/tm_injuries.html
curl -sA "Mozilla/5.0 (research; contact you@example.com)" \
  "https://www.transfermarkt.com/khvicha-kvaratskhelia/profil/spieler/537398" \
  -o tests/fixtures/tm_profile.html
```

Open both files and confirm they contain real content rather than a block page.
If blocked, the parsers below cannot be validated and this task cannot proceed —
stop and report it rather than writing parsers against guessed markup.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_scrape_tm.py
from pathlib import Path

import pandas as pd
import pytest
from src import scrape_tm

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def injuries_html() -> str:
    return (FIXTURES / "tm_injuries.html").read_text(encoding="utf-8")


@pytest.fixture
def profile_html() -> str:
    return (FIXTURES / "tm_profile.html").read_text(encoding="utf-8")


def test_parse_injuries_returns_expected_columns(injuries_html):
    out = scrape_tm.parse_injuries(injuries_html)
    assert list(out.columns) == ["season", "injury", "days_missed", "games_missed"]


def test_parse_injuries_yields_numeric_days(injuries_html):
    out = scrape_tm.parse_injuries(injuries_html)
    assert pd.api.types.is_integer_dtype(out["days_missed"])
    assert (out["days_missed"] >= 0).all()


def test_parse_injuries_normalises_season_format(injuries_html):
    out = scrape_tm.parse_injuries(injuries_html)
    assert out["season"].str.fullmatch(r"\d{4}").all()


def test_parse_injuries_on_player_with_no_injuries():
    html = "<html><body><div>No injuries recorded</div></body></html>"
    out = scrape_tm.parse_injuries(html)
    assert out.empty
    assert list(out.columns) == ["season", "injury", "days_missed", "games_missed"]


def test_parse_market_value_history_is_chronological(profile_html):
    out = scrape_tm.parse_market_value_history(profile_html)
    assert not out.empty
    assert out["date"].is_monotonic_increasing
    assert (out["value_eur"] > 0).all()


def test_parse_player_identity(profile_html):
    ident = scrape_tm.parse_player_identity(profile_html)
    assert set(ident) == {"name", "birth_date", "nationality"}
    assert ident["birth_date"] is not None


def test_days_to_int_handles_transfermarkt_formatting():
    assert scrape_tm._days_to_int("42 days") == 42
    assert scrape_tm._days_to_int("-") == 0
    assert scrape_tm._days_to_int("") == 0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_scrape_tm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.scrape_tm'`

- [ ] **Step 4: Write `src/scrape_tm.py`**

Inspect the saved fixtures before writing the selectors below — adjust them to
the real markup rather than trusting these. The tests are the contract; the
selectors are an implementation detail that will need tuning.

```python
import json
import re
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src import cache, config

BASE = "https://www.transfermarkt.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; contact you@example.com)"}

_limiter = cache.RateLimiter(config.REQUEST_DELAY_SECONDS)

INJURY_COLUMNS = ["season", "injury", "days_missed", "games_missed"]


def fetch(url: str, key: str) -> str:
    """Rate-limited, disk-cached GET."""

    def build() -> str:
        _limiter.wait()
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        return response.text

    return cache.cached_text(f"tm/{key}", build)


def _days_to_int(text: str) -> int:
    match = re.search(r"(\d+)", text or "")
    return int(match.group(1)) if match else 0


def _normalise_season(text: str) -> str:
    """'21/22' -> '2122'."""
    digits = re.findall(r"\d{2}", text or "")
    return "".join(digits[:2]) if len(digits) >= 2 else ""


def parse_injuries(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table.items")
    if table is None:
        return pd.DataFrame(columns=INJURY_COLUMNS)
    rows = []
    for tr in table.select("tbody tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.select("td")]
        if len(cells) < 6:
            continue
        rows.append({
            "season": _normalise_season(cells[0]),
            "injury": cells[1],
            "days_missed": _days_to_int(cells[4]),
            "games_missed": _days_to_int(cells[5]),
        })
    frame = pd.DataFrame(rows, columns=INJURY_COLUMNS)
    if frame.empty:
        return pd.DataFrame(columns=INJURY_COLUMNS)
    frame["days_missed"] = frame["days_missed"].astype(int)
    frame["games_missed"] = frame["games_missed"].astype(int)
    return frame[frame["season"] != ""].reset_index(drop=True)


def parse_market_value_history(html: str) -> pd.DataFrame:
    """Extract the highcharts series embedded in the profile page."""
    match = re.search(r"'data':(\[\{.*?\}\])", html, re.DOTALL)
    if not match:
        return pd.DataFrame(columns=["date", "value_eur"])
    raw = match.group(1).replace("'", '"')
    try:
        points = json.loads(raw)
    except json.JSONDecodeError:
        return pd.DataFrame(columns=["date", "value_eur"])
    rows = [
        {"date": pd.to_datetime(p["datum_mw"]), "value_eur": float(p["y"])}
        for p in points
        if p.get("y") and p.get("datum_mw")
    ]
    frame = pd.DataFrame(rows, columns=["date", "value_eur"])
    return frame.sort_values("date").reset_index(drop=True)


def parse_club_squad_value(html: str) -> float:
    """Total squad market value in euros, used for the step-up check."""
    soup = BeautifulSoup(html, "lxml")
    node = soup.select_one("a.data-header__market-value-wrapper, .data-header__market-value-wrapper")
    text = node.get_text(" ", strip=True) if node else ""
    match = re.search(r"([\d,.]+)\s*(bn|m|k)?", text, re.IGNORECASE)
    if not match:
        return float("nan")
    amount = float(match.group(1).replace(",", "."))
    scale = {"bn": 1e9, "m": 1e6, "k": 1e3}.get((match.group(2) or "").lower(), 1.0)
    return amount * scale


def parse_player_identity(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    name_node = soup.select_one("h1.data-header__headline-wrapper")
    birth_node = soup.find("span", string=re.compile(r"Date of birth", re.I))
    nat_node = soup.select_one("span.data-header__content img.flaggenrahmen")
    birth_text = ""
    if birth_node and birth_node.find_next("span"):
        birth_text = birth_node.find_next("span").get_text(" ", strip=True)
    birth_date = None
    for fmt in ("%b %d, %Y", "%d.%m.%Y"):
        try:
            birth_date = datetime.strptime(birth_text.split("(")[0].strip(), fmt).date()
            break
        except ValueError:
            continue
    return {
        "name": name_node.get_text(" ", strip=True) if name_node else "",
        "birth_date": birth_date,
        "nationality": nat_node.get("title") if nat_node else "",
    }
```

- [ ] **Step 5: Run tests, tuning selectors against the fixtures until green**

Run: `.venv/bin/pytest tests/test_scrape_tm.py -v`
Expected: 7 passed

If a selector test fails, open the fixture, find the real markup, and fix the
selector. Do not weaken the assertion to match a broken parser.

- [ ] **Step 6: Commit**

```bash
git add src/scrape_tm.py tests/test_scrape_tm.py tests/fixtures/
git commit -m "feat: Transfermarkt client and page parsers"
```

---

### Task 6: Entity resolution

**Files:**
- Create: `src/entities.py`, `tests/test_entities.py`

**Interfaces:**
- Consumes: `config.MANUAL_DIR`
- Produces: `entities.normalise_name(name: str) -> str`, `entities.match_players(fbref: pd.DataFrame, tm: pd.DataFrame, threshold: int = 88) -> tuple[pd.DataFrame, pd.DataFrame]` returning `(matches, unmatched)` where `matches` has columns `fbref_player, tm_id, score`, `entities.load_manual_map() -> pd.DataFrame`, `entities.apply_manual_overrides(matches, unmatched) -> tuple[pd.DataFrame, pd.DataFrame]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_entities.py
import pandas as pd
import pytest
from src import entities


def test_normalise_name_strips_accents_and_case():
    assert entities.normalise_name("Khvicha Kvaratskhelia") == "khvicha kvaratskhelia"
    assert entities.normalise_name("Ciro Immobile") == "ciro immobile"
    assert entities.normalise_name("Nicolò Barella") == "nicolo barella"
    assert entities.normalise_name("Zlatan Ibrahimović") == "zlatan ibrahimovic"


def test_normalise_name_collapses_whitespace_and_punctuation():
    assert entities.normalise_name("  De  Ligt, Matthijs ") == "de ligt matthijs"


def _pair():
    # FBref reports birth YEAR only (`born`), never a full date.
    fbref = pd.DataFrame({
        "player": ["Nicolo Barella", "Khvicha Kvaratskhelia"],
        "born": [1997, 2001],
        "nationality": ["ITA", "GEO"],
    })
    tm = pd.DataFrame({
        "tm_id": [1, 2],
        "name": ["Nicolò Barella", "Khvicha Kvaratskhelia"],
        "birth_date": ["1997-02-07", "2001-02-12"],
        "nationality": ["ITA", "GEO"],
    })
    return fbref, tm


def test_match_players_pairs_on_name_and_birth_date():
    fbref, tm = _pair()
    matches, unmatched = entities.match_players(fbref, tm)
    assert len(matches) == 2
    assert unmatched.empty
    assert matches.loc[matches.fbref_player == "Nicolo Barella", "tm_id"].item() == 1


def test_match_players_rejects_same_name_different_birth_year():
    fbref = pd.DataFrame({"player": ["Luis Silva"], "born": [1990], "nationality": ["BRA"]})
    tm = pd.DataFrame({"tm_id": [9], "name": ["Luis Silva"], "birth_date": ["1998-05-05"], "nationality": ["BRA"]})
    matches, unmatched = entities.match_players(fbref, tm)
    assert matches.empty
    assert list(unmatched["player"]) == ["Luis Silva"]


def test_match_players_rejects_same_name_and_year_different_nationality():
    fbref = pd.DataFrame({"player": ["Luis Silva"], "born": [1990], "nationality": ["BRA"]})
    tm = pd.DataFrame({"tm_id": [9], "name": ["Luis Silva"], "birth_date": ["1990-05-05"], "nationality": ["POR"]})
    matches, unmatched = entities.match_players(fbref, tm)
    assert matches.empty


def test_match_players_reports_unmatched():
    fbref, tm = _pair()
    fbref = pd.concat([fbref, pd.DataFrame({
        "player": ["Unknown Guy"], "born": [1995], "nationality": ["ARG"],
    })], ignore_index=True)
    matches, unmatched = entities.match_players(fbref, tm)
    assert list(unmatched["player"]) == ["Unknown Guy"]


def test_manual_overrides_take_precedence(tmp_path, monkeypatch):
    monkeypatch.setattr(entities.config, "MANUAL_DIR", tmp_path)
    (tmp_path / "id_map.csv").write_text("fbref_player,tm_id\nUnknown Guy,42\n")
    matches = pd.DataFrame(columns=["fbref_player", "tm_id", "score"])
    unmatched = pd.DataFrame({"player": ["Unknown Guy"]})
    matches, unmatched = entities.apply_manual_overrides(matches, unmatched)
    assert matches.loc[0, "tm_id"] == 42
    assert unmatched.empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_entities.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.entities'`

- [ ] **Step 3: Write `src/entities.py`**

```python
import re
import unicodedata

import pandas as pd
from rapidfuzz import fuzz, process

from src import config

MANUAL_MAP = "id_map.csv"


def normalise_name(name: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    decomposed = unicodedata.normalize("NFKD", str(name))
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", ascii_only)
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def match_players(fbref: pd.DataFrame, tm: pd.DataFrame, threshold: int = 88):
    """Match on fuzzy name, gated by birth YEAR and nationality.

    FBref reports only a birth year (`born`), so an exact-date gate can never
    match. Year plus nationality is the strongest gate the sources actually
    support. It stays a gate rather than a tiebreak: footballers share names
    constantly, and a wrong join silently attaches another player's injury
    history to this one.
    """
    left = fbref.copy()
    right = tm.copy()
    left["_key"] = left["player"].map(normalise_name)
    right["_key"] = right["name"].map(normalise_name)
    left["_year"] = pd.to_numeric(left["born"], errors="coerce")
    right["_year"] = pd.to_datetime(right["birth_date"], errors="coerce").dt.year
    left["_nat"] = left["nationality"].astype(str).str.upper().str[:3]
    right["_nat"] = right["nationality"].astype(str).str.upper().str[:3]

    rows = []
    matched_players = set()
    for _, candidate in left.iterrows():
        pool = right[
            (right["_year"] == candidate["_year"]) & (right["_nat"] == candidate["_nat"])
        ]
        if pool.empty:
            continue
        best = process.extractOne(
            candidate["_key"], pool["_key"].tolist(), scorer=fuzz.token_sort_ratio
        )
        if best is None or best[1] < threshold:
            continue
        tm_id = pool.iloc[best[2]]["tm_id"]
        rows.append({"fbref_player": candidate["player"], "tm_id": tm_id, "score": best[1]})
        matched_players.add(candidate["player"])

    matches = pd.DataFrame(rows, columns=["fbref_player", "tm_id", "score"])
    unmatched = fbref[~fbref["player"].isin(matched_players)].copy()
    return matches, unmatched


def load_manual_map() -> pd.DataFrame:
    path = config.MANUAL_DIR / MANUAL_MAP
    if not path.exists():
        return pd.DataFrame(columns=["fbref_player", "tm_id"])
    return pd.read_csv(path)


def apply_manual_overrides(matches: pd.DataFrame, unmatched: pd.DataFrame):
    """Fold hand-reviewed pairings in and drop them from the unmatched list."""
    manual = load_manual_map()
    if manual.empty:
        return matches, unmatched
    manual = manual.assign(score=100)
    combined = pd.concat([matches, manual], ignore_index=True)
    combined = combined.drop_duplicates(subset=["fbref_player"], keep="last").reset_index(drop=True)
    remaining = unmatched[~unmatched["player"].isin(manual["fbref_player"])].reset_index(drop=True)
    return combined, remaining
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_entities.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/entities.py tests/test_entities.py
git commit -m "feat: cross-source entity resolution with manual override map"
```

---

### Task 6b: Transfermarkt bulk pull

**Files:**
- Create: `src/pull_tm.py`, `tests/test_pull_tm.py`
- Modify: `src/scrape_tm.py` (add `parse_search_results`, `parse_transfer_history`)

**Interfaces:**
- Consumes: `scrape_tm.fetch`, `scrape_tm.parse_injuries`, `scrape_tm.parse_market_value_history`, `scrape_tm.parse_club_squad_value`, `entities.match_players`, `entities.apply_manual_overrides`
- Produces: `scrape_tm.parse_search_results(html) -> pd.DataFrame` with columns `tm_id, name, birth_date, nationality`; `scrape_tm.parse_club_search_results(html) -> list[int]`; `scrape_tm.parse_transfer_history(html) -> pd.DataFrame` with columns `season, from_club, to_club, fee_eur`; `pull_tm.search_player(name) -> pd.DataFrame`; `pull_tm.search_club(name) -> int | None`; `pull_tm.pull_club_values(team_names) -> None`; `pull_tm.build() -> None` writing `injuries.parquet`, `market_values.parquet`, `club_values.parquet`, `fees.parquet` to `data/processed/`
- Depends on `data/processed/fbref_identities.parquet` and `transfers.parquet`, both written by Task 7's `build()`. **Run Task 7 before running this task's `build()`.**

Tasks 10, 12 and 18 all read these four files. Without this task they do not
exist and the pipeline cannot run — this is the step that turns the Task 5
parsers into actual data.

- [ ] **Step 1: Save a search-results fixture**

```bash
curl -sA "Mozilla/5.0 (research; contact you@example.com)" \
  "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query=kvaratskhelia" \
  -o tests/fixtures/tm_search.html
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_pull_tm.py
from pathlib import Path

import pandas as pd
import pytest
from src import pull_tm, scrape_tm

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_search_results_returns_candidate_identities():
    html = (FIXTURES / "tm_search.html").read_text(encoding="utf-8")
    out = scrape_tm.parse_search_results(html)
    assert list(out.columns) == ["tm_id", "name", "birth_date", "nationality"]
    assert len(out) >= 1
    assert out["tm_id"].notna().all()


def test_parse_search_results_on_empty_page():
    out = scrape_tm.parse_search_results("<html><body>no hits</body></html>")
    assert out.empty
    assert list(out.columns) == ["tm_id", "name", "birth_date", "nationality"]


def test_parse_club_search_results_returns_ids_best_first():
    html = ('<a href="/ssc-napoli/startseite/verein/6195">Napoli</a>'
            '<a href="/us-lecce/startseite/verein/1005">Lecce</a>')
    assert scrape_tm.parse_club_search_results(html) == [6195, 1005]


def test_parse_club_search_results_on_empty_page():
    assert scrape_tm.parse_club_search_results("<html></html>") == []


def test_pull_club_values_records_nan_for_unresolvable_club(tmp_path, monkeypatch):
    monkeypatch.setattr(pull_tm.config, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(pull_tm, "search_club", lambda name: None)
    pull_tm.pull_club_values(["Nonexistent FC"])
    out = pd.read_parquet(tmp_path / "club_values.parquet")
    assert pd.isna(out["squad_value_eur"].iloc[0])


def test_parse_transfer_history_extracts_numeric_fees():
    html = (FIXTURES / "tm_profile.html").read_text(encoding="utf-8")
    out = scrape_tm.parse_transfer_history(html)
    assert list(out.columns) == ["season", "from_club", "to_club", "fee_eur"]
    assert pd.api.types.is_float_dtype(out["fee_eur"])


def test_fee_to_eur_parses_transfermarkt_notation():
    assert scrape_tm._fee_to_eur("€70.00m") == pytest.approx(70e6)
    assert scrape_tm._fee_to_eur("€450k") == pytest.approx(450e3)
    assert scrape_tm._fee_to_eur("free transfer") == 0.0
    assert scrape_tm._fee_to_eur("loan transfer") == 0.0
    assert pd.isna(scrape_tm._fee_to_eur("-"))


def test_build_skips_players_already_pulled(tmp_path, monkeypatch):
    monkeypatch.setattr(pull_tm.config, "PROCESSED_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(pull_tm, "_pull_one", lambda tm_id, name: calls.append(tm_id) or {
        "injuries": pd.DataFrame(columns=["season", "injury", "days_missed", "games_missed"]),
        "values": pd.DataFrame(columns=["date", "value_eur"]),
        "fees": pd.DataFrame(columns=["season", "from_club", "to_club", "fee_eur"]),
    })
    matches = pd.DataFrame({"fbref_player": ["A"], "tm_id": [1], "score": [100]})
    pull_tm.pull_players(matches)
    pull_tm.pull_players(matches)
    assert calls == [1], "second run must reuse cached parquet, not refetch"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_pull_tm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.pull_tm'`

- [ ] **Step 4: Add the two parsers to `src/scrape_tm.py`**

```python
SEARCH_COLUMNS = ["tm_id", "name", "birth_date", "nationality"]
TRANSFER_COLUMNS = ["season", "from_club", "to_club", "fee_eur"]

_FREE = ("free transfer", "loan", "end of loan", "?")


def _fee_to_eur(text: str) -> float:
    """'€70.00m' -> 70000000.0. Free and loan moves are zero, unknown is NaN."""
    lowered = (text or "").strip().lower()
    if any(token in lowered for token in _FREE):
        return 0.0
    match = re.search(r"([\d.,]+)\s*(bn|m|k)?", lowered)
    if not match:
        return float("nan")
    amount = float(match.group(1).replace(",", ""))
    scale = {"bn": 1e9, "m": 1e6, "k": 1e3}.get(match.group(2) or "", 1.0)
    return amount * scale


def parse_search_results(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for tr in soup.select("table.items tbody tr"):
        link = tr.select_one("a[href*='/profil/spieler/']")
        if link is None:
            continue
        tm_id = link["href"].rstrip("/").split("/")[-1]
        cells = [td.get_text(" ", strip=True) for td in tr.select("td")]
        birth = next((c for c in cells if re.search(r"\d{4}", c)), "")
        flag = tr.select_one("img.flaggenrahmen")
        rows.append({
            "tm_id": int(tm_id),
            "name": link.get_text(" ", strip=True),
            "birth_date": pd.to_datetime(birth, errors="coerce"),
            "nationality": flag.get("title") if flag else "",
        })
    return pd.DataFrame(rows, columns=SEARCH_COLUMNS)


def parse_club_search_results(html: str) -> list[int]:
    """Transfermarkt club ids from a quick-search page, best match first."""
    soup = BeautifulSoup(html, "lxml")
    ids = []
    for link in soup.select("a[href*='/startseite/verein/']"):
        parts = link["href"].rstrip("/").split("/")
        if parts and parts[-1].isdigit():
            club_id = int(parts[-1])
            if club_id not in ids:
                ids.append(club_id)
    return ids


def parse_transfer_history(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for tr in soup.select("div.tm-transfer-history tr, table.items tbody tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.select("td")]
        if len(cells) < 5 or not re.match(r"\d{2}/\d{2}", cells[0]):
            continue
        rows.append({
            "season": _normalise_season(cells[0]),
            "from_club": cells[-3],
            "to_club": cells[-2],
            "fee_eur": _fee_to_eur(cells[-1]),
        })
    frame = pd.DataFrame(rows, columns=TRANSFER_COLUMNS)
    frame["fee_eur"] = frame["fee_eur"].astype(float)
    return frame
```

- [ ] **Step 5: Write `src/pull_tm.py`**

```python
import pandas as pd

from src import config, entities, scrape_tm

BASE = scrape_tm.BASE
FILES = {
    "injuries": "injuries.parquet",
    "values": "market_values.parquet",
    "fees": "fees.parquet",
}


def search_player(name: str) -> pd.DataFrame:
    html = scrape_tm.fetch(
        f"{BASE}/schnellsuche/ergebnis/schnellsuche?query={name.replace(' ', '+')}",
        key=f"search/{entities.normalise_name(name).replace(' ', '_')}",
    )
    return scrape_tm.parse_search_results(html)


def _pull_one(tm_id: int, name: str) -> dict:
    slug = entities.normalise_name(name).replace(" ", "-")
    injuries = scrape_tm.parse_injuries(
        scrape_tm.fetch(f"{BASE}/{slug}/verletzungen/spieler/{tm_id}", f"inj/{tm_id}")
    )
    profile = scrape_tm.fetch(f"{BASE}/{slug}/profil/spieler/{tm_id}", f"prof/{tm_id}")
    return {
        "injuries": injuries,
        "values": scrape_tm.parse_market_value_history(profile),
        "fees": scrape_tm.parse_transfer_history(profile),
    }


def _existing(kind: str) -> pd.DataFrame:
    path = config.PROCESSED_DIR / FILES[kind]
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def pull_players(matches: pd.DataFrame) -> None:
    """Fetch every matched player. Resumable: players already stored are skipped."""
    stored = {kind: _existing(kind) for kind in FILES}
    done = set(stored["injuries"]["player"]) if len(stored["injuries"]) else set()

    collected = {kind: [stored[kind]] for kind in FILES}
    for _, row in matches.iterrows():
        if row["fbref_player"] in done:
            continue
        pulled = _pull_one(int(row["tm_id"]), row["fbref_player"])
        for kind, frame in pulled.items():
            frame = frame.copy()
            frame["player"] = row["fbref_player"]
            collected[kind].append(frame)

    for kind, frames in collected.items():
        combined = pd.concat([f for f in frames if len(f)], ignore_index=True)
        combined.to_parquet(config.PROCESSED_DIR / FILES[kind])


def search_club(name: str) -> int | None:
    """Resolve a club name to a Transfermarkt club id via the quick search."""
    html = scrape_tm.fetch(
        f"{BASE}/schnellsuche/ergebnis/schnellsuche?query={name.replace(' ', '+')}",
        key=f"clubsearch/{entities.normalise_name(name).replace(' ', '_')}",
    )
    ids = scrape_tm.parse_club_search_results(html)
    return int(ids[0]) if ids else None


def pull_club_values(team_names) -> None:
    """Squad market value per club, keyed by the team names FBref uses.

    The step-up check in Task 12 compares the selling club's squad value with
    the destination's. Without this file that comparison cannot run and the
    retention ladder silently collapses to three rungs — which would score a
    profitable sale to a bigger club as a failure, exactly what the spec forbids.
    """
    rows = []
    for name in team_names:
        club_id = search_club(name)
        if club_id is None:
            rows.append({"team": name, "squad_value_eur": float("nan")})
            continue
        html = scrape_tm.fetch(
            f"{BASE}/x/startseite/verein/{club_id}", key=f"club/{club_id}"
        )
        rows.append({
            "team": name,
            "squad_value_eur": scrape_tm.parse_club_squad_value(html),
        })
    pd.DataFrame(rows).to_parquet(config.PROCESSED_DIR / "club_values.parquet")


def build() -> None:
    fbref_players = pd.read_parquet(config.PROCESSED_DIR / "fbref_identities.parquet")
    candidates = pd.concat(
        [search_player(name) for name in fbref_players["player"].unique()],
        ignore_index=True,
    ).drop_duplicates(subset=["tm_id"])

    matches, unmatched = entities.match_players(fbref_players, candidates)
    matches, unmatched = entities.apply_manual_overrides(matches, unmatched)
    unmatched.to_csv(config.MANUAL_DIR / "unmatched_review.csv", index=False)
    print(f"matched {len(matches)}, unmatched {len(unmatched)} "
          f"(review data/manual/unmatched_review.csv)")

    pull_players(matches)

    teams = pd.read_parquet(config.PROCESSED_DIR / "transfers.parquet")
    club_names = sorted(set(teams["to_team"]) | set(teams["from_team"]))
    pull_club_values(club_names)


if __name__ == "__main__":
    build()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_pull_tm.py -v`
Expected: 8 passed

- [ ] **Step 7: Run the real pull, then work the unmatched list**

```bash
.venv/bin/python -m src.pull_tm
```

Open `data/manual/unmatched_review.csv`. For each player who genuinely exists on
Transfermarkt, add a `fbref_player,tm_id` line to `data/manual/id_map.csv` and
re-run. Players who are legitimately absent stay unmatched and will carry null
injury and value features, filled to the cohort median in Task 18.

Commit `id_map.csv` — it is hand-curated work and must not be regenerated.

- [ ] **Step 8: Commit**

```bash
git add src/pull_tm.py src/scrape_tm.py tests/test_pull_tm.py tests/fixtures/tm_search.html data/manual/id_map.csv
git commit -m "feat: Transfermarkt bulk pull for injuries, values and fees"
```

---

### Task 7: Arrival detection

**Files:**
- Create: `src/transfers.py`, `tests/test_transfers.py`

**Interfaces:**
- Consumes: `fbref.player_season_stats`, `config.SERIE_A`, `config.MIN_MINUTES_PRIOR_SEASON`
- Produces: `transfers.previous_season(season: str) -> str`, `transfers.detect_arrivals(all_seasons: pd.DataFrame) -> pd.DataFrame` with columns `player, arrival_season, to_team, from_team, from_league, prior_minutes`

A player in a club's season-S squad who was absent from its season S-1 squad is
an arrival. His pre-move club is wherever he appears in S-1.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transfers.py
import pandas as pd
import pytest
from src import transfers


def _rows(records):
    frame = pd.DataFrame(records, columns=["player", "team", "league", "season", "minutes", "pos"])
    frame["nationality"] = "ITA"
    return frame


def test_previous_season_rolls_back_correctly():
    assert transfers.previous_season("2223") == "2122"
    assert transfers.previous_season("2021") == "1920"
    assert transfers.previous_season("1819") == "1718"


def test_detects_arrival_from_another_league():
    frame = _rows([
        ("Kvara", "Dinamo Batumi", "GEO-Erovnuli", "2122", 2000, "FW"),
        ("Kvara", "Napoli", "ITA-Serie A", "2223", 2500, "FW"),
    ])
    out = transfers.detect_arrivals(frame)
    assert len(out) == 1
    assert out.iloc[0]["to_team"] == "Napoli"
    assert out.iloc[0]["from_team"] == "Dinamo Batumi"
    assert out.iloc[0]["arrival_season"] == "2223"


def test_ignores_player_who_stayed_at_the_same_club():
    frame = _rows([
        ("Barella", "Inter", "ITA-Serie A", "2122", 2500, "MF"),
        ("Barella", "Inter", "ITA-Serie A", "2223", 2600, "MF"),
    ])
    assert transfers.detect_arrivals(frame).empty


def test_detects_intra_serie_a_transfer():
    frame = _rows([
        ("Player", "Sassuolo", "ITA-Serie A", "2122", 2000, "MF"),
        ("Player", "Juventus", "ITA-Serie A", "2223", 1800, "MF"),
    ])
    out = transfers.detect_arrivals(frame)
    assert len(out) == 1
    assert out.iloc[0]["from_league"] == "ITA-Serie A"


def test_excludes_arrival_with_too_few_prior_minutes():
    frame = _rows([
        ("Kid", "Ajax", "NED-Eredivisie", "2122", 100, "FW"),
        ("Kid", "Milan", "ITA-Serie A", "2223", 400, "FW"),
    ])
    assert transfers.detect_arrivals(frame).empty


def test_excludes_player_with_no_prior_season_on_record():
    frame = _rows([("Ghost", "Roma", "ITA-Serie A", "2223", 1500, "FW")])
    assert transfers.detect_arrivals(frame).empty


def test_arrival_carries_nationality_for_the_adaptation_bundle():
    frame = _rows([
        ("Kvara", "Batumi", "GEO-Erovnuli", "2122", 2000, "FW"),
        ("Kvara", "Napoli", "ITA-Serie A", "2223", 2500, "FW"),
    ])
    frame.loc[frame.season == "2122", "nationality"] = "GEO"
    out = transfers.detect_arrivals(frame)
    assert out.iloc[0]["nationality"] == "GEO"


def test_emits_one_row_per_arrival_not_per_prior_season():
    frame = _rows([
        ("P", "Porto", "POR-Liga", "2021", 2000, "FW"),
        ("P", "Porto", "POR-Liga", "2122", 2100, "FW"),
        ("P", "Lazio", "ITA-Serie A", "2223", 1900, "FW"),
    ])
    out = transfers.detect_arrivals(frame)
    assert len(out) == 1
```

The last test pins an easy mistake: a player with three prior seasons is one
arrival. The per-season expansion happens in the feature layer (Task 9), not here.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_transfers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.transfers'`

- [ ] **Step 3: Write `src/transfers.py`**

```python
import pandas as pd

from src import config

ARRIVAL_COLUMNS = [
    "player", "arrival_season", "to_team", "from_team", "from_league",
    "prior_minutes", "nationality",
]


def previous_season(season: str) -> str:
    """'2223' -> '2122'."""
    start = int(season[:2])
    return f"{start - 1:02d}{start:02d}"


def detect_arrivals(all_seasons: pd.DataFrame) -> pd.DataFrame:
    """Serie A arrivals, found by diffing squad membership season over season."""
    serie_a = all_seasons[all_seasons["league"] == config.SERIE_A]
    rows = []
    for _, row in serie_a.iterrows():
        prior = previous_season(row["season"])
        was_here = all_seasons[
            (all_seasons["player"] == row["player"])
            & (all_seasons["season"] == prior)
            & (all_seasons["team"] == row["team"])
        ]
        if not was_here.empty:
            continue  # already at this club last season
        prior_rows = all_seasons[
            (all_seasons["player"] == row["player"]) & (all_seasons["season"] == prior)
        ]
        if prior_rows.empty:
            continue  # no pre-move season on record
        source = prior_rows.sort_values("minutes", ascending=False).iloc[0]
        if source["minutes"] < config.MIN_MINUTES_PRIOR_SEASON:
            continue
        rows.append({
            "player": row["player"],
            "arrival_season": row["season"],
            "to_team": row["team"],
            "from_team": source["team"],
            "from_league": source["league"],
            "prior_minutes": float(source["minutes"]),
            # Required by the adaptation bundle. Without it that whole bundle
            # silently evaluates to zero instead of failing loudly.
            "nationality": source.get("nationality", ""),
        })
    return pd.DataFrame(rows, columns=ARRIVAL_COLUMNS).drop_duplicates(
        subset=["player", "arrival_season"]
    ).reset_index(drop=True)


def build() -> pd.DataFrame:
    from src import scrape_fbref

    frames = [
        fbref.player_season_stats(league, season)
        for league in config.CANDIDATE_LEAGUES
        for season in config.SEASONS
    ]
    combined = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    arrivals = detect_arrivals(combined)
    arrivals.to_parquet(config.PROCESSED_DIR / "transfers.parquet")

    # Task 6b matches these against Transfermarkt.
    identities = (
        combined[["player", "born", "nationality"]]
        .dropna(subset=["player"])
        .drop_duplicates(subset=["player"])
        .reset_index(drop=True)
    )
    identities.to_parquet(config.PROCESSED_DIR / "fbref_identities.parquet")
    return arrivals


if __name__ == "__main__":
    print(build().head(30).to_string(index=False))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_transfers.py -v`
Expected: 8 passed

- [ ] **Step 5: Sanity-check against reality**

```bash
.venv/bin/python -m src.transfers
```

Verify by eye that several arrivals you know are real appear with the correct
`from_team`. A pipeline that produces plausible-looking nonsense here will
produce plausible-looking nonsense all the way to the CLI.

- [ ] **Step 6: Commit**

```bash
git add src/transfers.py tests/test_transfers.py
git commit -m "feat: Serie A arrival detection by squad-diffing"
```

---

### Task 8: Percentile normalisation

**Files:**
- Create: `src/normalize.py`, `tests/test_normalize.py`

**Interfaces:**
- Consumes: `config.POSITION_GROUPS`
- Produces: `normalize.position_group(pos: str) -> str`, `normalize.per_90(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame`, `normalize.percentile_within(frame: pd.DataFrame, columns: list[str], by=("league", "season", "position_group")) -> pd.DataFrame`

This is the most leakage-prone code in the project and it gets its own module and
its own test file. Percentiles must be computed **within a single season's data
only** — a rank computed across seasons leaks future information backward.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_normalize.py
import numpy as np
import pandas as pd
import pytest
from src import normalize


def test_position_group_maps_primary_position():
    assert normalize.position_group("FW") == "attacker"
    assert normalize.position_group("FW,MF") == "attacker"
    assert normalize.position_group("DF,MF") == "defender"
    assert normalize.position_group("") == "unknown"


def test_per_90_scales_by_minutes():
    frame = pd.DataFrame({"minutes": [900, 450], "goals": [10, 10]})
    out = normalize.per_90(frame, ["goals"])
    assert out["goals_p90"].tolist() == [1.0, 2.0]


def test_per_90_guards_against_zero_minutes():
    frame = pd.DataFrame({"minutes": [0], "goals": [3]})
    out = normalize.per_90(frame, ["goals"])
    assert pd.isna(out["goals_p90"].iloc[0])


def test_percentile_within_group_is_zero_to_one():
    frame = pd.DataFrame({
        "league": ["A"] * 4, "season": ["2223"] * 4, "position_group": ["attacker"] * 4,
        "x": [1.0, 2.0, 3.0, 4.0],
    })
    out = normalize.percentile_within(frame, ["x"])
    assert out["x_pct"].min() >= 0.0
    assert out["x_pct"].max() <= 1.0
    assert out["x_pct"].is_monotonic_increasing


def test_percentile_is_computed_separately_per_league():
    frame = pd.DataFrame({
        "league": ["A", "A", "B", "B"],
        "season": ["2223"] * 4,
        "position_group": ["attacker"] * 4,
        "x": [1.0, 2.0, 100.0, 200.0],
    })
    out = normalize.percentile_within(frame, ["x"])
    a = out[out.league == "A"]["x_pct"].tolist()
    b = out[out.league == "B"]["x_pct"].tolist()
    assert a == b  # identical ranks despite wildly different raw values


def test_percentile_does_not_pool_across_seasons():
    frame = pd.DataFrame({
        "league": ["A"] * 4,
        "season": ["2122", "2122", "2223", "2223"],
        "position_group": ["attacker"] * 4,
        "x": [1.0, 2.0, 50.0, 100.0],
    })
    out = normalize.percentile_within(frame, ["x"])
    assert out[out.season == "2122"]["x_pct"].tolist() == out[out.season == "2223"]["x_pct"].tolist()


def test_percentile_handles_single_member_group():
    frame = pd.DataFrame({
        "league": ["A"], "season": ["2223"], "position_group": ["keeper"], "x": [5.0],
    })
    out = normalize.percentile_within(frame, ["x"])
    assert out["x_pct"].iloc[0] == 0.5


def test_percentile_preserves_nulls():
    frame = pd.DataFrame({
        "league": ["A"] * 3, "season": ["2223"] * 3, "position_group": ["attacker"] * 3,
        "x": [1.0, np.nan, 3.0],
    })
    out = normalize.percentile_within(frame, ["x"])
    assert pd.isna(out["x_pct"].iloc[1])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_normalize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.normalize'`

- [ ] **Step 3: Write `src/normalize.py`**

```python
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
        sizes = grouped[column].transform("size")
        out[f"{column}_pct"] = np.where(sizes <= 1, 0.5, ranked)
        out.loc[out[column].isna(), f"{column}_pct"] = np.nan
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_normalize.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/normalize.py tests/test_normalize.py
git commit -m "feat: within-league-season-position percentile normalisation"
```

---
### Task 9: On-field features

**Files:**
- Create: `src/features_onfield.py`, `tests/test_features_onfield.py`

**Interfaces:**
- Consumes: `normalize.position_group`, `normalize.per_90`, `normalize.percentile_within`
- Produces: `features_onfield.COUNTING_COLUMNS: tuple[str, ...]`, `features_onfield.RATE_COLUMNS: tuple[str, ...]`, `features_onfield.build(seasons: pd.DataFrame) -> pd.DataFrame`, `features_onfield.FEATURE_COLUMNS: list[str]` (populated at import time — every column the model consumes from this bundle)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features_onfield.py
import pandas as pd
import pytest
from src import features_onfield


def _seasons(n=6):
    return pd.DataFrame({
        "player": [f"p{i}" for i in range(n)],
        "team": ["T"] * n,
        "league": ["ITA-Serie A"] * n,
        "season": ["2223"] * n,
        "pos": ["FW"] * n,
        "minutes": [900] * n,
        "age": [24] * n,
        "expected_npxg": [float(i) for i in range(n)],
        "expected_xag": [float(i) * 0.5 for i in range(n)],
        "progression_prgp": [float(i) * 2 for i in range(n)],
        "progression_prgc": [float(i) * 3 for i in range(n)],
        "possession_take_ons_succ": [float(i) for i in range(n)],
        "possession_touches_att_pen": [float(i) * 4 for i in range(n)],
        "possession_carries_mis": [float(i) for i in range(n)],
        "possession_carries_dis": [float(i) for i in range(n)],
        "tackles_tkl": [float(i) for i in range(n)],
        "int": [float(i) for i in range(n)],
        "aerial_duels_won_pct": [50.0] * n,
        "performance_crdy": [float(i) for i in range(n)],
        "performance_fls": [float(i) for i in range(n)],
    })


def test_build_adds_position_group():
    out = features_onfield.build(_seasons())
    assert set(out["position_group"]) == {"attacker"}


def test_build_emits_percentile_columns_for_every_feature():
    out = features_onfield.build(_seasons())
    for column in features_onfield.FEATURE_COLUMNS:
        assert column in out.columns, column


def test_percentile_features_are_bounded():
    out = features_onfield.build(_seasons())
    values = out[features_onfield.FEATURE_COLUMNS].to_numpy()
    assert ((values >= 0.0) & (values <= 1.0)).all()


def test_missing_optional_column_is_filled_not_fatal():
    frame = _seasons().drop(columns=["possession_touches_att_pen"])
    out = features_onfield.build(frame)
    assert "possession_touches_att_pen_p90_pct" in out.columns
    assert out["possession_touches_att_pen_p90_pct"].notna().all()


def test_build_is_row_preserving():
    frame = _seasons(n=7)
    assert len(features_onfield.build(frame)) == 7
```

`test_missing_optional_column_is_filled_not_fatal` matters: FBref tables differ
by league and era, and the pipeline must degrade to a neutral 0.5 rather than
crashing halfway through a multi-hour run.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_features_onfield.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.features_onfield'`

- [ ] **Step 3: Write `src/features_onfield.py`**

```python
import numpy as np
import pandas as pd

from src import normalize

# Raw counting stats, converted to per-90 before ranking.
COUNTING_COLUMNS = (
    "expected_npxg", "expected_xag", "progression_prgp", "progression_prgc",
    "possession_take_ons_succ", "possession_touches_att_pen",
    "possession_carries_mis", "possession_carries_dis",
    "tackles_tkl", "int", "performance_crdy", "performance_fls",
)

# Already rates or shares; ranked directly.
RATE_COLUMNS = ("aerial_duels_won_pct",)

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_features_onfield.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/features_onfield.py tests/test_features_onfield.py
git commit -m "feat: on-field percentile feature bundle"
```

---

### Task 10: Off-field feature bundles

**Files:**
- Create: `src/features_offfield.py`, `tests/test_features_offfield.py`

**Interfaces:**
- Consumes: `transfers.previous_season`
- Produces: `features_offfield.LANGUAGE_FAMILY: dict[str, str]`, `features_offfield.build_adaptation(arrivals, seasons, squad_nationalities) -> pd.DataFrame`, `features_offfield.build_trajectory(arrivals, seasons) -> pd.DataFrame`, `features_offfield.build_injury(arrivals, injuries) -> pd.DataFrame`, `features_offfield.build_availability(arrivals, seasons, injuries) -> pd.DataFrame`, `features_offfield.build_value(arrivals, values) -> pd.DataFrame`, `features_offfield.BUNDLES: dict[str, list[str]]` mapping bundle name to its feature columns

`BUNDLES` is the registry the ablation study in Task 16 iterates over. Every
off-field column must appear in exactly one bundle.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features_offfield.py
import pandas as pd
import pytest
from src import features_offfield as off


def test_language_family_classifies_romance_nations():
    for nat in ("ESP", "ARG", "BRA", "FRA", "POR", "ROU"):
        assert off.LANGUAGE_FAMILY[nat] == "romance"


def test_language_family_defaults_to_other():
    assert off.LANGUAGE_FAMILY.get("XXX", "other") == "other"


def test_adaptation_flags_first_move_abroad():
    arrivals = pd.DataFrame({
        "player": ["A"], "arrival_season": ["2223"], "to_team": ["Napoli"],
        "from_league": ["GEO-Erovnuli"], "nationality": ["GEO"],
    })
    seasons = pd.DataFrame({
        "player": ["A", "A"], "league": ["GEO-Erovnuli", "GEO-Erovnuli"],
        "season": ["2021", "2122"], "team": ["Batumi", "Batumi"], "minutes": [1800, 2000],
    })
    squads = pd.DataFrame({"team": ["Napoli"], "season": ["2223"], "nationality": ["ITA"], "n": [20]})
    out = off.build_adaptation(arrivals, seasons, squads)
    assert out.loc[0, "first_move_abroad"] == 1.0
    assert out.loc[0, "n_countries"] == 1.0
    assert out.loc[0, "compatriots_in_squad"] == 0.0


def test_adaptation_counts_compatriots():
    arrivals = pd.DataFrame({
        "player": ["A"], "arrival_season": ["2223"], "to_team": ["Napoli"],
        "from_league": ["GEO-Erovnuli"], "nationality": ["GEO"],
    })
    seasons = pd.DataFrame({
        "player": ["A"], "league": ["GEO-Erovnuli"], "season": ["2122"],
        "team": ["Batumi"], "minutes": [2000],
    })
    squads = pd.DataFrame({
        "team": ["Napoli", "Napoli"], "season": ["2223", "2223"],
        "nationality": ["GEO", "ITA"], "n": [2, 18],
    })
    out = off.build_adaptation(arrivals, seasons, squads)
    assert out.loc[0, "compatriots_in_squad"] == 2.0


def test_trajectory_detects_loan_round_trip():
    arrivals = pd.DataFrame({"player": ["A"], "arrival_season": ["2324"]})
    seasons = pd.DataFrame({
        "player": ["A", "A", "A"],
        "team": ["Home", "Away", "Home"],
        "league": ["ITA-Serie A"] * 3,
        "season": ["2021", "2122", "2223"],
        "minutes": [1000, 1500, 1200],
    })
    out = off.build_trajectory(arrivals, seasons)
    assert out.loc[0, "loan_count"] == 1.0


def test_injury_features_exclude_seasons_after_the_move():
    arrivals = pd.DataFrame({"player": ["A"], "arrival_season": ["2223"]})
    injuries = pd.DataFrame({
        "player": ["A", "A"],
        "season": ["2122", "2324"],
        "days_missed": [30, 200],
        "games_missed": [5, 40],
    })
    out = off.build_injury(arrivals, injuries)
    assert out.loc[0, "injury_days_prior"] == 30.0
    assert out.loc[0, "injury_longest_absence"] == 30.0


def test_availability_separates_missed_while_fit_from_injury():
    arrivals = pd.DataFrame({"player": ["A"], "arrival_season": ["2223"]})
    seasons = pd.DataFrame({
        "player": ["A"], "season": ["2122"], "team": ["X"], "league": ["ITA-Serie A"],
        "minutes": [900], "playing_time_mp": [10], "team_matches": [38],
    })
    injuries = pd.DataFrame({"player": ["A"], "season": ["2122"], "days_missed": [30], "games_missed": [5]})
    out = off.build_availability(arrivals, seasons, injuries)
    assert out.loc[0, "appearance_share"] == pytest.approx(10 / 38)
    assert out.loc[0, "fixtures_missed_while_fit"] == 23.0


def test_value_uses_most_recent_valuation_before_the_move():
    arrivals = pd.DataFrame({"player": ["A"], "arrival_season": ["2223"]})
    values = pd.DataFrame({
        "player": ["A", "A", "A"],
        "date": pd.to_datetime(["2020-07-01", "2022-05-01", "2023-01-01"]),
        "value_eur": [1e6, 8e6, 40e6],
    })
    out = off.build_value(arrivals, values)
    assert out.loc[0, "value_at_move"] == 8e6
    assert out.loc[0, "value_trend_2y"] > 1.0


def test_bundles_partition_all_off_field_columns():
    all_columns = [c for cols in off.BUNDLES.values() for c in cols]
    assert len(all_columns) == len(set(all_columns))
```

`test_injury_features_exclude_seasons_after_the_move` and
`test_value_uses_most_recent_valuation_before_the_move` are the leakage tests.
They are the reason this module is worth writing test-first.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_features_offfield.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.features_offfield'`

- [ ] **Step 3: Write `src/features_offfield.py`**

```python
import numpy as np
import pandas as pd

from src import transfers

_ROMANCE = ["ITA", "ESP", "ARG", "BRA", "FRA", "POR", "ROU", "URU", "COL", "CHI",
            "MEX", "PER", "ECU", "PAR", "VEN", "BEL", "SUI", "MDA", "CPV", "ANG"]
_GERMANIC = ["GER", "NED", "ENG", "SCO", "WAL", "IRL", "DEN", "SWE", "NOR", "ISL",
             "AUT", "USA", "CAN", "AUS", "NZL", "RSA"]
_SLAVIC = ["POL", "SRB", "CRO", "SVN", "SVK", "CZE", "RUS", "UKR", "BUL", "BIH",
           "MKD", "MNE", "BLR"]

LANGUAGE_FAMILY: dict[str, str] = (
    {n: "romance" for n in _ROMANCE}
    | {n: "germanic" for n in _GERMANIC}
    | {n: "slavic" for n in _SLAVIC}
)

BUNDLES = {
    "adaptation": ["first_move_abroad", "n_countries", "language_romance",
                   "compatriots_in_squad", "age_first_pro_season"],
    "trajectory": ["clubs_per_year", "prior_failed_transfer", "loan_count",
                   "lost_and_regained_place", "career_minutes"],
    "injury": ["injury_days_prior", "injury_spells_prior", "injury_longest_absence"],
    "availability": ["appearance_share", "longest_missed_run", "fixtures_missed_while_fit"],
    "value": ["value_at_move", "value_trend_2y"],
}


def _prior(seasons: pd.DataFrame, player: str, arrival_season: str) -> pd.DataFrame:
    """Every recorded season strictly before the move. The leakage boundary."""
    cutoff = int(arrival_season[:2])
    rows = seasons[seasons["player"] == player]
    return rows[rows["season"].str[:2].astype(int) < cutoff]


def build_adaptation(arrivals, seasons, squad_nationalities) -> pd.DataFrame:
    rows = []
    for _, arrival in arrivals.iterrows():
        history = _prior(seasons, arrival["player"], arrival["arrival_season"])
        countries = history["league"].str[:3].unique()
        nat = arrival.get("nationality", "")
        compatriots = squad_nationalities[
            (squad_nationalities["team"] == arrival["to_team"])
            & (squad_nationalities["season"] == arrival["arrival_season"])
            & (squad_nationalities["nationality"] == nat)
        ]["n"].sum()
        rows.append({
            "player": arrival["player"],
            "arrival_season": arrival["arrival_season"],
            "first_move_abroad": float(len(countries) <= 1),
            "n_countries": float(max(len(countries), 1)),
            "language_romance": float(LANGUAGE_FAMILY.get(nat, "other") == "romance"),
            "compatriots_in_squad": float(compatriots),
            "age_first_pro_season": float(len(history)),
        })
    return pd.DataFrame(rows)


def build_trajectory(arrivals, seasons) -> pd.DataFrame:
    rows = []
    for _, arrival in arrivals.iterrows():
        history = _prior(seasons, arrival["player"], arrival["arrival_season"]).sort_values("season")
        clubs = history["team"].tolist()
        n_seasons = max(len(history), 1)
        loans = sum(
            1 for i in range(1, len(clubs) - 1)
            if clubs[i] != clubs[i - 1] and clubs[i + 1] == clubs[i - 1]
        )
        minutes = history["minutes"].tolist()
        regained = 0.0
        for i in range(1, len(minutes) - 1):
            if minutes[i] < 0.6 * minutes[i - 1] and minutes[i + 1] > minutes[i - 1]:
                regained = 1.0
        rows.append({
            "player": arrival["player"],
            "arrival_season": arrival["arrival_season"],
            "clubs_per_year": len(set(clubs)) / n_seasons,
            "prior_failed_transfer": float(loans > 0 and len(set(clubs)) > 2),
            "loan_count": float(loans),
            "lost_and_regained_place": regained,
            "career_minutes": float(history["minutes"].sum()),
        })
    return pd.DataFrame(rows)


def build_injury(arrivals, injuries) -> pd.DataFrame:
    rows = []
    for _, arrival in arrivals.iterrows():
        prior = _prior(injuries, arrival["player"], arrival["arrival_season"])
        rows.append({
            "player": arrival["player"],
            "arrival_season": arrival["arrival_season"],
            "injury_days_prior": float(prior["days_missed"].sum()),
            "injury_spells_prior": float(len(prior)),
            "injury_longest_absence": float(prior["days_missed"].max()) if len(prior) else 0.0,
        })
    return pd.DataFrame(rows)


def build_availability(arrivals, seasons, injuries) -> pd.DataFrame:
    rows = []
    for _, arrival in arrivals.iterrows():
        history = _prior(seasons, arrival["player"], arrival["arrival_season"])
        played = history["playing_time_mp"].sum() if "playing_time_mp" in history else 0.0
        available = history["team_matches"].sum() if "team_matches" in history else 0.0
        injury_rows = _prior(injuries, arrival["player"], arrival["arrival_season"])
        missed_injured = float(injury_rows["games_missed"].sum())
        missed_total = float(available - played)
        rows.append({
            "player": arrival["player"],
            "arrival_season": arrival["arrival_season"],
            "appearance_share": float(played / available) if available else np.nan,
            "longest_missed_run": missed_total,
            "fixtures_missed_while_fit": max(missed_total - missed_injured, 0.0),
        })
    return pd.DataFrame(rows)


def build_value(arrivals, values) -> pd.DataFrame:
    """Value as of the move, plus two-year growth ratio."""
    rows = []
    for _, arrival in arrivals.iterrows():
        cutoff = pd.Timestamp(year=2000 + int(arrival["arrival_season"][:2]), month=7, day=1)
        history = values[(values["player"] == arrival["player"]) & (values["date"] < cutoff)]
        history = history.sort_values("date")
        if history.empty:
            rows.append({
                "player": arrival["player"], "arrival_season": arrival["arrival_season"],
                "value_at_move": np.nan, "value_trend_2y": np.nan,
            })
            continue
        latest = history.iloc[-1]["value_eur"]
        two_years_ago = history[history["date"] >= cutoff - pd.Timedelta(days=730)]
        baseline = two_years_ago.iloc[0]["value_eur"] if len(two_years_ago) else latest
        rows.append({
            "player": arrival["player"],
            "arrival_season": arrival["arrival_season"],
            "value_at_move": float(latest),
            "value_trend_2y": float(latest / baseline) if baseline else 1.0,
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_features_offfield.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/features_offfield.py tests/test_features_offfield.py
git commit -m "feat: adaptation, trajectory, injury, availability and value bundles"
```

---

### Task 11: Club style vectors and tactical presets

**Files:**
- Create: `src/clubs.py`, `src/presets.py`, `tests/test_clubs.py`

**Interfaces:**
- Consumes: `fbref.team_season_stats`, `normalize.percentile_within`
- Produces: `clubs.STYLE_COLUMNS: tuple[str, ...]`, `clubs.build_style_vectors(team_stats: pd.DataFrame) -> pd.DataFrame` with columns `team, season` plus one `*_pct` column per style dimension, `clubs.vector_for(styles, team, season) -> np.ndarray`, `presets.PRESETS: dict[str, np.ndarray]`, `presets.get(name: str) -> np.ndarray`

Presets and real clubs must live in the same vector space and the same ordering,
or a preset query silently means something other than it says.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_clubs.py
import numpy as np
import pandas as pd
import pytest
from src import clubs, presets


def _team_stats(n=4):
    return pd.DataFrame({
        "team": [f"T{i}" for i in range(n)],
        "league": ["ITA-Serie A"] * n,
        "season": ["2223"] * n,
        "possession": [40.0 + i * 5 for i in range(n)],
        "total_passes_att": [400.0 + i * 50 for i in range(n)],
        "total_pass_len_avg": [22.0 - i for i in range(n)],
        "progression_prgp": [30.0 + i * 5 for i in range(n)],
        "pass_types_crs": [15.0 + i for i in range(n)],
        "tackles_att_3rd": [3.0 + i for i in range(n)],
        "expected_npxg": [1.0 + i * 0.3 for i in range(n)],
    })


def test_style_vectors_are_percentiles():
    out = clubs.build_style_vectors(_team_stats())
    values = out[[f"{c}_pct" for c in clubs.STYLE_COLUMNS]].to_numpy()
    assert ((values >= 0.0) & (values <= 1.0)).all()


def test_vector_for_returns_correct_length():
    out = clubs.build_style_vectors(_team_stats())
    vector = clubs.vector_for(out, "T0", "2223")
    assert vector.shape == (len(clubs.STYLE_COLUMNS),)


def test_vector_for_raises_on_unknown_club():
    out = clubs.build_style_vectors(_team_stats())
    with pytest.raises(KeyError):
        clubs.vector_for(out, "Nonexistent", "2223")


def test_every_preset_matches_the_club_vector_space():
    for name, vector in presets.PRESETS.items():
        assert vector.shape == (len(clubs.STYLE_COLUMNS),), name
        assert ((vector >= 0.0) & (vector <= 1.0)).all(), name


def test_presets_cover_the_five_named_in_the_spec():
    assert set(presets.PRESETS) == {
        "high-press-possession", "low-block-counter", "direct-vertical",
        "positional-control", "wide-overload",
    }


def test_preset_get_rejects_unknown_name():
    with pytest.raises(KeyError):
        presets.get("tiki-taka")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_clubs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.clubs'`

- [ ] **Step 3: Write `src/clubs.py`**

```python
import numpy as np
import pandas as pd

from src import normalize

STYLE_COLUMNS = (
    "possession",            # share of the ball
    "total_passes_att",      # passing volume
    "total_pass_len_avg",    # directness (high = long)
    "progression_prgp",      # progressive passing
    "pass_types_crs",        # crossing volume
    "tackles_att_3rd",       # pressing high up
    "expected_npxg",         # attacking quality
)

VECTOR_COLUMNS = [f"{c}_pct" for c in STYLE_COLUMNS]


def build_style_vectors(team_stats: pd.DataFrame) -> pd.DataFrame:
    """Percentile-rank each club against its own league-season."""
    frame = team_stats.copy()
    for column in STYLE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame["position_group"] = "team"
    ranked = normalize.percentile_within(frame, list(STYLE_COLUMNS))
    ranked[VECTOR_COLUMNS] = ranked[VECTOR_COLUMNS].fillna(0.5)
    return ranked[["team", "season", *VECTOR_COLUMNS]].reset_index(drop=True)


def vector_for(styles: pd.DataFrame, team: str, season: str) -> np.ndarray:
    match = styles[(styles["team"] == team) & (styles["season"] == season)]
    if match.empty:
        raise KeyError(f"no style vector for {team} in {season}")
    return match.iloc[0][VECTOR_COLUMNS].to_numpy(dtype=np.float32)
```

- [ ] **Step 4: Write `src/presets.py`**

```python
import numpy as np

from src.clubs import STYLE_COLUMNS

# Ordering matches clubs.STYLE_COLUMNS exactly:
# possession, passing volume, pass length, progressive passing, crossing,
# attacking-third tackles, npxG.
PRESETS: dict[str, np.ndarray] = {
    "high-press-possession": np.array([0.90, 0.90, 0.15, 0.85, 0.50, 0.95, 0.85], dtype=np.float32),
    "low-block-counter":     np.array([0.15, 0.15, 0.85, 0.30, 0.35, 0.10, 0.40], dtype=np.float32),
    "direct-vertical":       np.array([0.40, 0.35, 0.90, 0.70, 0.55, 0.55, 0.65], dtype=np.float32),
    "positional-control":    np.array([0.95, 0.95, 0.10, 0.80, 0.40, 0.60, 0.75], dtype=np.float32),
    "wide-overload":         np.array([0.60, 0.65, 0.45, 0.60, 0.95, 0.60, 0.70], dtype=np.float32),
}

assert all(v.shape == (len(STYLE_COLUMNS),) for v in PRESETS.values())


def get(name: str) -> np.ndarray:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
    return PRESETS[name]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_clubs.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add src/clubs.py src/presets.py tests/test_clubs.py
git commit -m "feat: club style vectors and tactical preset definitions"
```

---

### Task 12: The three targets and the step-up check

**Files:**
- Create: `src/labels.py`, `tests/test_labels.py`

**Interfaces:**
- Consumes: `transfers.previous_season`, `config.LABEL_WINDOW_SEASONS`, `config.TARGETS`
- Produces: `labels.next_season(season: str) -> str`, `labels.window_seasons(arrival_season: str) -> list[str]`, `labels.minutes_target(...) -> float`, `labels.output_target(...) -> float`, `labels.retention_target(...) -> float`, `labels.build(arrivals, seasons, club_values, fees) -> pd.DataFrame` with columns `player, arrival_season, minutes, output, retention`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_labels.py
import pandas as pd
import pytest
from src import labels


def test_window_is_two_seasons_after_arrival():
    assert labels.window_seasons("2223") == ["2223", "2324"]


def test_minutes_target_is_share_of_available():
    value = labels.minutes_target(minutes_played=3420, team_matches=76)
    assert value == pytest.approx(0.5)


def test_minutes_target_clamps_at_one():
    assert labels.minutes_target(minutes_played=99999, team_matches=76) == 1.0


def test_minutes_outside_serie_a_do_not_count():
    seasons = pd.DataFrame({
        "player": ["A", "A"],
        "season": ["2223", "2324"],
        "league": ["ITA-Serie A", "ENG-Premier League"],
        "team": ["Napoli", "Chelsea"],
        "minutes": [2000, 3000],
        "team_matches": [38, 38],
    })
    played = labels.serie_a_minutes(seasons, "A", ["2223", "2324"])
    assert played == 2000


def test_retention_stayed_at_signing_club():
    assert labels.retention_target(
        end_team="Napoli", signing_team="Napoli", end_league="ITA-Serie A",
        fee_in=10e6, fee_out=None, club_value_from=500e6, club_value_to=None,
    ) == 1.0


def test_retention_step_up_exit_scores_full_marks():
    # Kvaratskhelia: Napoli -> PSG at a profit. Must not read as failure.
    assert labels.retention_target(
        end_team="Paris Saint-Germain", signing_team="Napoli", end_league="FRA-Ligue 1",
        fee_in=10e6, fee_out=70e6, club_value_from=500e6, club_value_to=1100e6,
    ) == 1.0


def test_retention_move_within_serie_a():
    assert labels.retention_target(
        end_team="Torino", signing_team="Napoli", end_league="ITA-Serie A",
        fee_in=10e6, fee_out=8e6, club_value_from=500e6, club_value_to=200e6,
    ) == pytest.approx(0.66, abs=0.01)


def test_retention_lateral_exit_scores_partial():
    assert labels.retention_target(
        end_team="Fenerbahce", signing_team="Napoli", end_league="TUR-Super Lig",
        fee_in=10e6, fee_out=10e6, club_value_from=500e6, club_value_to=200e6,
    ) == pytest.approx(0.33, abs=0.01)


def test_retention_sold_at_a_loss_to_lesser_club_is_zero():
    assert labels.retention_target(
        end_team="Salernitana B", signing_team="Napoli", end_league="ESP-Segunda",
        fee_in=30e6, fee_out=3e6, club_value_from=500e6, club_value_to=40e6,
    ) == 0.0


def test_retention_free_exit_is_zero_regardless_of_destination():
    assert labels.retention_target(
        end_team="Real Madrid", signing_team="Napoli", end_league="ESP-La Liga",
        fee_in=30e6, fee_out=0.0, club_value_from=500e6, club_value_to=1200e6,
    ) == 0.0


def test_post_output_ranks_attackers_against_attackers():
    arrivals = pd.DataFrame({
        "player": ["A", "B"], "arrival_season": ["2223", "2223"],
    })
    seasons = pd.DataFrame({
        "player": ["A", "B"], "season": ["2223", "2223"],
        "league": ["ITA-Serie A", "ITA-Serie A"], "pos": ["FW", "FW"],
        "minutes": [900, 900],
        "expected_npxg": [10.0, 1.0], "expected_xag": [5.0, 1.0],
        "progression_prgp": [0.0, 0.0], "progression_prgc": [0.0, 0.0],
    })
    out = labels.post_output_percentile(seasons, arrivals)
    a = out.loc[out.player == "A", "post_output_pct"].item()
    b = out.loc[out.player == "B", "post_output_pct"].item()
    assert a > b


def test_post_output_is_zero_when_player_never_played_in_serie_a():
    arrivals = pd.DataFrame({"player": ["A"], "arrival_season": ["2223"]})
    seasons = pd.DataFrame(columns=["player", "season", "league", "pos", "minutes",
                                    "expected_npxg", "expected_xag",
                                    "progression_prgp", "progression_prgc"])
    out = labels.post_output_percentile(seasons, arrivals)
    assert out["post_output_pct"].iloc[0] == 0.0
```

The three Kvaratskhelia-shaped tests are the point of this task. If
`test_retention_step_up_exit_scores_full_marks` ever goes red, the label is
wrong, not the test.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.labels'`

- [ ] **Step 3: Write `src/labels.py`**

```python
import numpy as np
import pandas as pd

from src import config

MINUTES_PER_MATCH = 90.0


def next_season(season: str) -> str:
    end = int(season[2:])
    return f"{end:02d}{end + 1:02d}"


def window_seasons(arrival_season: str) -> list[str]:
    seasons = [arrival_season]
    for _ in range(config.LABEL_WINDOW_SEASONS - 1):
        seasons.append(next_season(seasons[-1]))
    return seasons


def serie_a_minutes(seasons: pd.DataFrame, player: str, window: list[str]) -> float:
    rows = seasons[
        (seasons["player"] == player)
        & (seasons["season"].isin(window))
        & (seasons["league"] == config.SERIE_A)
    ]
    return float(rows["minutes"].sum())


def minutes_target(minutes_played: float, team_matches: float) -> float:
    available = team_matches * MINUTES_PER_MATCH
    if available <= 0:
        return 0.0
    return float(min(minutes_played / available, 1.0))


def post_output_percentile(seasons: pd.DataFrame, arrivals: pd.DataFrame) -> pd.DataFrame:
    """Position-adjusted Serie A production over the label window.

    Attackers are judged on npxG+xAG per 90; everyone else on progressive
    actions per 90, because ranking a centre-back on expected goals measures
    nothing. Ranked within position group across the cohort so the two scales
    are comparable.
    """
    from src import normalize

    rows = []
    for _, arrival in arrivals.iterrows():
        window = window_seasons(arrival["arrival_season"])
        played = seasons[
            (seasons["player"] == arrival["player"])
            & (seasons["season"].isin(window))
            & (seasons["league"] == config.SERIE_A)
        ]
        minutes = float(played["minutes"].sum()) if len(played) else 0.0
        if minutes <= 0:
            rows.append({"player": arrival["player"],
                         "arrival_season": arrival["arrival_season"],
                         "position_group": "unknown", "raw": np.nan})
            continue
        group = normalize.position_group(played.iloc[0]["pos"])
        if group == "attacker":
            total = played["expected_npxg"].sum() + played["expected_xag"].sum()
        else:
            total = played["progression_prgp"].sum() + played["progression_prgc"].sum()
        rows.append({"player": arrival["player"],
                     "arrival_season": arrival["arrival_season"],
                     "position_group": group, "raw": float(total) / minutes * 90.0})

    frame = pd.DataFrame(rows)
    frame["post_output_pct"] = frame.groupby("position_group")["raw"].rank(pct=True)
    # No Serie A minutes at all is a real outcome, not missing data: zero output.
    frame["post_output_pct"] = frame["post_output_pct"].fillna(0.0)
    return frame[["player", "arrival_season", "post_output_pct"]]


def output_target(percentile: float) -> float:
    """Clamp the ranked production value from `post_output_percentile`."""
    return 0.0 if pd.isna(percentile) else float(np.clip(percentile, 0.0, 1.0))


def retention_target(
    end_team: str,
    signing_team: str,
    end_league: str,
    fee_in: float | None,
    fee_out: float | None,
    club_value_from: float | None,
    club_value_to: float | None,
) -> float:
    """Four-rung ladder. See the spec's step-up section for the reasoning.

    A profitable sale to a bigger club is a signing that worked — Kvaratskhelia
    leaving Napoli for PSG is the canonical case, and scoring it as failure would
    label the cohort's most celebrated transfer a bust.
    """
    if end_team == signing_team:
        return 1.0

    left_for_free = fee_out is not None and fee_out <= 0
    if left_for_free:
        return 0.0

    if end_league == config.SERIE_A:
        return 0.66

    bigger = (
        club_value_to is not None
        and club_value_from is not None
        and club_value_to >= club_value_from
    )
    profitable = fee_out is not None and fee_in is not None and fee_out > fee_in

    if bigger and profitable:
        return 1.0
    if profitable or bigger:
        return 0.33
    return 0.0


def build(arrivals, seasons, club_values, fees) -> pd.DataFrame:
    arrivals = arrivals.merge(
        post_output_percentile(seasons, arrivals),
        on=["player", "arrival_season"], how="left",
    )
    rows = []
    for _, arrival in arrivals.iterrows():
        window = window_seasons(arrival["arrival_season"])
        played = serie_a_minutes(seasons, arrival["player"], window)
        team_rows = seasons[
            (seasons["team"] == arrival["to_team"]) & (seasons["season"].isin(window))
        ]
        team_matches = float(team_rows["team_matches"].max() or 0) * len(window)

        after = seasons[
            (seasons["player"] == arrival["player"]) & (seasons["season"] == window[-1])
        ]
        end_team = after.iloc[0]["team"] if len(after) else ""
        end_league = after.iloc[0]["league"] if len(after) else ""

        fee_row = fees[
            (fees["player"] == arrival["player"]) & (fees["season"] == arrival["arrival_season"])
        ]
        fee_in = float(fee_row["fee_eur"].iloc[0]) if len(fee_row) else None
        exit_row = fees[
            (fees["player"] == arrival["player"]) & (fees["season"] == next_season(window[-1]))
        ]
        fee_out = float(exit_row["fee_eur"].iloc[0]) if len(exit_row) else None

        def club_value(team: str) -> float | None:
            match = club_values[club_values["team"] == team]
            return float(match["squad_value_eur"].iloc[0]) if len(match) else None

        rows.append({
            "player": arrival["player"],
            "arrival_season": arrival["arrival_season"],
            "minutes": minutes_target(played, team_matches),
            "output": output_target(arrival.get("post_output_pct", np.nan)),
            "retention": retention_target(
                end_team=end_team,
                signing_team=arrival["to_team"],
                end_league=end_league,
                fee_in=fee_in,
                fee_out=fee_out,
                club_value_from=club_value(arrival["to_team"]),
                club_value_to=club_value(end_team),
            ),
        })
    return pd.DataFrame(rows, columns=["player", "arrival_season", *config.TARGETS])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_labels.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add src/labels.py tests/test_labels.py
git commit -m "feat: three prediction targets with step-up-aware retention ladder"
```

---
### Task 13: Dataset assembly, panel exclusion and grouped splits

**Files:**
- Create: `src/dataset.py`, `eval/panel.yaml`, `tests/test_dataset.py`

**Interfaces:**
- Consumes: `features_onfield.FEATURE_COLUMNS`, `features_offfield.BUNDLES`, `clubs.VECTOR_COLUMNS`, `config.TARGETS`
- Produces: `dataset.load_panel(path) -> list[str]`, `dataset.exclude_panel(frame, panel) -> pd.DataFrame`, `dataset.grouped_folds(frame, n_splits=5, seed=0) -> list[tuple[np.ndarray, np.ndarray]]`, `dataset.TransferDataset(frame, player_columns, club_columns)` (a `torch.utils.data.Dataset` yielding `(player_x, club_x, y)` float32 tensors), `dataset.fit_scaler(train_frame, columns)`, `dataset.apply_scaler(frame, columns, scaler)`

- [ ] **Step 1: Write `eval/panel.yaml`**

```yaml
# Players withheld from training entirely — every row, every fold.
# A qualitative smell test, never a metric.
#
# Verify each name exists in data/processed/transfers.parquet before relying on
# it. Kvaratskhelia's Dinamo Batumi season is almost certainly absent from FBref
# (the Georgian Erovnuli Liga has no advanced stats); his Rubin Kazan seasons may
# serve instead. If neither is present, drop him and note it here.
great:
  - Khvicha Kvaratskhelia
  - Rafael Leao
  - Sofyan Amrabat
ok:
  - Jens Petter Hauge
  - Emil Audero
bust:
  - Lucas Paqueta
  - Gonzalo Higuain
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_dataset.py
import numpy as np
import pandas as pd
import pytest
import torch
from src import dataset


def _frame(n=20):
    return pd.DataFrame({
        "player": [f"p{i % 10}" for i in range(n)],
        "arrival_season": ["2223"] * n,
        "f1": np.linspace(0, 1, n),
        "f2": np.linspace(1, 0, n),
        "club_c1_pct": np.linspace(0, 1, n),
        "minutes": np.linspace(0, 1, n),
        "output": np.linspace(0, 1, n),
        "retention": [0.0, 0.33, 0.66, 1.0] * (n // 4),
    })


def test_load_panel_flattens_all_tiers(tmp_path):
    path = tmp_path / "panel.yaml"
    path.write_text("great:\n  - A\nok:\n  - B\nbust:\n  - C\n")
    assert dataset.load_panel(path) == ["A", "B", "C"]


def test_exclude_panel_removes_every_row_for_those_players():
    frame = _frame()
    out = dataset.exclude_panel(frame, ["p0", "p1"])
    assert not out["player"].isin(["p0", "p1"]).any()
    assert len(out) < len(frame)


def test_grouped_folds_never_split_a_player_across_train_and_test():
    frame = _frame()
    for train_idx, test_idx in dataset.grouped_folds(frame, n_splits=5):
        train_players = set(frame.iloc[train_idx]["player"])
        test_players = set(frame.iloc[test_idx]["player"])
        assert train_players.isdisjoint(test_players)


def test_grouped_folds_cover_every_row_exactly_once_across_folds():
    frame = _frame()
    seen = np.concatenate([test for _, test in dataset.grouped_folds(frame, n_splits=5)])
    assert sorted(seen.tolist()) == list(range(len(frame)))


def test_dataset_yields_correct_tensor_shapes():
    frame = _frame()
    ds = dataset.TransferDataset(frame, ["f1", "f2"], ["club_c1_pct"])
    player_x, club_x, y = ds[0]
    assert player_x.shape == (2,)
    assert club_x.shape == (1,)
    assert y.shape == (3,)
    assert player_x.dtype == torch.float32


def test_scaler_is_fitted_on_train_only():
    train = _frame(20)
    test = _frame(20)
    test["f1"] = test["f1"] * 100
    scaler = dataset.fit_scaler(train, ["f1", "f2"])
    scaled = dataset.apply_scaler(test, ["f1", "f2"], scaler)
    # Test values far outside the training range must NOT be squashed into [0,1].
    assert scaled["f1"].max() > 2.0
```

The last test guards a subtle leak: fitting a scaler on the full frame before
splitting lets test-set statistics inform training.

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.dataset'`

- [ ] **Step 4: Write `src/dataset.py`**

```python
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from src import config


def load_panel(path: Path | str) -> list[str]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    names: list[str] = []
    for tier in ("great", "ok", "bust"):
        names.extend(data.get(tier) or [])
    return names


def exclude_panel(frame: pd.DataFrame, panel: list[str]) -> pd.DataFrame:
    return frame[~frame["player"].isin(panel)].reset_index(drop=True)


def grouped_folds(frame: pd.DataFrame, n_splits: int = 5, seed: int = 0):
    """Splits grouped by player. Never split by row — see Global Constraints."""
    splitter = GroupKFold(n_splits=n_splits)
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
    def __init__(self, frame: pd.DataFrame, player_columns: list[str], club_columns: list[str]):
        self.player_x = frame[player_columns].to_numpy(dtype=np.float32)
        self.club_x = frame[club_columns].to_numpy(dtype=np.float32)
        self.y = frame[list(config.TARGETS)].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.player_x[index]),
            torch.from_numpy(self.club_x[index]),
            torch.from_numpy(self.y[index]),
        )
```

Add `pyyaml` to the venv: `.venv/bin/pip install pyyaml`

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_dataset.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add src/dataset.py eval/panel.yaml tests/test_dataset.py
git commit -m "feat: dataset assembly, panel exclusion, player-grouped splits"
```

---

### Task 14: Baselines

**Files:**
- Create: `src/baselines.py`, `tests/test_baselines.py`

**Interfaces:**
- Consumes: `dataset.grouped_folds`, `dataset.fit_scaler`, `config.TARGETS`
- Produces: `baselines.spearman(y_true, y_pred) -> float`, `baselines.evaluate(frame, feature_columns, model_name: str) -> dict[str, dict[str, float]]` keyed by target then metric, `baselines.MODELS: dict[str, Callable]`

Built **before** the network, per Global Constraints. These numbers are the bar.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baselines.py
import numpy as np
import pandas as pd
import pytest
from src import baselines


def _frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    f1 = rng.random(n)
    return pd.DataFrame({
        "player": [f"p{i}" for i in range(n)],
        "f1": f1,
        "f2": rng.random(n),
        "minutes": f1 * 0.8 + rng.random(n) * 0.2,
        "output": f1 * 0.5 + rng.random(n) * 0.5,
        "retention": rng.choice([0.0, 0.33, 0.66, 1.0], n),
    })


def test_spearman_of_perfect_ranking_is_one():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert baselines.spearman(y, y * 10) == pytest.approx(1.0)


def test_spearman_of_reversed_ranking_is_minus_one():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert baselines.spearman(y, -y) == pytest.approx(-1.0)


def test_spearman_of_constant_prediction_is_zero_not_nan():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert baselines.spearman(y, np.zeros(4)) == 0.0


def test_models_registry_has_ridge_and_gbm():
    assert set(baselines.MODELS) == {"ridge", "gbm"}


def test_evaluate_reports_every_target():
    result = baselines.evaluate(_frame(), ["f1", "f2"], "ridge")
    assert set(result) == {"minutes", "output", "retention"}
    for metrics in result.values():
        assert set(metrics) == {"spearman", "mae"}


def test_evaluate_recovers_a_planted_signal():
    result = baselines.evaluate(_frame(n=120), ["f1", "f2"], "ridge")
    assert result["minutes"]["spearman"] > 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_baselines.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.baselines'`

- [ ] **Step 3: Write `src/baselines.py`**

```python
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from src import config, dataset

MODELS = {
    "ridge": lambda: Ridge(alpha=10.0),
    "gbm": lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05),
}


def spearman(y_true, y_pred) -> float:
    """Rank correlation, with a constant prediction reported as 0 rather than NaN."""
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        return 0.0
    value = spearmanr(y_true, y_pred).statistic
    return 0.0 if np.isnan(value) else float(value)


def evaluate(frame: pd.DataFrame, feature_columns: list[str], model_name: str) -> dict:
    """Cross-validated per-target metrics on player-grouped folds."""
    predictions = {t: np.zeros(len(frame)) for t in config.TARGETS}
    for train_idx, test_idx in dataset.grouped_folds(frame):
        train, test = frame.iloc[train_idx], frame.iloc[test_idx]
        scaler = dataset.fit_scaler(train, feature_columns)
        x_train = scaler.transform(train[feature_columns].to_numpy(dtype=np.float64))
        x_test = scaler.transform(test[feature_columns].to_numpy(dtype=np.float64))
        for target in config.TARGETS:
            model = MODELS[model_name]()
            model.fit(x_train, train[target].to_numpy())
            predictions[target][test_idx] = model.predict(x_test)

    return {
        target: {
            "spearman": spearman(frame[target].to_numpy(), predictions[target]),
            "mae": float(np.mean(np.abs(frame[target].to_numpy() - predictions[target]))),
        }
        for target in config.TARGETS
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_baselines.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/baselines.py tests/test_baselines.py
git commit -m "feat: ridge and gradient-boosting baselines with grouped CV"
```

---

### Task 15: The conditioned two-tower network

**Files:**
- Create: `src/model.py`, `tests/test_model.py`

**Interfaces:**
- Consumes: `config.TARGETS`
- Produces: `model.TwoTowerNet(n_player_features: int, n_club_features: int, hidden: int = 64, embed: int = 32, dropout: float = 0.3)` with `forward(player_x, club_x) -> torch.Tensor` of shape `(batch, 3)`, `model.combined_loss(pred, target) -> torch.Tensor`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py
import pytest
import torch
from src import model


def test_forward_returns_one_value_per_target():
    net = model.TwoTowerNet(n_player_features=20, n_club_features=7)
    out = net(torch.randn(8, 20), torch.randn(8, 7))
    assert out.shape == (8, 3)


def test_outputs_are_bounded_to_unit_interval():
    net = model.TwoTowerNet(n_player_features=20, n_club_features=7)
    out = net(torch.randn(64, 20) * 50, torch.randn(64, 7) * 50)
    assert (out >= 0).all() and (out <= 1).all()


def test_club_input_actually_changes_the_prediction():
    torch.manual_seed(0)
    net = model.TwoTowerNet(n_player_features=20, n_club_features=7).eval()
    player = torch.randn(4, 20)
    a = net(player, torch.zeros(4, 7))
    b = net(player, torch.ones(4, 7))
    assert not torch.allclose(a, b), "conditioning on the club has no effect"


def test_gradients_reach_both_towers():
    net = model.TwoTowerNet(n_player_features=20, n_club_features=7)
    out = net(torch.randn(4, 20), torch.randn(4, 7))
    out.sum().backward()
    assert net.player_tower[0].weight.grad is not None
    assert net.club_tower[0].weight.grad is not None
    assert net.club_tower[0].weight.grad.abs().sum() > 0


def test_combined_loss_is_zero_for_perfect_prediction():
    y = torch.tensor([[0.5, 0.5, 1.0]])
    assert model.combined_loss(y, y).item() == pytest.approx(0.0, abs=1e-6)


def test_combined_loss_weights_all_three_heads_equally():
    y = torch.tensor([[0.0, 0.0, 0.0]])
    only_first = model.combined_loss(torch.tensor([[0.5, 0.0, 0.0]]), y)
    only_second = model.combined_loss(torch.tensor([[0.0, 0.5, 0.0]]), y)
    assert only_first.item() == pytest.approx(only_second.item())
```

`test_club_input_actually_changes_the_prediction` is the one that matters most.
A wiring bug that ignores the club tower would leave every other test green while
silently deleting the entire tactical-fit capability.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.model'`

- [ ] **Step 3: Write `src/model.py`**

```python
import torch
from torch import nn

from src import config

N_TARGETS = len(config.TARGETS)


def _tower(n_in: int, hidden: int, embed: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(n_in, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, embed),
        nn.ReLU(),
    )


class TwoTowerNet(nn.Module):
    """Player and club embedded separately, combined by concatenation plus an
    elementwise product. The product term is where player-by-club interactions
    live — it is the reason the model can answer 'fits THIS side' rather than
    only 'good player'.

    Three heads share one trunk, which is multi-task regularisation: the trunk
    must serve all three outcomes, which is harder to overfit than any one.
    """

    def __init__(self, n_player_features: int, n_club_features: int,
                 hidden: int = 64, embed: int = 32, dropout: float = 0.3):
        super().__init__()
        self.player_tower = _tower(n_player_features, hidden, embed, dropout)
        self.club_tower = _tower(n_club_features, hidden, embed, dropout)
        self.head = nn.Sequential(
            nn.Linear(embed * 3, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, N_TARGETS),
        )

    def forward(self, player_x: torch.Tensor, club_x: torch.Tensor) -> torch.Tensor:
        p = self.player_tower(player_x)
        c = self.club_tower(club_x)
        combined = torch.cat([p, c, p * c], dim=1)
        return torch.sigmoid(self.head(combined))


def combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Unweighted sum across heads.

    Training weights are equal by design and are unrelated to the query-time
    weights the CLI exposes — those combine already-trained predictions.
    """
    return nn.functional.huber_loss(pred, target, delta=0.2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_model.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "feat: conditioned two-tower network with three heads"
```

---

### Task 16: Training, seed ensembling, ablations

**Files:**
- Create: `src/train.py`, `tests/test_train.py`

**Interfaces:**
- Consumes: `model.TwoTowerNet`, `model.combined_loss`, `dataset.*`, `baselines.spearman`, `features_offfield.BUNDLES`, `config.ENSEMBLE_SEEDS`
- Produces: `train.train_one(train_frame, val_frame, player_columns, club_columns, seed, max_epochs=300) -> TwoTowerNet`, `train.predict(nets, frame, player_columns, club_columns) -> tuple[np.ndarray, np.ndarray]` returning `(mean, std)` each shaped `(n, 3)`, `train.cross_validate(frame, player_columns, club_columns, n_seeds) -> dict`, `train.run_ablations(frame, player_columns, club_columns) -> pd.DataFrame`, `train.report(frame, player_columns, club_columns, panel) -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train.py
import numpy as np
import pandas as pd
import pytest
import torch
from src import train


def _frame(n=80, seed=0):
    rng = np.random.default_rng(seed)
    f1 = rng.random(n)
    return pd.DataFrame({
        "player": [f"p{i}" for i in range(n)],
        "f1": f1,
        "f2": rng.random(n),
        "club_a_pct": rng.random(n),
        "minutes": np.clip(f1 * 0.9 + rng.random(n) * 0.1, 0, 1),
        "output": np.clip(f1 * 0.6 + rng.random(n) * 0.4, 0, 1),
        "retention": rng.choice([0.0, 0.33, 0.66, 1.0], n),
    })


PLAYER_COLS = ["f1", "f2"]
CLUB_COLS = ["club_a_pct"]


def test_train_one_returns_a_module_in_eval_mode():
    frame = _frame()
    net = train.train_one(frame[:60], frame[60:], PLAYER_COLS, CLUB_COLS, seed=0, max_epochs=20)
    assert isinstance(net, torch.nn.Module)
    assert not net.training


def test_predict_returns_mean_and_spread_per_target():
    frame = _frame()
    nets = [train.train_one(frame[:60], frame[60:], PLAYER_COLS, CLUB_COLS, s, max_epochs=10)
            for s in range(2)]
    mean, std = train.predict(nets, frame[60:], PLAYER_COLS, CLUB_COLS)
    assert mean.shape == (20, 3)
    assert std.shape == (20, 3)
    assert (std >= 0).all()


def test_training_reduces_loss_on_a_learnable_signal():
    frame = _frame(n=200)
    net = train.train_one(frame[:150], frame[150:], PLAYER_COLS, CLUB_COLS, seed=0, max_epochs=200)
    mean, _ = train.predict([net], frame[150:], PLAYER_COLS, CLUB_COLS)
    from src import baselines
    assert baselines.spearman(frame[150:]["minutes"].to_numpy(), mean[:, 0]) > 0.4


def test_cross_validate_reports_every_target():
    result = train.cross_validate(_frame(), PLAYER_COLS, CLUB_COLS, n_seeds=2, max_epochs=10)
    assert set(result) == {"minutes", "output", "retention"}
    assert set(result["minutes"]) == {"spearman", "mae"}


def test_save_ensemble_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(train.config, "ROOT", tmp_path)
    frame = _frame()
    nets = [train.train_one(frame[:60], frame[60:], PLAYER_COLS, CLUB_COLS, 0, max_epochs=5)]
    train.save_ensemble(nets, len(PLAYER_COLS), len(CLUB_COLS))
    payload = torch.load(tmp_path / "models" / "ensemble.pt", weights_only=False)
    assert payload["n_player"] == 2
    assert payload["n_club"] == 1
    assert len(payload["states"]) == 1


def test_run_ablations_produces_one_row_per_bundle_plus_full():
    frame = _frame()
    out = train.run_ablations(frame, PLAYER_COLS, CLUB_COLS, n_seeds=1, max_epochs=5)
    assert "full" in out["variant"].tolist()
    assert len(out) >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.train'`

- [ ] **Step 3: Write `src/train.py`**

```python
import copy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import baselines, config, dataset, features_offfield, model


def train_one(train_frame, val_frame, player_columns, club_columns,
              seed: int, max_epochs: int = 300, patience: int = 25):
    """Train a single seed with early stopping on the validation fold."""
    torch.manual_seed(seed)
    net = model.TwoTowerNet(len(player_columns), len(club_columns))
    optimiser = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)

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
                   n_seeds: int = config.ENSEMBLE_SEEDS, max_epochs: int = 300) -> dict:
    predictions = np.zeros((len(frame), len(config.TARGETS)))
    for train_idx, test_idx in dataset.grouped_folds(frame):
        train_frame = frame.iloc[train_idx].reset_index(drop=True)
        test_frame = frame.iloc[test_idx].reset_index(drop=True)
        cut = max(int(len(train_frame) * 0.85), 1)
        nets = [
            train_one(train_frame[:cut], train_frame[cut:], player_columns,
                      club_columns, seed, max_epochs)
            for seed in range(n_seeds)
        ]
        mean, _ = predict(nets, test_frame, player_columns, club_columns)
        predictions[test_idx] = mean

    return {
        target: {
            "spearman": baselines.spearman(frame[target].to_numpy(), predictions[:, i]),
            "mae": float(np.mean(np.abs(frame[target].to_numpy() - predictions[:, i]))),
        }
        for i, target in enumerate(config.TARGETS)
    }


def run_ablations(frame, player_columns, club_columns, n_seeds: int = 3,
                  max_epochs: int = 150) -> pd.DataFrame:
    """Full model versus each feature bundle removed."""
    rows = []
    variants = {"full": player_columns}
    for bundle, columns in features_offfield.BUNDLES.items():
        remaining = [c for c in player_columns if c not in columns]
        if remaining and len(remaining) < len(player_columns):
            variants[f"no_{bundle}"] = remaining

    for name, columns in variants.items():
        result = cross_validate(frame, columns, club_columns, n_seeds, max_epochs)
        rows.append({"variant": name, "n_features": len(columns),
                     **{f"{t}_spearman": result[t]["spearman"] for t in config.TARGETS}})
    return pd.DataFrame(rows)


def save_ensemble(nets, n_player: int, n_club: int) -> None:
    """Persist in the exact shape `query.load_ensemble` expects."""
    path = config.ROOT / "models" / "ensemble.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"states": [net.state_dict() for net in nets],
         "n_player": n_player, "n_club": n_club},
        path,
    )


def report(frame, player_columns, club_columns, panel: list[str]) -> None:
    """The full evaluation the spec requires, printed in one place."""
    labelled = dataset.exclude_panel(frame, panel)

    print("=== baselines ===")
    for name in baselines.MODELS:
        print(name, baselines.evaluate(labelled, player_columns, name))

    print("\n=== network ===")
    net_metrics = cross_validate(labelled, player_columns, club_columns)
    print(net_metrics)

    print("\n=== ablations ===")
    print(run_ablations(labelled, player_columns, club_columns).to_string(index=False))

    print("\n=== withheld panel (qualitative, not a metric) ===")
    held = frame[frame["player"].isin(panel)]
    if held.empty:
        print("no panel players present in the dataset — check eval/panel.yaml names")
        return
    cut = max(int(len(labelled) * 0.85), 1)
    nets = [train_one(labelled[:cut], labelled[cut:], player_columns, club_columns, s)
            for s in range(config.ENSEMBLE_SEEDS)]
    mean, std = predict(nets, held, player_columns, club_columns)
    for i, (_, row) in enumerate(held.iterrows()):
        print(f"{row['player']:<28} "
              + "  ".join(f"{t}={mean[i, j]:.2f}±{std[i, j]:.2f}"
                          for j, t in enumerate(config.TARGETS)))

    save_ensemble(nets, len(player_columns), len(club_columns))
    from src import archetypes, features_onfield

    archetypes.fit_all(frame, features_onfield.FEATURE_COLUMNS)
    print(f"\nensemble and archetypes saved under {config.ROOT / 'models'}")


if __name__ == "__main__":
    from src import pipeline

    frame = pd.read_parquet(config.PROCESSED_DIR / "model_frame.parquet")
    report(
        frame,
        pipeline.PLAYER_FEATURE_COLUMNS,
        pipeline.CLUB_FEATURE_COLUMNS,
        dataset.load_panel(config.ROOT / "eval" / "panel.yaml"),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_train.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "feat: training loop, seed ensembling, ablations and evaluation report"
```

---

### Task 17: Archetype clustering

**Files:**
- Create: `src/archetypes.py`, `tests/test_archetypes.py`

**Interfaces:**
- Consumes: `features_onfield.FEATURE_COLUMNS`
- Produces: `archetypes.fit(frame, feature_columns, position_group, k_range=(4, 7)) -> GaussianMixture`, `archetypes.best_k(frame, feature_columns, k_range) -> int`, `archetypes.assign(frame, models: dict[str, GaussianMixture], feature_columns) -> pd.Series`, `archetypes.describe(mixture, feature_columns, top_n=4) -> list[list[str]]`

Descriptive only. Archetypes label CLI output; they never enter the model.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_archetypes.py
import numpy as np
import pandas as pd
import pytest
from src import archetypes


def _clustered(n=120, seed=0):
    rng = np.random.default_rng(seed)
    half = n // 2
    return pd.DataFrame({
        "player": [f"p{i}" for i in range(n)],
        "position_group": ["attacker"] * n,
        "a": np.concatenate([rng.normal(0.2, 0.05, half), rng.normal(0.8, 0.05, n - half)]),
        "b": np.concatenate([rng.normal(0.8, 0.05, half), rng.normal(0.2, 0.05, n - half)]),
    })


def test_best_k_finds_the_planted_cluster_count():
    assert archetypes.best_k(_clustered(), ["a", "b"], k_range=(2, 5)) == 2


def test_assign_labels_every_row():
    frame = _clustered()
    models = {"attacker": archetypes.fit(frame, ["a", "b"], "attacker", k_range=(2, 4))}
    labels = archetypes.assign(frame, models, ["a", "b"])
    assert len(labels) == len(frame)
    assert labels.notna().all()


def test_assign_separates_the_two_planted_groups():
    frame = _clustered()
    models = {"attacker": archetypes.fit(frame, ["a", "b"], "attacker", k_range=(2, 4))}
    labels = archetypes.assign(frame, models, ["a", "b"])
    assert labels[:60].nunique() == 1
    assert labels[:60].iloc[0] != labels[60:].iloc[0]


def test_describe_names_the_dominant_features():
    frame = _clustered()
    mixture = archetypes.fit(frame, ["a", "b"], "attacker", k_range=(2, 4))
    described = archetypes.describe(mixture, ["a", "b"], top_n=1)
    assert len(described) == mixture.n_components
    assert all(len(d) == 1 for d in described)


def test_fit_all_round_trips_through_disk(tmp_path, monkeypatch):
    from src import config as cfg
    monkeypatch.setattr(cfg, "ROOT", tmp_path)
    frame = _clustered()
    archetypes.fit_all(frame, ["a", "b"])
    models, features = archetypes.load_all()
    assert "attacker" in models
    assert features == ["a", "b"]


def test_load_all_returns_empty_when_never_fitted(tmp_path, monkeypatch):
    from src import config as cfg
    monkeypatch.setattr(cfg, "ROOT", tmp_path)
    models, features = archetypes.load_all()
    assert models == {}
    assert features == []


def test_assign_handles_unknown_position_group():
    frame = _clustered()
    frame.loc[0, "position_group"] = "keeper"
    models = {"attacker": archetypes.fit(frame, ["a", "b"], "attacker", k_range=(2, 4))}
    labels = archetypes.assign(frame, models, ["a", "b"])
    assert labels.iloc[0] == "unclustered"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_archetypes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.archetypes'`

- [ ] **Step 3: Write `src/archetypes.py`**

```python
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


def best_k(frame: pd.DataFrame, feature_columns: list[str], k_range=(4, 7)) -> int:
    x = frame[feature_columns].to_numpy(dtype=np.float64)
    scores = {}
    for k in range(k_range[0], k_range[1] + 1):
        if k >= len(x):
            continue
        labels = GaussianMixture(n_components=k, random_state=0).fit_predict(x)
        if len(set(labels)) < 2:
            continue
        scores[k] = silhouette_score(x, labels)
    return max(scores, key=scores.get) if scores else k_range[0]


def fit(frame, feature_columns, position_group: str, k_range=(4, 7)) -> GaussianMixture:
    subset = frame[frame["position_group"] == position_group]
    k = best_k(subset, feature_columns, k_range)
    x = subset[feature_columns].to_numpy(dtype=np.float64)
    return GaussianMixture(n_components=k, random_state=0).fit(x)


def assign(frame, models: dict, feature_columns: list[str]) -> pd.Series:
    labels = pd.Series("unclustered", index=frame.index, dtype=object)
    for group, mixture in models.items():
        mask = frame["position_group"] == group
        if not mask.any():
            continue
        x = frame.loc[mask, feature_columns].to_numpy(dtype=np.float64)
        labels.loc[mask] = [f"{group}-{c}" for c in mixture.predict(x)]
    return labels


def fit_all(frame, feature_columns: list[str]) -> dict:
    """One mixture per position group, persisted for the CLI."""
    import joblib

    from src import config

    models = {}
    for group in frame["position_group"].unique():
        if group == "unknown" or (frame["position_group"] == group).sum() < 20:
            continue
        models[group] = fit(frame, feature_columns, group)
    path = config.ROOT / "models" / "archetypes.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "features": feature_columns}, path)
    return models


def load_all() -> tuple[dict, list[str]]:
    import joblib

    from src import config

    path = config.ROOT / "models" / "archetypes.joblib"
    if not path.exists():
        return {}, []
    payload = joblib.load(path)
    return payload["models"], payload["features"]


def describe(mixture: GaussianMixture, feature_columns: list[str], top_n: int = 4):
    """Features on which each cluster centre sits furthest above the mean —
    the raw material for naming clusters by hand."""
    overall = mixture.means_.mean(axis=0)
    out = []
    for centre in mixture.means_:
        deltas = centre - overall
        order = np.argsort(-deltas)[:top_n]
        out.append([feature_columns[i] for i in order])
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_archetypes.py -v`
Expected: 7 passed

Add `joblib` to the venv: `.venv/bin/pip install joblib`

- [ ] **Step 5: Name the clusters by hand**

After the first real training run, print `describe(...)` for each position group
and write human names into a `ARCHETYPE_NAMES: dict[str, str]` in
`src/archetypes.py` mapping `"attacker-0"` to e.g. `"poacher"`. The CLI prints
whichever name is available and falls back to the raw cluster id.

- [ ] **Step 6: Commit**

```bash
git add src/archetypes.py tests/test_archetypes.py
git commit -m "feat: Gaussian-mixture style archetype clustering"
```

---
### Task 18: Pipeline assembly

**Files:**
- Create: `src/pipeline.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: every builder from Tasks 7–12
- Produces: `pipeline.expand_per_season(arrivals, seasons) -> pd.DataFrame` (one row per player × arrival × pre-move season), `pipeline.assemble(...) -> pd.DataFrame`, `pipeline.PLAYER_FEATURE_COLUMNS: list[str]`, `pipeline.CLUB_FEATURE_COLUMNS: list[str]`, `pipeline.build() -> pd.DataFrame` writing `data/processed/model_frame.parquet`

This is where the spec's "expanded per-player" cohort decision is actually
implemented, and where every bundle is joined into the single frame the model
consumes. It has no clever logic — its whole job is to join things without
losing or duplicating rows, which is exactly what its tests check.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline.py
import pandas as pd
import pytest
from src import pipeline


def _arrivals():
    return pd.DataFrame({
        "player": ["A"], "arrival_season": ["2223"], "to_team": ["Napoli"],
        "from_team": ["Porto"], "from_league": ["POR-Liga"], "prior_minutes": [2000.0],
    })


def _seasons():
    return pd.DataFrame({
        "player": ["A", "A", "A"],
        "team": ["Porto", "Porto", "Napoli"],
        "league": ["POR-Liga", "POR-Liga", "ITA-Serie A"],
        "season": ["2021", "2122", "2223"],
        "minutes": [1800, 2000, 2200],
    })


def test_expand_emits_one_row_per_pre_move_season():
    out = pipeline.expand_per_season(_arrivals(), _seasons())
    assert len(out) == 2
    assert sorted(out["season"]) == ["2021", "2122"]


def test_expand_never_includes_the_arrival_season_itself():
    out = pipeline.expand_per_season(_arrivals(), _seasons())
    assert "2223" not in out["season"].tolist()


def test_expand_keeps_arrival_metadata_on_every_row():
    out = pipeline.expand_per_season(_arrivals(), _seasons())
    assert set(out["to_team"]) == {"Napoli"}
    assert set(out["arrival_season"]) == {"2223"}


def test_expand_drops_arrivals_with_no_pre_move_seasons():
    seasons = _seasons()[_seasons()["season"] == "2223"]
    assert pipeline.expand_per_season(_arrivals(), seasons).empty


def test_assemble_produces_no_duplicate_rows():
    frame = pipeline.expand_per_season(_arrivals(), _seasons())
    assert not frame.duplicated(subset=["player", "arrival_season", "season"]).any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.pipeline'`

- [ ] **Step 3: Write `src/pipeline.py`**

```python
import pandas as pd

from src import (clubs, config, features_offfield, features_onfield, labels,
                 fbref, transfers)

PLAYER_FEATURE_COLUMNS = features_onfield.FEATURE_COLUMNS + [
    column for columns in features_offfield.BUNDLES.values() for column in columns
]
CLUB_FEATURE_COLUMNS = clubs.VECTOR_COLUMNS

_ARRIVAL_META = ["arrival_season", "to_team", "from_team", "from_league"]


def expand_per_season(arrivals: pd.DataFrame, seasons: pd.DataFrame) -> pd.DataFrame:
    """One row per (player, arrival, pre-move season).

    This is the spec's 'expanded per-player' cohort. Rows within a player are
    correlated, which is why every split is grouped by player.
    """
    rows = []
    for _, arrival in arrivals.iterrows():
        cutoff = int(arrival["arrival_season"][:2])
        history = seasons[
            (seasons["player"] == arrival["player"])
            & (seasons["season"].str[:2].astype(int) < cutoff)
        ]
        for _, season_row in history.iterrows():
            rows.append({**season_row.to_dict(),
                         **{k: arrival[k] for k in _ARRIVAL_META}})
    if not rows:
        return pd.DataFrame(columns=[*seasons.columns, *_ARRIVAL_META])
    return pd.DataFrame(rows).reset_index(drop=True)


def assemble(arrivals, seasons, team_stats, squad_nationalities,
             injuries, values, club_values, fees) -> pd.DataFrame:
    on_field = features_onfield.build(seasons)
    expanded = expand_per_season(arrivals, on_field)

    key = ["player", "arrival_season"]
    bundles = [
        features_offfield.build_adaptation(arrivals, seasons, squad_nationalities),
        features_offfield.build_trajectory(arrivals, seasons),
        features_offfield.build_injury(arrivals, injuries),
        features_offfield.build_availability(arrivals, seasons, injuries),
        features_offfield.build_value(arrivals, values),
        labels.build(arrivals, seasons, club_values, fees),
    ]
    for bundle in bundles:
        expanded = expanded.merge(bundle, on=key, how="left")

    styles = clubs.build_style_vectors(team_stats)
    expanded = expanded.merge(
        styles.rename(columns={"team": "to_team", "season": "arrival_season"}),
        on=["to_team", "arrival_season"], how="left",
    )

    expanded[PLAYER_FEATURE_COLUMNS] = expanded[PLAYER_FEATURE_COLUMNS].fillna(
        expanded[PLAYER_FEATURE_COLUMNS].median()
    )
    expanded[CLUB_FEATURE_COLUMNS] = expanded[CLUB_FEATURE_COLUMNS].fillna(0.5)
    return expanded.dropna(subset=list(config.TARGETS)).reset_index(drop=True)


def build() -> pd.DataFrame:
    seasons = pd.concat(
        [fbref.player_season_stats(league, season)
         for league in config.CANDIDATE_LEAGUES for season in config.SEASONS],
        ignore_index=True,
    )
    team_stats = pd.concat(
        [fbref.team_season_stats(config.SERIE_A, season) for season in config.SEASONS],
        ignore_index=True,
    )
    arrivals = transfers.detect_arrivals(seasons)

    squad_nationalities = (
        seasons.groupby(["team", "season", "nationality"]).size()
        .reset_index(name="n")
    )
    injuries = pd.read_parquet(config.PROCESSED_DIR / "injuries.parquet")
    values = pd.read_parquet(config.PROCESSED_DIR / "market_values.parquet")
    club_values = pd.read_parquet(config.PROCESSED_DIR / "club_values.parquet")
    fees = pd.read_parquet(config.PROCESSED_DIR / "fees.parquet")

    frame = assemble(arrivals, seasons, team_stats, squad_nationalities,
                     injuries, values, club_values, fees)
    frame.to_parquet(config.PROCESSED_DIR / "model_frame.parquet")
    # query.py --club reads this to build a real club's style vector.
    team_stats.to_parquet(config.PROCESSED_DIR / "clubs.parquet")
    print(f"{len(frame)} rows, {frame['player'].nunique()} distinct players")
    return frame


if __name__ == "__main__":
    build()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_pipeline.py -v`
Expected: 5 passed

- [ ] **Step 5: Build the real frame and check the shape against the spec**

```bash
.venv/bin/python -m src.pipeline
```

The spec predicts 900–1,200 rows over a few hundred distinct players. If the
distinct-player count is far below that, transfer detection or entity resolution
is dropping people — investigate before training anything.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline.py tests/test_pipeline.py
git commit -m "feat: end-to-end dataset assembly into model_frame.parquet"
```

---

### Task 19: The CLI

**Files:**
- Create: `query.py`, `tests/test_query.py`

**Interfaces:**
- Consumes: `train.predict`, `presets.get`, `clubs.vector_for`, `archetypes.assign`, `pipeline.PLAYER_FEATURE_COLUMNS`, `pipeline.CLUB_FEATURE_COLUMNS`
- Produces: `query.parse_weights(text: str) -> dict[str, float]`, `query.combine(predictions: np.ndarray, weights: dict) -> np.ndarray`, `query.bucket(score: float) -> str`, `query.standout_features(row, cohort_median, feature_columns, top_n=3) -> list[str]`, `query.format_row(...) -> str`, `query.main(argv) -> int`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_query.py
import numpy as np
import pandas as pd
import pytest
import query


def test_parse_weights_reads_the_documented_format():
    assert query.parse_weights("minutes=0.5,output=0.3,retention=0.2") == {
        "minutes": 0.5, "output": 0.3, "retention": 0.2,
    }


def test_parse_weights_defaults_when_empty():
    assert query.parse_weights("") == {"minutes": 0.40, "output": 0.35, "retention": 0.25}


def test_parse_weights_rejects_unknown_target():
    with pytest.raises(ValueError, match="unknown target"):
        query.parse_weights("vibes=1.0")


def test_parse_weights_rejects_weights_that_do_not_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        query.parse_weights("minutes=0.5,output=0.9,retention=0.2")


def test_combine_applies_weights_in_target_order():
    predictions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out = query.combine(predictions, {"minutes": 0.5, "output": 0.3, "retention": 0.2})
    assert out.tolist() == [0.5, 0.3]


def test_bucket_thresholds():
    assert query.bucket(0.80) == "great"
    assert query.bucket(0.50) == "ok"
    assert query.bucket(0.20) == "bust"


def test_standout_features_returns_largest_positive_deviations():
    row = pd.Series({"a": 0.9, "b": 0.5, "c": 0.1})
    median = pd.Series({"a": 0.5, "b": 0.5, "c": 0.5})
    assert query.standout_features(row, median, ["a", "b", "c"], top_n=1) == ["a"]


def test_main_rejects_unknown_preset(capsys):
    code = query.main(["--preset", "tiki-taka"])
    assert code != 0
    assert "unknown preset" in capsys.readouterr().err.lower()


def test_main_requires_either_preset_or_club(capsys):
    code = query.main([])
    assert code != 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_query.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'query'`

- [ ] **Step 3: Write `query.py`**

```python
import argparse
import sys

import numpy as np
import pandas as pd
import torch

from src import archetypes, clubs, config, model, pipeline, presets

DEFAULT_WEIGHTS = {"minutes": 0.40, "output": 0.35, "retention": 0.25}
MODEL_PATH = config.ROOT / "models" / "ensemble.pt"


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


def bucket(score: float) -> str:
    if score >= 0.66:
        return "great"
    if score >= 0.33:
        return "ok"
    return "bust"


def standout_features(row, cohort_median, feature_columns, top_n: int = 3) -> list[str]:
    deltas = {c: float(row[c]) - float(cohort_median[c]) for c in feature_columns}
    return [c for c, _ in sorted(deltas.items(), key=lambda kv: -kv[1])[:top_n]]


def load_ensemble():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"no trained model at {MODEL_PATH}. Run: .venv/bin/python -m src.train"
        )
    payload = torch.load(MODEL_PATH, weights_only=False)
    nets = []
    for state in payload["states"]:
        net = model.TwoTowerNet(payload["n_player"], payload["n_club"])
        net.load_state_dict(state)
        net.eval()
        nets.append(net)
    return nets


def format_row(rank, row, predictions, spread, score, archetype, standouts) -> str:
    per_head = "  ".join(
        f"{t}={predictions[i]:.2f}±{spread[i]:.2f}" for i, t in enumerate(config.TARGETS)
    )
    return (
        f"{rank:>3}. {row['player']:<26} {row.get('team', ''):<18} "
        f"age {row.get('age', '?'):>2}  score {score:.2f} ({bucket(score)})\n"
        f"      {per_head}\n"
        f"      archetype: {archetype}   standout: {', '.join(standouts)}"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Rank Serie A transfer candidates.")
    parser.add_argument("--preset", help=f"one of: {sorted(presets.PRESETS)}")
    parser.add_argument("--club", help="a real Serie A club name")
    parser.add_argument("--season", default=config.SEASONS[-1])
    parser.add_argument("--position", help="attacker, midfielder, defender, keeper")
    parser.add_argument("--max-age", type=int)
    parser.add_argument("--weights", default="")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args(argv)

    if not args.preset and not args.club:
        print("error: one of --preset or --club is required", file=sys.stderr)
        return 2

    try:
        weights = parse_weights(args.weights)
        frame = pd.read_parquet(config.PROCESSED_DIR / "model_frame.parquet")
        if args.preset:
            club_vector = presets.get(args.preset)
        else:
            styles = clubs.build_style_vectors(
                pd.read_parquet(config.PROCESSED_DIR / "clubs.parquet")
            )
            club_vector = clubs.vector_for(styles, args.club, args.season)
        nets = load_ensemble()
    except (KeyError, ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.position:
        frame = frame[frame["position_group"] == args.position]
    if args.max_age:
        frame = frame[frame["age"] <= args.max_age]
    if frame.empty:
        print("no candidates match those filters", file=sys.stderr)
        return 1

    player_x = torch.tensor(
        frame[pipeline.PLAYER_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    )
    club_x = torch.tensor(
        np.tile(club_vector, (len(frame), 1)).astype(np.float32)
    )
    with torch.no_grad():
        stacked = np.stack([net(player_x, club_x).numpy() for net in nets])
    mean, spread = stacked.mean(axis=0), stacked.std(axis=0)

    scores = combine(mean, weights)
    order = np.argsort(-scores)[: args.top]
    median = frame[pipeline.PLAYER_FEATURE_COLUMNS].median()
    models, archetype_features = archetypes.load_all()
    labels = (
        archetypes.assign(frame, models, archetype_features)
        if models else pd.Series("unclustered", index=frame.index)
    )

    for rank, i in enumerate(order, start=1):
        row = frame.iloc[i]
        print(format_row(
            rank, row, mean[i], spread[i], scores[i], labels.iloc[i],
            standout_features(row, median, pipeline.PLAYER_FEATURE_COLUMNS),
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_query.py -v`
Expected: 9 passed

- [ ] **Step 5: Run the whole suite and the real CLI**

```bash
.venv/bin/pytest -v
```

```bash
.venv/bin/python query.py --preset high-press-possession --position attacker --top 20
```

- [ ] **Step 6: Commit**

```bash
git add query.py tests/test_query.py
git commit -m "feat: CLI for ranking candidates by club or tactical preset"
```

---

## Reading the results

Before treating any output as meaningful, check three things in the Task 16 report:

1. **Did the network beat both baselines on Spearman, per head?** If not, the
   honest conclusion is that ridge is the model. That is a finding, not a failure
   — with a few hundred independent players it is a live possibility.
2. **What does the `no_value` ablation do?** If removing market value collapses
   performance, the model is largely rediscovering that expensive players get
   picked, and the rankings carry less scouting information than they appear to.
3. **Does the withheld panel read sensibly?** Not "is it accurate" — seven players
   cannot measure accuracy — but "is anything absurd". A model ranking a known
   disaster in the top decile has something wrong upstream of the model.
