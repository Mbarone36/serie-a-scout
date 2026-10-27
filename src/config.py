import os
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

# `output` is raw attacking production per 90 at the new club; `output_share`
# is the same measured as a share of the team's attack (see
# `labels.post_output_percentile`). Both are kept: the first says how much a
# player produces at that club, the second how central he is to it.
TARGETS = ("minutes", "output", "output_share", "retention")

LABEL_WINDOW_SEASONS = 2
MIN_MINUTES_PRIOR_SEASON = 450

# A Big-5 season runs 34-38 matches per team; a player frame whose max
# `team_matches` falls short of this was exported (or scraped) mid-season
# and must not be treated as an observable/complete season downstream (see
# `fbref.season_is_complete`).
MIN_COMPLETE_SEASON_MATCHES = 34
REQUEST_DELAY_SECONDS = 4.0
# API-Football paid plans allow far more than one request a second; 1s keeps
# a full pull (~1,400 pages) to about 25 minutes. The free plan's limit is
# 10 a minute, so raise this to 6.5 if pulling on a free key.
API_FOOTBALL_DELAY_SECONDS = 1.0
ENSEMBLE_SEEDS = 10

# Task 3b: pre-exported FBref Big-5 dump (R .rds files) as an offline data
# source, selected instead of the live soccerdata scraper.
FBREF_SOURCE = "dump"
DUMP_DIR = RAW_DIR / "dump"

# Scrapers identify themselves politely; set SCRAPER_CONTACT to add a contact address.
SCRAPER_CONTACT = os.environ.get("SCRAPER_CONTACT", "").strip()
RESEARCH_USER_AGENT = "Mozilla/5.0 (research project" + (f"; contact {SCRAPER_CONTACT}" if SCRAPER_CONTACT else "") + ")"
