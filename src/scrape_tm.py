"""Transfermarkt client and page/JSON parsers.

Covers player identity, injuries, market-value history, transfer history and
club squad value. The identity/injuries parsers work off the classic HTML
profile pages; market-value history and transfer history are pulled from
Transfermarkt's internal ``ceapi`` JSON endpoints, which is what the site's
own front-end uses and is far more stable than scraping embedded chart
markup out of the HTML profile page.
"""

import copy
import json
import logging
import re
import time
from datetime import date, datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src import cache, config

logger = logging.getLogger(__name__)

BASE = "https://www.transfermarkt.com"
HEADERS = {"User-Agent": config.RESEARCH_USER_AGENT}

MV_URL = BASE + "/ceapi/marketValueDevelopment/graph/{id}"
TRANSFERS_URL = BASE + "/ceapi/transferHistory/list/{id}"

_limiter = cache.RateLimiter(config.REQUEST_DELAY_SECONDS)

INJURY_COLUMNS = ["season", "injury", "days_missed", "games_missed"]
MV_COLUMNS = ["date", "value_eur"]
TRANSFER_COLUMNS = ["season", "from_club", "to_club", "fee_eur", "is_loan", "transfer_date"]
SEARCH_COLUMNS = ["tm_id", "name", "birth_date", "nationality", "slug"]

# fetch() retry policy for transient network failures (FINDING addendum):
# up to 3 attempts, escalating per-attempt timeouts, backoff sleep between
# attempts. The rate limiter is re-armed before EVERY attempt so the 4s
# floor between actual requests to the site holds even during a retry.
_FETCH_TIMEOUTS = (30, 60, 90)
_FETCH_BACKOFFS = (10, 20)

_SCALE = {"bn": 1e9, "m": 1e6, "k": 1e3}

# Transfermarkt's quick-search only reports each candidate's age, not a birth
# date, and its nationality column is a full country name (a flag `title`),
# not the 3-letter code `entities.match_players` gates on. This maps the
# country names actually seen for the footballing nations in
# `data/processed/fbref_identities.parquet` to the exact trigraphs FBref
# uses there (which mostly, but not always, follow FIFA codes -- e.g. Niger
# is "NIG" and Nigeria is "NGA", Ivory Coast is "CIV" not "IVO"). Anything
# missing falls back to the first three uppercase letters of the name.
COUNTRY_TO_CODE = {
    "Albania": "ALB", "Algeria": "ALG", "Angola": "ANG", "Argentina": "ARG",
    "Armenia": "ARM", "Burundi": "BDI", "Belgium": "BEL", "Benin": "BEN",
    "Bermuda": "BER", "Burkina Faso": "BFA", "Bosnia-Herzegovina": "BIH",
    "Bosnia and Herzegovina": "BIH", "Bolivia": "BOL", "Brazil": "BRA",
    "Bulgaria": "BUL", "Canada": "CAN", "Congo": "CGO", "Chad": "CHA",
    "Chile": "CHI", "China": "CHN", "Ivory Coast": "CIV",
    "Côte d'Ivoire": "CIV", "Cote d'Ivoire": "CIV", "Cameroon": "CMR", "DR Congo": "COD",
    "Congo DR": "COD", "Colombia": "COL", "Comoros": "COM",
    "Cape Verde": "CPV", "Costa Rica": "CRC", "Croatia": "CRO",
    "Central African Republic": "CTA", "Cuba": "CUB", "Curacao": "CUW",
    "Curaçao": "CUW", "Cyprus": "CYP", "Czech Republic": "CZE",
    "Denmark": "DEN", "Dominican Republic": "DOM", "Ecuador": "ECU",
    "Egypt": "EGY", "England": "ENG", "Equatorial Guinea": "EQG",
    "Spain": "ESP", "Estonia": "EST", "Finland": "FIN", "France": "FRA",
    "Faroe Islands": "FRO", "Gabon": "GAB", "Gambia": "GAM",
    "The Gambia": "GAM",
    "Georgia": "GEO", "Germany": "GER", "Ghana": "GHA",
    "Guadeloupe": "GLP", "Guinea-Bissau": "GNB", "Greece": "GRE",
    "Grenada": "GRN", "French Guiana": "GUF", "Guinea": "GUI",
    "Haiti": "HAI", "Honduras": "HON", "Hungary": "HUN",
    "Republic of Ireland": "IRL", "Ireland": "IRL", "Iran": "IRN",
    "Iraq": "IRQ", "Iceland": "ISL", "Israel": "ISR", "Italy": "ITA",
    "Jamaica": "JAM", "Japan": "JPN", "Kenya": "KEN",
    "South Korea": "KOR", "Korea, South": "KOR", "Saudi Arabia": "KSA",
    "Kosovo": "KVX", "Laos": "LAO", "Libya": "LBY",
    "Liechtenstein": "LIE", "Lithuania": "LTU", "Luxembourg": "LUX",
    "Madagascar": "MAD", "Morocco": "MAR", "Moldova": "MDA",
    "Mexico": "MEX", "North Macedonia": "MKD", "Mali": "MLI",
    "Montenegro": "MNE", "Mozambique": "MOZ", "Mauritania": "MTN",
    "Martinique": "MTQ", "New Caledonia": "NCL", "Netherlands": "NED",
    "Nigeria": "NGA", "Niger": "NIG", "Northern Ireland": "NIR",
    "Norway": "NOR", "New Zealand": "NZL", "Panama": "PAN",
    "Paraguay": "PAR", "Peru": "PER", "Philippines": "PHI",
    "Poland": "POL", "Portugal": "POR", "North Korea": "PRK",
    "Korea, North": "PRK", "Réunion": "REU", "Romania": "ROU",
    "South Africa": "RSA", "Russia": "RUS", "Scotland": "SCO",
    "Senegal": "SEN", "St. Kitts & Nevis": "SKN",
    "St Kitts and Nevis": "SKN", "Sierra Leone": "SLE", "Serbia": "SRB",
    "Switzerland": "SUI", "Suriname": "SUR", "Slovakia": "SVK",
    "Slovenia": "SVN", "Sweden": "SWE", "Tanzania": "TAN", "Togo": "TOG",
    "Trinidad and Tobago": "TRI", "Tunisia": "TUN", "Turkey": "TUR",
    "Türkiye": "TUR",
    "Ukraine": "UKR", "Uruguay": "URU", "United States": "USA",
    "USA": "USA", "Uzbekistan": "UZB", "Venezuela": "VEN", "Wales": "WAL",
    "Zambia": "ZAM", "Zimbabwe": "ZIM",
}


def fetch(url: str, key: str) -> str:
    """Rate-limited, disk-cached GET with retry on transient failures.

    ``ceapi`` endpoints return JSON and are asked for it explicitly via the
    ``Accept`` header; ordinary profile pages get the plain research
    User-Agent only.

    Retries up to 3 attempts on a read timeout, connection error, or an
    HTTP 5xx response, with escalating per-attempt timeouts (30/60/90s) and
    a 10/20s backoff sleep between attempts, logging a WARNING before each
    retry. Any other HTTP error (4xx) is not retried. The rate limiter is
    re-armed before EVERY attempt -- including retries -- so the 4s floor
    between requests to the site holds even during a retry storm. On a
    third failure the exception propagates; `cache.cached_text` only ever
    persists the return value of a successful `build()`, so a hard failure
    never writes a partial/error response to the disk cache.
    """

    def build() -> str:
        headers = dict(HEADERS)
        if "/ceapi/" in url:
            headers["Accept"] = "application/json"

        last_exc: Exception | None = None
        for attempt in range(3):
            _limiter.wait()
            try:
                response = requests.get(
                    url, headers=headers, timeout=_FETCH_TIMEOUTS[attempt]
                )
                response.raise_for_status()
                return response.text
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as exc:
                last_exc = exc
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status is None or not (500 <= status < 600):
                    raise
                last_exc = exc

            if attempt < 2:
                logger.warning(
                    "fetch retry %d/3 for %s after %r", attempt + 1, url, last_exc
                )
                time.sleep(_FETCH_BACKOFFS[attempt])

        raise last_exc

    return cache.cached_text(f"tm/{key}", build)


def _days_to_int(text: str) -> int:
    match = re.search(r"(\d+)", text or "")
    return int(match.group(1)) if match else 0


def _normalise_season(text: str) -> str:
    """'21/22' -> '2122'."""
    digits = re.findall(r"\d{2}", text or "")
    return "".join(digits[:2]) if len(digits) >= 2 else ""


def _fee_to_eur(text) -> float:
    """Parse a Transfermarkt fee string into euros.

    Handles plain amounts ('€80.00m', '€450k'), free/loan moves (0.0), and
    unknown/unspecified fees ('-', '?') as NaN. Some rows embed the fee in
    HTML (e.g. a loan fee wrapped in ``<i>``) — tags are stripped first.
    """
    if text is None:
        return float("nan")
    clean = re.sub(r"<[^>]+>", " ", str(text))
    clean = re.sub(r"\s+", " ", clean).strip()
    if not clean:
        return float("nan")
    low = clean.lower()
    if low in ("free transfer", "loan transfer", "end of loan"):
        return 0.0
    if clean in ("-", "?"):
        return float("nan")
    match = re.search(r"€\s*([\d.,]+)\s*(bn|m|k)?", clean, re.IGNORECASE)
    if not match:
        return float("nan")
    amount_text = match.group(1).replace(",", "")
    try:
        amount = float(amount_text)
    except ValueError:
        return float("nan")
    scale = _SCALE.get((match.group(2) or "").lower(), 1.0)
    return amount * scale


def _is_loan(text) -> bool:
    """Whether a raw fee string describes a loan move.

    Covers "loan transfer", "Loan fee: <br/><i>€100k</i>" (a loan WITH a
    disclosed loan fee, which still isn't a permanent sale) and "End of
    loan". Case-insensitive; HTML tags are stripped first.
    """
    if text is None:
        return False
    clean = re.sub(r"<[^>]+>", " ", str(text))
    return "loan" in clean.lower()


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


def parse_player_identity(html: str) -> dict:
    """Extract name, birth date and nationality from a player profile page."""
    soup = BeautifulSoup(html, "lxml")

    name = ""
    h1 = soup.select_one("h1.data-header__headline-wrapper")
    if h1 is not None:
        h1_copy = copy.copy(h1)
        for shirt_number in h1_copy.select(".data-header__shirt-number"):
            shirt_number.decompose()
        name = h1_copy.get_text(" ", strip=True)

    birth_date = None
    birth_node = soup.find(attrs={"itemprop": "birthDate"})
    if birth_node is not None:
        text = birth_node.get_text(" ", strip=True)
        match = re.search(r"(\d{2}/\d{2}/\d{4})", text)
        if match:
            try:
                birth_date = datetime.strptime(match.group(1), "%d/%m/%Y").date()
            except ValueError:
                birth_date = None

    nationality = ""
    nat_node = soup.select_one('span[itemprop="nationality"]')
    if nat_node is not None:
        nationality = nat_node.get_text(" ", strip=True)
    if not nationality:
        flag = soup.select_one('li.data-header__label img.flaggenrahmen')
        if flag is not None:
            nationality = flag.get("title", "")

    return {"name": name, "birth_date": birth_date, "nationality": nationality}


def parse_market_value_history(json_text: str) -> pd.DataFrame:
    """Parse the ceapi marketValueDevelopment/graph JSON into a tidy frame.

    Each point's date comes from ``datum_mw`` (dd/mm/yyyy); if that is
    missing or unparseable, falls back to the ``x`` millisecond epoch.
    """
    try:
        data = json.loads(json_text)
    except (ValueError, TypeError):
        return pd.DataFrame(columns=MV_COLUMNS)

    if not isinstance(data, dict):
        return pd.DataFrame(columns=MV_COLUMNS)
    points = data.get("list")
    if not points:
        return pd.DataFrame(columns=MV_COLUMNS)

    rows = []
    for point in points:
        if not isinstance(point, dict):
            continue
        point_date = None
        datum_mw = point.get("datum_mw")
        if datum_mw:
            try:
                point_date = datetime.strptime(datum_mw, "%d/%m/%Y")
            except ValueError:
                point_date = None
        if point_date is None and point.get("x") is not None:
            try:
                point_date = datetime.fromtimestamp(int(point["x"]) / 1000.0)
            except (ValueError, TypeError, OverflowError, OSError):
                point_date = None
        if point_date is None:
            continue
        y = point.get("y")
        if y is None:
            continue
        try:
            value_eur = float(y)
        except (TypeError, ValueError):
            continue
        rows.append({"date": pd.Timestamp(point_date), "value_eur": value_eur})

    frame = pd.DataFrame(rows, columns=MV_COLUMNS)
    if frame.empty:
        return frame
    return frame.sort_values("date").reset_index(drop=True)


def parse_transfer_history(json_text: str) -> pd.DataFrame:
    """Parse the ceapi transferHistory/list JSON into a tidy frame."""
    try:
        data = json.loads(json_text)
    except (ValueError, TypeError):
        return pd.DataFrame(columns=TRANSFER_COLUMNS)

    if not isinstance(data, dict):
        return pd.DataFrame(columns=TRANSFER_COLUMNS)
    transfers = data.get("transfers")
    if not transfers:
        return pd.DataFrame(columns=TRANSFER_COLUMNS)

    rows = []
    for t in transfers:
        if not isinstance(t, dict):
            continue
        from_club = (t.get("from") or {}).get("clubName", "")
        to_club = (t.get("to") or {}).get("clubName", "")
        rows.append({
            "season": _normalise_season(t.get("season", "")),
            "from_club": from_club,
            "to_club": to_club,
            "fee_eur": _fee_to_eur(t.get("fee")),
            "is_loan": _is_loan(t.get("fee")),
            "transfer_date": pd.to_datetime(t.get("dateUnformatted"), errors="coerce"),
        })

    frame = pd.DataFrame(rows, columns=TRANSFER_COLUMNS)
    frame["fee_eur"] = frame["fee_eur"].astype(float)
    frame["is_loan"] = frame["is_loan"].astype(bool)
    frame["transfer_date"] = pd.to_datetime(frame["transfer_date"], errors="coerce")
    return frame


def parse_club_squad_value(html: str) -> float:
    """Total squad/player market value in euros.

    No club-page fixture was available at implementation time; this has
    only been verified against the player profile fixture's identical
    ``.data-header__market-value-wrapper`` markup and against synthetic
    strings covering bn/m/k suffixes.
    """
    soup = BeautifulSoup(html, "lxml")
    node = soup.select_one(
        "a.data-header__market-value-wrapper, .data-header__market-value-wrapper"
    )
    text = node.get_text(" ", strip=True) if node else ""
    match = re.search(r"€\s*([\d,.]+)\s*(bn|m|k)?", text, re.IGNORECASE)
    if not match:
        return float("nan")
    amount = float(match.group(1).replace(",", ""))
    scale = _SCALE.get((match.group(2) or "").lower(), 1.0)
    return amount * scale


def _country_to_code(name: str) -> str:
    """Map a Transfermarkt flag `title` (a full country name) to the
    3-letter code `entities.match_players` expects. Falls back to the
    first three uppercase letters for any nation not in the lookup."""
    if not name:
        return ""
    if name in COUNTRY_TO_CODE:
        return COUNTRY_TO_CODE[name]
    return re.sub(r"[^A-Za-z]", "", name).upper()[:3]


def _today() -> date:
    """Seam for tests: the reference date used to turn a quick-search
    'age' figure back into an approximate birth year."""
    return date.today()


def _birth_date_from_age(text: str):
    """Transfermarkt's quick search reports age, not a birth date. This
    inverts it into an approximate birth date (year is what matters --
    `entities.match_players` only gates on birth year) using today's date
    as the reference point. The month/day are an arbitrary mid-year
    placeholder and carry no real signal, only the year does."""
    match = re.search(r"\d+", text or "")
    if not match:
        return pd.NaT
    age = int(match.group())
    year = _today().year - age
    return pd.Timestamp(year=year, month=7, day=1)


def parse_search_results(html: str) -> pd.DataFrame:
    """Candidate player identities from a Transfermarkt quick-search page.

    The results table (`table.items`) lists Name/Position, Position, Club,
    Age, Nat. (a flag image) and Market Value, in that column order -- but
    column order isn't guaranteed across page variants, so the Age and
    Nat. columns are located by their header text rather than hardcoded
    positions. Age is the only birth signal Transfermarkt exposes here (no
    exact birth date), so `birth_date` is reconstructed from it via
    `_birth_date_from_age`; nationality is mapped from the flag's country
    name to a 3-letter code via `_country_to_code`.
    """
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table.items")
    if table is None:
        return pd.DataFrame(columns=SEARCH_COLUMNS)

    header_cells = [th.get_text(" ", strip=True).lower() for th in table.select("thead th")]

    def _column_index(*keywords):
        for i, text in enumerate(header_cells):
            if all(keyword in text for keyword in keywords):
                return i
        return None

    age_idx = _column_index("age")
    nat_idx = _column_index("nat")

    rows = []
    for tr in table.select("tbody tr"):
        cells = tr.find_all("td", recursive=False)
        if not cells:
            continue
        link = cells[0].select_one("a[href*='/profil/spieler/']")
        if link is None:
            continue
        href = link["href"]
        id_match = re.search(r"/profil/spieler/(\d+)", href)
        if not id_match:
            continue
        slug_match = re.match(r"/?([^/]+)/profil/spieler/", href)
        slug = slug_match.group(1) if slug_match else ""

        age_text = ""
        if age_idx is not None and age_idx < len(cells):
            age_text = cells[age_idx].get_text(" ", strip=True)

        nationality = ""
        if nat_idx is not None and nat_idx < len(cells):
            flag = cells[nat_idx].select_one("img")
            if flag is not None:
                nationality = _country_to_code(flag.get("title", ""))

        rows.append({
            "tm_id": int(id_match.group(1)),
            "name": link.get_text(" ", strip=True),
            "birth_date": _birth_date_from_age(age_text),
            "nationality": nationality,
            "slug": slug,
        })

    return pd.DataFrame(rows, columns=SEARCH_COLUMNS)


def parse_club_search_results(html: str) -> list[tuple[int, str]]:
    """Transfermarkt (club_id, label) pairs from a quick-search page, best
    match first.

    FINDING D: `search_club` used to trust the first hit outright, but the
    quick search's top hit isn't always the right club (a reserve/youth
    side, a similarly-named unrelated club, ...). The label -- the link
    text, or the slug segment of the href when the link has no text -- lets
    `search_club` verify a hit is a plausible name match before trusting it.
    """
    soup = BeautifulSoup(html, "lxml")
    seen = set()
    results: list[tuple[int, str]] = []
    for link in soup.select("a[href*='/startseite/verein/']"):
        parts = link["href"].rstrip("/").split("/")
        if not parts or not parts[-1].isdigit():
            continue
        club_id = int(parts[-1])
        if club_id in seen:
            continue
        seen.add(club_id)
        label = link.get_text(" ", strip=True)
        if not label and len(parts) > 1:
            label = parts[1].replace("-", " ")
        results.append((club_id, label))
    return results
