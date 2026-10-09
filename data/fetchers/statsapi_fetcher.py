"""
TheStatsAPI fetcher for World Cup 2026 predictor — advanced tactical / xG stats.

Data source: TheStatsAPI (https://api.thestatsapi.com/api)
Auth: Authorization: Bearer {STATS_API_KEY}  (set in config.py)

Fetches per-match stats for a team's last N international fixtures before
June 11 2026, then aggregates: xG for/against, possession, shots on target,
and PPDA (passes per defensive action — a pressing-intensity proxy).

Caching strategy:
  - Team name → ID mapping: data/cache/statsapi_team_ids.json (permanent)
  - Match list per team: data/cache/statsapi_matches_{sha}.json (7 days)
  - Per-match stats: data/cache/statsapi_stats_{match_id}.json (permanent)
  - Every live HTTP call is appended to data/api_call_log.txt with a UTC timestamp.
  - Warnings (team not found, API errors) go to data/fetcher_warnings.log.

PPDA calculation:
  PPDA = opponent_passes_allowed / (our_tackles + our_interceptions)
  Lower PPDA = more aggressive pressing (elite pressers sit at 6–9).
  Higher PPDA = deeper block / less press (defensive sides sit at 14–20).

Graceful degradation:
  - No active subscription → all live calls raise HTTP 403; fetcher returns
    games_found=0 so advanced_tactical_model falls back to ranking proxy.
  - Team not in API → returns games_found=0 for the same reason.
  - Individual match stats missing → that match is skipped; others are used.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TypedDict

# ── Paths ─────────────────────────────────────────────────────────────────────

_REPO_ROOT  = Path(__file__).parent.parent.parent
_CACHE_DIR  = _REPO_ROOT / "data" / "cache"
_API_LOG    = _REPO_ROOT / "data" / "api_call_log.txt"
_WARN_LOG   = _REPO_ROOT / "data" / "fetcher_warnings.log"

sys.path.insert(0, str(_REPO_ROOT))
from config import STATS_API_KEY  # noqa: E402

# ── API constants ─────────────────────────────────────────────────────────────

BASE_URL        = "https://api.thestatsapi.com/api"
FIXTURE_CUTOFF  = "2026-06-11"    # only fixtures strictly before WC group stage
_CACHE_MAX_DAYS = 7               # refresh match lists after this many days
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0"
)

# ── Team name aliases ─────────────────────────────────────────────────────────
# Maps project names → search strings recognised by TheStatsAPI.

TEAM_NAME_MAP: dict[str, str] = {
    # North America
    "USA":                      "United States",
    # Asia / Middle East
    "IR Iran":                  "Iran",
    "Korea Republic":           "South Korea",
    "Korea DPR":                "North Korea",
    # Africa
    "Côte d'Ivoire":            "Ivory Coast",
    "Cote d'Ivoire":            "Ivory Coast",
    "Congo DR":                 "DR Congo",
    "Cape Verde Islands":       "Cape Verde",
    # Europe
    "Bosnia-Herzegovina":       "Bosnia",
    "Trinidad & Tobago":        "Trinidad and Tobago",
    # Encoding variants
    "Curacao":                  "Curaçao",
}


# ── TypedDicts ────────────────────────────────────────────────────────────────

class AdvancedStats(TypedDict):
    """Aggregated advanced tactical stats across N matches."""
    xg_for_avg:                   float   # expected goals scored per game
    xg_against_avg:               float   # expected goals conceded per game
    xg_diff_avg:                  float   # xg_for - xg_against per game
    possession_avg:               float   # ball possession % (0–100)
    shots_on_target_avg:          float   # shots on target per game
    shots_on_target_against_avg:  float   # opponent shots on target per game
    ppda_avg:                     float   # passes per defensive action
    games_found:                  int     # number of matches with usable stats


# ── Logging helpers ───────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger(__name__)


def _warn(msg: str) -> None:
    """Append a timestamped warning to fetcher_warnings.log and stderr."""
    _logger.warning(msg)
    _WARN_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(_WARN_LOG, "a", encoding="utf-8") as f:
        f.write(f"{ts}  [statsapi] {msg}\n")


def _log_call(url: str) -> None:
    """Append every live HTTP call to api_call_log.txt."""
    _API_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(_API_LOG, "a", encoding="utf-8") as f:
        f.write(f"{ts}  GET {url}\n")


# ── HTTP helper ───────────────────────────────────────────────────────────────

def _get(path: str, params: dict[str, str] | None = None) -> dict:
    """
    Make an authenticated GET to TheStatsAPI, log it, return parsed JSON.

    Raises urllib.error.HTTPError on non-2xx (caller handles gracefully).
    """
    qs = "&".join(f"{k}={v}" for k, v in (params or {}).items())
    url = f"{BASE_URL}{path}?{qs}" if qs else f"{BASE_URL}{path}"
    _log_call(url)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent":    _UA,
            "Authorization": f"Bearer {STATS_API_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _sha(s: str) -> str:
    """First 20 hex chars of SHA-256 — used as a compact cache filename stem."""
    return hashlib.sha256(s.encode()).hexdigest()[:20]


def _cache_path(key: str) -> Path:
    return _CACHE_DIR / f"statsapi_{key}.json"


def _cache_fresh(key: str, max_days: int) -> bool:
    """Return True if the cache file exists and is younger than max_days."""
    p = _cache_path(key)
    if not p.exists():
        return False
    mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime) < timedelta(days=max_days)


def _load_cache(key: str) -> dict | list | None:
    p = _cache_path(key)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _save_cache(key: str, data: dict | list) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(key).write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )


# ── Team ID resolution ────────────────────────────────────────────────────────

_TEAM_ID_CACHE_KEY = "team_ids"


def _load_id_cache() -> dict[str, str]:
    """Load the persistent team-name → team-ID mapping from disk."""
    cached = _load_cache(_TEAM_ID_CACHE_KEY)
    return cached if isinstance(cached, dict) else {}


def _save_id_cache(mapping: dict[str, str]) -> None:
    _save_cache(_TEAM_ID_CACHE_KEY, mapping)


def get_team_id(team_name: str) -> str | None:
    """
    Resolve a team name to TheStatsAPI's internal team ID.

    Checks the persistent disk cache first (data/cache/statsapi_team_ids.json).
    On cache miss, calls GET /football/teams?search=<name> and caches the result.

    Prefers an exact (case-insensitive) name match; falls back to the first
    search result when no exact match exists.

    Args:
        team_name: Project team name, e.g. "France", "USA", "IR Iran".

    Returns:
        Team ID string like "tm_0406", or None if lookup fails.
    """
    canonical = TEAM_NAME_MAP.get(team_name, team_name)
    id_cache  = _load_id_cache()

    if canonical in id_cache:
        return id_cache[canonical]

    try:
        resp  = _get("/football/teams", {"search": canonical, "per_page": "5"})
        teams = resp.get("data", [])
    except Exception as exc:
        _warn(f"Team ID lookup failed for '{canonical}': {exc}")
        return None

    if not teams:
        _warn(f"No teams found for '{canonical}' (resolved from '{team_name}').")
        return None

    # Prefer an exact case-insensitive name match; fall back to first result.
    team_id: str | None = None
    for t in teams:
        if t.get("name", "").lower() == canonical.lower():
            team_id = t["id"]
            break
    if team_id is None:
        team_id = teams[0]["id"]
        _warn(
            f"No exact match for '{canonical}'; using first result "
            f"'{teams[0].get('name')}' (id={team_id})."
        )

    id_cache[canonical] = team_id
    _save_id_cache(id_cache)
    return team_id


# ── Match list fetching ───────────────────────────────────────────────────────

def _get_team_matches(team_id: str, fetch_n: int = 20) -> list[dict]:
    """
    Return up to fetch_n finished matches for team_id before FIXTURE_CUTOFF.

    Cached per (team_id, fetch_n) with a 7-day TTL — the data changes slowly
    as new qualifiers/friendlies are played leading up to the tournament.

    Args:
        team_id: TheStatsAPI team ID, e.g. "tm_0406".
        fetch_n: Maximum matches to request (fetched more than needed to
                 account for matches with missing per-match stats).

    Returns:
        List of match dicts from the API response. Empty on failure.
    """
    cache_key = f"matches_{_sha(team_id + str(fetch_n))}"

    if _cache_fresh(cache_key, _CACHE_MAX_DAYS):
        cached = _load_cache(cache_key)
        if isinstance(cached, list):
            return cached

    try:
        resp = _get(
            "/football/matches",
            {
                "team_id":  team_id,
                "date_to":  FIXTURE_CUTOFF,
                "status":   "finished",
                "per_page": str(fetch_n),
            },
        )
        matches = resp.get("data", [])
    except Exception as exc:
        _warn(f"Match list failed for team_id={team_id}: {exc}")
        # Return stale cache if available.
        stale = _load_cache(cache_key)
        return stale if isinstance(stale, list) else []

    _save_cache(cache_key, matches)
    return matches


# ── Per-match stats fetching ──────────────────────────────────────────────────

def _get_match_stats(match_id: str) -> dict | None:
    """
    Return stats for a single completed match. Cached permanently —
    finished-match stats never change.

    Args:
        match_id: TheStatsAPI match ID, e.g. "mt_838955483".

    Returns:
        MatchStats dict (overview, shots, defending, …) or None on failure.
    """
    cache_key = f"stats_{match_id}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    try:
        resp = _get(f"/football/matches/{match_id}/stats")
        # API may return {"data": <MatchStats>} or the MatchStats directly.
        data = resp.get("data", resp)
    except Exception as exc:
        _warn(f"Match stats failed for {match_id}: {exc}")
        return None

    _save_cache(cache_key, data)
    return data


# ── Stat extraction helpers ───────────────────────────────────────────────────

def _safe(d: dict | None, *keys: str, default: float = 0.0) -> float:
    """
    Safely traverse a nested dict by successive keys, returning `default`
    when any key is absent or the value is None/non-numeric.
    """
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k)  # type: ignore[assignment]
    if d is None:
        return default
    try:
        return float(d)
    except (TypeError, ValueError):
        return default


def _extract_match_row(
    match: dict,
    stats: dict,
    team_id: str,
) -> dict | None:
    """
    Pull per-team stats from a match stats payload.

    Determines whether the team played home or away from match.home_team.id,
    then reads xG, possession, shots, and PPDA from the appropriate side.

    PPDA = opponent_passes / max(1, our_tackles + our_interceptions).
    Returns None if the match stats payload is empty / malformed.

    Args:
        match: Match dict from /football/matches (contains home/away team IDs).
        stats: MatchStats dict from /football/matches/{id}/stats.
        team_id: The team we are extracting stats for.

    Returns:
        Dict with raw per-match values, or None if stats look empty.
    """
    home_id  = match.get("home_team", {}).get("id", "")
    is_home  = home_id == team_id
    our_side = "home" if is_home else "away"
    opp_side = "away" if is_home else "home"

    overview  = stats.get("overview",  {})
    defending = stats.get("defending", {})

    xg_for  = _safe(overview,  "expected_goals",   "all", our_side)
    xg_ag   = _safe(overview,  "expected_goals",   "all", opp_side)
    poss    = _safe(overview,  "ball_possession",   "all", our_side)
    sot     = _safe(overview,  "shots_on_target",   "all", our_side)
    sot_ag  = _safe(overview,  "shots_on_target",   "all", opp_side)

    # PPDA: how many opponent passes we allow per defensive action.
    opp_passes    = _safe(overview,  "passes",        "all", opp_side)
    our_tackles   = _safe(defending, "tackles",       "all", our_side)
    our_intercepts = _safe(defending, "interceptions", "all", our_side)
    ppda = opp_passes / max(1.0, our_tackles + our_intercepts)

    # Flag whether this match actually has xG data (not all matches do).
    has_xg = (xg_for > 0.0 or xg_ag > 0.0)

    # Skip completely empty rows (stats not available for this match).
    if poss == 0.0 and sot == 0.0 and not has_xg:
        return None

    return {
        "xg_for":                  xg_for,
        "xg_against":              xg_ag,
        "possession":              poss,
        "shots_on_target":         sot,
        "shots_on_target_against": sot_ag,
        "ppda":                    ppda,
        "has_xg":                  has_xg,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def get_team_advanced_stats(team_name: str, n: int = 10) -> AdvancedStats:
    """
    Return advanced tactical stats averaged across the last N international
    fixtures before June 11 2026 for which per-match stat data is available.

    Pipeline:
      1. Resolve team_name → team_id via get_team_id()
      2. Fetch last 2×n finished matches (extra buffer for missing stats)
      3. For each match, fetch per-match stats from /matches/{id}/stats
      4. Extract the team's side (home/away) and aggregate

    xG averages are computed only over matches that have xG data; other stats
    are averaged across all matches with any stat data.

    Args:
        team_name: e.g. "France", "USA", "Morocco". Alias-resolved via TEAM_NAME_MAP.
        n: Target number of recent fixtures to aggregate over (default 10).

    Returns:
        AdvancedStats TypedDict. games_found=0 when no data is available
        (team not in API, no active subscription, or no matches found).
    """
    _EMPTY: AdvancedStats = AdvancedStats(
        xg_for_avg=0.0, xg_against_avg=0.0, xg_diff_avg=0.0,
        possession_avg=0.0, shots_on_target_avg=0.0,
        shots_on_target_against_avg=0.0, ppda_avg=0.0, games_found=0,
    )

    team_id = get_team_id(team_name)
    if not team_id:
        return _EMPTY

    # Fetch extra matches to account for some having no per-match stats.
    raw_matches = _get_team_matches(team_id, fetch_n=n * 2)
    if not raw_matches:
        return _EMPTY

    # Sort newest-first, then walk until we have n usable rows.
    raw_matches.sort(key=lambda m: m.get("utc_date", ""), reverse=True)

    rows: list[dict] = []
    for match in raw_matches:
        if len(rows) >= n:
            break
        match_id = match.get("id")
        if not match_id:
            continue
        stats = _get_match_stats(match_id)
        if not stats:
            continue
        row = _extract_match_row(match, stats, team_id)
        if row:
            rows.append(row)

    if not rows:
        _warn(f"No usable per-match stats found for '{team_name}'.")
        return _EMPTY

    total = len(rows)

    # Only average xG from matches that actually carry xG data.
    xg_rows = [r for r in rows if r["has_xg"]] or rows

    xg_for_avg   = sum(r["xg_for"]     for r in xg_rows) / len(xg_rows)
    xg_ag_avg    = sum(r["xg_against"] for r in xg_rows) / len(xg_rows)

    return AdvancedStats(
        xg_for_avg=                 round(xg_for_avg, 3),
        xg_against_avg=             round(xg_ag_avg,  3),
        xg_diff_avg=                round(xg_for_avg - xg_ag_avg, 3),
        possession_avg=             round(sum(r["possession"]              for r in rows) / total, 3),
        shots_on_target_avg=        round(sum(r["shots_on_target"]         for r in rows) / total, 3),
        shots_on_target_against_avg=round(sum(r["shots_on_target_against"] for r in rows) / total, 3),
        ppda_avg=                   round(sum(r["ppda"]                    for r in rows) / total, 3),
        games_found=                total,
    )
