"""International results from the bundled, frozen portfolio snapshot.

Only dated rows before July 5, 2026 are exposed to models. The dataset
contains scores and competition names, not measured possession or shots.
Those fields use documented heuristic fallbacks in the models.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from functools import lru_cache

from config import SNAPSHOT_DATE
from pathlib import Path
from typing import TypedDict

# ── Paths ────────────────────────────────────────────────────────────────────

_REPO_ROOT    = Path(__file__).parent.parent.parent
_CACHE_DIR    = _REPO_ROOT / "data" / "cache"
_RESULTS_CACHE = _CACHE_DIR / "intl_results.json"
_WARN_LOG     = _REPO_ROOT / "data" / "fetcher_warnings.log"

# ── Source ────────────────────────────────────────────────────────────────────


FIXTURE_CUTOFF: str = SNAPSHOT_DATE

# Weight by competition importance.
# Friendlies use rotated squads and have no stakes, so they contribute far less
# than competitive qualifiers or tournaments.  Substring matching is used when
# resolving weights (see _competition_weight) so regional suffixes like
# "(UEFA)" or "(CONMEBOL)" still resolve to the correct bucket.
# Unrecognised tournaments default to 1.0 (assumed competitive).
# FIFA World Cup matches are weighted highest (1.5) so in-tournament performance
# dominates recent form as soon as WC games appear in the dataset.
COMPETITION_WEIGHTS: dict[str, float] = {
    "FIFA World Cup":                  1.5,
    "FIFA World Cup qualification":    1.0,
    "UEFA Nations League":             0.85,
    "CONMEBOL":                        0.85,
    "AFC Asian Cup":                   0.90,
    "Friendly":                        0.40,
    "International friendly":          0.40,
}

_DEFAULT_COMPETITION_WEIGHT: float = 1.0


def _competition_weight(tournament: str) -> float:
    """
    Return the importance weight for a given tournament string.

    Iterates COMPETITION_WEIGHTS keys longest-first so more-specific entries
    (e.g. "FIFA World Cup qualification") beat shorter ones ("FIFA World Cup").
    Falls back to _DEFAULT_COMPETITION_WEIGHT (1.0) for unrecognised names.
    """
    for key in sorted(COMPETITION_WEIGHTS, key=len, reverse=True):
        if key.lower() in tournament.lower():
            return COMPETITION_WEIGHTS[key]
    return _DEFAULT_COMPETITION_WEIGHT

# ── Team name map ─────────────────────────────────────────────────────────────
# Maps names used elsewhere in the project → exact names in the martj42 dataset.
# The dataset uses common English names, NOT FIFA/FBref abbreviations.

TEAM_NAME_MAP: dict[str, str] = {
    # North America
    "USA":                       "United States",
    # Asia
    "South Korea":               "South Korea",   # dataset already uses this
    "IR Iran":                   "Iran",
    "Korea Republic":            "South Korea",
    "Korea DPR":                 "North Korea",
    # Africa
    "Ivory Coast":               "Ivory Coast",   # dataset already uses this
    "Côte d'Ivoire":             "Ivory Coast",
    "Cote d'Ivoire":             "Ivory Coast",
    "Congo DR":                  "DR Congo",
    "Republic of Ireland":       "Republic of Ireland",
    "Cape Verde Islands":        "Cape Verde",
    # Europe
    "Bosnia-Herzegovina":        "Bosnia and Herzegovina",
    "Bosnia":                    "Bosnia and Herzegovina",
    "Trinidad & Tobago":         "Trinidad and Tobago",
    # Misc encoding variants
    "Curacao":                   "Curaçao",
}


# ── TypedDict for a single fixture ────────────────────────────────────────────

class FixtureDict(TypedDict):
    date: str
    opponent: str
    goals_for: int
    goals_against: int
    result: str              # "W", "D", or "L"
    tournament: str
    competition_weight: float  # from COMPETITION_WEIGHTS; 1.0 for competitive, 0.40 for friendlies
    possession: float          # always 0.0 — not in dataset; caller uses fallback
    shots_on_target: float     # always 0.0 — not in dataset; caller uses fallback


class H2HDict(TypedDict):
    date: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    result: str   # from team1's perspective: "W", "D", "L"
    tournament: str


# ── Logging helpers ───────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger(__name__)


def _warn(msg: str) -> None:
    """Log a warning to both stderr and the persistent fetcher_warnings.log."""
    _logger.warning(msg)
    _WARN_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(_WARN_LOG, "a", encoding="utf-8") as f:
        f.write(f"{ts}  {msg}\n")


# ── CSV download and caching ──────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_results() -> list[dict]:
    """Load the supplied results once; exclude every post-cutoff row."""
    if not _RESULTS_CACHE.exists():
        raise RuntimeError(
            "Bundled data/cache/intl_results.json is missing. "
            "Restore the snapshot from the repository before running."
        )
    with _RESULTS_CACHE.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    return [row for row in rows if row.get("date", "") and row["date"] < FIXTURE_CUTOFF]


# ── Name resolution ───────────────────────────────────────────────────────────

def _resolve(team_name: str) -> str:
    """Map a common project name to the dataset's canonical name."""
    return TEAM_NAME_MAP.get(team_name, team_name)


# ── Row parsing helpers ───────────────────────────────────────────────────────

def _is_played(row: dict) -> bool:
    """Return True only for matches that have a valid integer score."""
    hs, as_ = row.get("home_score", ""), row.get("away_score", "")
    if hs in ("", "NA", None) or as_ in ("", "NA", None):
        return False
    try:
        int(hs)
        int(as_)
        return True
    except ValueError:
        return False


def _result(goals_for: int, goals_against: int) -> str:
    """Return "W", "D", or "L" from the perspective of the team with goals_for."""
    if goals_for > goals_against:
        return "W"
    if goals_for == goals_against:
        return "D"
    return "L"


# ── Public API ────────────────────────────────────────────────────────────────

def get_team_last_n_fixtures(
    team_name: str,
    n: int = 10,
    source_label: str = "intl_results",
) -> list[FixtureDict]:
    """
    Return the last N completed international fixtures for a team before
    FIXTURE_CUTOFF, sorted oldest→newest.

    Args:
        team_name:    Common team name (e.g. "USA", "France"). Mapped via
                      TEAM_NAME_MAP to the dataset's canonical name.
        n:            Maximum number of most-recent fixtures to return.
        source_label: Ignored — kept for interface parity with future fetchers.

    Returns:
        List of FixtureDict. Empty list if the team is not found or has no
        qualifying matches (a warning is logged in that case).

    Note:
        possession and shots_on_target are always 0.0 — this dataset does not
        carry in-game stats. tactical_model._derive() uses its 0.15 fallback
        for attack_efficiency and 50.0 for possession in that scenario.
    """
    canonical = _resolve(team_name)
    rows = _load_results()

    # Filter: team played, match completed, before cutoff.
    team_rows = [
        r for r in rows
        if canonical in (r["home_team"], r["away_team"])
        and _is_played(r)
        and r["date"] < FIXTURE_CUTOFF
    ]

    if not team_rows:
        _warn(f"No matches found for '{team_name}' (resolved: '{canonical}').")
        return []

    # Sort by date ascending, then take the last N.
    team_rows.sort(key=lambda r: r["date"])
    recent = team_rows[-n:]

    fixtures: list[FixtureDict] = []
    for row in recent:
        is_home = row["home_team"] == canonical
        gf = int(row["home_score"]) if is_home else int(row["away_score"])
        ga = int(row["away_score"]) if is_home else int(row["home_score"])
        opponent = row["away_team"] if is_home else row["home_team"]

        fixtures.append(FixtureDict(
            date=row["date"],
            opponent=opponent,
            goals_for=gf,
            goals_against=ga,
            result=_result(gf, ga),
            tournament=row["tournament"],
            competition_weight=_competition_weight(row["tournament"]),
            possession=0.0,       # not available in this dataset
            shots_on_target=0.0,  # not available in this dataset
        ))

    return fixtures


def get_team_stats_summary(team_name: str, n: int = 10) -> dict:
    """
    Aggregate tactical stats across a team's last N international matches.

    Returns a dict with keys used by tactical_model.extract_tactical_stats:
        games_played, win_rate, draw_rate, loss_rate,
        goals_scored_avg, goals_conceded_avg,
        possession_avg, shots_on_target_avg,
        clean_sheet_rate, failed_to_score_rate

    possession_avg and shots_on_target_avg are always 0.0 (data unavailable).
    Returns an all-zero dict if the team has no qualifying data.
    """
    fixtures = get_team_last_n_fixtures(team_name, n=n)
    total = len(fixtures)

    if total == 0:
        return {
            "games_played": 0,
            "win_rate": 0.0,
            "draw_rate": 0.0,
            "loss_rate": 0.0,
            "goals_scored_avg": 0.0,
            "goals_conceded_avg": 0.0,
            "possession_avg": 0.0,
            "shots_on_target_avg": 0.0,
            "clean_sheet_rate": 0.0,
            "failed_to_score_rate": 0.0,
        }

    # Use competition_weight so a friendly (0.40) counts far less than a
    # World Cup qualifier (1.0).  games_played is the raw fixture count for
    # diagnostic clarity; all rates and averages are divided by total_weight.
    total_weight = sum(f["competition_weight"] for f in fixtures)

    wins   = sum(f["competition_weight"] for f in fixtures if f["result"] == "W")
    draws  = sum(f["competition_weight"] for f in fixtures if f["result"] == "D")
    losses = sum(f["competition_weight"] for f in fixtures if f["result"] == "L")
    gf_total       = sum(f["goals_for"]     * f["competition_weight"] for f in fixtures)
    ga_total       = sum(f["goals_against"] * f["competition_weight"] for f in fixtures)
    clean_sheets   = sum(f["competition_weight"] for f in fixtures if f["goals_against"] == 0)
    failed_to_score = sum(f["competition_weight"] for f in fixtures if f["goals_for"]    == 0)

    return {
        "games_played":         total,
        "win_rate":             round(wins          / total_weight, 3),
        "draw_rate":            round(draws         / total_weight, 3),
        "loss_rate":            round(losses        / total_weight, 3),
        "goals_scored_avg":     round(gf_total      / total_weight, 3),
        "goals_conceded_avg":   round(ga_total      / total_weight, 3),
        "possession_avg":       0.0,   # not available in this dataset
        "shots_on_target_avg":  0.0,   # not available in this dataset
        "clean_sheet_rate":     round(clean_sheets    / total_weight, 3),
        "failed_to_score_rate": round(failed_to_score / total_weight, 3),
    }


def get_h2h(team1: str, team2: str, last_n: int = 10) -> list[H2HDict]:
    """
    Return the last N head-to-head matches between two teams, sorted
    oldest→newest.

    Args:
        team1:  First team (results framed from their perspective).
        team2:  Second team.
        last_n: Maximum number of most-recent H2H matches to return.

    Returns:
        List of H2HDict. Empty if no H2H history exists.
    """
    c1, c2 = _resolve(team1), _resolve(team2)
    rows = _load_results()

    h2h_rows = [
        r for r in rows
        if {r["home_team"], r["away_team"]} == {c1, c2}
        and _is_played(r)
        and r["date"] < FIXTURE_CUTOFF
    ]

    if not h2h_rows:
        _warn(f"No H2H history found between '{team1}' and '{team2}'.")
        return []

    h2h_rows.sort(key=lambda r: r["date"])
    recent = h2h_rows[-last_n:]

    results: list[H2HDict] = []
    for row in recent:
        hg, ag = int(row["home_score"]), int(row["away_score"])
        is_home = row["home_team"] == c1
        gf = hg if is_home else ag
        ga = ag if is_home else hg

        results.append(H2HDict(
            date=row["date"],
            home_team=row["home_team"],
            away_team=row["away_team"],
            home_goals=hg,
            away_goals=ag,
            result=_result(gf, ga),
            tournament=row["tournament"],
        ))

    return results
