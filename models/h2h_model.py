"""
H2H model for World Cup 2026 match prediction.

Captures historical head-to-head patterns between two specific teams —
psychological edge, tactical familiarity, and historical dominance — from
their last 10 H2H meetings before June 11 2026.

Data source: footballdata_fetcher → get_h2h(), which queries the martj42
international_results CSV for all matches between the two named teams.

Composite formula (from team1/home perspective):
    composite = (
        weighted_win_rate  * 0.40  +   # recency-weighted H2H win rate
        norm_goal_diff     * 0.25  +   # avg goal diff capped ±3 then /3
        last_match_result  * 0.20  +   # W=+1, D=0, L=-1
        big_win_in_last10  * 0.15      # 1 if any 3+ goal win in last 10
    )

Confidence tiers (H2H data is sparse for many international matchups):
    10+ matches → 0.80
    5–9  matches → 0.60
    3–4  matches → 0.40
    <3   matches → 0.20 — FIFA ranking delta proxy used

Fallback: fewer than 3 H2H matches → composite = ranking_delta × 0.15
(confidence = 0.20 so the ensemble downweights this model heavily).

Sigmoid calibration: SIGMOID_K = 3.0, DRAW_DECAY = 0.8 (same as other models).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import TypedDict

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.fetchers import footballdata_fetcher as fetcher
from data.fetchers.footballdata_fetcher import H2HDict

# ── Tuning constants ──────────────────────────────────────────────────────────

SIGMOID_K: float = 3.0
BASE_DRAW_RATE: float = 0.27
DRAW_DECAY: float = 0.8
PROB_MIN: float = 0.18
PROB_MAX: float = 0.82

# ── Confidence tiers ──────────────────────────────────────────────────────────

_CONF_FALLBACK = 0.20   # <3 matches — ranking proxy used
_CONF_SPARSE   = 0.40   # 3–4 matches
_CONF_MID      = 0.60   # 5–9 matches
_CONF_FULL     = 0.80   # 10+ matches

# Minimum H2H matches required before using the formula; below this we fall
# back to the FIFA ranking delta as a proxy composite.
_MIN_H2H: int = 3

# ── FIFA rankings (for the ranking-delta fallback) ────────────────────────────
# Keys match both common project names AND martj42 canonical dataset names where
# they differ, so the lookup works regardless of which form the caller passes.

_FIFA_RANKINGS: dict[str, int] = {
    "Argentina":       1,
    "France":          2,
    "England":         3,
    "Belgium":         4,
    "Brazil":          5,
    "Portugal":        6,
    "Netherlands":     7,
    "Spain":           8,
    "Germany":         9,
    "Italy":          10,
    "United States":  11,
    "USA":            11,
    "Mexico":         12,
    "Colombia":       13,
    "Uruguay":        14,
    "Croatia":        15,
    "Switzerland":    16,
    "Japan":          17,
    "Morocco":        18,
    "Senegal":        19,
    "Denmark":        20,
    "Austria":        21,
    "Ecuador":        22,
    "South Korea":    23,
    "Canada":         24,
    "Norway":         25,
    "Serbia":         26,
    "Poland":         27,
    "Ukraine":        28,
    "Australia":      29,
    "Turkey":         30,
    "Iran":           40,
    "Saudi Arabia":   45,
    "Ghana":          50,
    "Cameroon":       55,
    "Iraq":           70,
    "Panama":         75,
    "Bolivia":        80,
    "Jamaica":        85,
}

_DEFAULT_RANK = 100  # assigned to teams absent from the table


# ── TypedDicts ────────────────────────────────────────────────────────────────

class H2HStats(TypedDict):
    """All H2H features computed from team1's perspective."""
    total_matches:       int
    team1_win_rate:      float   # wins / total
    draw_rate:           float   # draws / total
    team1_loss_rate:     float   # losses / total
    goals_scored_avg:    float   # team1 avg goals per H2H game
    goals_conceded_avg:  float   # team1 avg goals conceded per H2H game
    clean_sheet_rate:    float   # team1 clean sheets / total
    weighted_win_rate:   float   # linearly-decayed win rate (recent=1.0, oldest=0.1)
    avg_goal_diff:       float   # avg (gf - ga) per game
    last_match_result:   int     # +1 W, 0 D, -1 L from team1 perspective
    big_win_in_last10:   int     # 1 if any win by 3+ goals in last 10


class PredictionOutput(TypedDict):
    team1_win:   float
    draw:        float
    team2_win:   float
    confidence:  float
    key_factors: list[str]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """Maps composite to (0, 1). Positive x favours team1."""
    return 1.0 / (1.0 + math.exp(-SIGMOID_K * x))


def _clamp(value: float) -> float:
    return max(PROB_MIN, min(PROB_MAX, value))


def _confidence(n: int) -> float:
    """Return H2H confidence score based on how many meetings were found."""
    if n >= 10:
        return _CONF_FULL
    if n >= 5:
        return _CONF_MID
    if n >= _MIN_H2H:
        return _CONF_SPARSE
    return _CONF_FALLBACK


def _resolve(name: str) -> str:
    """Resolve project team name to the martj42 dataset canonical name."""
    return fetcher.TEAM_NAME_MAP.get(name, name)


def _team1_goals(entry: H2HDict, canonical_t1: str) -> tuple[int, int]:
    """Return (goals_for, goals_against) from team1's perspective for one H2H entry."""
    if entry["home_team"] == canonical_t1:
        return entry["home_goals"], entry["away_goals"]
    return entry["away_goals"], entry["home_goals"]


def _linear_weight(idx: int, total: int) -> float:
    """
    Linear decay weight for match at position idx (0 = oldest, total-1 = newest).
    Most recent → 1.0, oldest → 0.1. Single-match case always returns 1.0.
    """
    if total == 1:
        return 1.0
    return 0.1 + (idx / (total - 1)) * 0.9


def _ranking_fallback_composite(team1: str, team2: str) -> float:
    """
    FIFA ranking delta proxy for when H2H history is too sparse.
    Returns ranking_delta × 0.15 so the sparse signal has minimal ensemble weight.
    ranking_delta = (team2_rank - team1_rank) / 40.0  (positive → team1 is higher-ranked).
    """
    t1_key = _resolve(team1)
    t2_key = _resolve(team2)
    r1 = _FIFA_RANKINGS.get(t1_key, _FIFA_RANKINGS.get(team1, _DEFAULT_RANK))
    r2 = _FIFA_RANKINGS.get(t2_key, _FIFA_RANKINGS.get(team2, _DEFAULT_RANK))
    delta = (r2 - r1) / 40.0
    return delta * 0.15


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_h2h_stats(
    team1: str,
    team2: str,
    verbose: bool = False,
) -> tuple[H2HStats | None, list[H2HDict]]:
    """
    Compute all H2H features from the last 10 meetings between team1 and team2.

    Returns (H2HStats, raw_entries). Returns (None, []) when no data is found.
    Features are framed entirely from team1's perspective.

    Args:
        team1:   The team whose perspective defines W/D/L and goal direction.
        team2:   The opponent.
        verbose: Print match count found before computing.
    """
    try:
        entries = fetcher.get_h2h(team1, team2, last_n=10)
    except Exception as exc:
        if verbose:
            print(f"  [H2H {team1} vs {team2}] ERROR: {exc}")
        return None, []

    n = len(entries)
    if verbose:
        print(f"  [H2H {team1} vs {team2}] {n} matches found")

    if n == 0:
        return None, []

    canonical_t1 = _resolve(team1)

    # Per-match goal and result data from team1's perspective.
    goals_for     = []
    goals_against = []
    results       = []   # "W", "D", "L" — already encoded in H2HDict
    for entry in entries:  # sorted oldest → newest
        gf, ga = _team1_goals(entry, canonical_t1)
        goals_for.append(gf)
        goals_against.append(ga)
        results.append(entry["result"])

    wins   = results.count("W")
    draws  = results.count("D")
    losses = results.count("L")

    # ── Weighted win rate (linear decay, oldest=0.1, newest=1.0) ─────────────
    weights      = [_linear_weight(i, n) for i in range(n)]
    total_weight = sum(weights)
    weighted_wins = sum(w for w, r in zip(weights, results) if r == "W")
    weighted_win_rate = weighted_wins / total_weight

    # ── Avg goal diff (normalised before composite application) ───────────────
    diffs = [gf - ga for gf, ga in zip(goals_for, goals_against)]
    avg_goal_diff = sum(diffs) / n

    # ── Last match result ─────────────────────────────────────────────────────
    last_result_map = {"W": 1, "D": 0, "L": -1}
    last_match_result = last_result_map[results[-1]]

    # ── Big win in last 10 (3+ goal margin for team1) ────────────────────────
    big_win_in_last10 = int(any(d >= 3 for d in diffs))

    return H2HStats(
        total_matches=n,
        team1_win_rate=round(wins / n, 3),
        draw_rate=round(draws / n, 3),
        team1_loss_rate=round(losses / n, 3),
        goals_scored_avg=round(sum(goals_for) / n, 3),
        goals_conceded_avg=round(sum(goals_against) / n, 3),
        clean_sheet_rate=round(sum(1 for g in goals_against if g == 0) / n, 3),
        weighted_win_rate=round(weighted_win_rate, 3),
        avg_goal_diff=round(avg_goal_diff, 3),
        last_match_result=last_match_result,
        big_win_in_last10=big_win_in_last10,
    ), entries


# ── Composite score ───────────────────────────────────────────────────────────

def h2h_composite(stats: H2HStats) -> float:
    """
    Combine H2H features into a single signed score from team1's perspective.
    Positive → team1 has historical dominance; negative → team2 does.
    Typical range: [-0.45, +1.0].

    avg_goal_diff is normalised to [-1, 1] by capping at ±3 then dividing by 3
    before the weighted sum.
    """
    norm_goal_diff = max(-1.0, min(1.0, stats["avg_goal_diff"] / 3.0))

    return (
        stats["weighted_win_rate"]  * 0.40
        + norm_goal_diff            * 0.25
        + stats["last_match_result"] * 0.20
        + stats["big_win_in_last10"] * 0.15
    )


# ── Key factor generation ─────────────────────────────────────────────────────

def _format_score_line(entry: H2HDict) -> str:
    """Format a single H2H entry as 'HomeTeam N-M AwayTeam (YYYY Tournament)'."""
    year = entry["date"][:4]
    return (
        f"{entry['home_team']} {entry['home_goals']}-{entry['away_goals']}"
        f" {entry['away_team']} ({year} {entry['tournament']})"
    )


def _build_key_factors(
    team1: str,
    team2: str,
    stats: H2HStats,
    entries: list[H2HDict],
    used_fallback: bool,
) -> list[str]:
    """
    Generate up to 4 human-readable H2H insight strings in priority order:

    Fallback path:
      1. "Insufficient H2H history — ranking proxy used"

    Normal path:
      1. Sparse data notice (< 5 matches)
      2. Historical dominance (win_rate ≥ 0.70 or ≤ 0.30)
      3. Never beaten notice (win_rate == 0 and ≥ 3 matches)
      4. Last meeting detail with score and tournament
      5. Big win signal (3+ goal margin)
    """
    if used_fallback:
        n = stats["total_matches"] if stats else 0
        if n == 0:
            return [f"{team1} vs {team2} — no H2H history found, ranking proxy used"]
        label = "meeting" if n == 1 else "meetings"
        return [
            f"{team1} vs {team2} — only {n} historical {label}, low confidence",
            "Insufficient H2H history — ranking proxy used",
        ]

    n = stats["total_matches"]
    factors: list[str] = []

    # ── Sparse data notice ────────────────────────────────────────────────────
    if n < 5:
        label = "meeting" if n == 1 else "meetings"
        factors.append(
            f"{team1} vs {team2} — only {n} historical {label}, low confidence"
        )

    # ── Historical dominance / never beaten ──────────────────────────────────
    wins   = round(stats["team1_win_rate"] * n)
    losses = round(stats["team1_loss_rate"] * n)

    if stats["team1_win_rate"] == 0.0 and n >= _MIN_H2H:
        factors.append(
            f"{team1} have never beaten {team2} in {n} recorded meetings"
        )
    elif stats["team1_loss_rate"] == 0.0 and n >= _MIN_H2H:
        factors.append(
            f"{team2} have never beaten {team1} in {n} recorded meetings"
        )
    elif stats["team1_win_rate"] >= 0.70:
        factors.append(
            f"{team1} won {wins} of last {n} meetings vs {team2}"
            f" — historical dominance"
        )
    elif stats["team1_win_rate"] <= 0.30:
        factors.append(
            f"{team1} won only {wins} of last {n} meetings vs {team2}"
            f" — historical dominance ({team2})"
        )

    # ── Last meeting detail ───────────────────────────────────────────────────
    if entries:
        last = entries[-1]
        score_line = _format_score_line(last)
        # Psychological edge: mention it for decisive results (not draws).
        if last["result"] in ("W", "L"):
            factors.append(
                f"Last meeting: {score_line} — psychological edge"
            )
        else:
            factors.append(f"Last meeting: {score_line}")

    # ── Big win signal ────────────────────────────────────────────────────────
    if stats["big_win_in_last10"] and len(factors) < 4:
        canonical_t1 = _resolve(team1)
        best_margin = -1
        best_entry: H2HDict | None = None
        for entry in entries:
            gf, ga = _team1_goals(entry, canonical_t1)
            margin = gf - ga
            if margin >= 3 and margin > best_margin:
                best_margin = margin
                best_entry = entry
        if best_entry is not None:
            factors.append(
                f"{team1} recorded a {best_margin}+ goal win vs {team2}"
                f" in last {n} — historical dominance"
            )

    return factors[:4] if factors else ["Insufficient H2H data for detailed factors"]


# ── Public predict() ──────────────────────────────────────────────────────────

def predict(
    home_team: str,
    away_team: str,
    verbose: bool = False,
) -> PredictionOutput:
    """
    Predict match outcome probabilities from head-to-head historical patterns.

    Frames all features from home_team (team1) perspective. Falls back to FIFA
    ranking delta × 0.15 if fewer than 3 H2H meetings are found.

    Args:
        home_team: The team treated as "team1" — their H2H record drives the signal.
        away_team: The opponent.
        verbose:   Print H2H match count found before computing.

    Returns:
        PredictionOutput with team1_win, draw, team2_win, confidence, key_factors.
        Low confidence (0.20) flags sparse data so the ensemble can downweight.
    """
    stats, entries = extract_h2h_stats(home_team, away_team, verbose=verbose)

    n = stats["total_matches"] if stats else 0
    confidence = _confidence(n)
    used_fallback = n < _MIN_H2H

    if used_fallback:
        composite = _ranking_fallback_composite(home_team, away_team)
    else:
        composite = h2h_composite(stats)  # type: ignore[arg-type]

    # Draw prior decays as composite magnitude grows (strong dominance → fewer draws).
    draw = BASE_DRAW_RATE * math.exp(-DRAW_DECAY * abs(composite))
    remaining = 1.0 - draw
    raw_home = _sigmoid(composite)

    team1_win = _clamp(raw_home * remaining)
    team2_win = _clamp((1.0 - raw_home) * remaining)
    draw = round(max(0.0, 1.0 - team1_win - team2_win), 4)

    key_factors = _build_key_factors(
        home_team, away_team, stats, entries, used_fallback
    )

    return PredictionOutput(
        team1_win=round(team1_win, 4),
        draw=round(draw, 4),
        team2_win=round(team2_win, 4),
        confidence=round(confidence, 3),
        key_factors=key_factors,
    )


# ── Test runner ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TEST_CASES = [
        ("France",  "Morocco"),   # France won 2022 WC semifinal, strong H2H
        ("France",  "Iraq"),      # likely sparse or one-sided
        ("Brazil",  "Morocco"),   # Morocco reached 2022 semis beating top teams
        ("Norway",  "Iraq"),      # probably very sparse
    ]

    print(f"\n{'Match':<28} {'team1_win':>8} {'draw':>6} {'team2_win':>9} {'conf':>6}")
    print("-" * 120)

    for home, away in TEST_CASES:
        print(f"\n{home} vs {away}")
        try:
            result = predict(home, away, verbose=True)
            label = f"→ {home} vs {away}"
            print(
                f"{label:<28} "
                f"{result['team1_win']:>8.4f} "
                f"{result['draw']:>6.4f} "
                f"{result['team2_win']:>9.4f} "
                f"{result['confidence']:>6.3f}"
            )
            for factor in result["key_factors"]:
                print(f"    • {factor}")
        except Exception as exc:
            print(f"  ERROR: {exc}")
