"""
Momentum model for World Cup 2026 match prediction.

Captures each team's trajectory heading into the tournament: form trend,
win rate, current streak, defensive solidity, and notable scalps — all from
their last 10 international results before June 11 2026.

Data source: footballdata_fetcher → martj42/international_results CSV,
same as tactical_model.py. All three pipeline models share the same fetcher
and the same output interface (team1_win, draw, team2_win, confidence,
key_factors).

Composite formula (home momentum score minus away momentum score):
    momentum_score = (
        win_rate         * 0.30  +
        recent_form      * 0.25  +   # last-5 win_rate minus prev-5 win_rate
        (-loss_rate)     * 0.20  +
        clean_sheet_rate * 0.15  +
        big_scalp        * 0.05  +   # beat a top-20 FIFA ranked team in last 10
        playoff_winner   * 0.05       # hardcoded intercontinental-playoff bonus
    )

Sigmoid calibration: SIGMOID_K=3.0, DRAW_DECAY=0.8.
These match tactical_model.py — both operate on composites in the ≈ ±1.0 range.
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

# ── Tuning constants ──────────────────────────────────────────────────────────
SIGMOID_K: float = 3.0
BASE_DRAW_RATE: float = 0.27
DRAW_DECAY: float = 0.8
PROB_MIN: float = 0.18
PROB_MAX: float = 0.82

# Maximum absolute value for the current-streak feature.
STREAK_CAP: int = 5

# ── Confidence anchors (match-data richness) ──────────────────────────────────
_CONF_SPARSE = 0.30   # < 3 matches
_CONF_MID    = 0.65   # 5–9 matches
_CONF_FULL   = 0.85   # 10+ matches
_GAMES_MID   = 5
_GAMES_FULL  = 10
_MIN_GAMES   = 3      # below this → FIFA ranking fallback

# ── Static bonuses ────────────────────────────────────────────────────────────
# Teams that won their intercontinental qualifying playoff get a momentum bonus
# reflecting the psychological lift and high-pressure match experience.
PLAYOFF_WINNERS: dict[str, int] = {
    "Iraq":        1,   # beat Bolivia in intercontinental playoff
    "New Zealand": 1,
}

# Teams that won a major confederation tournament within the last 24 months.
# These represent peak tactical cohesion and proven tournament-winning mentality
# that historical results alone don't fully capture.
TOURNAMENT_WINNERS: dict[str, float] = {
    "Spain":     0.08,  # UEFA Euro 2024 champions
    "Argentina": 0.05,  # Copa America 2024 champions (already strong in model)
    "Morocco":   0.03,  # AFCON 2025 — recent continental title
}
# Note: Argentina's bonus is lower because their World Cup 2022 + Copa America
# success is already well-represented through H2H history and momentum data.
# Spain's Euro 2024 win is the most underrepresented signal in the current model.

# ── FIFA rankings for big-scalp detection ─────────────────────────────────────
# Top 20 only — opponents outside this set are never a "scalp".
# Keys must match the names used in the martj42 dataset (canonical English names).
# "USA" → "United States" in the fixture data; both are included for safety.
_FIFA_RANKINGS: dict[str, int] = {
    "Argentina":          1,
    "France":             2,
    "England":            3,
    "Belgium":            4,
    "Brazil":             5,
    "Portugal":           6,
    "Netherlands":        7,
    "Spain":              8,
    "Germany":            9,
    "Italy":             10,
    "United States":     11,   # dataset canonical name
    "USA":               11,   # project alias — same rank
    "Mexico":            12,
    "Colombia":          13,
    "Uruguay":           14,
    "Croatia":           15,
    "Switzerland":       16,
    "Japan":             17,
    "Morocco":           18,
    "Senegal":           19,
    "Denmark":           20,
}
_TOP_20: frozenset[str] = frozenset(
    name for name, rank in _FIFA_RANKINGS.items() if rank <= 20
)


# ── TypedDicts ────────────────────────────────────────────────────────────────

class MomentumStats(TypedDict):
    """All momentum features for one team, computed from their last N fixtures."""
    games_played:       int
    win_rate:           float   # W / games
    loss_rate:          float   # L / games
    clean_sheet_rate:   float
    recent_form:        float   # last-5 win_rate − prev-5 win_rate
    current_streak:     int     # +N = N-game win streak, −N = loss streak
    big_scalp:          int     # 1 if beat a top-20 team in last 10, else 0
    playoff_winner:     int     # 1 if in PLAYOFF_WINNERS, else 0
    tournament_winner:  float   # additive bonus from TOURNAMENT_WINNERS, else 0.0


class PredictionOutput(TypedDict):
    team1_win:   float
    draw:        float
    team2_win:   float
    confidence:  float
    key_factors: list[str]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """Maps a composite score to (0, 1). Positive x favours team1."""
    return 1.0 / (1.0 + math.exp(-SIGMOID_K * x))


def _clamp(value: float) -> float:
    return max(PROB_MIN, min(PROB_MAX, value))


def _confidence_from_games(n: int) -> float:
    """Piecewise linear confidence from match count."""
    if n < _MIN_GAMES:
        return _CONF_SPARSE
    if n >= _GAMES_FULL:
        return _CONF_FULL
    if n >= _GAMES_MID:
        return _CONF_MID
    # 3–4 games: linear ramp from sparse → mid.
    t = (n - _MIN_GAMES) / (_GAMES_MID - _MIN_GAMES)
    return _CONF_SPARSE + t * (_CONF_MID - _CONF_SPARSE)


def _combined_confidence(home_games: int, away_games: int) -> float:
    """Geometric mean of each team's individual confidence score."""
    h = _confidence_from_games(home_games)
    a = _confidence_from_games(away_games)
    return round(math.sqrt(h * a), 3)


def _compute_recent_form(results: list[str]) -> float:
    """
    Trend signal: last-5 win rate minus the previous-5 win rate.
    Positive = improving, negative = declining.

    If fewer than 10 results are available the split is proportional;
    if fewer than 5, recent_form = 0.0 (insufficient data).

    Args:
        results: Ordered oldest → newest list of "W"/"D"/"L" strings.
    """
    if len(results) < 5:
        return 0.0
    last5 = results[-5:]
    prev5 = results[-10:-5]
    last5_wr = last5.count("W") / len(last5)
    if not prev5:
        return 0.0
    prev5_wr = prev5.count("W") / len(prev5)
    return round(last5_wr - prev5_wr, 3)


def _compute_current_streak(results: list[str]) -> int:
    """
    Consecutive-result streak entering the tournament.
    Scan from most recent backward; stop at the first break or draw.

    Returns:
        Positive int for a win streak, negative for a loss streak, 0 if the
        most recent result is a draw or the sequence alternates immediately.
        Capped at ±STREAK_CAP.

    Examples (oldest → newest):
        [W, W, L, W, W] → +2   (two most-recent consecutive wins)
        [W, W, W, L, L] → −2   (two most-recent consecutive losses)
        [W, L, W, W, D] → 0    (most recent is a draw)
    """
    if not results:
        return 0
    streak = 0
    direction: str | None = None
    for result in reversed(results):
        if result == "D":
            # A draw breaks any streak; if streak hasn't started, streak stays 0.
            break
        if direction is None:
            direction = result
        if result != direction:
            break
        streak += 1 if direction == "W" else -1
        if abs(streak) >= STREAK_CAP:
            break
    return streak


def _compute_big_scalp(fixtures: list[dict]) -> int:
    """
    Return 1 if the team beat a top-20 FIFA-ranked nation in their last 10
    fixtures, 0 otherwise.

    Args:
        fixtures: List of FixtureDict from get_team_last_n_fixtures (last 10).
    """
    for fix in fixtures:
        if fix["result"] == "W" and fix["opponent"] in _TOP_20:
            return 1
    return 0


# ── Fallback: FIFA ranking composite ─────────────────────────────────────────

_DEFAULT_RANKING = 100   # for teams absent from the dict

# Full FIFA ranking table for the fallback path.
_ALL_FIFA_RANKINGS: dict[str, int] = {
    **_FIFA_RANKINGS,
    "Austria":     21, "Ecuador":    22, "South Korea": 23,
    "Canada":      24, "Norway":     25, "Serbia":      26,
    "Poland":      27, "Ukraine":    28, "Australia":   29,
    "Turkey":      30, "Iran":       40, "Saudi Arabia": 45,
    "Ghana":       50, "Cameroon":   55, "Iraq":        70,
    "Panama":      75, "Bolivia":    80, "Jamaica":     85,
}


def _ranking_composite(home_team: str, away_team: str) -> float:
    """
    FIFA ranking delta proxy used when fewer than _MIN_GAMES results exist.
    A 40-place gap → composite ≈ 1.0 for clean sigmoid separation at k=3.0.
    """
    name_map = getattr(fetcher, "TEAM_NAME_MAP", {})
    h_key = name_map.get(home_team, home_team)
    a_key = name_map.get(away_team, away_team)
    home_rank = _ALL_FIFA_RANKINGS.get(h_key,
                _ALL_FIFA_RANKINGS.get(home_team, _DEFAULT_RANKING))
    away_rank = _ALL_FIFA_RANKINGS.get(a_key,
                _ALL_FIFA_RANKINGS.get(away_team, _DEFAULT_RANKING))
    return (away_rank - home_rank) / 40.0


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_momentum_stats(team_name: str, verbose: bool = False) -> MomentumStats:
    """
    Compute all momentum features from the team's last 10 international results.

    Args:
        team_name: e.g. "France", "Iraq". Resolved via fetcher's TEAM_NAME_MAP.
        verbose:   Print match count and source before aggregating.

    Returns:
        MomentumStats TypedDict. games_played=0 triggers the FIFA ranking
        fallback path in predict().
    """
    try:
        fixtures = fetcher.get_team_last_n_fixtures(team_name, n=10)
    except Exception as exc:
        if verbose:
            print(f"  [{team_name}] ERROR: {exc}")
        return _zeroed_momentum(team_name)

    n = len(fixtures)
    if verbose:
        source = "intl_results CSV" if n > 0 else "none"
        print(f"  [{team_name}] {n} matches found via {source}")

    if n == 0:
        return _zeroed_momentum(team_name)

    results        = [f["result"] for f in fixtures]        # oldest → newest
    goals_scored   = [f["goals_for"]     for f in fixtures]
    goals_conceded = [f["goals_against"] for f in fixtures]

    win_rate         = round(results.count("W") / n, 3)
    loss_rate        = round(results.count("L") / n, 3)
    clean_sheet_rate = round(sum(1 for g in goals_conceded if g == 0) / n, 3)
    recent_form      = _compute_recent_form(results)
    current_streak   = _compute_current_streak(results)
    big_scalp        = _compute_big_scalp(fixtures)
    playoff_winner   = PLAYOFF_WINNERS.get(team_name, 0)
    tournament_winner = TOURNAMENT_WINNERS.get(team_name, 0.0)

    return MomentumStats(
        games_played=n,
        win_rate=win_rate,
        loss_rate=loss_rate,
        clean_sheet_rate=clean_sheet_rate,
        recent_form=recent_form,
        current_streak=current_streak,
        big_scalp=big_scalp,
        playoff_winner=playoff_winner,
        tournament_winner=tournament_winner,
    )


def _zeroed_momentum(team_name: str) -> MomentumStats:
    """Return empty stats (triggers ranking fallback). Preserves static bonuses."""
    return MomentumStats(
        games_played=0,
        win_rate=0.0,
        loss_rate=0.0,
        clean_sheet_rate=0.0,
        recent_form=0.0,
        current_streak=0,
        big_scalp=0,
        playoff_winner=PLAYOFF_WINNERS.get(team_name, 0),
        tournament_winner=TOURNAMENT_WINNERS.get(team_name, 0.0),
    )


# ── Composite score ───────────────────────────────────────────────────────────

def team_momentum_score(stats: MomentumStats) -> float:
    """
    Single signed score summarising a team's current momentum.
    Higher = more in-form, more clinical, heading into the tournament well.
    Typical range: roughly [−0.45, +0.88].
    """
    score = (
        stats["win_rate"]           * 0.30
        + stats["recent_form"]      * 0.25
        + (-stats["loss_rate"])     * 0.20
        + stats["clean_sheet_rate"] * 0.15
        + stats["big_scalp"]        * 0.05
        + stats["playoff_winner"]   * 0.05
    )
    # Add tournament winner bonus — applied after composite so it doesn't
    # interact with the weighting formula. Capped at 0.08 by construction.
    tournament_bonus = stats["tournament_winner"]
    score = score + tournament_bonus
    return score


# ── Key factor generation ─────────────────────────────────────────────────────

def _build_key_factors(
    home_name: str,
    away_name: str,
    home: MomentumStats,
    away: MomentumStats,
    used_fallback: bool,
) -> list[str]:
    """
    Generate up to 4 human-readable insight strings in priority order:
      1. Playoff bonus (always shown when present)
      2. Form trend — improving or declining (threshold ≥ ±0.20)
      3. Big scalp — beat a top-20 nation
      4. Win streak ≥ 3 or loss streak ≤ -3
      5. Overall win rate ≥ 0.70 (filler if nothing else fires)

    Only fires when signals clear meaningful thresholds to avoid noise.
    """
    if used_fallback:
        return ["Limited pre-tournament data — prediction based on FIFA ranking"]

    factors: list[str] = []

    # ── 1. Playoff winner ────────────────────────────────────────────────────
    for name, stats in [(home_name, home), (away_name, away)]:
        if stats["playoff_winner"]:
            s = stats["current_streak"]
            streak_str = (
                f" — on {s}-match winning streak"   if s >= 2  else
                f" — but on {abs(s)}-match losing streak" if s <= -2 else ""
            )
            factors.append(f"{name} playoff momentum bonus{streak_str}")

    # ── 1b. Tournament winner bonus ──────────────────────────────────────────
    for name, stats in [(home_name, home), (away_name, away)]:
        tournament_bonus = stats.get("tournament_winner", 0.0)
        if tournament_bonus > 0:
            tournament_name = (
                "Euro 2024"        if name == "Spain"     else
                "Copa America 2024" if name == "Argentina" else
                "AFCON 2025"
            )
            factors.append(
                f"{name} tournament winner bonus ({tournament_bonus:.2f}) — "
                f"{tournament_name} champions"
            )

    # ── 2. Form trend (threshold: ±0.20) ────────────────────────────────────
    # last5_wr = win_rate + recent_form/2  (exact when games_played == 10)
    for name, stats in [(home_name, home), (away_name, away)]:
        rf = stats["recent_form"]
        if abs(rf) < 0.20:
            continue
        last5_wr = round(stats["win_rate"] + rf / 2, 2)
        prev5_wr = round(stats["win_rate"] - rf / 2, 2)
        if rf > 0:
            factors.append(
                f"{name} improving — last 5 win rate {last5_wr:.2f} "
                f"vs previous 5 at {max(prev5_wr, 0.0):.2f}"
            )
        else:
            factors.append(
                f"{name} declining — last 5 win rate {max(last5_wr, 0.0):.2f} "
                f"vs previous 5 at {prev5_wr:.2f}"
            )

    # ── 3. Big scalp ─────────────────────────────────────────────────────────
    # Re-fetch is free — CSV is already in memory after the first call.
    for name, stats in [(home_name, home), (away_name, away)]:
        if not stats["big_scalp"]:
            continue
        try:
            fixtures = fetcher.get_team_last_n_fixtures(name, n=10)
            # Report the highest-profile scalp (lowest FIFA rank number).
            scalps = [
                f for f in fixtures
                if f["result"] == "W" and f["opponent"] in _TOP_20
            ]
            if scalps:
                best = min(scalps, key=lambda f: _FIFA_RANKINGS.get(f["opponent"], 99))
                opp  = best["opponent"]
                rank = _FIFA_RANKINGS.get(opp, "?")
                tier = "top 10" if isinstance(rank, int) and rank <= 10 else "top 20"
                factors.append(
                    f"{name} beat {opp} ({tier}, FIFA #{rank}) "
                    f"— big scalp in last {stats['games_played']}"
                )
        except Exception:
            factors.append(f"{name} beat a top-20 nation — big scalp")

    # ── 4. Win / loss streak ─────────────────────────────────────────────────
    for name, stats in [(home_name, home), (away_name, away)]:
        if stats["playoff_winner"]:
            continue   # streak already embedded in the playoff message
        s = stats["current_streak"]
        if s >= 3:
            factors.append(
                f"{name} on {s}-match winning streak heading into the tournament"
            )
        elif s <= -3:
            factors.append(
                f"{name} on {abs(s)}-match losing streak — form concern"
            )

    # ── 5. High win rate (filler) ────────────────────────────────────────────
    for name, stats in [(home_name, home), (away_name, away)]:
        if len(factors) >= 4:
            break
        wr   = stats["win_rate"]
        gp   = stats["games_played"]
        wins = round(wr * gp)
        if wr >= 0.70:
            factors.append(
                f"{name} won {wins} of last {gp} heading into tournament"
            )

    return factors[:4] if factors else ["Momentum is evenly matched"]


# ── Public predict() ──────────────────────────────────────────────────────────

def predict(
    home_team: str,
    away_team: str,
    verbose: bool = False,
) -> PredictionOutput:
    """
    Predict match outcome probabilities from pre-tournament momentum signals.

    Uses each team's last 10 international results before June 11 2026.
    Falls back to FIFA ranking delta if fewer than 3 results are found.

    Args:
        home_team: e.g. "France", "Iraq". Resolved via fetcher's TEAM_NAME_MAP.
        away_team: Same.
        verbose:   Print match count + source per team while fetching.

    Returns:
        PredictionOutput with team1_win, draw, team2_win, confidence, key_factors.
    """
    home_stats = extract_momentum_stats(home_team, verbose=verbose)
    away_stats = extract_momentum_stats(away_team, verbose=verbose)

    home_games = home_stats["games_played"]
    away_games = away_stats["games_played"]
    used_fallback = min(home_games, away_games) < _MIN_GAMES

    if used_fallback:
        composite = _ranking_composite(home_team, away_team)
    else:
        composite = team_momentum_score(home_stats) - team_momentum_score(away_stats)
        # Quality anchor: raw momentum signals (winning streaks, recent results)
        # are schedule-biased — teams with easy qualifying schedules appear
        # stronger than those who played harder opponents.  A small log-rank
        # anchor ensures quality mismatches aren't flipped by schedule artifacts.
        name_map = getattr(fetcher, "TEAM_NAME_MAP", {})
        h_key = name_map.get(home_team, home_team)
        a_key = name_map.get(away_team, away_team)
        r1 = _ALL_FIFA_RANKINGS.get(h_key, _ALL_FIFA_RANKINGS.get(home_team, _DEFAULT_RANKING))
        r2 = _ALL_FIFA_RANKINGS.get(a_key, _ALL_FIFA_RANKINGS.get(away_team, _DEFAULT_RANKING))
        if r1 > 0 and r2 > 0:
            composite += math.log(r2 / r1) * 0.14

    # Independent draw prior — decays exponentially as composite grows.
    draw = BASE_DRAW_RATE * math.exp(-DRAW_DECAY * abs(composite))

    remaining = 1.0 - draw
    raw_home  = _sigmoid(composite)

    team1_win = _clamp(raw_home * remaining)
    team2_win = _clamp((1.0 - raw_home) * remaining)
    # Recompute draw from the remainder so the three always sum to exactly 1.0.
    draw = round(1.0 - team1_win - team2_win, 4)
    draw = max(0.0, draw)

    confidence = _combined_confidence(home_games, away_games)

    key_factors = _build_key_factors(
        home_team, away_team, home_stats, away_stats, used_fallback
    )

    return PredictionOutput(
        team1_win=round(team1_win, 4),
        draw=round(draw, 4),
        team2_win=round(team2_win, 4),
        confidence=confidence,
        key_factors=key_factors,
    )


# ── Test runner ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TEST_CASES = [
        ("France",      "Iraq"),
        ("Algeria",     "Netherlands"),
        ("Brazil",      "Morocco"),
    ]

    print(f"\n{'Match':<28} {'team1_win':>8} {'draw':>6} {'team2_win':>9} {'conf':>6}")
    print("-" * 120)

    for home, away in TEST_CASES:
        print(f"\n{home} vs {away}")
        try:
            result = predict(home, away, verbose=True)
            label  = f"→ {home} vs {away}"
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
