"""
Form & results model for World Cup 2026 match prediction.

Renamed from tactical_model — scores matchups based on raw results form:
goal scoring, defensive solidity, and clean-sheet records using each team's
last 10 international results across all competitions (qualifiers, friendlies,
Nations League, continental cups) before June 11 2026.

Data source: footballdata_fetcher → martj42/international_results on GitHub.
This CSV covers every international match from 1872 to present and is updated
daily. It provides: date, teams, scores, tournament. Possession and shots on
target are NOT in the dataset; the model uses a constant shots denominator
(4.5 shots/game, the international average) so attack_efficiency still
differentiates teams by their actual scoring rates.

Cross-matchup logic mirrors team_value_model.py's positional approach:
  - home attack efficiency vs away defensive solidity
  - away attack efficiency vs home defensive solidity
  - midfield possession delta (defaults to 0 — data unavailable)

Sigmoid + draw-prior approach is identical to Session 1. SIGMOID_K and
DRAW_DECAY are recalibrated for the tactical composite scale (≈ ±1.0) rather
than the squad-value scale (≈ ±500). BASE_DRAW_RATE stays at 0.27.

Fallback: fewer than 3 results found → FIFA ranking delta proxy.
Confidence: 10 results = 0.85, 5–9 = 0.65, < 3 = 0.30.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import TypedDict

# Ensure project root is on sys.path for sibling imports.
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.fetchers import footballdata_fetcher as fetcher

# ── Tuning constants ─────────────────────────────────────────────────────────
# BASE_DRAW_RATE is shared across all pipeline models (World Cup average).
BASE_DRAW_RATE: float = 0.27
PROB_MIN: float = 0.18
PROB_MAX: float = 0.82

# SIGMOID_K and DRAW_DECAY are recalibrated for the tactical composite range.
# Team-value composites sit in the ±500 range (million-euro squad gaps).
# Tactical composites sit in the ±1.0 range (fractional stat differences).
# Using the Session-1 value of 0.0025 here would give sigmoid ≈ 0.500375
# for a composite of 0.3 — completely flat. The values below produce the same
# probabilistic shape but operate on the correct input scale.
SIGMOID_K: float = 3.0    # sigmoid(3.0 × 0.35) ≈ 0.65 for a moderate advantage
DRAW_DECAY: float = 0.8   # draw drops from 0.27 → ~0.19 at composite 0.35

# When shots-on-target data is unavailable (this dataset doesn't carry it),
# we use the international average as a fixed denominator so that
# attack_efficiency = goals_scored / AVG_SHOTS still differentiates teams
# by how many goals they score rather than collapsing everyone to 0.15.
_INTL_AVG_SHOTS_ON_TARGET: float = 4.5  # ~4–5 shots on target/game, international level

# Below this many matches we fall back to the FIFA ranking proxy.
MIN_GAMES_THRESHOLD: int = 3

# Confidence anchors derived from match-data richness.
#   < 3 matches  → 0.30 (sparse historical record, using ranking proxy)
#   3–4 matches  → 0.30 → 0.65 (linear ramp)
#   5–9 matches  → 0.65 (good qualifier dataset)
#   10+ matches  → 0.85 (full pre-tournament picture)
_CONF_SPARSE  = 0.30
_CONF_MID     = 0.65
_CONF_FULL    = 0.85
_GAMES_MID    = 5
_GAMES_FULL   = 10


# ── TypedDicts ───────────────────────────────────────────────────────────────

class TacticalStats(TypedDict):
    """Raw per-team tactical features extracted from API-Football."""
    games_played: int
    avg_possession: float       # e.g. 55.3 (percent)
    avg_shots_on_target: float
    avg_goals_scored: float
    avg_goals_conceded: float
    clean_sheet_rate: float     # fraction [0, 1]
    failed_to_score_rate: float # fraction [0, 1]


class DerivedFeatures(TypedDict):
    """Computed features used in the cross-matchup composite."""
    attack_efficiency: float    # goals scored / shots on target
    defensive_solidity: float   # clean_sheet_rate - failed_to_score_rate
    avg_possession: float


class PredictionOutput(TypedDict):
    team1_win: float
    draw: float
    team2_win: float
    confidence: float
    key_factors: list[str]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """Maps composite score to (0, 1); positive x favours team1."""
    return 1.0 / (1.0 + math.exp(-SIGMOID_K * x))


def _clamp(value: float) -> float:
    return max(PROB_MIN, min(PROB_MAX, value))


def _confidence_from_games(games: int) -> float:
    """
    Piecewise linear confidence derived from how many pre-tournament matches
    were found. Three anchors:
      0–2  → 0.30  (ranking proxy territory)
      3–4  → 0.30 → 0.65  (ramping up as qualifier data grows)
      5–9  → 0.65  (solid qualifier sample)
      10+  → 0.85  (full pre-tournament picture)
    """
    if games < MIN_GAMES_THRESHOLD:
        return _CONF_SPARSE
    if games >= _GAMES_FULL:
        return _CONF_FULL
    if games >= _GAMES_MID:
        # 5–9 matches: flat at mid confidence.
        return _CONF_MID
    # 3–4 matches: linear ramp from sparse to mid.
    t = (games - MIN_GAMES_THRESHOLD) / (_GAMES_MID - MIN_GAMES_THRESHOLD)
    return _CONF_SPARSE + t * (_CONF_MID - _CONF_SPARSE)


def _combined_confidence(home_games: int, away_games: int) -> float:
    """Geometric mean of both teams' confidence scores."""
    h = _confidence_from_games(home_games)
    a = _confidence_from_games(away_games)
    return round(math.sqrt(h * a), 3)


# ── Data fetching ────────────────────────────────────────────────────────────

# FIFA rankings — fallback proxy when fewer than MIN_GAMES_THRESHOLD matches found.
# Lower rank = better. Source: approximate FIFA rankings entering WC 2026.
_FIFA_RANKINGS: dict[str, int] = {
    "Argentina":   1,  "France":      2,  "England":     3,
    "Belgium":     4,  "Brazil":      5,  "Portugal":    6,
    "Netherlands": 7,  "Spain":       8,  "Germany":     9,
    "Italy":      10,  "USA":        11,  "Mexico":     12,
    "Colombia":   13,  "Uruguay":    14,  "Croatia":    15,
    "Switzerland":16,  "Japan":      17,  "Morocco":    18,
    "Senegal":    19,  "Denmark":    20,  "Austria":    21,
    "Ecuador":    22,  "South Korea":23,  "Canada":     24,
    "Norway":     25,  "Serbia":     26,  "Poland":     27,
    "Ukraine":    28,  "Australia":  29,  "Turkey":     30,
    "Iran":       40,  "Saudi Arabia":45, "Ghana":      50,
    "Cameroon":   55,  "Iraq":       70,  "Panama":     75,
    "Bolivia":    80,  "Jamaica":    85,
}
_DEFAULT_RANKING = 100  # fallback for teams absent from the table


def extract_tactical_stats(team_name: str, verbose: bool = False) -> TacticalStats:
    """
    Aggregate tactical stats from the team's last 10 international matches.

    Data source: footballdata_fetcher → martj42/international_results CSV.
    Covers qualifiers, friendlies, Nations League, continental cups — all
    completed matches before June 11 2026.

    Goals, clean sheets, and failed-to-score come directly from match scores.
    Possession is unavailable in this dataset and defaults to 50.0 (neutral).
    Shots on target are unavailable; avg_shots_on_target is set to the
    international average (_INTL_AVG_SHOTS_ON_TARGET = 4.5) so that
    attack_efficiency = goals / 4.5 still differentiates teams by scoring rate
    rather than collapsing to a flat fallback.

    Falls back to zeroed stats (→ FIFA ranking proxy in predict()) if the team
    has no qualifying matches.

    Args:
        team_name: e.g. "France" or "USA". Resolved via TEAM_NAME_MAP in fetcher.
        verbose:   If True, print match count and data source before aggregating.
    """
    try:
        fixtures = fetcher.get_team_last_n_fixtures(team_name, n=10)
    except Exception as exc:
        if verbose:
            print(f"  [{team_name}] ERROR fetching: {exc}")
        return TacticalStats(
            games_played=0, avg_possession=50.0, avg_shots_on_target=0.0,
            avg_goals_scored=0.0, avg_goals_conceded=0.0,
            clean_sheet_rate=0.0, failed_to_score_rate=0.0,
        )

    n = len(fixtures)
    source = "intl_results CSV" if n > 0 else "none"

    if verbose:
        print(f"  [{team_name}] {n} matches found via {source}")

    if n == 0:
        return TacticalStats(
            games_played=0, avg_possession=50.0, avg_shots_on_target=0.0,
            avg_goals_scored=0.0, avg_goals_conceded=0.0,
            clean_sheet_rate=0.0, failed_to_score_rate=0.0,
        )

    goals_scored   = [f["goals_for"]     for f in fixtures]
    goals_conceded = [f["goals_against"] for f in fixtures]
    avg_goals_scored   = round(sum(goals_scored)   / n, 3)
    avg_goals_conceded = round(sum(goals_conceded)  / n, 3)
    clean_sheet_rate    = round(sum(1 for g in goals_conceded if g == 0) / n, 3)
    failed_to_score_rate = round(sum(1 for g in goals_scored  if g == 0) / n, 3)

    # Possession and shots on target are not in this dataset.
    # avg_shots_on_target = _INTL_AVG_SHOTS_ON_TARGET (constant denominator) so
    # _derive() produces attack_efficiency = goals_avg / 4.5, differentiating
    # teams by scoring rate rather than returning a flat 0.15 fallback for all.
    return TacticalStats(
        games_played=n,
        avg_possession=50.0,                       # not available
        avg_shots_on_target=_INTL_AVG_SHOTS_ON_TARGET,  # constant proxy
        avg_goals_scored=avg_goals_scored,
        avg_goals_conceded=avg_goals_conceded,
        clean_sheet_rate=clean_sheet_rate,
        failed_to_score_rate=failed_to_score_rate,
    )


# ── Derived features ─────────────────────────────────────────────────────────

def _safe_div(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Division with a safe fallback for zero denominator."""
    return numerator / denominator if denominator else fallback


def _derive(stats: TacticalStats) -> DerivedFeatures:
    """
    Compute second-order features from raw tactical stats.

    attack_efficiency:
        When real shots-on-target data is present: goals / shots (conversion rate).
        When avg_shots_on_target == _INTL_AVG_SHOTS_ON_TARGET (constant proxy):
        efficiency = goals / 4.5, which scales with scoring rate so France (2.5
        gf/game → 0.56) still beats Iraq (1.5 gf/game → 0.33) meaningfully.

    defensive_solidity:
        Signed balance of keeping clean sheets vs failing to score.
        Range ≈ [−1, +1].

    avg_possession:
        Defaults to 50.0 when unavailable; midfield_dominance = 0.
    """
    attack_efficiency = _safe_div(
        stats["avg_goals_scored"],
        stats["avg_shots_on_target"],
        fallback=0.15,  # last resort if both goals and shots are 0
    )
    defensive_solidity = stats["clean_sheet_rate"] - stats["failed_to_score_rate"]

    return DerivedFeatures(
        attack_efficiency=attack_efficiency,
        defensive_solidity=defensive_solidity,
        avg_possession=stats["avg_possession"],
    )


# ── Composite score ───────────────────────────────────────────────────────────

def _composite_from_features(
    home: DerivedFeatures,
    away: DerivedFeatures,
) -> float:
    """
    Cross-matchup composite score (positive = team1 advantage).

    home_attack_vs_away_defense: home's clinical finishing against away defensive record.
    away_attack_vs_home_defense: away's clinical finishing against home defensive record (inverted).
    midfield_dominance:          possession delta drives tempo and territory.

    Weights sum to 1.0; attack matchups each get 0.35 (70% combined) because
    goals ultimately decide matches, possession gets 0.30.
    """
    home_attack_vs_away_defense = home["attack_efficiency"] - away["defensive_solidity"]
    away_attack_vs_home_defense = away["attack_efficiency"] - home["defensive_solidity"]
    midfield_dominance = home["avg_possession"] - away["avg_possession"]

    return (
        home_attack_vs_away_defense    * 0.35
        + (-away_attack_vs_home_defense) * 0.35
        + midfield_dominance             * 0.30
    )


def _ranking_composite(home_team: str, away_team: str) -> float:
    """
    FIFA ranking delta as a proxy composite when match-data is unavailable.

    Names are resolved through the fetcher's TEAM_NAME_MAP first so aliases
    like "USA" correctly look up rank 11.
    Scaled so a 20-place ranking gap ≈ composite 0.5, which at SIGMOID_K=3.0
    gives sigmoid ≈ 0.78 — a realistic edge for a clearly stronger side.
    """
    name_map = getattr(fetcher, "TEAM_NAME_MAP", {})
    # Rankings keyed by common name; resolve from alias → common → ranking.
    h_key = name_map.get(home_team, home_team)
    a_key = name_map.get(away_team, away_team)
    home_rank = _FIFA_RANKINGS.get(h_key, _FIFA_RANKINGS.get(home_team, _DEFAULT_RANKING))
    away_rank = _FIFA_RANKINGS.get(a_key, _FIFA_RANKINGS.get(away_team, _DEFAULT_RANKING))
    # Positive composite → home team ranked better (lower number).
    rank_delta = away_rank - home_rank
    # Divide by 40 so a 20-place gap → 0.5, a 40-place gap → 1.0.
    return rank_delta / 40.0


# ── Key factor generation ────────────────────────────────────────────────────

def _build_key_factors(
    home_name: str,
    away_name: str,
    home_stats: TacticalStats,
    away_stats: TacticalStats,
    home_feat: DerivedFeatures,
    away_feat: DerivedFeatures,
    used_fallback: bool,
) -> list[str]:
    """Generate human-readable insight strings mirroring Session 1 style."""
    factors: list[str] = []

    if used_fallback:
        factors.append(
            f"Limited WC 2026 data — predictions based on FIFA ranking proxy"
        )
        return factors

    # Goals conceded
    if away_stats["avg_goals_conceded"] < 0.6:
        factors.append(
            f"{away_name} concede only {away_stats['avg_goals_conceded']:.1f} "
            f"goals/game — elite defensive solidity"
        )
    elif home_stats["avg_goals_conceded"] < 0.6:
        factors.append(
            f"{home_name} concede only {home_stats['avg_goals_conceded']:.1f} "
            f"goals/game — elite defensive solidity"
        )

    # Failed to score
    if away_stats["failed_to_score_rate"] >= 0.35:
        factors.append(
            f"{away_name} failed to score in "
            f"{away_stats['failed_to_score_rate']*100:.0f}% of recent games "
            f"— attack vulnerability"
        )
    elif home_stats["failed_to_score_rate"] >= 0.35:
        factors.append(
            f"{home_name} failed to score in "
            f"{home_stats['failed_to_score_rate']*100:.0f}% of recent games "
            f"— attack vulnerability"
        )

    # Possession dominance
    poss_gap = abs(home_feat["avg_possession"] - away_feat["avg_possession"])
    if poss_gap >= 8:
        dominant = home_name if home_feat["avg_possession"] > away_feat["avg_possession"] else away_name
        dom_poss = max(home_feat["avg_possession"], away_feat["avg_possession"])
        other = away_name if dominant == home_name else home_name
        other_poss = min(home_feat["avg_possession"], away_feat["avg_possession"])
        factors.append(
            f"{dominant} possession ({dom_poss:.0f}%) dominates midfield "
            f"vs {other} ({other_poss:.0f}%)"
        )

    # Attack efficiency
    if home_feat["attack_efficiency"] > away_feat["attack_efficiency"] * 1.25:
        factors.append(
            f"{home_name} convert {home_feat['attack_efficiency']*100:.0f}% of shots on target "
            f"— more clinical than {away_name} ({away_feat['attack_efficiency']*100:.0f}%)"
        )
    elif away_feat["attack_efficiency"] > home_feat["attack_efficiency"] * 1.25:
        factors.append(
            f"{away_name} convert {away_feat['attack_efficiency']*100:.0f}% of shots on target "
            f"— more clinical than {home_name} ({home_feat['attack_efficiency']*100:.0f}%)"
        )

    # Clean sheet comparison
    if home_stats["clean_sheet_rate"] > 0.5:
        factors.append(
            f"{home_name} kept clean sheets in "
            f"{home_stats['clean_sheet_rate']*100:.0f}% of games — defensive resilience"
        )
    elif away_stats["clean_sheet_rate"] > 0.5:
        factors.append(
            f"{away_name} kept clean sheets in "
            f"{away_stats['clean_sheet_rate']*100:.0f}% of games — defensive resilience"
        )

    return factors if factors else ["Teams are tactically evenly matched"]


# ── Public predict() ─────────────────────────────────────────────────────────

def predict(
    home_team: str,
    away_team: str,
    verbose: bool = False,
) -> PredictionOutput:
    """
    Predict match outcome probabilities from pre-tournament international form.

    Fetches each team's last 10 results (qualifiers, friendlies, Nations
    League, continental cups) before June 11 2026. Falls back to FIFA ranking
    delta if fewer than 3 results are found for either team.

    Args:
        home_team: e.g. "France", "USA". Resolved via fetcher's TEAM_NAME_MAP.
        away_team: Same.
        verbose:   Print match count + source per team before computing.

    Returns:
        PredictionOutput with team1_win, draw, team2_win, confidence, key_factors.
    """
    home_stats = extract_tactical_stats(home_team, verbose=verbose)
    away_stats = extract_tactical_stats(away_team, verbose=verbose)

    games_home = home_stats["games_played"]
    games_away = away_stats["games_played"]
    min_games = min(games_home, games_away)

    used_fallback = min_games < MIN_GAMES_THRESHOLD

    if used_fallback:
        # Use FIFA ranking delta scaled to a composite value.
        composite = _ranking_composite(home_team, away_team)
    else:
        home_feat = _derive(home_stats)
        away_feat = _derive(away_stats)
        composite = _composite_from_features(home_feat, away_feat)
        # Quality anchor: raw form stats are schedule-biased (teams in weak
        # confederations inflate their numbers vs weak opponents).  Blend in
        # a small log-rank signal so extreme mismatches (e.g. France #2 vs
        # Iraq #70) aren't dominated by Congolese defensive stats from AFCON
        # qualifying or Uzbek wins over Singapore.
        name_map = getattr(fetcher, "TEAM_NAME_MAP", {})
        h_key = name_map.get(home_team, home_team)
        a_key = name_map.get(away_team, away_team)
        r1 = _FIFA_RANKINGS.get(h_key, _FIFA_RANKINGS.get(home_team, _DEFAULT_RANKING))
        r2 = _FIFA_RANKINGS.get(a_key, _FIFA_RANKINGS.get(away_team, _DEFAULT_RANKING))
        if r1 > 0 and r2 > 0:
            composite += math.log(r2 / r1) * 0.14

    # Draw is an independent exponential prior that decays with composite magnitude.
    draw = BASE_DRAW_RATE * math.exp(-DRAW_DECAY * abs(composite))

    remaining = 1.0 - draw
    raw_home = _sigmoid(composite)

    team1_win = _clamp(raw_home * remaining)
    team2_win = _clamp((1.0 - raw_home) * remaining)
    # Recompute draw from leftovers so sum is exactly 1.0 regardless of clamping.
    draw = round(1.0 - team1_win - team2_win, 4)
    draw = max(0.0, draw)

    confidence = _combined_confidence(games_home, games_away)

    if used_fallback:
        home_feat = DerivedFeatures(
            attack_efficiency=0.0, defensive_solidity=0.0,
            avg_possession=home_stats["avg_possession"],
        )
        away_feat = DerivedFeatures(
            attack_efficiency=0.0, defensive_solidity=0.0,
            avg_possession=away_stats["avg_possession"],
        )

    key_factors = _build_key_factors(
        home_team, away_team,
        home_stats, away_stats,
        home_feat, away_feat,
        used_fallback,
    )

    return PredictionOutput(
        team1_win=round(team1_win, 4),
        draw=round(draw, 4),
        team2_win=round(team2_win, 4),
        confidence=confidence,
        key_factors=key_factors,
    )


# ── Test runner ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TEST_CASES = [
        ("France",  "Iraq"),
        ("Brazil",  "Morocco"),
        ("USA",     "England"),
        ("Norway",  "Senegal"),
    ]

    print(f"\n{'Match':<25} {'team1_win':>8} {'draw':>6} {'team2_win':>9} "
          f"{'conf':>6}  key_factors")
    print("-" * 120)

    for home, away in TEST_CASES:
        print(f"\n{home} vs {away}")
        try:
            result = predict(home, away, verbose=True)
            factors_str = " | ".join(result["key_factors"][:2])
            print(
                f"{'→ ' + home + ' vs ' + away:<25} "
                f"{result['team1_win']:>8.4f} "
                f"{result['draw']:>6.4f} "
                f"{result['team2_win']:>9.4f} "
                f"{result['confidence']:>6.3f}  "
                f"{factors_str}"
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
