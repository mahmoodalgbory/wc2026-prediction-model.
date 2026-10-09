"""
Team market value model for World Cup 2026 match prediction.

Scores each match via cross-positional matchups (home attack vs away defence,
etc.) and maps the composite advantage to win/draw/loss probabilities through
a sigmoid function.

Tuning note: SIGMOID_K = 0.0025 is deliberately conservative. If predicted
probabilities feel too close to 50/50 for high-mismatch games, raise k toward
0.005–0.006. The clamp at [0.18, 0.82] prevents probabilities from hitting
certainty regardless of k.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TypedDict

DATA_PATH = Path(__file__).parent.parent / "data" / "squad_values.json"

PROB_MAX = 0.82
PROB_MIN = 0.18

# Controls how steeply the sigmoid responds to the composite score.
# At k=0.0050 a composite of ~275 reaches ~0.80, which is appropriate for
# extreme mismatches like France (€1280m) vs Iraq (€28m).
SIGMOID_K = 0.0050

# Only the three co-host nations receive a home-crowd nudge; all other
# World Cup matches are played on neutral ground.
HOST_NATIONS: frozenset[str] = frozenset({"United States", "USA", "Canada", "Mexico"})
HOME_NUDGE = 0.05

# Draw probability is modelled as an independent exponential decay:
#   draw = BASE_DRAW_RATE * exp(-DRAW_DECAY * |composite|)
# At composite=0 (perfectly even) draw ≈ 27% (World Cup group-stage average).
# At composite=200 (large gap) draw falls to ~5%.
# Home/away are then scaled into the remaining (1 - draw) mass.
BASE_DRAW_RATE: float = 0.27
DRAW_DECAY: float = 0.008


class PositionalValues(TypedDict):
    goalkeeper: float
    defense: float
    midfield: float
    attack: float


class TeamStats(TypedDict):
    total_value_eur_millions: float
    starting_xi_value: float
    bench_value: float
    positional_values: PositionalValues
    avg_age: float
    top5_league_pct: float
    star_concentration: float


class PredictionOutput(TypedDict):
    team1_win: float
    draw: float
    team2_win: float
    confidence: float
    key_factors: list[str]


def _load_squad_data(path: Path = DATA_PATH) -> dict[str, TeamStats]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sigmoid(x: float, k: float = SIGMOID_K) -> float:
    """Maps a composite score to (0, 1). Positive x favours team1."""
    return 1.0 / (1.0 + math.exp(-k * x))


def _clamp(value: float, lo: float = PROB_MIN, hi: float = PROB_MAX) -> float:
    return max(lo, min(hi, value))


def _composite_advantage(home: TeamStats, away: TeamStats) -> float:
    """
    Returns a signed score (positive = home advantage) using cross-positional
    matchups that mirror actual on-pitch contests.

        attack_vs_defense: home attacking threat against away back-line
        defense_vs_attack: home defensive solidity against away forwards
        midfield_battle:   control of the centre of the pitch
        gk_pressure:       away attacking load on home goalkeeper (inverted)
    """
    hp = home["positional_values"]
    ap = away["positional_values"]

    attack_vs_defense = hp["attack"]    - ap["defense"]
    defense_vs_attack = hp["defense"]   - ap["attack"]
    midfield_battle   = hp["midfield"]  - ap["midfield"]
    gk_pressure       = ap["attack"]    - hp["goalkeeper"]  # positive = bad for home

    return (
        attack_vs_defense * 0.30
        + defense_vs_attack * 0.30
        + midfield_battle   * 0.25
        + (-gk_pressure)    * 0.15
    )


def _confidence(
    home: TeamStats,
    away: TeamStats,
    squad_data: dict[str, TeamStats],
) -> float:
    """
    Flat baseline confidence of 0.65.  Squad value data is considered reliable
    regardless of squad size mismatch, so a geometric-mean penalty is
    unnecessary and was suppressing confidence for smaller nations unfairly.
    """
    return 0.65


def _fmt(value: float) -> str:
    """Format a euro value as a compact string, e.g. 258.0 → '€258m'."""
    return f"€{value:.0f}m"


def _build_key_factors(
    home_name: str,
    away_name: str,
    home: TeamStats,
    away: TeamStats,
    is_host: bool,
) -> list[str]:
    hp = home["positional_values"]
    ap = away["positional_values"]
    factors: list[str] = []

    # Attack vs opposing defence
    atk_def_gap = hp["attack"] - ap["defense"]
    if atk_def_gap > 0:
        label = "comfortable advantage" if atk_def_gap > 80 else "closer than it looks"
        factors.append(
            f"{home_name}'s {_fmt(hp['attack'])} attack faces "
            f"{away_name}'s {_fmt(ap['defense'])} defense — {label}"
        )
    else:
        factors.append(
            f"{away_name}'s {_fmt(ap['defense'])} defense holds firm against "
            f"{home_name}'s {_fmt(hp['attack'])} attack"
        )

    # Midfield battle
    mid_gap = abs(hp["midfield"] - ap["midfield"])
    if mid_gap >= 5:
        dominant = home_name if hp["midfield"] > ap["midfield"] else away_name
        dominant_val = max(hp["midfield"], ap["midfield"])
        other_val = min(hp["midfield"], ap["midfield"])
        other = away_name if dominant == home_name else home_name
        factors.append(
            f"{dominant} midfield ({_fmt(dominant_val)}) vs "
            f"{other} ({_fmt(other_val)}) — expect game control"
        )

    # Away attack pressure on home GK
    gk_pressure = ap["attack"] - hp["goalkeeper"]
    if gk_pressure > 10:
        factors.append(
            f"{away_name}'s {_fmt(ap['attack'])} attack puts "
            f"{home_name} GK ({_fmt(hp['goalkeeper'])}) under pressure"
        )
    elif gk_pressure < -20:
        factors.append(
            f"{home_name} GK ({_fmt(hp['goalkeeper'])}) is well-protected; "
            f"{away_name} attack ({_fmt(ap['attack'])}) poses little threat"
        )

    # Home defensive solidity vs away attack
    def_atk_gap = hp["defense"] - ap["attack"]
    if abs(def_atk_gap) >= 15:
        if def_atk_gap > 0:
            factors.append(
                f"{home_name}'s {_fmt(hp['defense'])} defense comfortably "
                f"outvalues {away_name}'s {_fmt(ap['attack'])} attack"
            )
        else:
            factors.append(
                f"{away_name}'s {_fmt(ap['attack'])} attack threatens "
                f"{home_name}'s {_fmt(hp['defense'])} defense"
            )

    # Host nation nudge
    if is_host:
        factors.append(
            f"{home_name} receive a home-crowd boost as co-host (+{int(HOME_NUDGE*100)}%)"
        )

    return factors if factors else ["Squads are closely matched across all positions"]


def predict(
    home_team: str,
    away_team: str,
    data_path: Path = DATA_PATH,
) -> PredictionOutput:
    """
    Predict match outcome probabilities based on squad positional market values.

    Args:
        home_team: Key matching an entry in squad_values.json.
        away_team: Key matching an entry in squad_values.json.
        data_path: Override for the JSON data file (useful in tests).

    Returns:
        PredictionOutput with team1_win, draw, team2_win, confidence, key_factors.

    Raises:
        KeyError: If either team is absent from the data file.
    """
    squad_data = _load_squad_data(data_path)

    if home_team not in squad_data:
        raise KeyError(f"Team '{home_team}' not found in squad data.")
    if away_team not in squad_data:
        raise KeyError(f"Team '{away_team}' not found in squad data.")

    home = squad_data[home_team]
    away = squad_data[away_team]

    composite = _composite_advantage(home, away)

    # Draw is an independent prior that decays as the composite grows.
    # This ensures draw is never zero even in high-mismatch games.
    draw = BASE_DRAW_RATE * math.exp(-DRAW_DECAY * abs(composite))

    # Home and away share the remaining probability mass, preserving their
    # relative ratio from the sigmoid.
    remaining = 1.0 - draw
    raw_home = _sigmoid(composite)

    is_host = home_team in HOST_NATIONS
    if is_host:
        raw_home = min(raw_home + HOME_NUDGE, 1.0)

    # Scale into (1 - draw) space so the three outcomes always sum to 1.0.
    team1_win = _clamp(raw_home * remaining)
    team2_win = _clamp((1.0 - raw_home) * remaining)

    # Re-derive draw from leftovers so rounding never breaks the sum.
    draw = round(1.0 - team1_win - team2_win, 4)
    draw = max(0.0, draw)

    return PredictionOutput(
        team1_win=round(team1_win, 4),
        draw=round(draw, 4),
        team2_win=round(team2_win, 4),
        confidence=_confidence(home, away, squad_data),
        key_factors=_build_key_factors(home_team, away_team, home, away, is_host),
    )


if __name__ == "__main__":
    TEST_CASES = [
        ("France",  "Iraq"),
        ("USA",     "England"),
        ("Germany", "Curaçao"),
        ("Brazil",  "Morocco"),
    ]
    EXPECTED = {
        ("France",  "Iraq"):    (0.78, 0.82),
        ("USA",     "England"): (0.38, 0.42),
        ("Germany", "Curaçao"): (0.80, 0.82),
        ("Brazil",  "Morocco"): (0.62, 0.70),
    }

    print(f"{'Match':<25} {'team1_win':>8} {'draw':>6} {'team2_win':>9} "
          f"{'conf':>6}  {'in_range':>8}  key_factors")
    print("-" * 110)

    for home, away in TEST_CASES:
        result = predict(home, away)
        lo, hi = EXPECTED[(home, away)]
        in_range = lo <= result["team1_win"] <= hi
        print(
            f"{home+' vs '+away:<25} "
            f"{result['team1_win']:>8.4f} "
            f"{result['draw']:>6.4f} "
            f"{result['team2_win']:>9.4f} "
            f"{result['confidence']:>6.3f}  "
            f"{'✓' if in_range else '✗ (expected '+str(lo)+'-'+str(hi)+')':>8}  "
            f"{result['key_factors'][0]}"
        )
