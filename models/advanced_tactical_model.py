"""
Advanced tactical model for World Cup 2026 match prediction.

Computes opponent-quality-adjusted proxy metrics from the martj42 international
results dataset (already in data/fetchers/footballdata_fetcher.py) — no external
API required.  All six features are derived purely from goals and match results
weighted by opponent FIFA ranking:

    quality_adj_xg_for     — weighted goals scored  / total opponent weight
    quality_adj_xg_against — weighted goals conceded / total opponent weight
    quality_adj_xg_diff    — xg_for − xg_against
    proxy_ppda             — goals_conceded_avg / (clean_sheet_rate + 0.1)
                             (lower = more aggressive press proxy)
    clinical_efficiency    — simple_goals_for_avg / quality_adj_xg_for
                             (> 1.0 = overperforming opposition quality)
    neutral_form_score     — (away_win_rate × 1.2 + home_win_rate × 0.8) / 2
                             (WC is neutral, away results weighted up)
    press_success_rate     — wins_vs_top20 / max(games_vs_top20, 1)

Opponent quality buckets (FIFA ranking → weight):
    rank  1–10  → 1.50   rank 11–20 → 1.25
    rank 21–40  → 1.00   rank 41–70 → 0.75   rank 71+  → 0.50

Cross-matchup composite (team1 perspective):
    xg_attack_vs_defense  = team1.quality_adj_xg_for   − team2.quality_adj_xg_against
    xg_defense_vs_attack  = −(team1.quality_adj_xg_against − team2.quality_adj_xg_for)
    ppda_battle           = team2.proxy_ppda − team1.proxy_ppda   (+ = team1 presses harder)
    neutral_form          = team1.neutral_form_score − team2.neutral_form_score
    press_success         = team1.press_success_rate  − team2.press_success_rate

    composite = (
        xg_attack_vs_defense × 0.30
      + xg_defense_vs_attack  × 0.30
      + ppda_battle           × 0.20
      + neutral_form          × 0.10
      + press_success         × 0.10
    )

Sigmoid + draw prior identical to all other pipeline models:
    SIGMOID_K = 3.0,  BASE_DRAW_RATE = 0.27,  DRAW_DECAY = 0.8

Confidence tiers:
    10+ games, 3+ vs top-20 → 0.85
    10+ games, 0–2 vs top-20 → 0.70
    5–9 games              → 0.60
    < 3 games              → 0.30 (FIFA ranking delta proxy)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import TypedDict

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import the fetcher module AND its internal helpers needed for raw row access.
# The FixtureDict does not carry is_home, so we reconstruct fixtures from raw
# rows — exactly the same technique used inside the fetcher itself.
from data.fetchers.footballdata_fetcher import (   # noqa: E402
    FIXTURE_CUTOFF,
    TEAM_NAME_MAP,
    _competition_weight,   # tournament name → float weight
    _is_played,            # row has valid integer scores
    _load_results,         # full martj42 CSV as list of row-dicts
    _resolve,              # project team name → dataset canonical name
)

# ── Tuning constants ──────────────────────────────────────────────────────────

SIGMOID_K:      float = 3.0
BASE_DRAW_RATE: float = 0.27
DRAW_DECAY:     float = 0.8
PROB_MIN:       float = 0.18
PROB_MAX:       float = 0.82

# Minimum fixtures before using the ranking proxy instead of the full formula.
_MIN_GAMES: int = 3

# Confidence anchors.
_CONF_FALLBACK: float = 0.30   # < _MIN_GAMES
_CONF_MID:      float = 0.60   # 5–9 games
_CONF_GOOD:     float = 0.70   # 10+ games, few top-20 opponents
_CONF_FULL:     float = 0.85   # 10+ games, 3+ top-20 opponents

# Threshold of top-20 games that unlocks the FULL confidence tier.
_TOP20_THRESHOLD: int = 3

# ── Opponent quality weights ──────────────────────────────────────────────────

OPPONENT_WEIGHTS: dict[str, float] = {
    "top10": 1.50,   # rank  1–10
    "top20": 1.25,   # rank 11–20
    "top40": 1.00,   # rank 21–40
    "top70": 0.75,   # rank 41–70
    "rest":  0.50,   # rank 71+
}


def _opp_quality_weight(rank: int) -> float:
    """Return the quality multiplier for an opponent at the given FIFA rank."""
    if rank <= 10:
        return OPPONENT_WEIGHTS["top10"]
    if rank <= 20:
        return OPPONENT_WEIGHTS["top20"]
    if rank <= 40:
        return OPPONENT_WEIGHTS["top40"]
    if rank <= 70:
        return OPPONENT_WEIGHTS["top70"]
    return OPPONENT_WEIGHTS["rest"]


# ── FIFA rankings ─────────────────────────────────────────────────────────────
# Covers all 48 WC 2026 teams plus common international opponents.
# Both project names (e.g. "USA") and martj42 dataset names (e.g. "United States")
# are included so _opponent_rank() works regardless of which form appears in rows.

_FIFA_RANKINGS: dict[str, int] = {
    # Top 10
    "Argentina":             1,
    "France":                2,
    "England":               3,
    "Belgium":               4,
    "Brazil":                5,
    "Portugal":              6,
    "Netherlands":           7,
    "Spain":                 8,
    "Germany":               9,
    "Italy":                10,
    # 11–20
    "United States":        11,   "USA":              11,
    "Mexico":               12,
    "Colombia":             13,
    "Uruguay":              14,
    "Croatia":              15,
    "Switzerland":          16,
    "Japan":                17,
    "Morocco":              18,
    "Senegal":              19,
    "Denmark":              20,
    # 21–40
    "Austria":              21,
    "Ecuador":              22,
    "South Korea":          23,   "Korea Republic":   23,
    "Canada":               24,
    "Norway":               25,
    "Serbia":               26,
    "Poland":               27,
    "Ukraine":              28,
    "Australia":            29,
    "Turkey":               30,
    "Sweden":               31,
    "Scotland":             34,
    "Tunisia":              35,
    "Iran":                 38,   "IR Iran":          38,
    "Czech Republic":       40,
    # 41–70
    "Saudi Arabia":         44,
    "Egypt":                46,
    "Ghana":                50,
    "Algeria":              51,
    "Paraguay":             54,
    "Cameroon":             56,
    "Ivory Coast":          57,
    "Bosnia and Herzegovina": 58, "Bosnia-Herzegovina": 58,
    "Curaçao":              60,   "Curacao":          60,
    "DR Congo":             62,   "Congo DR":         62,
    "Qatar":                61,
    "South Africa":         65,
    "Cape Verde":           68,
    "Iraq":                 70,
    # 71+
    "New Zealand":          71,
    "Panama":               75,
    "Jordan":               77,
    "Uzbekistan":           75,
    "Bolivia":              80,
    "Haiti":                84,
    "Jamaica":              85,
    "North Korea":          90,   "Korea DPR":        90,
    "Scotland":             34,
}
_DEFAULT_RANK: int = 100


def _opponent_rank(opponent_name: str) -> int:
    """
    Return the FIFA rank for a given opponent name.

    Tries the name as-is, then common project aliases, then defaults to
    _DEFAULT_RANK (100) so unknown nations get the lowest quality weight.
    """
    # Direct lookup (handles both dataset and project canonical forms).
    if opponent_name in _FIFA_RANKINGS:
        return _FIFA_RANKINGS[opponent_name]
    # Try resolving through the fetcher's alias map (maps project → dataset).
    # We need the reverse: dataset → project → ranking.
    # Build a reverse pass: if a TEAM_NAME_MAP value matches, look up the key.
    for proj_name, dataset_name in TEAM_NAME_MAP.items():
        if dataset_name == opponent_name and proj_name in _FIFA_RANKINGS:
            return _FIFA_RANKINGS[proj_name]
    return _DEFAULT_RANK


# ── TypedDicts ────────────────────────────────────────────────────────────────

class TeamAdvancedStats(TypedDict):
    """All advanced proxy features for one team, computed from last N fixtures."""
    quality_adj_xg_for:    float   # quality-weighted goals scored per game
    quality_adj_xg_against: float  # quality-weighted goals conceded per game
    quality_adj_xg_diff:   float   # xg_for − xg_against
    proxy_ppda:            float   # goals_conceded / (clean_sheet_rate + 0.1)
    clinical_efficiency:   float   # actual_goals_for / quality_adj_xg_for
    neutral_form_score:    float   # (away_wr × 1.2 + home_wr × 0.8) / 2
    press_success_rate:    float   # wins vs top-20 / games vs top-20
    games_found:           int
    games_vs_top20:        int


class PredictionOutput(TypedDict):
    team1_win:   float
    draw:        float
    team2_win:   float
    confidence:  float
    key_factors: list[str]


# ── Math helpers ──────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """Logistic sigmoid — positive x favours team1."""
    return 1.0 / (1.0 + math.exp(-SIGMOID_K * x))


def _clamp(v: float) -> float:
    return max(PROB_MIN, min(PROB_MAX, v))


# ── Confidence ────────────────────────────────────────────────────────────────

def _confidence(games: int, games_vs_top20: int) -> float:
    """
    Tiered confidence based on how much data we have and how strong it is.
    Games vs top-20 opponents are the richest signal for WC predictions.
    """
    if games < _MIN_GAMES:
        return _CONF_FALLBACK
    if games < 5:
        return _CONF_FALLBACK
    if games < 10:
        return _CONF_MID
    # 10+ games — quality of schedule decides top vs good tier.
    return _CONF_FULL if games_vs_top20 >= _TOP20_THRESHOLD else _CONF_GOOD


def _combined_confidence(g1: int, t20_1: int, g2: int, t20_2: int) -> float:
    """Geometric mean of both teams' individual confidence scores."""
    c1 = _confidence(g1, t20_1)
    c2 = _confidence(g2, t20_2)
    return round(math.sqrt(c1 * c2), 3)


# ── Ranking proxy fallback ────────────────────────────────────────────────────

def _ranking_composite(team1: str, team2: str) -> float:
    """
    FIFA ranking delta — used when either team has < _MIN_GAMES data points.
    Positive = team1 higher ranked.  Scaled so a 20-place gap ≈ 0.20.
    """
    r1 = _FIFA_RANKINGS.get(team1, _DEFAULT_RANK)
    r2 = _FIFA_RANKINGS.get(team2, _DEFAULT_RANK)
    return (r2 - r1) / 100.0


# ── Probability conversion ────────────────────────────────────────────────────

def _probs(composite: float) -> tuple[float, float, float]:
    """
    Convert a composite score to (team1_win, draw, team2_win) using the
    sigmoid + draw-decay prior shared across all pipeline models.
    """
    raw_win = _sigmoid(composite)
    draw    = BASE_DRAW_RATE * math.exp(-DRAW_DECAY * abs(composite))

    t1_raw = _clamp(raw_win       * (1.0 - draw))
    t2_raw = _clamp((1 - raw_win) * (1.0 - draw))

    total     = t1_raw + draw + t2_raw
    team1_win = round(t1_raw / total, 4)
    team2_win = round(t2_raw / total, 4)
    draw_out  = round(max(0.0, 1.0 - team1_win - team2_win), 4)
    return team1_win, draw_out, team2_win


# ── Feature extraction ────────────────────────────────────────────────────────

def _compute_team_stats(team_name: str, n: int = 10) -> TeamAdvancedStats:
    """
    Compute all advanced proxy features for one team from their last N
    international fixtures before FIXTURE_CUTOFF.

    Uses footballdata_fetcher._load_results() directly (instead of the
    FixtureDict API) so we retain the home/away flag needed for
    neutral_form_score — that field is absent from FixtureDict.

    Args:
        team_name: Project or dataset name, e.g. "France", "USA".
        n:         Maximum number of recent fixtures to aggregate over.

    Returns:
        TeamAdvancedStats with all features set to 0.0 and games_found=0
        when the team has no qualifying fixtures.
    """
    _EMPTY = TeamAdvancedStats(
        quality_adj_xg_for=0.0, quality_adj_xg_against=0.0,
        quality_adj_xg_diff=0.0, proxy_ppda=0.0,
        clinical_efficiency=0.0, neutral_form_score=0.0,
        press_success_rate=0.0, games_found=0, games_vs_top20=0,
    )

    canonical = _resolve(team_name)
    rows      = _load_results()

    # Filter: team participated, match completed, before WC start.
    team_rows = [
        r for r in rows
        if canonical in (r["home_team"], r["away_team"])
        and _is_played(r)
        and r["date"] < FIXTURE_CUTOFF
    ]

    if not team_rows:
        return _EMPTY

    # Take the N most recent fixtures.
    team_rows.sort(key=lambda r: r["date"])
    recent = team_rows[-n:]

    # Build augmented fixture dicts that include the home/away flag and
    # opponent quality weight — both absent from the public FixtureDict API.
    fixtures: list[dict] = []
    for row in recent:
        is_home  = row["home_team"] == canonical
        gf       = int(row["home_score"]) if is_home else int(row["away_score"])
        ga       = int(row["away_score"]) if is_home else int(row["home_score"])
        opponent = row["away_team"]       if is_home else row["home_team"]
        result   = "W" if gf > ga else ("D" if gf == ga else "L")
        opp_rank = _opponent_rank(opponent)

        fixtures.append({
            "is_home":     is_home,
            "opponent":    opponent,
            "goals_for":   gf,
            "goals_against": ga,
            "result":      result,
            "opp_rank":    opp_rank,
            "opp_weight":  _opp_quality_weight(opp_rank),
            "comp_weight": _competition_weight(row["tournament"]),
        })

    if not fixtures:
        return _EMPTY

    total_opp_weight = sum(f["opp_weight"] for f in fixtures)
    n_games          = len(fixtures)

    # ── Quality-adjusted xG proxies ───────────────────────────────────────────
    # Weighted average of goals with opponent quality as weight.
    # Scoring against a top-10 team (weight 1.5) counts more than against a
    # weak team (weight 0.5), so teams with harder schedules are rewarded.
    quality_adj_xg_for     = sum(f["goals_for"]     * f["opp_weight"] for f in fixtures) / total_opp_weight
    quality_adj_xg_against = sum(f["goals_against"] * f["opp_weight"] for f in fixtures) / total_opp_weight
    quality_adj_xg_diff    = quality_adj_xg_for - quality_adj_xg_against

    # ── Proxy PPDA (pressing intensity) ──────────────────────────────────────
    # Lower proxy_ppda = more aggressive pressing.
    # Uses simple (unweighted) averages: pressing is a consistent style trait
    # not inflated/deflated by opponent quality the way scoring rates are.
    goals_conceded_avg = sum(f["goals_against"] for f in fixtures) / n_games
    clean_sheet_rate   = sum(1 for f in fixtures if f["goals_against"] == 0) / n_games
    proxy_ppda         = goals_conceded_avg / (clean_sheet_rate + 0.1)

    # ── Clinical efficiency ───────────────────────────────────────────────────
    # Ratio of actual scoring rate to quality-adjusted expectation.
    # > 1.0 means the team overperforms the quality of opponents faced.
    actual_goals_for_avg = sum(f["goals_for"] for f in fixtures) / n_games
    clinical_efficiency  = actual_goals_for_avg / max(quality_adj_xg_for, 0.01)

    # ── Neutral venue form ────────────────────────────────────────────────────
    # WC is played at neutral venues, so away performance is more predictive.
    # Weight away wins 1.2×, home wins 0.8× to approximate neutral-venue form.
    home_fix = [f for f in fixtures if     f["is_home"]]
    away_fix = [f for f in fixtures if not f["is_home"]]

    home_win_rate = (sum(1 for f in home_fix if f["result"] == "W") / len(home_fix)
                     if home_fix else 0.0)
    away_win_rate = (sum(1 for f in away_fix if f["result"] == "W") / len(away_fix)
                     if away_fix else 0.0)
    neutral_form_score = (away_win_rate * 1.2 + home_win_rate * 0.8) / 2

    # ── Press success vs top-20 ───────────────────────────────────────────────
    # Win rate specifically against top-20 opponents — a proxy for ability to
    # press and break down elite defences under tournament conditions.
    top20_fix       = [f for f in fixtures if f["opp_rank"] <= 20]
    games_vs_top20  = len(top20_fix)
    wins_vs_top20   = sum(1 for f in top20_fix if f["result"] == "W")
    press_success_rate = wins_vs_top20 / max(games_vs_top20, 1)

    return TeamAdvancedStats(
        quality_adj_xg_for=    round(quality_adj_xg_for,    3),
        quality_adj_xg_against=round(quality_adj_xg_against,3),
        quality_adj_xg_diff=   round(quality_adj_xg_diff,   3),
        proxy_ppda=            round(proxy_ppda,             3),
        clinical_efficiency=   round(clinical_efficiency,    3),
        neutral_form_score=    round(neutral_form_score,     3),
        press_success_rate=    round(press_success_rate,     3),
        games_found=           n_games,
        games_vs_top20=        games_vs_top20,
    )


# ── Key factors ───────────────────────────────────────────────────────────────

def _build_key_factors(
    t1: TeamAdvancedStats, t2: TeamAdvancedStats,
    team1: str, team2: str,
) -> list[str]:
    """
    Build up to 4 key factors with real computed numbers, mirroring the
    spec format:
      "France quality-adj xG: 2.1 vs Iraq 0.4 — elite attacking output"
      "Morocco proxy PPDA: 0.31 — elite press intensity vs Brazil 0.48"
      "Japan neutral venue form: 0.71 — stronger away than home"
      "Iraq press success vs top-20: 0% — no wins against elite opposition"
    """
    factors: list[str] = []

    # ── Quality-adjusted xG attacking matchup ─────────────────────────────────
    if t1["quality_adj_xg_for"] > 0 or t2["quality_adj_xg_for"] > 0:
        xg_adv = t1["quality_adj_xg_for"] - t2["quality_adj_xg_against"]
        if abs(xg_adv) >= 0.4:
            better   = team1 if xg_adv > 0 else team2
            strength = "elite attacking output vs top opposition" if abs(xg_adv) >= 1.0 else "xG attacking edge"
            factors.append(
                f"{team1} quality-adj xG: {t1['quality_adj_xg_for']:.2f} "
                f"vs {team2} {t2['quality_adj_xg_against']:.2f} conceded "
                f"— {strength} for {better}"
            )

    # ── Proxy PPDA pressing battle ────────────────────────────────────────────
    if t1["proxy_ppda"] > 0 and t2["proxy_ppda"] > 0:
        ppda_diff = t2["proxy_ppda"] - t1["proxy_ppda"]   # positive = team1 presses harder
        if abs(ppda_diff) >= 0.15:
            presser   = team1 if ppda_diff > 0 else team2
            intensity = "elite press intensity" if min(t1["proxy_ppda"], t2["proxy_ppda"]) < 0.5 else "pressing edge"
            factors.append(
                f"{team1} proxy PPDA: {t1['proxy_ppda']:.2f} "
                f"vs {team2} {t2['proxy_ppda']:.2f} "
                f"— {intensity} for {presser}"
            )

    # ── Neutral venue form ────────────────────────────────────────────────────
    if t1["neutral_form_score"] > 0 or t2["neutral_form_score"] > 0:
        form_diff = t1["neutral_form_score"] - t2["neutral_form_score"]
        if abs(form_diff) >= 0.08:
            stronger = team1 if form_diff > 0 else team2
            factors.append(
                f"{stronger} neutral venue form: {max(t1['neutral_form_score'], t2['neutral_form_score']):.2f} "
                f"— stronger away than home"
            )
        else:
            # Even if tied, show both numbers when scores are notable
            if t1["neutral_form_score"] >= 0.40:
                factors.append(
                    f"{team1} neutral form: {t1['neutral_form_score']:.2f} "
                    f"vs {team2} {t2['neutral_form_score']:.2f} — evenly matched"
                )

    # ── Press success vs top-20 ───────────────────────────────────────────────
    g20_1, g20_2 = t1["games_vs_top20"], t2["games_vs_top20"]
    ps1,   ps2   = t1["press_success_rate"], t2["press_success_rate"]
    if g20_1 > 0 or g20_2 > 0:
        if ps1 == 0 and g20_1 > 0:
            factors.append(
                f"{team1} press success vs top-20: 0% "
                f"({g20_1} game{'s' if g20_1 > 1 else ''}, 0 wins) — no wins against elite opposition"
            )
        elif ps2 == 0 and g20_2 > 0:
            factors.append(
                f"{team2} press success vs top-20: 0% "
                f"({g20_2} game{'s' if g20_2 > 1 else ''}, 0 wins) — no wins against elite opposition"
            )
        elif abs(ps1 - ps2) >= 0.15:
            better = team1 if ps1 > ps2 else team2
            better_rate = max(ps1, ps2)
            factors.append(
                f"{team1} top-20 win rate: {ps1*100:.0f}% ({g20_1}g) "
                f"vs {team2} {ps2*100:.0f}% ({g20_2}g) — {better} wins more vs elite"
            )

    return factors[:4]


def _fallback_key_factors(team1: str, team2: str) -> list[str]:
    """Key factors used when falling back to the FIFA ranking proxy."""
    r1 = _FIFA_RANKINGS.get(team1, _DEFAULT_RANK)
    r2 = _FIFA_RANKINGS.get(team2, _DEFAULT_RANK)
    better = team1 if r1 < r2 else team2
    return [
        f"Insufficient fixture data — using FIFA ranking proxy "
        f"(#{r1} {team1} vs #{r2} {team2})",
        f"Ranking proxy favours {better}",
    ]


# ── Public predict() ──────────────────────────────────────────────────────────

def predict(home_team: str, away_team: str) -> PredictionOutput:
    """
    Advanced tactical prediction using quality-adjusted proxy metrics from the
    martj42 international results dataset — no external API required.

    Computes six features per team (quality-adj xG, proxy PPDA, clinical
    efficiency, neutral venue form, press success rate) and combines them
    into a cross-matchup composite using the shared sigmoid + draw-prior.

    Falls back to the FIFA ranking delta proxy when either team has fewer
    than _MIN_GAMES (3) qualifying fixtures.

    Args:
        home_team: e.g. "France". Resolved via footballdata_fetcher.TEAM_NAME_MAP.
        away_team: e.g. "Iraq".

    Returns:
        PredictionOutput with team1_win, draw, team2_win summing to 1.0,
        plus confidence and key_factors (max 4 entries with real numbers).
    """
    t1 = _compute_team_stats(home_team)
    t2 = _compute_team_stats(away_team)

    g1, g2 = t1["games_found"], t2["games_found"]

    # ── Fallback: too few fixtures for either team ────────────────────────────
    if g1 < _MIN_GAMES or g2 < _MIN_GAMES:
        composite   = _ranking_composite(home_team, away_team)
        confidence  = _CONF_FALLBACK
        t1w, drw, t2w = _probs(composite)
        return PredictionOutput(
            team1_win=t1w, draw=drw, team2_win=t2w,
            confidence=confidence,
            key_factors=_fallback_key_factors(home_team, away_team),
        )

    # ── Cross-matchup composite ───────────────────────────────────────────────

    # How well team1 attacks vs how well team2 defends (higher = team1 advantage).
    xg_attack_vs_defense = t1["quality_adj_xg_for"]     - t2["quality_adj_xg_against"]
    # How well team1 defends vs how well team2 attacks (negated: higher = team1 advantage).
    xg_defense_vs_attack = -(t1["quality_adj_xg_against"] - t2["quality_adj_xg_for"])
    # Pressing battle: positive means team1 presses harder (lower ppda = more press).
    ppda_battle          = t2["proxy_ppda"]              - t1["proxy_ppda"]
    # Neutral venue advantage.
    neutral_form         = t1["neutral_form_score"]      - t2["neutral_form_score"]
    # Win rate vs top opposition.
    press_success        = t1["press_success_rate"]      - t2["press_success_rate"]

    composite = (
        xg_attack_vs_defense * 0.30
        + xg_defense_vs_attack * 0.30
        + ppda_battle          * 0.20
        + neutral_form         * 0.10
        + press_success        * 0.10
    )

    confidence  = _combined_confidence(g1, t1["games_vs_top20"], g2, t2["games_vs_top20"])
    t1w, drw, t2w = _probs(composite)
    key_factors = _build_key_factors(t1, t2, home_team, away_team)

    return PredictionOutput(
        team1_win=t1w,
        draw=drw,
        team2_win=t2w,
        confidence=confidence,
        key_factors=key_factors,
    )
