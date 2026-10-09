"""
Tactical style model for World Cup 2026 match prediction.

Identifies each team's playing style from squad value positional ratios
(data/squad_values.json) and recent results (martj42 via footballdata_fetcher),
then scores the style matchup using a clash matrix and validates the
classification against actual form.

Style classification — two-pass hybrid:

  Pass 1 (positional ratios + star concentration from squad_values.json):
    attack_dominant   : atk_ratio >= 0.33  (e.g. Norway — Haaland concentration)
                        OR star_concentration >= 0.38  (e.g. Brazil)
                        OR (goals_scored_avg >= 2.5 AND star_concentration >= 0.33)

    defensive_block   : goals_conceded_avg <= 0.50 AND clean_sheet_rate >= 0.55
                        (results dominate because squad value split is near-uniform)

    midfield_control  : win_rate >= 0.70 AND goals_conceded_avg < 1.0
                        (efficient winning without massive scoring margins)

    counter_attacking : goals_conceded_avg <= 0.80 AND goals_scored_avg <= 2.0
                        AND neutral_form_score >= 0.50

    balanced          : none of the above

  Teams can carry a secondary style (e.g. France = attack_dominant + midfield_control,
  Morocco = defensive_block + counter_attacking) used for key-factor narrative.

Style clash matrix — how styles interact (team1 perspective):
  Positive = team1 benefits, negative = team1 harmed.
  Values auto-negated for reversed matchups.

Validation scores — how well recent results confirm the classified style:
  attack_confirmation  = goals_scored_avg / 1.5   (1.5 g/g → 1.0)
  defense_confirmation = clean_sheet_rate  / 0.40  (40% cs → 1.0)
  mid_confirmation     = (1 − goals_conceded/2) × (win_rate + 0.5)
  counter_confirmation = neutral_form_score
  All capped at [0.50, 1.50] — can boost or reduce by up to 50%.

Final composite:
  validated_advantage = style_advantage × team1_confirmation
                        − style_advantage × team2_confirmation
  composite           = raw_clash + validated_advantage
  (pure style signal — no results component; goals/win-rate handled by form_results + quality_form)

Same sigmoid + draw prior as all other pipeline models:
  SIGMOID_K = 3.0,  BASE_DRAW_RATE = 0.27,  DRAW_DECAY = 0.8

Confidence:
  squad data found + 10+ match results = 0.85
  squad data found + 5–9 results       = 0.70
  squad data missing                   = 0.45
  fewer than 3 results                 = 0.35
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import TypedDict

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.fetchers.footballdata_fetcher import (   # noqa: E402
    FIXTURE_CUTOFF,
    TEAM_NAME_MAP,
    _competition_weight,
    _is_played,
    _load_results,
    _resolve,
    get_team_stats_summary,
)
from models.quality_form_model import _FIFA_RANKINGS, _DEFAULT_RANK  # noqa: E402

# ── Squad values data ─────────────────────────────────────────────────────────

_SQUAD_VALUES_PATH = _REPO_ROOT / "data" / "squad_values.json"
_squad_cache: dict | None = None


def _load_squad_values() -> dict:
    """Load squad_values.json once, then serve from in-memory cache."""
    global _squad_cache
    if _squad_cache is None:
        _squad_cache = json.loads(_SQUAD_VALUES_PATH.read_text(encoding="utf-8"))
    return _squad_cache


def _squad_data(team_name: str) -> dict | None:
    """Return the squad value dict for a team, or None if not found."""
    sq = _load_squad_values()
    # Try direct name first, then common aliases.
    aliases = [team_name, TEAM_NAME_MAP.get(team_name, "")]
    for alias in aliases:
        if alias in sq:
            return sq[alias]
    return None


# ── Tuning constants ──────────────────────────────────────────────────────────

SIGMOID_K:      float = 3.0
BASE_DRAW_RATE: float = 0.27
DRAW_DECAY:     float = 0.8
PROB_MIN:       float = 0.18
PROB_MAX:       float = 0.82

# Quality anchor: log(rank_away / rank_home) × RANKING_WEIGHT added to the
# tactical composite so that extreme quality mismatches (e.g. France #2 vs
# Iraq #70) don't get dragged to near-50% by pure style considerations alone.
RANKING_WEIGHT: float = 0.14

_MIN_GAMES: int = 3

# Confidence tiers.
_CONF_NO_DATA: float = 0.35    # <3 results
_CONF_NO_SQUAD: float = 0.45   # squad data missing
_CONF_MID:     float = 0.70    # 5–9 results
_CONF_FULL:    float = 0.85    # 10+ results with squad data


# ── Style constants ───────────────────────────────────────────────────────────

# Style labels — five mutually exclusive primary styles.
STYLES = ("attack_dominant", "defensive_block", "midfield_control",
          "counter_attacking", "balanced")

# Classification thresholds (positional ratio — from squad_values.json).
# Most teams have near-identical splits (atk≈0.28, mid≈0.32, def≈0.30) so
# only the outliers (Norway 0.37, Argentina 0.36) are caught by positional
# ratio alone.  star_concentration and results data handle the rest.
_ATK_RATIO_THRESHOLD: float = 0.33   # flags Norway / Argentina
_STAR_CONC_THRESHOLD: float = 0.38   # high concentration = attack-focused (Brazil etc.)
_ATK_STAR_COMBO_GSC:  float = 2.50   # gsc threshold when star_conc >= 0.33 (France)
_ATK_STAR_COMBO_SC:   float = 0.33

# Results-based classification thresholds.
_DEF_BLOCK_GCN: float = 0.50   # goals conceded / game
_DEF_BLOCK_CS:  float = 0.55   # clean sheet rate
_MID_WR:        float = 0.70   # win rate
_MID_GCN:       float = 1.00   # max goals conceded / game
_CTR_GCN:       float = 0.80
_CTR_GSC:       float = 2.00
_CTR_NFS:       float = 0.50   # neutral form score threshold

# Style clash matrix — team1_style vs team2_style → team1 composite advantage.
# Unlisted pair → 0.  Mirror handled automatically (auto-negate).
STYLE_CLASH_MATRIX: dict[tuple[str, str], float] = {
    ("attack_dominant",  "defensive_block"):   -0.08,  # defence neutralises pure attack
    ("attack_dominant",  "midfield_control"):  +0.04,  # attack exploits space behind mid
    ("attack_dominant",  "counter_attacking"): +0.06,  # attack dominates possession
    ("attack_dominant",  "balanced"):          +0.05,
    ("midfield_control", "defensive_block"):   +0.07,  # possession suffocates the block
    ("midfield_control", "counter_attacking"): +0.03,  # control limits counter threat
    ("midfield_control", "balanced"):          +0.05,
    ("defensive_block",  "counter_attacking"): +0.04,  # two defensive sides — block wins
    ("defensive_block",  "balanced"):          +0.02,
    ("counter_attacking","balanced"):          +0.03,
}

# Confirmation normalisation denominators.
_CONF_ATK_DENOM:  float = 1.50   # goals/game that maps to confirmation = 1.0
_CONF_DEF_DENOM:  float = 0.40   # clean-sheet rate that maps to confirmation = 1.0
_CONF_MIN: float = 0.50
_CONF_MAX: float = 1.50


# ── TypedDicts ────────────────────────────────────────────────────────────────

class StyleProfile(TypedDict):
    """Classified style and validation for one team."""
    primary_style:     str
    secondary_style:   str | None
    confirmation:      float          # capped at [0.5, 1.5]
    goals_scored_avg:  float
    goals_conceded_avg: float
    win_rate:          float
    clean_sheet_rate:  float
    neutral_form_score: float
    games_found:       int
    has_squad_data:    bool


class PredictionOutput(TypedDict):
    team1_win:   float
    draw:        float
    team2_win:   float
    confidence:  float
    key_factors: list[str]


# ── Math helpers ──────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-SIGMOID_K * x))


def _clamp_prob(v: float) -> float:
    return max(PROB_MIN, min(PROB_MAX, v))


def _clamp_conf(v: float) -> float:
    """Cap a confirmation score to [_CONF_MIN, _CONF_MAX]."""
    return max(_CONF_MIN, min(_CONF_MAX, v))


# ── Neutral form score helper ─────────────────────────────────────────────────

def _neutral_form_score(team_name: str, n: int = 10) -> float:
    """
    Compute (away_win_rate × 1.2 + home_win_rate × 0.8) / 2 from raw rows.

    WC matches are at neutral venues so away performance is a better proxy.
    Returns 0.0 when no fixtures are found.
    """
    canonical = _resolve(team_name)
    rows = _load_results()
    team_rows = [
        r for r in rows
        if canonical in (r["home_team"], r["away_team"])
        and _is_played(r)
        and r["date"] < FIXTURE_CUTOFF
    ]
    if not team_rows:
        return 0.0
    team_rows.sort(key=lambda r: r["date"])
    recent = team_rows[-n:]

    home_wins = sum(
        1 for r in recent
        if r["home_team"] == canonical
        and int(r["home_score"]) > int(r["away_score"])
    )
    away_wins = sum(
        1 for r in recent
        if r["away_team"] == canonical
        and int(r["away_score"]) > int(r["home_score"])
    )
    home_games = sum(1 for r in recent if r["home_team"] == canonical)
    away_games = sum(1 for r in recent if r["away_team"] == canonical)

    hwr = home_wins / max(home_games, 1)
    awr = away_wins / max(away_games, 1)
    return round((awr * 1.2 + hwr * 0.8) / 2, 3)


# ── Style classification ──────────────────────────────────────────────────────

def _classify_style(
    squad: dict | None,
    gsc: float, gcn: float, wr: float, cs: float, nfs: float,
) -> tuple[str, str | None]:
    """
    Two-pass style classifier.

    Pass 1: positional ratios and star_concentration from squad_values.json.
    Pass 2: results-based classification for teams with near-uniform positional
            splits (i.e. nearly everyone except Norway / Argentina).

    Args:
        squad: squad_values.json entry for the team, or None.
        gsc:   goals scored avg (last 10 fixtures).
        gcn:   goals conceded avg.
        wr:    win rate.
        cs:    clean sheet rate.
        nfs:   neutral form score.

    Returns:
        (primary_style, secondary_style | None)
    """
    atk_ratio  = 0.0
    star_conc  = 0.25  # default to mid-range when no squad data
    if squad:
        xi = squad.get("starting_xi_value", 1) or 1
        pv = squad.get("positional_values", {})
        atk_ratio = pv.get("attack", 0) / xi
        star_conc = squad.get("star_concentration", 0.25)

    # ── Primary classification ────────────────────────────────────────────────

    primary: str

    # 1. Positional attack dominance (Haaland / Lautaro concentrations).
    if atk_ratio >= _ATK_RATIO_THRESHOLD:
        primary = "attack_dominant"

    # 2. Star-concentrated attack OR very high scoring.
    elif (star_conc >= _STAR_CONC_THRESHOLD
          or gsc >= 3.0
          or (gsc >= _ATK_STAR_COMBO_GSC and star_conc >= _ATK_STAR_COMBO_SC)):
        primary = "attack_dominant"

    # 3. Elite defensive record (results confirm pure defensive style).
    elif gcn <= _DEF_BLOCK_GCN and cs >= _DEF_BLOCK_CS:
        primary = "defensive_block"

    # 4. Efficient possession-based winning (midfield controls the game).
    elif wr >= _MID_WR and gcn < _MID_GCN:
        primary = "midfield_control"

    # 5. Defensive efficiency + counter threat (good away/neutral form, few conceded).
    elif gcn <= _CTR_GCN and gsc <= _CTR_GSC and nfs >= _CTR_NFS:
        primary = "counter_attacking"

    else:
        primary = "balanced"

    # ── Secondary classification ──────────────────────────────────────────────

    secondary: str | None = None

    if primary == "attack_dominant":
        # High-scoring + efficient → also midfield-control traits.
        if wr >= 0.70 and gcn < 1.5:
            secondary = "midfield_control"

    elif primary == "defensive_block":
        # Defensive teams that score regularly play counter-attack.
        if gsc >= 1.5:
            secondary = "counter_attacking"
        else:
            secondary = "midfield_control"

    elif primary == "midfield_control":
        # Prolific midfield-control teams have attack-dominant secondary.
        if gsc >= 2.3:
            secondary = "attack_dominant"
        elif cs >= 0.45:
            # Efficient and clean — defensive secondary.
            secondary = "defensive_block"

    elif primary == "counter_attacking":
        if wr >= 0.60:
            secondary = "midfield_control"

    return primary, secondary


# ── Validation score ──────────────────────────────────────────────────────────

def _validation_score(
    style: str,
    gsc: float, gcn: float, wr: float, cs: float, nfs: float,
) -> float:
    """
    How well does recent form confirm the team's classified playing style?
    Returns a value in [_CONF_MIN, _CONF_MAX] = [0.50, 1.50].

    1.0 = form exactly matches the style benchmark.
    > 1.0 = form EXCEEDS the benchmark (style well-confirmed).
    < 1.0 = form below the benchmark (style less confirmed).

    Formulas (spec-defined):
      attack_dominant   → goals_scored_avg / 1.5
      defensive_block   → clean_sheet_rate  / 0.40
      midfield_control  → (1 − gcn/2) × (win_rate + 0.5)
      counter_attacking → neutral_form_score
      balanced          → 1.0 (no specific confirmation metric)
    """
    if style == "attack_dominant":
        raw = gsc / _CONF_ATK_DENOM
    elif style == "defensive_block":
        raw = cs / _CONF_DEF_DENOM
    elif style == "midfield_control":
        raw = (1.0 - gcn / 2.0) * (wr + 0.5)
    elif style == "counter_attacking":
        raw = nfs
    else:
        raw = 1.0  # balanced — no directional signal

    return _clamp_conf(raw)


# ── Clash matrix lookup ───────────────────────────────────────────────────────

def _clash_score(style1: str, style2: str) -> float:
    """
    Return the clash matrix advantage for (style1, style2) from team1's perspective.

    Checks the direct entry first; falls back to the mirrored entry (negated).
    Returns 0.0 when the pair is not in the matrix.
    """
    direct  = STYLE_CLASH_MATRIX.get((style1, style2))
    if direct is not None:
        return direct
    mirror  = STYLE_CLASH_MATRIX.get((style2, style1))
    if mirror is not None:
        return -mirror
    return 0.0


# ── Full team profile ─────────────────────────────────────────────────────────

def _build_profile(team_name: str) -> StyleProfile:
    """
    Compute the complete StyleProfile for a team.

    1. Load squad_values.json for positional ratios + star_concentration.
    2. Load form stats from footballdata_fetcher (goals, win_rate, clean_sheets).
    3. Compute neutral_form_score from raw rows (requires home/away flag).
    4. Run two-pass style classifier.
    5. Compute validation (confirmation) score for the classified style.

    Returns a zeroed profile with games_found=0 when no fixture data exists.
    """
    squad = _squad_data(team_name)

    # Results stats from martj42.
    stats = get_team_stats_summary(team_name)
    games = stats["games_played"]

    if games == 0:
        # No data at all — return a neutral profile.
        primary, secondary = ("balanced", None)
        if squad:
            # Still attempt a rough classification from squad data alone.
            xi = squad.get("starting_xi_value", 1) or 1
            pv = squad.get("positional_values", {})
            atk = pv.get("attack", 0) / xi
            sc  = squad.get("star_concentration", 0.25)
            if atk >= _ATK_RATIO_THRESHOLD or sc >= _STAR_CONC_THRESHOLD:
                primary = "attack_dominant"
        return StyleProfile(
            primary_style=primary, secondary_style=secondary,
            confirmation=_CONF_MIN, goals_scored_avg=0.0,
            goals_conceded_avg=0.0, win_rate=0.0, clean_sheet_rate=0.0,
            neutral_form_score=0.0, games_found=0, has_squad_data=squad is not None,
        )

    gsc = stats["goals_scored_avg"]
    gcn = stats["goals_conceded_avg"]
    wr  = stats["win_rate"]
    cs  = stats["clean_sheet_rate"]
    nfs = _neutral_form_score(team_name)

    primary, secondary = _classify_style(squad, gsc, gcn, wr, cs, nfs)
    conf = _validation_score(primary, gsc, gcn, wr, cs, nfs)

    return StyleProfile(
        primary_style=primary, secondary_style=secondary,
        confirmation=conf, goals_scored_avg=gsc,
        goals_conceded_avg=gcn, win_rate=wr, clean_sheet_rate=cs,
        neutral_form_score=nfs, games_found=games,
        has_squad_data=squad is not None,
    )


# ── Confidence ────────────────────────────────────────────────────────────────

def _confidence(p1: StyleProfile, p2: StyleProfile) -> float:
    """
    Combined confidence based on data availability for both teams.
    Lower of the two teams' data richness drives the combined score.
    """
    def _single(p: StyleProfile) -> float:
        if p["games_found"] < _MIN_GAMES:
            return _CONF_NO_DATA
        if not p["has_squad_data"]:
            return _CONF_NO_SQUAD
        if p["games_found"] >= 10:
            return _CONF_FULL
        return _CONF_MID

    return round(math.sqrt(_single(p1) * _single(p2)), 3)


# ── Probabilities ─────────────────────────────────────────────────────────────

def _probs(composite: float) -> tuple[float, float, float]:
    """Convert composite to (team1_win, draw, team2_win) via sigmoid + draw decay."""
    raw  = _sigmoid(composite)
    draw = BASE_DRAW_RATE * math.exp(-DRAW_DECAY * abs(composite))

    t1 = _clamp_prob(raw        * (1.0 - draw))
    t2 = _clamp_prob((1 - raw)  * (1.0 - draw))

    total = t1 + draw + t2
    t1w   = round(t1 / total, 4)
    t2w   = round(t2 / total, 4)
    drw   = round(max(0.0, 1.0 - t1w - t2w), 4)
    return t1w, drw, t2w


# ── Key factors ───────────────────────────────────────────────────────────────

def _build_key_factors(
    p1: StyleProfile, p2: StyleProfile,
    team1: str, team2: str,
    raw_clash: float, validated_adv: float,
) -> list[str]:
    """
    Build up to 4 key factors explaining the style-based analysis.
    All factors include real computed numbers.
    """
    factors: list[str] = []
    s1 = p1["primary_style"]
    s2 = p2["primary_style"]
    sec1 = p1["secondary_style"]
    sec2 = p2["secondary_style"]

    # ── Style clash narrative ─────────────────────────────────────────────────
    if raw_clash != 0.0:
        sec1_str = f"+{sec1}" if sec1 else ""
        sec2_str = f"+{sec2}" if sec2 else ""
        benefit  = team1 if raw_clash > 0 else team2
        penalty  = f"{raw_clash:+.2f}"

        descriptions = {
            ("attack_dominant",  "defensive_block"):
                "defensive systems historically neutralise pure attack",
            ("defensive_block",  "attack_dominant"):
                "elite defence counters attack-dominant opponent",
            ("midfield_control", "counter_attacking"):
                "possession control limits counter-attack threat",
            ("attack_dominant",  "counter_attacking"):
                "attack-dominant side controls possession vs counter",
            ("midfield_control", "defensive_block"):
                "possession suffocates low defensive block",
            ("attack_dominant",  "midfield_control"):
                "attack exploits space left by advanced midfield",
        }
        desc = descriptions.get((s1, s2), descriptions.get((s2, s1), "stylistic edge in matchup"))
        factors.append(
            f"{team1} ({s1.replace('_','-')}{sec1_str}) vs "
            f"{team2} ({s2.replace('_','-')}{sec2_str}) — "
            f"{desc} (style clash: {penalty})"
        )
    else:
        sec1_str = f"+{sec1}" if sec1 else ""
        sec2_str = f"+{sec2}" if sec2 else ""
        factors.append(
            f"{team1} ({s1.replace('_','-')}{sec1_str}) vs "
            f"{team2} ({s2.replace('_','-')}{sec2_str}) — neutral stylistic matchup"
        )

    # ── Style confirmation for team1 ──────────────────────────────────────────
    conf1 = p1["confirmation"]
    gsc1, gcn1, wr1, cs1 = (p1["goals_scored_avg"], p1["goals_conceded_avg"],
                              p1["win_rate"], p1["clean_sheet_rate"])
    if s1 == "attack_dominant":
        factors.append(
            f"{team1} ({s1.replace('_','-')}) confirmed: "
            f"{gsc1:.1f} goals/game in last 10 (benchmark 1.5 → score {conf1:.2f})"
        )
    elif s1 == "defensive_block":
        factors.append(
            f"{team1} ({s1.replace('_','-')}) confirmed: "
            f"{cs1*100:.0f}% clean sheets / {gcn1:.1f} conceded/game (score {conf1:.2f})"
        )
    elif s1 == "midfield_control":
        factors.append(
            f"{team1} (midfield-control) confirmed: "
            f"{wr1*100:.0f}% win rate, {gcn1:.1f} conceded/game (score {conf1:.2f})"
        )
    elif s1 == "counter_attacking":
        nfs1 = p1["neutral_form_score"]
        factors.append(
            f"{team1} (counter-attacking): neutral venue form {nfs1:.2f} "
            f"— away/neutral results confirm counter style (score {conf1:.2f})"
        )

    # ── Style confirmation for team2 ──────────────────────────────────────────
    conf2 = p2["confirmation"]
    gsc2, gcn2, wr2, cs2 = (p2["goals_scored_avg"], p2["goals_conceded_avg"],
                              p2["win_rate"], p2["clean_sheet_rate"])
    if s2 == "attack_dominant":
        factors.append(
            f"{team2} ({s2.replace('_','-')}) confirmed: "
            f"{gsc2:.1f} goals/game (score {conf2:.2f})"
        )
    elif s2 == "defensive_block":
        factors.append(
            f"{team2} (defensive-block) confirmed: "
            f"{cs2*100:.0f}% clean sheets, {gcn2:.1f} conceded/game (score {conf2:.2f})"
        )
    elif s2 == "midfield_control":
        factors.append(
            f"{team2} (midfield-control) confirmed: "
            f"{wr2*100:.0f}% win rate, {gcn2:.1f} conceded/game (score {conf2:.2f})"
        )
    elif s2 == "counter_attacking":
        nfs2 = p2["neutral_form_score"]
        factors.append(
            f"{team2} (counter-attacking): neutral form {nfs2:.2f} "
            f"(score {conf2:.2f})"
        )

    # ── Validated advantage headline ──────────────────────────────────────────
    if abs(validated_adv) >= 0.02:
        beneficiary = team1 if validated_adv > 0 else team2
        factors.append(
            f"Style validation net: {validated_adv:+.3f} → {beneficiary} style "
            f"better confirmed in recent results"
        )

    return factors[:4]


# ── Public predict() ──────────────────────────────────────────────────────────

def predict(home_team: str, away_team: str) -> PredictionOutput:
    """
    Style-based tactical prediction for a WC 2026 matchup.

    1. Classify each team's primary (and secondary) playing style from
       positional ratios in squad_values.json + recent results from martj42.
    2. Score the style matchup using STYLE_CLASH_MATRIX.
    3. Validate the classification against actual form (confirmation scores).
    4. Composite = raw_clash + validated_adv (pure style signal, no results component).

    Args:
        home_team: e.g. "France".
        away_team: e.g. "Iraq".

    Returns:
        PredictionOutput with team1_win + draw + team2_win = 1.0,
        confidence, and up to 4 key factors with real computed numbers.
    """
    p1 = _build_profile(home_team)
    p2 = _build_profile(away_team)

    g1, g2 = p1["games_found"], p2["games_found"]

    # ── Style clash ───────────────────────────────────────────────────────────
    raw_clash = _clash_score(p1["primary_style"], p2["primary_style"])

    # Validated advantage: style advantage scaled by how well each team's
    # results confirm their classified style.
    # term1: team1's style advantage, amplified by team1's confirmation.
    # term2: the "counter-claim" — team2's same-direction style push, amplified
    #        by team2's confirmation.  Net > 0 when team1's style is more confirmed.
    validated_adv = (raw_clash * p1["confirmation"]
                     - raw_clash * p2["confirmation"])

    # Pure style signal + small quality anchor.
    # Raw goals/win-rate are already handled by form_results and quality_form.
    # Adding them here caused double-counting and inflated outlier teams
    # (e.g. Belgium 3.81 gsc from weak qualifiers dominated knockout predictions).
    # This model's unique contribution is ONLY the style clash + validation signal.
    # The ranking anchor is small but prevents extreme mismatches (e.g. France #2
    # vs Iraq #70) from collapsing to near-50% on tactical grounds alone.
    r1 = _FIFA_RANKINGS.get(home_team, _DEFAULT_RANK)
    r2 = _FIFA_RANKINGS.get(away_team, _DEFAULT_RANK)
    ranking_anchor = math.log(r2 / r1) * RANKING_WEIGHT if r1 > 0 and r2 > 0 else 0.0
    composite = raw_clash + validated_adv + ranking_anchor
    confidence = _confidence(p1, p2)
    t1w, drw, t2w = _probs(composite)

    key_factors = _build_key_factors(
        p1, p2, home_team, away_team, raw_clash, validated_adv
    )

    return PredictionOutput(
        team1_win=t1w, draw=drw, team2_win=t2w,
        confidence=confidence, key_factors=key_factors,
    )
