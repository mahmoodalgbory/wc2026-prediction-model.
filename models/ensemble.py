"""
Confidence-weighted ensemble for World Cup 2026 match prediction.

Combines six models (value, tactical, form_results, quality_form, momentum, h2h)
into one final prediction. Each model's base weight is scaled by its own
confidence score so data-sparse outputs (e.g. H2H) contribute little while
high-signal outputs (e.g. tactical style model at 0.85) dominate.

Model inventory:
  value         — squad market value positional matchup
  tactical      — style classification + clash matrix (NEW — highest weight)
  form_results  — raw form: goals, clean sheets, win rate  (was: tactical_model)
  quality_form  — quality-adjusted proxy xG + PPDA         (was: advanced_tactical_model)
  momentum      — recent form trend, streaks, big-scalp wins
  h2h           — head-to-head historical record

Weighting pipeline:
    adjusted_weight = base_weight × model_confidence
    final_weight    = adjusted_weight / Σ(all adjusted_weights)

Probabilities are a weighted sum of each model's team1_win / draw / team2_win,
re-normalised to sum exactly to 1.0.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import TypedDict


_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.team_names import canonical_team

from models import (  # noqa: E402
    form_results_model,
    h2h_model,
    momentum_model,
    quality_form_model,
    tactical_model,
    team_value_model,
)

# Base weights before confidence adjustment. Must sum to 1.0.
# tactical (new style model) gets the highest single weight — most novel signal.
# form_results and quality_form each drop to 0.15 (still meaningful but not dominant).
BASE_WEIGHTS: dict[str, float] = {
    "value":        0.20,
    "tactical":     0.25,   # new style + clash matrix model — highest weight
    "form_results": 0.15,   # renamed from tactical_model
    "quality_form": 0.15,   # renamed from advanced_tactical_model
    "momentum":     0.15,
    "h2h":          0.10,
}

# Short tags for labelling key factors with their source model.
_SOURCE_TAG: dict[str, str] = {
    "value":        "VALUE",
    "tactical":     "TACTICAL",
    "form_results": "FORM",
    "quality_form": "QUAL",
    "momentum":     "MOMENTUM",
    "h2h":          "H2H",
}


class EnsembleOutput(TypedDict):
    team1_win:     float
    draw:          float
    team2_win:     float
    confidence:    float          # base-weight-averaged model confidences
    model_weights: dict[str, float]   # final normalized weights used
    model_outputs: dict[str, dict]    # each model's raw output for transparency
    key_factors:   list[str]     # max 5, labelled "[SOURCE] ..."


# ── Model runner ──────────────────────────────────────────────────────────────

def _run_models(home_team: str, away_team: str) -> dict[str, dict]:
    """
    Call all six models and return their raw outputs keyed by model name.

    If a model raises (e.g. team missing from squad_values.json, or
    TheStatsAPI subscription inactive), it is omitted from the result.
    The ensemble re-normalises weights across surviving models so missing
    data degrades gracefully rather than crashing the whole prediction.
    """
    candidates: dict[str, object] = {
        "value":        team_value_model,
        "tactical":     tactical_model,
        "form_results": form_results_model,
        "quality_form": quality_form_model,
        "momentum":     momentum_model,
        "h2h":          h2h_model,
    }
    outputs: dict[str, dict] = {}
    for name, module in candidates.items():
        try:
            outputs[name] = dict(module.predict(home_team, away_team))  # type: ignore[attr-defined]
        except Exception as exc:
            print(f"  [ENSEMBLE] {name} model skipped: {exc}", file=sys.stderr)
    if not outputs:
        raise RuntimeError(
            f"All six models failed for {home_team} vs {away_team}. "
            "Check team names and data files."
        )
    return outputs


# ── Weight computation ────────────────────────────────────────────────────────

def _compute_weights(outputs: dict[str, dict]) -> dict[str, float]:
    """
    Multiply each base weight by the model's confidence score, then normalise
    so all final weights sum to 1.0.

    Only models present in `outputs` are included — if a model was skipped
    (e.g. team missing from squad data), its weight is redistributed
    proportionally across the surviving models.

    A model that reports confidence=0.20 (e.g. sparse H2H) ends up with
    roughly one-quarter of what it would have at confidence=0.80, giving
    data-rich models automatic dominance without manual overrides.
    """
    adjusted = {
        name: BASE_WEIGHTS[name] * outputs[name]["confidence"]
        for name in outputs  # only models that successfully ran
    }
    total = sum(adjusted.values())
    return {name: w / total for name, w in adjusted.items()}


# ── Probability aggregation ───────────────────────────────────────────────────

def _weighted_probs(
    outputs: dict[str, dict],
    weights: dict[str, float],
) -> tuple[float, float, float]:
    """
    Confidence-weighted average of each model's team1_win / draw / team2_win.
    Re-normalised after summing, then draw is derived from the remainder so
    the three outcomes always sum to exactly 1.0 despite 4dp rounding.
    """
    home_raw = sum(outputs[n]["team1_win"] * weights[n] for n in weights)
    draw_raw = sum(outputs[n]["draw"]     * weights[n] for n in weights)
    away_raw = sum(outputs[n]["team2_win"] * weights[n] for n in weights)

    total    = home_raw + draw_raw + away_raw
    team1_win = round(home_raw / total, 4)
    team2_win = round(away_raw / total, 4)
    # Derive draw as the remainder so rounding never causes sum ≠ 1.0.
    draw     = round(max(0.0, 1.0 - team1_win - team2_win), 4)

    return team1_win, draw, team2_win


# ── Ensemble confidence ───────────────────────────────────────────────────────

def _ensemble_confidence(outputs: dict[str, dict]) -> float:
    """
    Base-weight average of surviving model confidence scores, re-normalised
    over the base weights of the models that ran so missing models don't
    artificially depress the reported confidence.
    """
    active_base_total = sum(BASE_WEIGHTS[n] for n in outputs)
    if active_base_total == 0:
        return 0.0
    return round(
        sum(BASE_WEIGHTS[n] * outputs[n]["confidence"] for n in outputs)
        / active_base_total,
        3,
    )


# ── Key factor merging ────────────────────────────────────────────────────────

def _has_specific_stats(factor: str) -> bool:
    """Return True when a factor string references concrete numbers (stats priority)."""
    return any(c.isdigit() for c in factor) or "€" in factor or "%" in factor


def _merge_key_factors(
    outputs: dict[str, dict],
    home_team: str,
    away_team: str,
    team1_win: float,
    team2_win: float,
) -> list[str]:
    """
    Merge key factors from all models into at most 5 entries:
      - Label each factor with its source model: "[TACTICAL] ..."
      - Deduplicate identical (lowercased) factor strings
      - Prioritise factors that mention specific stats or numbers
      - Slot 5 is always the ensemble summary sentence

    Args:
        outputs:   Raw model outputs (each has "key_factors" list).
        home_team: Home team name, used in the summary sentence.
        away_team: Away team name, used in the summary sentence.
        team1_win:  Final ensemble home-win probability.
        team2_win:  Final ensemble away-win probability.
    """
    tagged: list[tuple[str, bool]] = []  # (labelled_factor, has_specific_stats)
    seen: set[str] = set()

    for name, output in outputs.items():
        tag = _SOURCE_TAG[name]
        for factor in output["key_factors"]:
            norm = factor.lower().strip()
            if norm in seen:
                continue
            seen.add(norm)
            labelled = f"[{tag}] {factor}"
            tagged.append((labelled, _has_specific_stats(factor)))

    # Factors with concrete stats first, generic observations second.
    with_stats    = [f for f, has in tagged if has]
    without_stats = [f for f, has in tagged if not has]
    ordered = with_stats + without_stats

    selected = ordered[:4]  # reserve slot 5 for ensemble summary
    selected.append(_ensemble_summary(outputs, home_team, away_team, team1_win, team2_win))
    return selected


def _ensemble_summary(
    outputs: dict[str, dict],
    home_team: str,
    away_team: str,
    team1_win: float,
    team2_win: float,
) -> str:
    """
    One-sentence ensemble agreement/split verdict.

    Agreement:   "Ensemble: 4 models agree France heavily favoured"
    Split:       "Ensemble: models split — value & tactical favour France,
                             momentum favour Iraq"
    """
    home_models = [n for n, o in outputs.items() if o["team1_win"] > o["team2_win"]]
    away_models = [n for n, o in outputs.items() if o["team2_win"] > o["team1_win"]]
    n_total     = len(outputs)
    winner      = home_team if team1_win > team2_win else away_team
    strength    = "heavily" if abs(team1_win - team2_win) > 0.30 else "slightly"

    if len(home_models) == n_total:
        return f"Ensemble: {n_total} models agree {winner} {strength} favoured"
    if len(away_models) == n_total:
        return f"Ensemble: {n_total} models agree {winner} {strength} favoured"

    # Split: describe which models favour each side.
    def _tags(names: list[str]) -> str:
        return " & ".join(_SOURCE_TAG[n].lower() for n in names)

    if home_models and away_models:
        return (
            f"Ensemble: models split — "
            f"{_tags(home_models)} favour {home_team}, "
            f"{_tags(away_models)} favour {away_team}"
        )
    return f"Ensemble: mixed signals — slight edge to {winner}"


# ── Insight helpers ───────────────────────────────────────────────────────────

def generate_style_clash_string(model_outputs: dict) -> str | None:
    """
    Extract the style matchup string from the tactical model's key_factors.

    Looks for a factor of the form:
        "Team1 (style1) vs Team2 (style2) — ..."
    and returns "style1 vs style2", or None when tactical model returned a
    neutral stylistic matchup or did not run.
    """
    tactical_factors = model_outputs.get("tactical", {}).get("key_factors", [])
    for factor in tactical_factors:
        if "neutral stylistic matchup" in factor.lower():
            return None
        # Match: any text followed by (style1) vs any text followed by (style2)
        m = re.search(r'\([^)]+\)\s+vs\s+[^(]+\(([^)]+)\)', factor)
        m2 = re.search(r'\(([^)]+)\)\s+vs\s+[^(]+\(([^)]+)\)', factor)
        if m2:
            style1 = m2.group(1)
            style2 = m2.group(2)
            # Skip if both balanced — no meaningful clash
            if style1 == "balanced" and style2 == "balanced":
                return None
            return f"{style1} vs {style2}"
    return None


def generate_match_insight(
    team1: str,
    team2: str,
    ensemble_output: dict,
    model_outputs: dict,
    final_weights: dict,
) -> dict:
    """
    Generates a structured analytical insight from six model outputs.

    Returns:
    {
      "verdict": str,            # one sentence summary
      "model_consensus": str,    # "strong" / "moderate" / "split"
      "models_for_team1": list,  # model names backing team1
      "models_for_team2": list,  # model names backing team2
      "models_even": list,       # models within 5pp
      "key_driver": str,         # top factor from highest-weight model
      "uncertainty_flag": bool,  # draw > 22% or margin < 8pp
      "analytical_note": str,    # 2-3 sentence programmatic paragraph
      "style_clash": str|None,   # style matchup string if available
      "value_gap_eur_m": float|None,  # team1 total squad value minus team2
    }
    """
    # ── Classify models ───────────────────────────────────────────────────────
    models_even: list[str] = []
    models_for_team1: list[str] = []
    models_for_team2: list[str] = []

    for name, output in model_outputs.items():
        diff = output["team1_win"] - output["team2_win"]
        if abs(diff) <= 0.05:
            models_even.append(name)
        elif diff > 0:
            models_for_team1.append(name)
        else:
            models_for_team2.append(name)

    n1, n2 = len(models_for_team1), len(models_for_team2)

    # ── Verdict sentence ──────────────────────────────────────────────────────
    if n1 >= 5:
        verdict = f"{team1} favoured across {n1} of 6 models"
    elif n2 >= 5:
        verdict = f"{team2} favoured across {n2} of 6 models"
    elif n1 == n2:
        verdict = f"Model split — {n1} models back {team1}, {n2} back {team2}"
    else:
        majority = team1 if n1 > n2 else team2
        minority = team2 if majority == team1 else team1
        verdict = (
            f"{majority} backed by {max(n1, n2)} of 6 models "
            f"with {minority} competitive"
        )

    # ── Consensus strength ────────────────────────────────────────────────────
    dominant = max(n1, n2)
    if dominant >= 5:
        model_consensus = "strong"
    elif abs(n1 - n2) <= 1 or len(models_even) >= 2:
        model_consensus = "split"
    else:
        model_consensus = "moderate"

    # ── Key driver ────────────────────────────────────────────────────────────
    top_model = max(final_weights, key=final_weights.get) if final_weights else None
    key_driver: str | None = None
    if top_model and top_model in model_outputs:
        top_factors = [
            f for f in model_outputs[top_model].get("key_factors", [])
            if not f.lower().startswith("ensemble:")
        ]
        key_driver = top_factors[0] if top_factors else None

    # ── Uncertainty flag ──────────────────────────────────────────────────────
    draw_pct = ensemble_output["draw"] * 100
    margin   = abs(ensemble_output["team1_win"] - ensemble_output["team2_win"]) * 100
    uncertainty_flag = draw_pct > 22 or margin < 8

    # ── Analytical note ───────────────────────────────────────────────────────
    # Sentence 1 — consensus or split
    if n1 >= 5:
        s1 = f"{team1} favoured across {n1} of 6 models."
    elif n2 >= 5:
        s1 = f"{team2} favoured across {n2} of 6 models."
    elif n1 == n2:
        s1 = f"Model split — {n1} models back {team1}, {n2} back {team2}."
    else:
        majority = team1 if n1 > n2 else team2
        minority = team2 if majority == team1 else team1
        s1 = (
            f"{majority} backed by {max(n1, n2)} of 6 models "
            f"with {minority} competitive."
        )

    # Sentence 2 — key driver from highest-weight model
    top_model_tag = top_model.upper() if top_model else "MODEL"
    s2 = f"Primary driver [{top_model_tag}]: {key_driver}." if key_driver else ""

    # Sentence 3 — uncertainty or confidence note
    if draw_pct > 22:
        s3 = (
            f"High draw probability ({draw_pct:.0f}%) — "
            f"evenly matched across most dimensions."
        )
    elif margin < 8:
        s3 = (
            f"Narrow margin ({margin:.0f}pp) — "
            f"outcome sensitive to match-day variables."
        )
    elif margin > 20:
        winner = team1 if ensemble_output["team1_win"] > ensemble_output["team2_win"] else team2
        s3 = (
            f"Clear analytical advantage for {winner} — "
            f"margin consistent across model dimensions."
        )
    else:
        s3 = (
            "Moderate confidence — data supports the favourite "
            "but upset window remains meaningful."
        )

    analytical_note = " ".join(filter(None, [s1, s2, s3]))

    # ── Style clash ───────────────────────────────────────────────────────────
    style_clash = generate_style_clash_string(model_outputs)

    # ── Value gap ─────────────────────────────────────────────────────────────
    value_gap_eur_m: float | None = None
    squad_values_path = _REPO_ROOT / "data" / "squad_values.json"
    if squad_values_path.exists():
        try:
            with open(squad_values_path) as _f:
                _sv = json.load(_f)
            t1_val = _sv.get(team1, {}).get("total_value_eur_millions")
            t2_val = _sv.get(team2, {}).get("total_value_eur_millions")
            if t1_val is not None and t2_val is not None:
                value_gap_eur_m = round(float(t1_val) - float(t2_val), 1)
        except Exception:
            pass

    return {
        "verdict":          verdict,
        "model_consensus":  model_consensus,
        "models_for_team1": models_for_team1,
        "models_for_team2": models_for_team2,
        "models_even":      models_even,
        "key_driver":       key_driver,
        "uncertainty_flag": uncertainty_flag,
        "analytical_note":  analytical_note,
        "style_clash":      style_clash,
        "value_gap_eur_m":  value_gap_eur_m,
    }


# ── Public predict() ──────────────────────────────────────────────────────────

def predict(home_team: str, away_team: str) -> EnsembleOutput:
    """
    Confidence-weighted ensemble prediction from all six pipeline models.

    Base weights: value=0.20, tactical=0.25, form_results=0.15,
    quality_form=0.15, momentum=0.15, h2h=0.10. Each weight is scaled by
    that model's confidence and re-normalised, so a sparse H2H contributes
    far less than a well-confirmed tactical style prediction (confidence=0.85).

    Args:
        home_team: e.g. "France". Must exist in squad_values.json.
        away_team: e.g. "Iraq".

    Returns:
        EnsembleOutput dict with team1_win, draw, team2_win (sum 1.0),
        confidence, model_weights, model_outputs, and key_factors (max 5).
    """
    home_team, away_team = canonical_team(home_team), canonical_team(away_team)
    outputs   = _run_models(home_team, away_team)
    weights   = _compute_weights(outputs)
    team1_win, draw, team2_win = _weighted_probs(outputs, weights)
    confidence = _ensemble_confidence(outputs)

    key_factors = _merge_key_factors(
        outputs, home_team, away_team, team1_win, team2_win
    )

    return EnsembleOutput(
        team1_win=team1_win,
        draw=draw,
        team2_win=team2_win,
        confidence=confidence,
        model_weights=weights,
        model_outputs=outputs,
        key_factors=key_factors,
    )


# ── Test runner ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TEST_CASES = [
        ("France",    "Iraq"),
        ("Brazil",    "Morocco"),
        ("USA",       "England"),
        ("Norway",    "Iraq"),
        ("Algeria",   "Netherlands"),
    ]

    for home, away in TEST_CASES:
        print(f"\n{'=' * 60}")
        print(f"  {home} vs {away}")
        print(f"{'=' * 60}")
        result = predict(home, away)

        print(f"  team1_win={result['team1_win']:.4f}  draw={result['draw']:.4f}  "
              f"team2_win={result['team2_win']:.4f}  "
              f"sum={result['team1_win']+result['draw']+result['team2_win']:.4f}")
        print(f"  confidence={result['confidence']:.3f}")

        print("\n  Model weights & outputs:")
        print(f"  {'model':<10} {'weight':>8} {'conf':>6} {'home':>7} {'draw':>7} {'away':>7}")
        for name, output in result["model_outputs"].items():
            w = result["model_weights"][name]
            print(
                f"  {name:<10} {w:>8.4f} {output['confidence']:>6.3f} "
                f"{output['team1_win']:>7.4f} {output['draw']:>7.4f} {output['team2_win']:>7.4f}"
            )

        print("\n  Key factors:")
        for factor in result["key_factors"]:
            print(f"    • {factor}")
