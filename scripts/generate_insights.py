#!/usr/bin/env python3
"""
generate_insights.py — Generate match-level analytical insights for all 72
WC2026 group-stage matches.

Reads:  data/group_predictions.json
Writes: data/match_insights.json
        scripts/wc2026_dashboard.html  (injects MATCH_INSIGHTS inline)
Errors: data/errors.log

Usage:
    cd wc2026-prediction-model
    python scripts/generate_insights.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.ensemble import generate_match_insight  # noqa: E402

PREDICTIONS_PATH = _REPO_ROOT / "data" / "group_predictions.json"
INSIGHTS_PATH    = _REPO_ROOT / "data" / "match_insights.json"
DASHBOARD_PATH   = _REPO_ROOT / "scripts" / "wc2026_dashboard.html"
ERRORS_LOG_PATH  = _REPO_ROOT / "data" / "errors.log"

logging.basicConfig(
    filename=str(ERRORS_LOG_PATH),
    filemode="a",
    level=logging.ERROR,
    format="%(asctime)s  %(levelname)s  %(message)s",
)

VERIFY_KEYS = ["France vs Iraq", "Brazil vs Morocco"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _inject_insights_into_dashboard(insights: dict) -> None:
    """Replace the MATCH_INSIGHTS placeholder in the dashboard HTML."""
    if not DASHBOARD_PATH.exists():
        print(f"  ⚠  Dashboard not found at {DASHBOARD_PATH} — skipping HTML injection")
        return

    html = DASHBOARD_PATH.read_text(encoding="utf-8")
    insights_js = "const MATCH_INSIGHTS = " + json.dumps(insights, indent=2) + ";"

    # Replace existing block (handles both placeholder and previous injection)
    pattern = r"const MATCH_INSIGHTS\s*=\s*\{[\s\S]*?\};"
    if re.search(pattern, html):
        html = re.sub(pattern, lambda _: insights_js, html, count=1)
    else:
        # Fall back: insert before buildNav() call
        html = html.replace(
            "buildNav();\nrenderContent();",
            insights_js + "\n\nbuildNav();\nrenderContent();",
        )

    DASHBOARD_PATH.write_text(html, encoding="utf-8")
    print(f"  ✓  Injected into dashboard → {DASHBOARD_PATH}")


def _print_insight(key: str, insight: dict) -> None:
    """Pretty-print a single insight for verification."""
    bar = "═" * 72
    print(f"\n{bar}")
    print(f"  INSIGHT: {key}")
    print(bar)
    fields = [
        "verdict", "model_consensus", "models_for_team1",
        "models_for_team2", "models_even", "key_driver",
        "uncertainty_flag", "style_clash", "value_gap_eur_m",
    ]
    for field in fields:
        print(f"  {field:<25}: {insight.get(field)}")
    print(f"\n  {'analytical_note':<25}:")
    note = insight.get("analytical_note", "")
    # Wrap at ~70 chars for terminal readability
    words, line = note.split(), ""
    for word in words:
        if len(line) + len(word) + 1 > 70:
            print(f"    {line}")
            line = word
        else:
            line = f"{line} {word}".lstrip()
    if line:
        print(f"    {line}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    with open(PREDICTIONS_PATH, encoding="utf-8") as f:
        group_predictions: dict = json.load(f)

    insights: dict[str, dict] = {}
    total = errors = 0

    for group_key in sorted(group_predictions.keys()):
        matches = group_predictions[group_key]
        print(f"\nGroup {group_key}  ({len(matches)} matches)")

        for match in matches:
            team1 = match["team1"]
            team2 = match["team2"]
            key   = f"{team1} vs {team2}"
            total += 1

            try:
                ensemble_output = {
                    "team1_win":  match["team1_win"],
                    "draw":       match["draw"],
                    "team2_win":  match["team2_win"],
                    "confidence": match.get("confidence", 0.0),
                }
                model_outputs = match.get("model_outputs", {})
                final_weights = match.get("model_weights", {})

                insight = generate_match_insight(
                    team1=team1,
                    team2=team2,
                    ensemble_output=ensemble_output,
                    model_outputs=model_outputs,
                    final_weights=final_weights,
                )
                insight["team1"] = team1
                insight["team2"] = team2
                insights[key]    = insight

                flag  = " ⚠" if insight["uncertainty_flag"] else "  "
                cons  = insight["model_consensus"].upper()
                verd  = insight["verdict"][:55]
                n1    = len(insight["models_for_team1"])
                n2    = len(insight["models_for_team2"])
                print(f"  ✓{flag} [{cons:<8}] {n1}–{n2}  {key:<38}  {verd}")

            except Exception as exc:
                errors += 1
                msg = f"ERROR generating insight for {key}: {exc}"
                print(f"  ✗   {msg}", file=sys.stderr)
                logging.error(msg, exc_info=True)

    # ── Persist insights JSON ─────────────────────────────────────────────────
    with open(INSIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2, ensure_ascii=False)
    print(f"\n✓  Saved {len(insights)} insights → {INSIGHTS_PATH}")

    # ── Inject into dashboard ─────────────────────────────────────────────────
    _inject_insights_into_dashboard(insights)

    # ── Verification printout ─────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("VERIFICATION PRINTOUT")
    for vk in VERIFY_KEYS:
        if vk in insights:
            _print_insight(vk, insights[vk])
        else:
            print(f"\n  ⚠  '{vk}' not found in generated insights")

    print(f"\n{'─' * 72}")
    print(f"Done.  {total} matches processed, {errors} error(s).")
    if errors:
        print(f"       See {ERRORS_LOG_PATH} for details.")


if __name__ == "__main__":
    main()
