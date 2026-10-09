#!/usr/bin/env python3
"""
predict.py — World Cup 2026 match prediction entry point.

Single match:
    python predict.py --home "France" --away "Iraq"

All built-in test cases:
    python predict.py --test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.ensemble import predict

TEST_CASES: list[tuple[str, str]] = [
    ("France",   "Iraq"),
    ("Brazil",   "Morocco"),
    ("USA",      "England"),
    ("Norway",   "Iraq"),
    ("Algeria",  "Netherlands"),
]


# ── Formatting helpers ────────────────────────────────────────────────────────

def _bar(prob: float, width: int = 28) -> str:
    """ASCII probability bar — filled blocks proportional to probability."""
    filled = round(prob * width)
    return "█" * filled + "░" * (width - filled)


def _pct(prob: float) -> str:
    return f"{prob * 100:5.1f}%"


# ── Output printer ────────────────────────────────────────────────────────────

def print_prediction(home: str, away: str) -> None:
    """
    Fetch an ensemble prediction for home vs away and print a formatted report
    that shows final probabilities, each model's contribution, confidence
    breakdown, and the top 5 key factors.
    """
    print(f"\n{'═' * 62}")
    print(f"  {home}  vs  {away}")
    print(f"{'═' * 62}\n")

    result = predict(home, away)

    # ── Final probabilities ───────────────────────────────────────────────────
    print("  FINAL PROBABILITIES")
    print(f"  {home:<20}  {_bar(result['team1_win'])}  {_pct(result['team1_win'])}")
    print(f"  {'Draw':<20}  {_bar(result['draw'])}  {_pct(result['draw'])}")
    print(f"  {away:<20}  {_bar(result['team2_win'])}  {_pct(result['team2_win'])}")

    prob_sum = round(result["team1_win"] + result["draw"] + result["team2_win"], 4)
    sum_ok   = "✓" if prob_sum == 1.0 else f"✗ ({prob_sum})"
    print(f"\n  Sum: {prob_sum:.4f} {sum_ok}    Ensemble confidence: {result['confidence']:.3f}")

    # ── Per-model breakdown ───────────────────────────────────────────────────
    print(f"\n  {'─' * 60}")
    print("  MODEL BREAKDOWN")
    print(f"  {'Model':<10}  {'Weight':>7}  {'Conf':>5}  {'Team1':>7}  {'Draw':>7}  {'Team2':>7}")
    print(f"  {'─' * 55}")
    for name, output in result["model_outputs"].items():
        w = result["model_weights"][name]
        c = output["confidence"]
        h = output["team1_win"]
        d = output["draw"]
        a = output["team2_win"]
        print(f"  {name:<10}  {w:>7.4f}  {c:>5.3f}  {h:>7.4f}  {d:>7.4f}  {a:>7.4f}")

    # ── Key factors ───────────────────────────────────────────────────────────
    print(f"\n  {'─' * 60}")
    print("  KEY FACTORS")
    for i, factor in enumerate(result["key_factors"], 1):
        print(f"  {i}. {factor}")

    print()


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="World Cup 2026 ensemble match predictor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python predict.py --home "France" --away "Iraq"\n'
            "  python predict.py --test"
        ),
    )
    parser.add_argument("--home", help='Home team, e.g. "France"')
    parser.add_argument("--away", help='Away team, e.g. "Iraq"')
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run all five built-in test cases and print full output.",
    )
    args = parser.parse_args()

    if args.test:
        for home, away in TEST_CASES:
            try:
                print_prediction(home, away)
            except Exception as exc:
                print(f"\n  ERROR [{home} vs {away}]: {exc}\n", file=sys.stderr)
        return

    if not args.home or not args.away:
        parser.error("--home and --away are required (or use --test for all cases).")

    try:
        print_prediction(args.home, args.away)
    except KeyError as exc:
        print(f"\n  ERROR: Team not found — {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
