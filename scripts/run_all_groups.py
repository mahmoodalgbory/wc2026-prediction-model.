#!/usr/bin/env python3
"""
scripts/run_all_groups.py

Run every group-stage matchup for WC 2026 (12 groups × 6 matchups = 72 total),
print a live summary table, and save full ensemble output to
data/group_predictions.json.

Errors are appended to data/errors.log without stopping execution.

Usage:
    python scripts/run_all_groups.py        (from repo root)
    python run_all_groups.py                (from scripts/)
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.ensemble import predict  # noqa: E402  (sys.path must be patched first)

# ── Group definitions ─────────────────────────────────────────────────────────

GROUPS: dict[str, list[str]] = {
    "A": ["Mexico",      "South Africa", "South Korea",  "Czech Republic"],
    "B": ["Canada",      "Bosnia-Herzegovina", "Qatar",  "Switzerland"],
    "C": ["Brazil",      "Morocco",      "Haiti",        "Scotland"],
    "D": ["United States", "Paraguay",   "Australia",    "Turkey"],
    "E": ["Germany",     "Curaçao",      "Ivory Coast",  "Ecuador"],
    "F": ["Netherlands", "Japan",        "Sweden",       "Tunisia"],
    "G": ["Belgium",     "Egypt",        "Iran",         "New Zealand"],
    "H": ["Spain",       "Cape Verde",   "Saudi Arabia", "Uruguay"],
    "I": ["France",      "Senegal",      "Norway",       "Iraq"],
    "J": ["Argentina",   "Algeria",      "Austria",      "Jordan"],
    "K": ["Portugal",    "Congo DR",     "Uzbekistan",   "Colombia"],
    "L": ["England",     "Croatia",      "Ghana",        "Panama"],
}

# Map display names → squad_values.json keys where they differ.
# All 48 group teams have entries in the bundled squad data.
_SQUAD_KEY: dict[str, str] = {
    "United States":      "United States",
    "Bosnia-Herzegovina": "Bosnia-Herzegovina",
}


def _resolve(name: str) -> str:
    """Return the squad_values.json key for a display name."""
    return _SQUAD_KEY.get(name, name)


# ── Error logging ─────────────────────────────────────────────────────────────

_ERRORS_LOG = _REPO_ROOT / "data" / "errors.log"


def _log_error(context: str, exc: Exception) -> None:
    ts  = datetime.now(timezone.utc).isoformat(timespec="seconds")
    msg = f"[{ts}]  {context}  →  {type(exc).__name__}: {exc}"
    print(f"    ERROR: {exc}", file=sys.stderr)
    _ERRORS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(_ERRORS_LOG, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


# ── Formatting helpers ────────────────────────────────────────────────────────

def _short(name: str, width: int = 11) -> str:
    """Abbreviate long names so the summary table stays readable."""
    abbrevs = {
        "South Africa":       "S.Africa",
        "South Korea":        "S.Korea",
        "Czech Republic":     "Czech Rep.",
        "Bosnia-Herzegovina": "Bosnia-Herzegovina",
        "United States":      "United States",
        "New Zealand":        "N.Zealand",
        "Saudi Arabia":       "S.Arabia",
        "Cape Verde":         "C.Verde",
        "Ivory Coast":        "I.Coast",
        "Congo DR":           "Congo DR",
    }
    return abbrevs.get(name, name[:width] if len(name) > width else name)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t_start      = time.monotonic()
    all_results: dict[str, list[dict]] = {}
    error_teams: set[str]              = set()
    total_run    = 0
    total_errors = 0

    for group_id, roster in GROUPS.items():
        print(f"\nGroup {group_id}")
        group_rows: list[dict] = []

        for t1_display, t2_display in itertools.combinations(roster, 2):
            t1 = _resolve(t1_display)
            t2 = _resolve(t2_display)
            context = f"Group {group_id}: {t1_display} vs {t2_display}"

            try:
                result = predict(t1, t2)
                total_run += 1

                t1w = result["team1_win"]
                drw = result["draw"]
                t2w = result["team2_win"]

                matchup_str = f"{t1_display} vs {t2_display}"
                t1s = _short(t1_display)
                t2s = _short(t2_display)
                print(
                    f"  {matchup_str:<40}  "
                    f"{t1s} {t1w * 100:.0f}%  "
                    f"Draw {drw * 100:.0f}%  "
                    f"{t2s} {t2w * 100:.0f}%"
                )

                group_rows.append({
                    "team1":        t1_display,
                    "team2":        t2_display,
                    "team1_win":    t1w,
                    "draw":         drw,
                    "team2_win":    t2w,
                    "confidence":   result["confidence"],
                    "key_factors":  result["key_factors"],
                    "model_weights": result["model_weights"],
                    "model_outputs": result["model_outputs"],
                })

            except Exception as exc:
                total_errors += 1
                error_teams.add(t1_display)
                error_teams.add(t2_display)
                _log_error(context, exc)

        all_results[group_id] = group_rows

    # ── Persist results ───────────────────────────────────────────────────────
    out_path = _REPO_ROOT / "data" / "group_predictions.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, ensure_ascii=False)

    elapsed = time.monotonic() - t_start

    # ── Footer ────────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"Total matchups run:   {total_run}")
    print(f"Errors:               {total_errors}")
    if error_teams:
        print(f"Teams with errors:    {', '.join(sorted(error_teams))}")
    print(f"Time taken:           {elapsed:.1f}s")
    print(f"Output saved →        {out_path}")
    if total_errors:
        print(f"Error log →           {_ERRORS_LOG}")


if __name__ == "__main__":
    main()
