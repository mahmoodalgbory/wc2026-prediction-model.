#!/usr/bin/env python3
"""
Patch wc2026_dashboard.html with fresh simulation data.

Reads:  data/simulation_results.json
Writes: scripts/wc2026_dashboard.html  (patches const KO_DATA in-place)

Usage:
    python scripts/update_dashboard_data.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH   = _REPO_ROOT / "data" / "simulation_results.json"
DASHBOARD_PATH = _REPO_ROOT / "scripts" / "wc2026_dashboard.html"


def _build_ko_data_js(sim: dict) -> str:
    """Return the full `const KO_DATA = {...};` replacement block."""
    ko   = sim["knockout_progression"]
    gadv = sim["group_advancement"]
    runs = sim["runs"]

    lines = ["const KO_DATA = {"]
    for team in sorted(ko.keys()):
        kp = ko[team]
        group_rate = gadv[team]["rate"] if team in gadv else kp["r32"]
        # Multiply by 100 and round to 2 dp
        r32  = round(kp["r32"]      * 100, 2)
        r16  = round(kp["r16"]      * 100, 2)
        qf   = round(kp["qf"]       * 100, 2)
        sf   = round(kp["sf"]       * 100, 2)
        fi   = round(kp["final"]    * 100, 2)
        ch   = round(kp["champion"] * 100, 2)
        grp  = round(group_rate     * 100, 2)
        lines.append(
            f'  "{team}": {{r32:{r32},r16:{r16},qf:{qf},sf:{sf},fi:{fi},ch:{ch},group:{grp}}},'
        )
    lines.append("};")
    return "\n".join(lines)


def patch_dashboard(sim: dict) -> dict:
    html = DASHBOARD_PATH.read_text(encoding="utf-8")

    new_block = _build_ko_data_js(sim)

    pattern = r"const KO_DATA\s*=\s*\{[\s\S]*?\};"
    match   = re.search(pattern, html)
    if not match:
        print("ERROR: could not find 'const KO_DATA' block in dashboard HTML.")
        sys.exit(1)

    old_block = match.group(0)
    patched   = html.replace(old_block, new_block, 1)

    DASHBOARD_PATH.write_text(patched, encoding="utf-8")

    old_lines = old_block.count("\n") + 1
    new_lines = new_block.count("\n") + 1
    return {
        "old_lines": old_lines,
        "new_lines": new_lines,
        "teams":     len(sim["knockout_progression"]),
        "runs":      sim["runs"],
    }


def _print_summary(sim: dict) -> None:
    champ = sim["champion_probabilities"]
    print("\nTop 10 champion odds in updated dashboard:")
    for e in champ[:10]:
        print(f"  {e['team']:<28}  {e['probability']*100:5.2f}%")


def main() -> None:
    if not RESULTS_PATH.exists():
        print(f"ERROR: {RESULTS_PATH} not found — run simulate_tournament.py first.")
        sys.exit(1)

    sim = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    print(f"Loaded simulation_results.json  ({sim['runs']:,} runs, "
          f"type={sim.get('simulation_type','?')})")

    info = patch_dashboard(sim)
    print(f"Patched {DASHBOARD_PATH.name}:")
    print(f"  Teams updated : {info['teams']}")
    print(f"  KO_DATA lines : {info['old_lines']} → {info['new_lines']}")
    print(f"  Simulation runs: {info['runs']:,}")
    print(f"  Written        : {DASHBOARD_PATH}")

    _print_summary(sim)
    print("\nDashboard updated successfully.")


if __name__ == "__main__":
    main()
