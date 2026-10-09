#!/usr/bin/env python3
"""
scripts/export_report.py

Reads data/group_predictions.json and produces:
  • data/report.html  — styled HTML with standings, probabilities, key
                         factors, and model confidence.  Open in any browser;
                         use browser Print → Save as PDF for a PDF copy.
  • data/report.csv   — flat CSV, one row per matchup, easy to sort/filter.

Usage:
    python scripts/export_report.py
    python scripts/export_report.py --open   # open HTML in default browser
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

_REPO_ROOT    = Path(__file__).resolve().parent.parent
_INPUT_JSON   = _REPO_ROOT / "data" / "group_predictions.json"
_OUTPUT_HTML  = _REPO_ROOT / "data" / "report.html"
_OUTPUT_CSV   = _REPO_ROOT / "data" / "report.csv"

# ── Standings simulation ──────────────────────────────────────────────────────

def _compute_standings(rows: list[dict]) -> list[dict]:
    """
    Compute projected standings for one group from its matchup rows.

    Expected points per match:
        team1_exp = 3 × P(win) + 1 × P(draw)
        team2_exp = 3 × P(win) + 1 × P(draw)

    Returns list of dicts sorted by exp_pts desc, with keys:
        team, played, exp_pts, exp_wins, exp_draws, exp_losses
    """
    totals: dict[str, dict] = {}

    def _init(name: str) -> None:
        if name not in totals:
            totals[name] = {
                "team": name, "played": 0,
                "exp_pts": 0.0, "exp_wins": 0.0,
                "exp_draws": 0.0, "exp_losses": 0.0,
            }

    for row in rows:
        t1, t2 = row["team1"], row["team2"]
        w1, dr, w2 = row["team1_win"], row["draw"], row["team2_win"]
        _init(t1); _init(t2)

        totals[t1]["played"]     += 1
        totals[t1]["exp_pts"]    += 3 * w1 + dr
        totals[t1]["exp_wins"]   += w1
        totals[t1]["exp_draws"]  += dr
        totals[t1]["exp_losses"] += w2

        totals[t2]["played"]     += 1
        totals[t2]["exp_pts"]    += 3 * w2 + dr
        totals[t2]["exp_wins"]   += w2
        totals[t2]["exp_draws"]  += dr
        totals[t2]["exp_losses"] += w1

    return sorted(totals.values(), key=lambda x: x["exp_pts"], reverse=True)


# ── HTML helpers ──────────────────────────────────────────────────────────────

def _prob_bar(prob: float, colour: str, width: int = 120) -> str:
    """Return an inline SVG probability bar."""
    px = max(2, round(prob * width))
    return (
        f'<div class="bar-wrap">'
        f'<div class="bar" style="width:{px}px;background:{colour}"></div>'
        f'<span class="bar-label">{prob*100:.1f}%</span>'
        f'</div>'
    )


def _medal(rank: int) -> str:
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "")


def _conf_badge(conf: float) -> str:
    if conf >= 0.75:
        cls, label = "badge-high",   "High"
    elif conf >= 0.60:
        cls, label = "badge-med",    "Med"
    else:
        cls, label = "badge-low",    "Low"
    return f'<span class="badge {cls}">{label} {conf:.2f}</span>'


def _factor_html(factors: list[str]) -> str:
    items = "".join(f"<li>{f}</li>" for f in factors)
    return f'<ul class="factors">{items}</ul>'


def _model_row(name: str, weight: float, output: dict) -> str:
    w1  = output.get("team1_win", 0)
    dr  = output.get("draw",      0)
    w2  = output.get("team2_win", 0)
    cnf = output.get("confidence", 0)
    return (
        f'<tr class="model-row">'
        f'<td class="model-name">{name}</td>'
        f'<td>{weight:.3f}</td>'
        f'<td>{cnf:.2f}</td>'
        f'<td>{w1*100:.1f}%</td>'
        f'<td>{dr*100:.1f}%</td>'
        f'<td>{w2*100:.1f}%</td>'
        f'</tr>'
    )


# ── HTML generation ───────────────────────────────────────────────────────────

_CSS = """
:root {
  --bg:      #0f1117;
  --surface: #1a1d27;
  --border:  #2a2d3a;
  --text:    #e2e8f0;
  --muted:   #94a3b8;
  --accent:  #6366f1;
  --green:   #22c55e;
  --amber:   #f59e0b;
  --red:     #ef4444;
  --t1:      #6366f1;
  --draw:    #64748b;
  --t2:      #f97316;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--bg); color: var(--text); font-size: 14px; line-height: 1.5;
}
a { color: var(--accent); }
.container { max-width: 1100px; margin: 0 auto; padding: 32px 20px; }
h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 4px; }
.subtitle { color: var(--muted); margin-bottom: 40px; font-size: 0.9rem; }

/* Group section */
.group-section { margin-bottom: 56px; }
.group-header {
  font-size: 1.2rem; font-weight: 700; letter-spacing: 0.05em;
  color: var(--accent); border-bottom: 1px solid var(--border);
  padding-bottom: 8px; margin-bottom: 20px;
}
.section-label {
  font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--muted); margin: 20px 0 8px;
}

/* Standings table */
.standings { width: 100%; border-collapse: collapse; margin-bottom: 28px; }
.standings th {
  text-align: left; font-size: 0.72rem; text-transform: uppercase;
  letter-spacing: 0.06em; color: var(--muted); padding: 6px 10px;
  border-bottom: 1px solid var(--border);
}
.standings td { padding: 8px 10px; border-bottom: 1px solid var(--border); }
.standings tr:last-child td { border-bottom: none; }
.standings tr:nth-child(-n+2) .team-name { font-weight: 600; }
.rank-cell { color: var(--muted); width: 28px; }
.team-name { font-weight: 500; }
.pts-cell { font-weight: 700; color: var(--accent); }
.adv-bar { height: 4px; border-radius: 2px; background: var(--accent); margin-top: 4px; }
.qual-badge {
  font-size: 0.65rem; background: rgba(99,102,241,0.2); color: var(--accent);
  border-radius: 3px; padding: 1px 5px; margin-left: 6px; vertical-align: middle;
}

/* Matchup cards */
.matchup-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; margin-bottom: 12px; overflow: hidden;
}
.matchup-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 16px; cursor: pointer; user-select: none;
}
.matchup-header:hover { background: rgba(255,255,255,0.03); }
.teams { font-weight: 600; font-size: 0.95rem; }
.vs { color: var(--muted); font-size: 0.8rem; margin: 0 8px; }
.probs-inline { display: flex; gap: 16px; align-items: center; }
.prob-chip {
  font-size: 0.78rem; font-weight: 600; padding: 2px 8px;
  border-radius: 4px; white-space: nowrap;
}
.chip-t1   { background: rgba(99,102,241,0.18);  color: #a5b4fc; }
.chip-draw { background: rgba(100,116,139,0.18); color: #94a3b8; }
.chip-t2   { background: rgba(249,115,22,0.18);  color: #fdba74; }

/* Bar visualisation */
.bar-section { padding: 12px 16px 6px; border-top: 1px solid var(--border); }
.bar-row { display: flex; align-items: center; margin-bottom: 8px; gap: 10px; }
.bar-team { width: 130px; font-size: 0.82rem; white-space: nowrap; overflow: hidden;
            text-overflow: ellipsis; }
.bar-wrap { display: flex; align-items: center; gap: 6px; }
.bar { height: 10px; border-radius: 3px; min-width: 2px; }
.bar-label { font-size: 0.78rem; color: var(--muted); }

/* Details (factors + model breakdown) */
.matchup-details { padding: 0 16px 14px; }
.factors { list-style: none; font-size: 0.82rem; color: var(--muted);
           border-left: 2px solid var(--border); padding-left: 12px; margin: 8px 0 12px; }
.factors li { margin-bottom: 4px; }
.model-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
.model-table th {
  text-align: left; color: var(--muted); font-size: 0.7rem;
  text-transform: uppercase; letter-spacing: 0.05em;
  padding: 4px 8px; border-bottom: 1px solid var(--border);
}
.model-table td { padding: 5px 8px; border-bottom: 1px solid rgba(255,255,255,0.04); }
.model-name { font-weight: 600; color: var(--text); }
.model-table tr:last-child td { border-bottom: none; }
.details-toggle { font-size: 0.72rem; color: var(--accent); }

/* Badges */
.badge { font-size: 0.68rem; padding: 2px 6px; border-radius: 10px; font-weight: 600; }
.badge-high { background: rgba(34,197,94,0.15);  color: #86efac; }
.badge-med  { background: rgba(245,158,11,0.15); color: #fcd34d; }
.badge-low  { background: rgba(239,68,68,0.15);  color: #fca5a5; }

/* Summary strip at top of each group */
.group-meta { display: flex; gap: 20px; margin-bottom: 16px; flex-wrap: wrap; }
.meta-chip {
  font-size: 0.75rem; color: var(--muted);
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 20px; padding: 3px 10px;
}

/* Print styles */
@media print {
  body { background: #fff; color: #000; }
  .matchup-card { border: 1px solid #ccc; break-inside: avoid; }
  .group-section { break-inside: avoid; }
  .bar { print-color-adjust: exact; -webkit-print-color-adjust: exact; }
  .matchup-details { display: block !important; }
}

/* Collapsible */
details > summary { list-style: none; }
details > summary::-webkit-details-marker { display: none; }
"""


def _matchup_html(row: dict, idx: int, group_id: str) -> str:
    t1, t2 = row["team1"], row["team2"]
    w1, dr, w2 = row["team1_win"], row["draw"], row["team2_win"]
    conf = row["confidence"]
    factors = row.get("key_factors", [])
    weights = row.get("model_weights", {})
    outputs = row.get("model_outputs", {})

    uid = f"m-{group_id}-{idx}"

    # Inline probability chips
    chips = (
        f'<span class="prob-chip chip-t1">{t1} {w1*100:.1f}%</span>'
        f'<span class="prob-chip chip-draw">Draw {dr*100:.1f}%</span>'
        f'<span class="prob-chip chip-t2">{t2} {w2*100:.1f}%</span>'
    )

    # Probability bars
    bars = (
        f'<div class="bar-row">'
        f'<div class="bar-team">{t1}</div>'
        f'{_prob_bar(w1, "#6366f1")}'
        f'</div>'
        f'<div class="bar-row">'
        f'<div class="bar-team">Draw</div>'
        f'{_prob_bar(dr, "#64748b")}'
        f'</div>'
        f'<div class="bar-row">'
        f'<div class="bar-team">{t2}</div>'
        f'{_prob_bar(w2, "#f97316")}'
        f'</div>'
    )

    # Model breakdown table
    model_rows = "".join(
        _model_row(name, weights.get(name, 0), outputs[name])
        for name in ("value", "tactical", "momentum", "h2h")
        if name in outputs
    )
    model_section = (
        f'<div class="section-label">Model breakdown</div>'
        f'<table class="model-table">'
        f'<tr><th>Model</th><th>Weight</th><th>Conf</th>'
        f'<th>{t1[:12]}</th><th>Draw</th><th>{t2[:12]}</th></tr>'
        f'{model_rows}'
        f'</table>'
    )

    return f"""
<div class="matchup-card">
  <details>
    <summary>
      <div class="matchup-header">
        <div class="teams">
          <span>{t1}</span><span class="vs">vs</span><span>{t2}</span>
        </div>
        <div style="display:flex;align-items:center;gap:12px">
          <div class="probs-inline">{chips}</div>
          {_conf_badge(conf)}
          <span class="details-toggle">▸ details</span>
        </div>
      </div>
    </summary>
    <div class="bar-section">{bars}</div>
    <div class="matchup-details">
      <div class="section-label">Key factors</div>
      {_factor_html(factors)}
      {model_section}
    </div>
  </details>
</div>"""


def _standings_html(standing_rows: list[dict], max_pts: float) -> str:
    rows_html = ""
    for rank, s in enumerate(standing_rows, 1):
        bar_w = round((s["exp_pts"] / max_pts) * 100) if max_pts > 0 else 0
        qual  = '<span class="qual-badge">QUAL</span>' if rank <= 2 else ""
        rows_html += f"""
<tr>
  <td class="rank-cell">{_medal(rank)}{rank}</td>
  <td class="team-name">{s["team"]}{qual}</td>
  <td>{s["played"]}</td>
  <td class="pts-cell">{s["exp_pts"]:.2f}
    <div class="adv-bar" style="width:{bar_w}%"></div>
  </td>
  <td>{s["exp_wins"]:.2f}</td>
  <td>{s["exp_draws"]:.2f}</td>
  <td>{s["exp_losses"]:.2f}</td>
</tr>"""
    return f"""
<table class="standings">
  <tr>
    <th>#</th><th>Team</th><th>P</th><th>xPts</th>
    <th>xW</th><th>xD</th><th>xL</th>
  </tr>
  {rows_html}
</table>"""


def generate_html(data: dict[str, list[dict]]) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total = sum(len(v) for v in data.values())

    groups_html = ""
    for group_id, rows in data.items():
        standings = _compute_standings(rows)
        max_pts   = standings[0]["exp_pts"] if standings else 1.0
        matchups  = "".join(_matchup_html(r, i, group_id) for i, r in enumerate(rows))

        avg_conf = sum(r["confidence"] for r in rows) / len(rows) if rows else 0
        groups_html += f"""
<section class="group-section">
  <div class="group-header">Group {group_id}</div>
  <div class="group-meta">
    <span class="meta-chip">{len(rows)} matchups</span>
    <span class="meta-chip">Avg confidence {avg_conf:.2f}</span>
  </div>
  <div class="section-label">Projected standings (expected points)</div>
  {_standings_html(standings, max_pts)}
  <div class="section-label">Matchups — click to expand</div>
  {matchups}
</section>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WC 2026 Group Stage Predictions</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">
  <h1>⚽ WC 2026 — Group Stage Predictions</h1>
  <p class="subtitle">
    Generated {ts} &nbsp;·&nbsp; {total} matchups across 12 groups &nbsp;·&nbsp;
    Ensemble: value 0.35 · tactical 0.25 · momentum 0.25 · h2h 0.15 &nbsp;·&nbsp;
    Confidence flat 0.65 (squad value)
  </p>
  {groups_html}
</div>
</body>
</html>"""


# ── CSV generation ────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "group", "team1", "team2",
    "team1_win", "draw", "team2_win", "confidence",
    "value_w", "tactical_w", "momentum_w", "h2h_w",
    "value_t1", "value_draw", "value_t2", "value_conf",
    "tactical_t1", "tactical_draw", "tactical_t2", "tactical_conf",
    "momentum_t1", "momentum_draw", "momentum_t2", "momentum_conf",
    "h2h_t1", "h2h_draw", "h2h_t2", "h2h_conf",
    "key_factor_1", "key_factor_2", "key_factor_3",
]


def generate_csv(data: dict[str, list[dict]]) -> list[dict]:
    csv_rows = []
    for group_id, rows in data.items():
        for row in rows:
            w  = row.get("model_weights", {})
            mo = row.get("model_outputs", {})
            factors = row.get("key_factors", [])

            def _m(model: str, key: str, fallback: float = 0.0) -> float:
                return mo.get(model, {}).get(key, fallback)

            csv_rows.append({
                "group":       group_id,
                "team1":       row["team1"],
                "team2":       row["team2"],
                "team1_win":   round(row["team1_win"], 4),
                "draw":        round(row["draw"],      4),
                "team2_win":   round(row["team2_win"], 4),
                "confidence":  round(row["confidence"], 3),
                # weights
                "value_w":    round(w.get("value",    0), 4),
                "tactical_w": round(w.get("tactical", 0), 4),
                "momentum_w": round(w.get("momentum", 0), 4),
                "h2h_w":      round(w.get("h2h",      0), 4),
                # per-model probs
                "value_t1":    round(_m("value",    "team1_win"), 4),
                "value_draw":  round(_m("value",    "draw"),      4),
                "value_t2":    round(_m("value",    "team2_win"), 4),
                "value_conf":  round(_m("value",    "confidence"),3),
                "tactical_t1":    round(_m("tactical",    "team1_win"), 4),
                "tactical_draw":  round(_m("tactical",    "draw"),      4),
                "tactical_t2":    round(_m("tactical",    "team2_win"), 4),
                "tactical_conf":  round(_m("tactical",    "confidence"),3),
                "momentum_t1":    round(_m("momentum",    "team1_win"), 4),
                "momentum_draw":  round(_m("momentum",    "draw"),      4),
                "momentum_t2":    round(_m("momentum",    "team2_win"), 4),
                "momentum_conf":  round(_m("momentum",    "confidence"),3),
                "h2h_t1":    round(_m("h2h", "team1_win"), 4),
                "h2h_draw":  round(_m("h2h", "draw"),      4),
                "h2h_t2":    round(_m("h2h", "team2_win"), 4),
                "h2h_conf":  round(_m("h2h", "confidence"),3),
                # key factors (first 3)
                "key_factor_1": factors[0] if len(factors) > 0 else "",
                "key_factor_2": factors[1] if len(factors) > 1 else "",
                "key_factor_3": factors[2] if len(factors) > 2 else "",
            })
    return csv_rows


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export WC 2026 group predictions to HTML + CSV."
    )
    parser.add_argument(
        "--open", action="store_true",
        help="Open the HTML report in the default browser after writing."
    )
    parser.add_argument(
        "--input", default=str(_INPUT_JSON),
        help=f"Path to group_predictions.json (default: {_INPUT_JSON})"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run scripts/run_all_groups.py first.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Reading {input_path} …")
    with open(input_path, encoding="utf-8") as f:
        data: dict[str, list[dict]] = json.load(f)

    total_matchups = sum(len(v) for v in data.values())
    print(f"  {len(data)} groups, {total_matchups} matchups")

    # ── HTML ──────────────────────────────────────────────────────────────────
    print(f"Generating HTML report …")
    html = generate_html(data)
    _OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_HTML.write_text(html, encoding="utf-8")
    size_kb = _OUTPUT_HTML.stat().st_size // 1024
    print(f"  ✓ {_OUTPUT_HTML}  ({size_kb} KB)")
    print(f"    → Open in browser and use Print → Save as PDF for a PDF copy")

    # ── CSV ───────────────────────────────────────────────────────────────────
    print(f"Generating CSV …")
    csv_rows = generate_csv(data)
    with open(_OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  ✓ {_OUTPUT_CSV}  ({len(csv_rows)} rows, {len(_CSV_FIELDS)} columns)")

    # ── Standings summary to terminal ─────────────────────────────────────────
    print()
    print("Projected group winners (by expected points):")
    print(f"  {'Grp':<5} {'Winner':<22} {'xPts':>6}  {'2nd place':<22} {'xPts':>6}")
    print(f"  {'─'*60}")
    for gid, rows in data.items():
        standings = _compute_standings(rows)
        if len(standings) >= 2:
            w, r = standings[0], standings[1]
            print(
                f"  {gid:<5} {w['team']:<22} {w['exp_pts']:>6.2f}  "
                f"{r['team']:<22} {r['exp_pts']:>6.2f}"
            )

    if args.open:
        url = _OUTPUT_HTML.as_uri()
        print(f"\nOpening {url} …")
        webbrowser.open(url)


if __name__ == "__main__":
    main()
