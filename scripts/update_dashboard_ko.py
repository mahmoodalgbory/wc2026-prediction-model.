#!/usr/bin/env python3
"""
update_dashboard_ko.py — Update KO_DATA in dashboard from new simulation_results.json

Usage:
    python scripts/update_dashboard_ko.py
"""
import json, re
from pathlib import Path

REPO = Path(__file__).parent.parent
SIM  = REPO / "data" / "simulation_results.json"
DASH = REPO / "scripts" / "wc2026_dashboard.html"

with open(SIM) as f:
    sim = json.load(f)

ko = sim["knockout_progression"]
# Build new KO_DATA block
lines = ["const KO_DATA = {"]
for team in sorted(ko.keys()):
    d = ko[team]
    r32  = round(d.get("r32",0)*100, 1)
    r16  = round(d.get("r16",0)*100, 1)
    qf   = round(d.get("qf",0)*100, 1)
    sf   = round(d.get("sf",0)*100, 1)
    fi   = round(d.get("final",0)*100, 1)
    ch   = round(d.get("champion",0)*100, 2)
    # Group advancement rate
    grp  = round(sim["group_advancement"].get(team,{}).get("rate",0)*100, 1)
    lines.append(f"  {json.dumps(team)}: {{r32:{r32},r16:{r16},qf:{qf},sf:{sf},fi:{fi},ch:{ch},group:{grp}}},")
lines.append("};")
new_block = "\n".join(lines)

html = DASH.read_text(encoding="utf-8")

# Replace existing KO_DATA block
pattern = r"const KO_DATA\s*=\s*\{[\s\S]*?\};"
if re.search(pattern, html):
    html = re.sub(pattern, lambda _: new_block, html, count=1)
    print(f"✓ Updated KO_DATA ({len(ko)} teams)")
else:
    print("⚠ KO_DATA pattern not found in dashboard — no changes made")

# Patch Most Likely Final
mlf = sim.get("most_likely_final", [])
if len(mlf) == 2:
    t1, t2 = mlf
    html = re.sub(
        r'(<div class="final-teams">)[^<]*(<span class="final-vs">vs</span>)[^<]*(</div>)',
        rf'\g<1>{t1} \g<2> {t2}\g<3>',
        html,
    )
    print(f"✓ Most Likely Final → {t1} vs {t2}")

# Patch simulation type badge (h-tag)
sim_type = sim.get("simulation_type", "")
if "r16" in sim_type:
    badge = "Round of 16"
elif "ko_only" in sim_type:
    badge = "Round of 32"
else:
    badge = "Pre-Tournament"
html = re.sub(r'(<span class="h-tag">)[^<]*(</span>)', rf'\g<1>{badge}\g<2>', html)
print(f"✓ Badge → {badge}")

DASH.write_text(html, encoding="utf-8")
print(f"✓ Dashboard written → {DASH}")

# Print top 10 champion probabilities
print("\nTop 10 champion probabilities:")
sorted_ch = sorted(ko.items(), key=lambda x: x[1].get("champion",0), reverse=True)
for t, d in sorted_ch[:10]:
    print(f"  {t:25} {d.get('champion',0)*100:.2f}%")

print("\nGroup D advancement rates:")
for t in ["United States", "Turkey", "Australia", "Paraguay"]:
    r32 = ko.get(t,{}).get("r32",0)*100
    ch  = ko.get(t,{}).get("champion",0)*100
    print(f"  {t:20} advance={r32:.1f}%  champion={ch:.2f}%")
