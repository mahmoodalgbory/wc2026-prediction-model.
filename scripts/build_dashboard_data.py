#!/usr/bin/env python3
"""Embed current group and simulation data in the standalone dashboard."""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.run_all_groups import GROUPS
from scripts.update_dashboard_data import patch_dashboard


def main():
    predictions = json.loads((ROOT / 'data/group_predictions.json').read_text())
    groups = {}
    for group, teams in GROUPS.items():
        groups[group] = {
            'teams': teams,
            'matches': [
                {'t1': m['team1'], 't2': m['team2'],
                 'p1': round(100*m['team1_win'], 2),
                 'pd': round(100*m['draw'], 2),
                 'p2': round(100*m['team2_win'], 2),
                 'conf': round(100*m['confidence'], 2)}
                for m in predictions[group]
            ],
            'insight': 'These six pairwise comparisons use the July 5 snapshot. Expected points are calculated as three times win probability plus draw probability for each matchup. They are retrospective comparisons, not recorded pre-tournament forecasts. Model confidence is an internal weighting score; accuracy and calibration have not been measured.',
        }
    path = ROOT / 'scripts/wc2026_dashboard.html'
    html = path.read_text()
    html, count = re.subn(r'const GROUPS\s*=\s*\{[\s\S]*?\};', lambda _: 'const GROUPS = '+json.dumps(groups,indent=2,ensure_ascii=False)+';', html, count=1)
    if count != 1:
        raise RuntimeError('GROUPS block not found')
    team_groups = {team: group for group, roster in GROUPS.items() for team in roster}
    declaration = 'const TEAM_GROUP = '+json.dumps(team_groups,ensure_ascii=False)+';'
    if 'const TEAM_GROUP =' in html:
        html = re.sub(r'const TEAM_GROUP = [^\n]+;', lambda _: declaration, html, count=1)
    else:
        html = html.replace('const KO_DATA =', declaration+'\nconst KO_DATA =',1)
    sim = json.loads((ROOT / 'data/simulation_results.json').read_text())
    pair = sim['most_likely_final']
    html = re.sub(r'<div class="final-teams">[^\n]*</div>',lambda _: f'<div class="final-teams">{pair[0]} <span class="final-vs">vs</span> {pair[1]}</div>',html,count=1)
    path.write_text(html)
    patch_dashboard(sim)
    print('Updated group comparisons, team labels, final pairing, and knockout probabilities.')

if __name__ == '__main__':
    main()
