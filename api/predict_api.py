#!/usr/bin/env python3
"""Serve the dashboard and read-only July 5, 2026 prediction API."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import SNAPSHOT_DATE
from data.team_names import canonical_team
from models.ensemble import predict as ensemble_predict, generate_match_insight
from scripts.run_all_groups import GROUPS

app = Flask(__name__)
CORS(app)
VALID_TEAMS = frozenset(team for roster in GROUPS.values() for team in roster)


@app.get('/')
def dashboard():
    return send_from_directory(_ROOT / 'scripts', 'wc2026_dashboard.html')


@app.get('/metadata')
def metadata():
    return jsonify(json.loads((_ROOT / 'data/snapshot_metadata.json').read_text()))


@app.get('/predict')
def predict_match():
    team1 = canonical_team(request.args.get('team1', ''))
    team2 = canonical_team(request.args.get('team2', ''))
    if not team1 or not team2:
        return jsonify(error='team1 and team2 are required'), 400
    if team1 not in VALID_TEAMS or team2 not in VALID_TEAMS:
        return jsonify(error='Choose two supported teams from the dashboard'), 400
    if team1 == team2:
        return jsonify(error='Choose two different teams'), 400
    try:
        result = ensemble_predict(team1, team2)
        insight = generate_match_insight(
            team1, team2, result, result['model_outputs'], result['model_weights']
        )
    except Exception:
        app.logger.exception('Prediction failed for %s vs %s', team1, team2)
        return jsonify(error='Prediction failed; check the local server log'), 500
    per_model = {
        name: {key: output[key] for key in ('team1_win', 'draw', 'team2_win', 'confidence')}
        for name, output in result['model_outputs'].items()
    }
    return jsonify({
        'team1': team1, 'team2': team2, 'snapshot_date': SNAPSHOT_DATE,
        'team1_win': result['team1_win'], 'draw': result['draw'],
        'team2_win': result['team2_win'], 'confidence': result['confidence'],
        'model_weights': result['model_weights'], 'model_outputs': per_model,
        'analytical_note': insight['analytical_note'],
        'style_clash': insight['style_clash'],
        'model_consensus': insight['model_consensus'],
        'models_for_team1': insight['models_for_team1'],
        'models_for_team2': insight['models_for_team2'],
        'models_even': insight['models_even'], 'key_factors': result['key_factors'],
    })


@app.get('/tournament_results')
def get_tournament_results():
    """Return only results supplied before the historical snapshot cutoff."""
    path = _ROOT / 'data/tournament_results.json'
    data = json.loads(path.read_text()) if path.exists() else {'results': []}
    data['results'] = [r for r in data.get('results', []) if r.get('date', '') < SNAPSHOT_DATE]
    return jsonify(data)


@app.post('/log_result')
@app.post('/refresh_predictions')
def frozen_snapshot():
    return jsonify(error='This portfolio demo is a fixed historical snapshot'), 410


if __name__ == '__main__':
    print('Dashboard and Matchup Tool: http://127.0.0.1:5001/')
    app.run(host='127.0.0.1', port=5001, debug=False)
