"""Meaningful checks for historical leakage, API behavior, and simulation state."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from api.predict_api import app
from data.fetchers import footballdata_fetcher as results
from models.ensemble import BASE_WEIGHTS
from scripts import simulate_tournament as sim

ROOT = Path(__file__).resolve().parent.parent


class SnapshotTests(unittest.TestCase):
    def test_loader_rejects_rows_on_and_after_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'results.json'
            path.write_text(json.dumps([
                {'date': '2026-07-04'}, {'date': '2026-07-05'},
                {'date': '2026-07-19'}, {'date': ''},
            ]))
            results._load_results.cache_clear()
            try:
                with patch.object(results, '_RESULTS_CACHE', path):
                    self.assertEqual(results._load_results(), [{'date': '2026-07-04'}])
            finally:
                results._load_results.cache_clear()

    def test_bundled_results_and_groups_have_full_coverage(self):
        rows = results._load_results()
        self.assertTrue(rows)
        self.assertLess(max(r['date'] for r in rows), '2026-07-05')
        groups = json.loads((ROOT / 'data/group_predictions.json').read_text())
        self.assertEqual(len(groups), 12)
        self.assertEqual(sum(map(len, groups.values())), 72)
        for matches in groups.values():
            self.assertEqual(len(matches), 6)
            for match in matches:
                self.assertEqual(set(match['model_outputs']), set(BASE_WEIGHTS))
                self.assertAlmostEqual(sum(match[k] for k in ('team1_win','draw','team2_win')), 1, places=7)

    def test_api_serves_dashboard_and_normalized_predictions(self):
        client = app.test_client()
        with client.get('/') as response:
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'Champion Probabilities', response.data)
        response = client.get('/predict?team1=USA&team2=France')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['team1'], 'United States')
        self.assertEqual(data['snapshot_date'], '2026-07-05')
        self.assertEqual(set(data['model_outputs']), set(BASE_WEIGHTS))
        self.assertAlmostEqual(sum(data[k] for k in ('team1_win','draw','team2_win')), 1, places=7)
        self.assertAlmostEqual(sum(data['model_weights'].values()), 1, places=7)
        for key in ('team1_win','draw','team2_win'):
            self.assertGreaterEqual(data[key], 0)
            self.assertLessEqual(data[key], 1)
        self.assertTrue(data['analytical_note'])

    def test_api_rejects_invalid_teams_and_snapshot_mutation(self):
        client = app.test_client()
        for query in ('', '?team1=France', '?team1=France&team2=France', '?team1=Atlantis&team2=France', '?team1=USA&team2=United%20States'):
            self.assertEqual(client.get('/predict'+query).status_code, 400)
        self.assertEqual(client.post('/log_result', json={}).status_code, 410)
        self.assertEqual(client.post('/refresh_predictions', json={}).status_code, 410)

    def test_seeded_simulation_and_locked_results(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'simulation.json'
            with patch.object(sim, 'RESULTS_PATH', output), contextlib.redirect_stdout(io.StringIO()):
                sim.run_ko_only_r16(str(ROOT / 'data/wc_actual_bracket_r16.json'), runs=300, seed=42)
                first = json.loads(output.read_text())
                sim.run_ko_only_r16(str(ROOT / 'data/wc_actual_bracket_r16.json'), runs=300, seed=42)
                second = json.loads(output.read_text())
            self.assertEqual(first, second)
            progression = first['knockout_progression']
            self.assertEqual(progression['France']['qf'], 1)
            self.assertEqual(progression['Morocco']['qf'], 1)
            self.assertEqual(progression['Canada']['champion'], 0)
            self.assertEqual(progression['Paraguay']['champion'], 0)
            self.assertAlmostEqual(sum(p['champion'] for p in progression.values()), 1, delta=.001)
            for p in progression.values():
                self.assertGreaterEqual(p['qf'],p['sf'])
                self.assertGreaterEqual(p['sf'],p['final'])
                self.assertGreaterEqual(p['final'],p['champion'])


if __name__ == '__main__':
    unittest.main()
