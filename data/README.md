# Data provenance

The reproducible default snapshot is the **start of July 5, 2026**. A match is eligible only if its date is strictly earlier than `2026-07-05`. The loader enforces that rule even if a newer row is accidentally added to the bundled file.

| File | Purpose and origin |
| --- | --- |
| `cache/intl_results.json` | Filtered copy of results from [martj42/international_results](https://github.com/martj42/international_results); CC0-1.0. Includes 49,495 dated rows through July 4, 2026. The original source commit was not recorded. |
| `squad_values.json` | Supplied squad and positional market-value estimates, plus summary assumptions. Original collection date and source commit were not recorded. |
| `wc_actual_bracket_r16.json` | Supplied project scenario: eight R16 matches, with two July 4 results locked. The filename is inherited; official fixture accuracy has not been independently verified. |
| `wc_actual_bracket.json` | Legacy supplied R32 project scenario, retained for the optional simulation mode. |
| `group_predictions.json` | Generated pairwise comparisons at the snapshot cutoff; not historical pre-tournament forecasts. |
| `match_insights.json` | Programmatically generated explanations from model outputs. |
| `simulation_results.json` | Seeded R16 simulation output: 10,000 runs, seed 42, progression and championship frequencies. |
| `snapshot_metadata.json` | Cutoff convention, sources, row count, latest date, seed, and run count. |
| `tournament_results.json` | Original local result log; the read-only API excludes entries on or after the cutoff. It is not an independent source of model form features. |

CC0-1.0 applies to the results dataset. It does not establish a license for every other input or for the project code. Raw squad PDFs and market-value downloads are not bundled. The model does not automatically refresh the historical snapshot.
