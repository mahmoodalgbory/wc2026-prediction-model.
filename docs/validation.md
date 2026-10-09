# Verification

Checked on October 9, 2026 using Python 3.12 and the versions in `requirements.txt`.

- All 72 group comparisons completed without errors and contain all six active model outputs.
- The saved R16 output uses 10,000 runs and seed 42; its champion probabilities sum to 1.
- Five automated checks passed: cutoff filtering, bundled-data coverage, dashboard/API response, invalid-input rejection and read-only behavior, and seeded simulation with locked results.
- Repeated seeded simulations produce identical output.
- Python modules compile and the embedded dashboard JavaScript passes Node syntax checking.
- The source archive was scanned for the original embedded credentials; no matches were present.
- The generated README chart was visually inspected. The HTML dashboard has not had a full visual browser review in this environment.

These checks verify implementation and reproducibility. They do not measure forecast accuracy.
