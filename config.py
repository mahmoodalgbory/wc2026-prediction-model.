"""Configuration for the frozen July 5, 2026 portfolio snapshot."""

import os

# Match results on this date or later are excluded (start-of-day snapshot).
SNAPSHOT_DATE = "2026-07-05"
API_FOOTBALL_KEY = os.environ.get("API_FOOTBALL_KEY", "")
STATS_API_KEY = os.environ.get("STATS_API_KEY", "")
API_FOOTBALL_BASE_URL = 'https://v3.football.api-sports.io'
API_FOOTBALL_HOST = 'v3.football.api-sports.io'
WC_LEAGUE_ID = 1
WC_SEASON = 2026
