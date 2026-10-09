#!/usr/bin/env python3
"""
WC 2026 Monte Carlo Simulator — chronological group stage, official FIFA R32 bracket.

Usage:
    python scripts/simulate_tournament.py            # run 10,000 simulations
    python scripts/simulate_tournament.py --sample   # one run + full trace, then exit

Output: data/simulation_results.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.ensemble import predict as _ensemble_predict
from models.form_results_model import extract_tactical_stats

DATA_DIR              = _REPO_ROOT / "data"
GROUP_PREDICTIONS_PATH = DATA_DIR / "group_predictions.json"
RESULTS_PATH          = DATA_DIR / "simulation_results.json"
RUNS                  = 10_000

ARGENTINA_GROUP = "J"   # Group Argentina is placed in

# ── Group stage schedule (chronological matchday order) ───────────────────────
WC2026_MATCHDAYS: dict[int, list[tuple[str, str]]] = {
    1: [
        ("Mexico", "South Africa"), ("South Korea", "Czech Republic"),
        ("Canada", "Bosnia-Herzegovina"), ("United States", "Paraguay"),
        ("Qatar", "Switzerland"), ("Germany", "Curaçao"),
        ("Argentina", "Algeria"), ("Austria", "Jordan"),
        ("Portugal", "Congo DR"), ("England", "Croatia"),
        ("Ghana", "Panama"), ("Uzbekistan", "Colombia"),
        ("Brazil", "Morocco"), ("Curaçao", "Ivory Coast"),
        ("Spain", "Cape Verde"), ("France", "Senegal"),
        ("Iraq", "Norway"), ("Belgium", "Egypt"),
        ("Iran", "New Zealand"), ("Japan", "Sweden"),
        ("Netherlands", "Tunisia"), ("South Africa", "South Korea"),
        ("Ecuador", "Germany"), ("Saudi Arabia", "Uruguay"),
    ],
    2: [
        ("Mexico", "South Korea"), ("South Africa", "Czech Republic"),
        ("Canada", "Qatar"), ("Bosnia-Herzegovina", "Switzerland"),
        ("United States", "Australia"), ("Paraguay", "Turkey"),
        ("Germany", "Ivory Coast"), ("Curaçao", "Ecuador"),
        ("Argentina", "Austria"), ("Jordan", "Algeria"),
        ("Portugal", "Uzbekistan"), ("Congo DR", "Colombia"),
        ("England", "Ghana"), ("Croatia", "Panama"),
        ("Spain", "Saudi Arabia"), ("Cape Verde", "Uruguay"),
        ("France", "Iraq"), ("Norway", "Senegal"),
        ("Belgium", "Iran"), ("Egypt", "New Zealand"),
        ("Japan", "Tunisia"), ("Netherlands", "Sweden"),
        ("Brazil", "Haiti"), ("Morocco", "Scotland"),
    ],
    3: [
        ("Mexico", "Czech Republic"), ("South Africa", "South Korea"),
        ("Canada", "Switzerland"), ("Bosnia-Herzegovina", "Qatar"),
        ("United States", "Turkey"), ("Paraguay", "Australia"),
        ("Germany", "Ecuador"), ("Curaçao", "Ivory Coast"),
        ("Argentina", "Jordan"), ("Algeria", "Austria"),
        ("Portugal", "Colombia"), ("Congo DR", "Uzbekistan"),
        ("England", "Panama"), ("Croatia", "Ghana"),
        ("Spain", "Uruguay"), ("Cape Verde", "Saudi Arabia"),
        ("Norway", "France"), ("Senegal", "Iraq"),
        ("Belgium", "New Zealand"), ("Egypt", "Iran"),
        ("Japan", "Tunisia"), ("Sweden", "Netherlands"),
        ("Brazil", "Scotland"), ("Morocco", "Haiti"),
    ],
}

# ── Official FIFA 2026 R32 bracket (Matches 73–88) ───────────────────────────
# Slot notation: W_X = Group X winner, R_X = runner-up, T_XXX = third-place slot
# NOTE: M76 uses T_LAST (8th advancing third) — corrected from the official
#       source which inadvertently listed R_F twice (R_F is also used in M88).
R32_FIXED: list[tuple[str, str]] = [
    ("R_A",  "R_B"),       # M73 — Half 1
    ("W_E",  "T_ABCDF"),   # M74 — Winner E vs best 3rd from A/B/C/D/F
    ("W_F",  "R_C"),       # M75
    ("W_C",  "T_LAST"),    # M76 — Winner C vs 8th (weakest) advancing third
    ("W_I",  "T_CDFGH"),   # M77 — France's slot
    ("R_E",  "R_I"),       # M78
    ("W_A",  "T_CEFHI"),   # M79
    ("W_L",  "T_EHIJK"),   # M80
    ("W_D",  "T_BEFIJ"),   # M81 — Half 2
    ("W_G",  "T_AEHIJ"),   # M82
    ("R_K",  "R_L"),       # M83
    ("W_H",  "R_J"),       # M84
    ("W_B",  "T_EFGIJ"),   # M85
    ("W_J",  "R_H"),       # M86 — Argentina's slot (winner Group J)
    ("W_K",  "R_G"),       # M87
    ("R_D",  "R_F"),       # M88
]
# Half 1 → SF1: M73–M80  |  Half 2 → SF2: M81–M88
# France (W_I) = M77 → Half 1, Argentina (W_J) = M86 → Half 2 → cannot meet until Final

# Third-place slot eligible groups (team's group must be in this set to fill the slot)
SLOT_ELIGIBLE_GROUPS: dict[str, set[str]] = {
    "T_ABCDF": set("ABCDF"),
    "T_CDFGH": set("CDFGH"),
    "T_CEFHI": set("CEFHI"),
    "T_EHIJK": set("EHIJK"),
    "T_BEFIJ": set("BEFIJ"),
    "T_AEHIJ": set("AEHIJ"),
    "T_EFGIJ": set("EFGIJ"),
}

# Hardcoded bracket progression (match winners flow into fixed next-round matches)
R16_BRACKET: list[tuple[str, str]] = [
    ("W_M73", "W_M74"),   # M89 — Half 1
    ("W_M75", "W_M76"),   # M90
    ("W_M77", "W_M78"),   # M91
    ("W_M79", "W_M80"),   # M92
    ("W_M81", "W_M82"),   # M93 — Half 2
    ("W_M83", "W_M84"),   # M94
    ("W_M85", "W_M86"),   # M95
    ("W_M87", "W_M88"),   # M96
]
QF_BRACKET: list[tuple[str, str]] = [
    ("W_M89", "W_M90"),   # M97 — Half 1
    ("W_M91", "W_M92"),   # M98
    ("W_M93", "W_M94"),   # M99 — Half 2
    ("W_M95", "W_M96"),   # M100
]
SF_BRACKET: list[tuple[str, str]] = [
    ("W_M97",  "W_M98"),  # M101 — SF1
    ("W_M99",  "W_M100"), # M102 — SF2
]
FINAL: tuple[str, str] = ("W_M101", "W_M102")  # M104


def _mnum(slot: str) -> int:
    """Parse 'W_M73' → 73."""
    return int(slot[3:])


# ── Third-place slot assignment ───────────────────────────────────────────────

def assign_third_place_slots(
    advancing_thirds: list[dict],
) -> dict[str, str]:
    """
    Most-constrained-slot-first greedy assignment.
    At each step, pick the slot with the fewest eligible remaining teams,
    then assign the highest-ranked eligible team to it.
    advancing_thirds: list[{team, group, pts, gd, gf}] sorted best → worst.
    """
    unassigned = list(advancing_thirds)
    assignments: dict[str, str] = {}
    remaining_slots = list(SLOT_ELIGIBLE_GROUPS.keys())

    while remaining_slots:
        def count_eligible(slot: str) -> int:
            elig = SLOT_ELIGIBLE_GROUPS[slot]
            return sum(1 for t in unassigned if t["group"] in elig)

        remaining_slots.sort(key=count_eligible)  # stable: ties keep insertion order
        slot = remaining_slots[0]
        eligible = SLOT_ELIGIBLE_GROUPS[slot]

        filled = False
        for i, t in enumerate(unassigned):
            if t["group"] in eligible:
                assignments[slot] = t["team"]
                unassigned.pop(i)
                remaining_slots.pop(0)
                filled = True
                break

        if not filled:
            # Fallback: force-assign best remaining (unexpected edge case)
            assignments[slot] = unassigned[0]["team"]
            unassigned.pop(0)
            remaining_slots.pop(0)

    if unassigned:
        assignments["T_LAST"] = unassigned[0]["team"]
    return assignments


# ── Poisson goal model ────────────────────────────────────────────────────────
_TEAM_STATS: dict[str, dict] = {}


def _load_team_stats(teams: list[str]) -> None:
    for team in teams:
        try:
            _TEAM_STATS[team] = extract_tactical_stats(team)
        except Exception:
            _TEAM_STATS[team] = {"avg_goals_scored": 1.2, "avg_goals_conceded": 1.2}


def _poisson_goals(team1: str, team2: str) -> tuple[int, int]:
    s1 = _TEAM_STATS.get(team1, {"avg_goals_scored": 1.2, "avg_goals_conceded": 1.2})
    s2 = _TEAM_STATS.get(team2, {"avg_goals_scored": 1.2, "avg_goals_conceded": 1.2})
    lam1 = max(0.2, (s1["avg_goals_scored"] + s2["avg_goals_conceded"]) / 2)
    lam2 = max(0.2, (s2["avg_goals_scored"] + s1["avg_goals_conceded"]) / 2)
    return int(np.random.poisson(lam1)), int(np.random.poisson(lam2))


# ── Ensemble predict (cached) ─────────────────────────────────────────────────
_predict_cache: dict[tuple[str, str], dict] = {}


def _cached_predict(team1: str, team2: str) -> dict:
    canonical = (min(team1, team2), max(team1, team2))
    if canonical not in _predict_cache:
        _predict_cache[canonical] = _ensemble_predict(canonical[0], canonical[1])
    cached = _predict_cache[canonical]
    if canonical[0] == team1:
        return cached
    return {
        "team1_win": cached["team2_win"],
        "draw":      cached["draw"],
        "team2_win": cached["team1_win"],
    }


# ── Group stage ───────────────────────────────────────────────────────────────

def _sim_match(p1: float, pd: float, team1: str, team2: str) -> tuple[str, str, int, int]:
    r = random.random()
    if r < p1:
        outcome = "w1"
    elif r < p1 + pd:
        outcome = "draw"
    else:
        outcome = "w2"
    g1, g2 = _poisson_goals(team1, team2)
    if outcome == "w1":
        if g1 <= g2: g1 = g2 + 1
    elif outcome == "draw":
        g2 = g1
    else:
        if g2 <= g1: g2 = g1 + 1
    return team1, team2, g1, g2


def _compute_standings(
    teams: set[str],
    pts: dict[str, int],
    gd:  dict[str, int],
    gf:  dict[str, int],
    results: list[tuple[str, str, int, int]],
) -> list[dict]:
    """Sort: pts → GD → GF → H2H pts → H2H GD → team name."""
    def primary(t: str) -> tuple:
        return (-pts[t], -gd[t], -gf[t])

    def h2h_among(subset: list[str]) -> dict[str, tuple[int, int]]:
        sub = set(subset)
        hp  = {t: 0 for t in subset}
        hgd = {t: 0 for t in subset}
        for (t1, t2, g1, g2) in results:
            if t1 not in sub or t2 not in sub:
                continue
            if g1 > g2:
                hp[t1] += 3
            elif g1 == g2:
                hp[t1] += 1; hp[t2] += 1
            else:
                hp[t2] += 3
            hgd[t1] += g1 - g2; hgd[t2] += g2 - g1
        return {t: (hp[t], hgd[t]) for t in subset}

    base  = sorted(teams, key=primary)
    final: list[str] = []
    i = 0
    while i < len(base):
        j = i + 1
        while j < len(base) and primary(base[j]) == primary(base[i]):
            j += 1
        tied = base[i:j]
        if len(tied) > 1:
            h2h  = h2h_among(tied)
            tied = sorted(tied, key=lambda t: (-h2h[t][0], -h2h[t][1], -gf[t], t))
        final.extend(tied)
        i = j

    return [{"team": t, "pts": pts[t], "gd": gd[t], "gf": gf[t], "rank": r + 1}
            for r, t in enumerate(final)]


def _sim_group(matches_data: list[dict]) -> tuple[list[dict], list[tuple]]:
    """Simulate all 6 matches in chronological WC2026_MATCHDAYS order.
    MD3 pairs within each group are simulated from static probabilities,
    so neither result can influence the other.
    """
    lookup: dict[frozenset, dict] = {
        frozenset([m["team1"], m["team2"]]): m
        for m in matches_data
    }
    teams: set[str] = set()
    for m in matches_data:
        teams.add(m["team1"]); teams.add(m["team2"])

    pts = {t: 0 for t in teams}
    gd  = {t: 0 for t in teams}
    gf  = {t: 0 for t in teams}
    raw: list[tuple[str, str, int, int]] = []
    done: set[frozenset] = set()

    def _play(t1: str, t2: str, m: dict) -> None:
        p1 = m["team1_win"] if m["team1"] == t1 else m["team2_win"]
        pd = m["draw"]
        _, _, g1, g2 = _sim_match(p1, pd, t1, t2)
        raw.append((t1, t2, g1, g2))
        gf[t1] += g1; gf[t2] += g2
        gd[t1] += g1 - g2; gd[t2] += g2 - g1
        if g1 > g2:    pts[t1] += 3
        elif g1 == g2: pts[t1] += 1; pts[t2] += 1
        else:           pts[t2] += 3

    # MD1 → MD2 → MD3 (deduplicating cross-group scheduling overlaps)
    for md in (1, 2, 3):
        for (t1, t2) in WC2026_MATCHDAYS[md]:
            key = frozenset([t1, t2])
            if key not in lookup or key in done:
                continue
            done.add(key); _play(t1, t2, lookup[key])

    # Fallback for any group match not listed in WC2026_MATCHDAYS
    for key, m in lookup.items():
        if key in done:
            continue
        done.add(key); _play(m["team1"], m["team2"], m)

    return _compute_standings(teams, pts, gd, gf, raw), raw


# ── Knockout ──────────────────────────────────────────────────────────────────

def _knockout_match(team1: str, team2: str) -> str:
    """Simulate a single-elimination match. Draws go to ET then penalties."""
    pred = _cached_predict(team1, team2)
    p1   = pred["team1_win"]
    pd   = pred["draw"]
    p2   = pred["team2_win"]

    r = random.random()
    if r < p1:
        return team1
    elif r < p1 + pd:
        # Extra time: draw probability reduced by 60%, remainder split equally
        et_draw = pd * 0.4
        et_t1   = p1 + (pd - et_draw) * 0.5
        et_t2   = p2 + (pd - et_draw) * 0.5
        r2 = random.random()
        if r2 < et_t1:
            return team1
        elif r2 < et_t1 + et_draw:
            return team1 if random.random() < 0.5 else team2  # penalties (50/50)
        else:
            return team2
    else:
        return team2


def _play_bracket(
    slot: dict[str, str],
    r32_matchups: list[tuple[str, str]],
    trace: bool = False,
) -> dict:
    """
    Run full knockout bracket using hardcoded progression.
    Returns match_results dict (match_num → winner) and participation sets.
    If trace=True prints each result.
    """
    results: dict[int, str] = {}   # match_num → winner
    in_match: dict[int, tuple[str, str]] = {}  # match_num → (t1, t2)

    # ── R32 (M73–M88) ────────────────────────────────────────────────────────
    for i, (t1, t2) in enumerate(r32_matchups):
        m = 73 + i
        in_match[m] = (t1, t2)
        w = _knockout_match(t1, t2)
        results[m] = w
        if trace:
            print(f"    M{m}: {t1} vs {t2}  →  {w}")

    def _get(slot_name: str) -> str:
        return results[_mnum(slot_name)]

    # ── R16 (M89–M96) ────────────────────────────────────────────────────────
    if trace: print()
    for i, (s1, s2) in enumerate(R16_BRACKET):
        m = 89 + i
        t1, t2 = _get(s1), _get(s2)
        in_match[m] = (t1, t2)
        w = _knockout_match(t1, t2)
        results[m] = w
        if trace:
            print(f"    M{m}: {t1} vs {t2}  →  {w}")

    # ── QF (M97–M100) ────────────────────────────────────────────────────────
    if trace: print()
    for i, (s1, s2) in enumerate(QF_BRACKET):
        m = 97 + i
        t1, t2 = _get(s1), _get(s2)
        in_match[m] = (t1, t2)
        w = _knockout_match(t1, t2)
        results[m] = w
        if trace:
            print(f"    M{m}: {t1} vs {t2}  →  {w}")

    # ── SF (M101–M102) ───────────────────────────────────────────────────────
    if trace: print()
    for i, (s1, s2) in enumerate(SF_BRACKET):
        m = 101 + i
        t1, t2 = _get(s1), _get(s2)
        in_match[m] = (t1, t2)
        w = _knockout_match(t1, t2)
        results[m] = w
        if trace:
            print(f"    M{m} SF{i+1}: {t1} vs {t2}  →  {w}")

    # ── Final (M104) ─────────────────────────────────────────────────────────
    t1, t2 = _get(FINAL[0]), _get(FINAL[1])
    in_match[104] = (t1, t2)
    champion = _knockout_match(t1, t2)
    results[104] = champion
    if trace:
        print(f"\n    M104 FINAL: {t1} vs {t2}  →  CHAMPION: {champion}")

    return {"results": results, "in_match": in_match}


# ── Single full run ───────────────────────────────────────────────────────────

def _run_once(group_data: dict, trace: bool = False) -> dict:
    group_keys = sorted(group_data.keys())   # A … L

    winners: dict[str, str] = {}
    runners: dict[str, str] = {}
    thirds:  list[dict]     = []

    if trace:
        print("\nGROUP STAGE")
        print("─" * 70)

    for gk in group_keys:
        standings, _ = _sim_group(group_data[gk])
        winners[gk]  = standings[0]["team"]
        runners[gk]  = standings[1]["team"]
        thirds.append({
            "team":  standings[2]["team"],
            "group": gk,
            "pts":   standings[2]["pts"],
            "gd":    standings[2]["gd"],
            "gf":    standings[2]["gf"],
        })
        if trace:
            row = "  |  ".join(
                f"{i+1}. {s['team']} {s['pts']}pts {s['gd']:+d}GD"
                for i, s in enumerate(standings)
            )
            print(f"  Group {gk}: {row}")

    # Best 8 of 12 third-place teams
    best_thirds = sorted(
        thirds, key=lambda x: (-x["pts"], -x["gd"], -x["gf"], x["team"])
    )[:8]

    if trace:
        print(f"\n3RD PLACE TEAMS ADVANCING (best 8 of 12):")
        for i, t in enumerate(best_thirds, 1):
            print(f"  {i}. {t['team']:<24} Group {t['group']}  "
                  f"{t['pts']}pts  {t['gd']:+d}GD")

    # Third-place slot assignments
    slot_assignments = assign_third_place_slots(best_thirds)

    if trace:
        print(f"\nSLOT ASSIGNMENTS:")
        for slot_name, team in slot_assignments.items():
            grp = next(t["group"] for t in best_thirds if t["team"] == team)
            print(f"  {slot_name:<12} →  {team}  (Group {grp})")

    # Build slot → team map
    slot: dict[str, str] = {}
    for gk in group_keys:
        slot[f"W_{gk}"] = winners[gk]
        slot[f"R_{gk}"] = runners[gk]
    slot.update(slot_assignments)

    # Build R32 matchups (deterministic)
    r32_matchups = [(slot[s1], slot[s2]) for (s1, s2) in R32_FIXED]

    if trace:
        print(f"\nR32 BRACKET:")
        half_label = {73: "── Half 1 ─────", 81: "── Half 2 ─────"}
        for i, ((s1, s2), (t1, t2)) in enumerate(zip(R32_FIXED, r32_matchups)):
            m = 73 + i
            if m in half_label:
                print(f"  {half_label[m]}")
            print(f"  M{m}  [{s1:<10} {t1:<26}] vs [{t2:<26} {s2}]")

        print(f"\nKNOCKOUT RESULTS:")

    # Run bracket
    bracket = _play_bracket(slot, r32_matchups, trace=trace)
    res     = bracket["results"]
    in_m    = bracket["in_match"]

    champion  = res[104]
    finalists = list(in_m[104])
    sf1_pair  = in_m[101]
    sf2_pair  = in_m[102]

    # Build participation sets for each stage
    r32_all  = set(t for pair in r32_matchups for t in pair)
    r16_all  = set(t for m in range(89, 97) for t in in_m[m])
    qf_all   = set(t for m in range(97, 101) for t in in_m[m])
    sf_all   = set(t for m in (101, 102) for t in in_m[m])
    final_all = set(finalists)

    # Advancement tracking
    adv_detail: dict[str, str] = {}
    for gk in group_keys:
        adv_detail[winners[gk]] = "winner"
        adv_detail[runners[gk]] = "runner_up"
    for t in best_thirds:
        adv_detail[t["team"]] = "best_third"

    # Argentina-specific: R32 opponent
    arg_r32_opp = None
    if "Argentina" in r32_all:
        for t1, t2 in r32_matchups:
            if t1 == "Argentina":
                arg_r32_opp = t2; break
            if t2 == "Argentina":
                arg_r32_opp = t1; break

    return {
        "adv_detail":   adv_detail,
        "r32":          r32_all,
        "r16":          r16_all,
        "qf":           qf_all,
        "sf":           sf_all,
        "final":        final_all,
        "champion":     {champion},
        "final_pair":   tuple(sorted(finalists)),
        "sf1_pair":     tuple(sorted(sf1_pair)),
        "sf2_pair":     tuple(sorted(sf2_pair)),
        "r32_matchups": r32_matchups,
        "arg_r32_opp":  arg_r32_opp,
        "winners":      winners,
        "runners":      runners,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def _print_summary_header(runs: int) -> None:
    print(f"\n{'═'*60}")
    print(f"  TOP 15 CHAMPION PROBABILITIES  ({runs:,} simulations)")
    print(f"{'═'*60}")


def run(runs: int = RUNS, sample_only: bool = False) -> None:
    group_data: dict = json.loads(GROUP_PREDICTIONS_PATH.read_text())

    all_teams: set[str] = set()
    for matches in group_data.values():
        for m in matches:
            all_teams.add(m["team1"]); all_teams.add(m["team2"])
    teams_sorted = sorted(all_teams)

    print(f"Loading tactical stats for {len(teams_sorted)} teams …")
    _load_team_stats(teams_sorted)

    # ── PART 2: sample trace ──────────────────────────────────────────────────
    print("\n" + "═"*70)
    print("SAMPLE RUN — full trace of one tournament simulation")
    print("═"*70)
    _run_once(group_data, trace=True)

    if sample_only:
        print("\n" + "─"*70)
        print("Sample run complete. Confirm the bracket looks correct,")
        print("then run without --sample to start 10,000 simulations.")
        print("─"*70)
        return

    # ── PART 3: 10,000 simulations ────────────────────────────────────────────
    print(f"\nStarting {runs:,} simulations …\n")

    adv_counts: dict[str, dict[str, int]] = {
        t: {"advanced": 0, "as_winner": 0, "as_runner_up": 0, "as_third": 0}
        for t in teams_sorted
    }
    ko_counts: dict[str, dict[str, int]] = {
        t: {"r32": 0, "r16": 0, "qf": 0, "sf": 0, "final": 0, "champion": 0}
        for t in teams_sorted
    }
    final_pairs: Counter  = Counter()
    sf1_counter: Counter  = Counter()
    sf2_counter: Counter  = Counter()

    # Argentina-specific
    arg_won_group: int    = 0
    arg_runner_up: int    = 0
    arg_opp_ctr: Counter  = Counter()

    t0 = time.time()
    for run_i in range(1, runs + 1):
        result = _run_once(group_data)

        for team, how in result["adv_detail"].items():
            adv_counts[team]["advanced"] += 1
            if how == "winner":       adv_counts[team]["as_winner"] += 1
            elif how == "runner_up":  adv_counts[team]["as_runner_up"] += 1
            else:                     adv_counts[team]["as_third"] += 1

        for stage, key in [("r32","r32"),("r16","r16"),("qf","qf"),
                           ("sf","sf"),("final","final"),("champion","champion")]:
            for team in result[stage]:
                ko_counts[team][key] += 1

        final_pairs[result["final_pair"]] += 1
        sf1_counter[result["sf1_pair"]]   += 1
        sf2_counter[result["sf2_pair"]]   += 1

        if result["winners"].get(ARGENTINA_GROUP) == "Argentina":
            arg_won_group += 1
        elif result["runners"].get(ARGENTINA_GROUP) == "Argentina":
            arg_runner_up += 1
        if result["arg_r32_opp"]:
            arg_opp_ctr[result["arg_r32_opp"]] += 1

        if run_i % 1000 == 0:
            elapsed = time.time() - t0
            eta     = (elapsed / run_i) * (runs - run_i)
            print(f"  {run_i:>6,}/{runs:,}  {elapsed:.0f}s elapsed  ETA {eta:.0f}s  "
                  f"cache {len(_predict_cache)}")

    total_time = time.time() - t0

    # ── Build output ──────────────────────────────────────────────────────────
    group_advancement = {
        t: {
            "rate":         round(adv_counts[t]["advanced"]     / runs, 4),
            "as_winner":    round(adv_counts[t]["as_winner"]    / runs, 4),
            "as_runner_up": round(adv_counts[t]["as_runner_up"] / runs, 4),
            "as_third":     round(adv_counts[t]["as_third"]     / runs, 4),
        }
        for t in teams_sorted
    }
    knockout_progression = {
        t: {s: round(ko_counts[t][s] / runs, 4)
            for s in ("r32", "r16", "qf", "sf", "final", "champion")}
        for t in teams_sorted
    }
    champion_probs = sorted(
        [{"team": t, "probability": knockout_progression[t]["champion"]}
         for t in teams_sorted],
        key=lambda x: -x["probability"],
    )
    most_likely_final = list(final_pairs.most_common(1)[0][0])

    arg_eliminated = runs - arg_won_group - arg_runner_up - \
                     adv_counts["Argentina"]["as_third"]
    argentina_path = {
        "won_group":  round(arg_won_group / runs, 4),
        "runner_up":  round(arg_runner_up  / runs, 4),
        "eliminated": round(max(0, arg_eliminated) / runs, 4),
        "r32_opponents": {
            t: round(c / runs, 4)
            for t, c in arg_opp_ctr.most_common()
        },
    }

    def _sf_entry(ctr: Counter) -> list:
        pair, cnt = ctr.most_common(1)[0]
        return [pair[0], pair[1], round(cnt / runs, 4)]

    most_common_sf_pairs = {
        "sf1": _sf_entry(sf1_counter),
        "sf2": _sf_entry(sf2_counter),
    }

    output = {
        "runs":                   runs,
        "simulation_type":        "chronological_seeded_bracket_v2",
        "group_advancement":      group_advancement,
        "knockout_progression":   knockout_progression,
        "most_likely_final":      most_likely_final,
        "champion_probabilities": champion_probs,
        "argentina_path":         argentina_path,
        "most_common_sf_pairs":   most_common_sf_pairs,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved → {RESULTS_PATH}  ({total_time:.1f}s total)")

    # ── Terminal report ───────────────────────────────────────────────────────
    _print_summary_header(runs)
    for e in champion_probs[:15]:
        bar = "█" * round(e["probability"] * 200)
        print(f"  {e['team']:<28} {bar:<20}  {e['probability']*100:5.2f}%")

    arg_kp = knockout_progression["Argentina"]
    arg_ap = argentina_path
    print(f"\nARGENTINA PATH:")
    print(f"  Won Group {ARGENTINA_GROUP}:        {arg_ap['won_group']*100:.1f}%")
    print(f"  Runner-up Group {ARGENTINA_GROUP}:  {arg_ap['runner_up']*100:.1f}%")
    print(f"  Eliminated (groups):  {arg_ap['eliminated']*100:.1f}%")
    opps = list(argentina_path["r32_opponents"].items())
    opp_str = "  |  ".join(f"{t} {p*100:.1f}%" for t, p in opps[:6])
    print(f"  R32 opponents: {opp_str}")

    frc_kp = knockout_progression["France"]
    print(f"\nFRANCE PATH:")
    print(f"  R32 rate: {frc_kp['r32']*100:.1f}%  |  "
          f"R16: {frc_kp['r16']*100:.1f}%  |  "
          f"SF: {frc_kp['sf']*100:.1f}%  |  "
          f"Champion: {frc_kp['champion']*100:.2f}%")

    mf_pair, mf_cnt = final_pairs.most_common(1)[0]
    print(f"\nMOST COMMON FINAL: {mf_pair[0]} vs {mf_pair[1]}  "
          f"({mf_cnt/runs*100:.1f}% of runs)")

    sf1 = most_common_sf_pairs["sf1"]
    sf2 = most_common_sf_pairs["sf2"]
    print(f"MOST COMMON SF1:   {sf1[0]} vs {sf1[1]}  ({sf1[2]*100:.1f}%)")
    print(f"MOST COMMON SF2:   {sf2[0]} vs {sf2[1]}  ({sf2[2]*100:.1f}%)")

    print(f"\nRuntime: {total_time:.1f}s  |  Cache: {len(_predict_cache)} matchups\n")


def run_ko_only(bracket_path: str, runs: int = RUNS) -> None:
    """
    Simulate the KO stage only from a known R32 bracket.

    bracket_path: path to wc_actual_bracket.json
    Format: {"r32": [{"match":73,"slot1":"...","slot2":"...","team1":"X","team2":"Y"}, ...]}
    """
    bracket_file = Path(bracket_path)
    if not bracket_file.exists():
        print(f"ERROR: Bracket file not found: {bracket_path}")
        return

    raw = json.loads(bracket_file.read_text())
    entries = raw.get("r32", [])
    if len(entries) != 16:
        print(f"ERROR: Expected 16 R32 matches, got {len(entries)}")
        return

    r32_matchups: list[tuple[str, str]] = []
    for e in entries:
        t1 = e.get("team1", "").strip()
        t2 = e.get("team2", "").strip()
        if not t1 or not t2:
            print(f"ERROR: Match {e.get('match','?')} missing team1 or team2 — fill in wc_actual_bracket.json first")
            return
        r32_matchups.append((t1, t2))

    all_teams = sorted(set(t for pair in r32_matchups for t in pair))
    print(f"KO-only simulation ({runs:,} runs) from: {bracket_path}")
    print(f"R32 field ({len(all_teams)} teams): {', '.join(all_teams)}")
    print(f"\nLoading tactical stats …")
    _load_team_stats(all_teams)

    ko_counts: dict[str, dict[str, int]] = {
        t: {"r32": 0, "r16": 0, "qf": 0, "sf": 0, "final": 0, "champion": 0}
        for t in all_teams
    }
    final_pairs: Counter = Counter()
    sf1_counter: Counter = Counter()
    sf2_counter: Counter = Counter()

    t0 = time.time()
    for run_i in range(1, runs + 1):
        br    = _play_bracket({}, r32_matchups)
        res   = br["results"]
        in_m  = br["in_match"]

        r32_all   = set(t for pair in r32_matchups for t in pair)
        r16_all   = set(t for m in range(89, 97)  for t in in_m[m])
        qf_all    = set(t for m in range(97, 101) for t in in_m[m])
        sf_all    = set(t for m in (101, 102)     for t in in_m[m])
        finalists = list(in_m[104])
        champion  = res[104]

        for stage, ts in [("r32", r32_all), ("r16", r16_all), ("qf", qf_all),
                          ("sf", sf_all), ("final", set(finalists)), ("champion", {champion})]:
            for team in ts:
                if team in ko_counts:
                    ko_counts[team][stage] += 1

        final_pairs[tuple(sorted(finalists))] += 1
        sf1_counter[tuple(sorted(in_m[101]))] += 1
        sf2_counter[tuple(sorted(in_m[102]))] += 1

        if run_i % 1000 == 0:
            elapsed = time.time() - t0
            eta     = (elapsed / run_i) * (runs - run_i)
            print(f"  {run_i:>6,}/{runs:,}  {elapsed:.0f}s elapsed  ETA {eta:.0f}s  "
                  f"cache {len(_predict_cache)}")

    total_time = time.time() - t0

    knockout_progression = {
        t: {s: round(ko_counts[t][s] / runs, 4)
            for s in ("r32", "r16", "qf", "sf", "final", "champion")}
        for t in all_teams
    }
    group_advancement = {
        t: {"rate": 1.0, "as_winner": 0.0, "as_runner_up": 0.0, "as_third": 0.0}
        for t in all_teams
    }
    champion_probs = sorted(
        [{"team": t, "probability": knockout_progression[t]["champion"]}
         for t in all_teams],
        key=lambda x: -x["probability"],
    )
    most_likely_final = list(final_pairs.most_common(1)[0][0])

    def _sf_entry(ctr: Counter) -> list:
        pair, cnt = ctr.most_common(1)[0]
        return [pair[0], pair[1], round(cnt / runs, 4)]

    most_common_sf_pairs = {
        "sf1": _sf_entry(sf1_counter),
        "sf2": _sf_entry(sf2_counter),
    }

    arg_in = "Argentina" in all_teams
    argentina_path = {
        "won_group":  1.0 if arg_in else 0.0,
        "runner_up":  0.0,
        "eliminated": 0.0 if arg_in else 1.0,
        "r32_opponents": {},
    }

    output = {
        "runs":                   runs,
        "simulation_type":        "ko_only_from_actual_bracket",
        "bracket_source":         str(Path(bracket_path).relative_to(_REPO_ROOT)) if Path(bracket_path).is_absolute() and Path(bracket_path).is_relative_to(_REPO_ROOT) else bracket_path,
        "group_advancement":      group_advancement,
        "knockout_progression":   knockout_progression,
        "most_likely_final":      most_likely_final,
        "champion_probabilities": champion_probs,
        "argentina_path":         argentina_path,
        "most_common_sf_pairs":   most_common_sf_pairs,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved → {RESULTS_PATH}  ({total_time:.1f}s total)")

    _print_summary_header(runs)
    for e in champion_probs[:15]:
        bar = "█" * round(e["probability"] * 200)
        print(f"  {e['team']:<28} {bar:<20}  {e['probability']*100:5.2f}%")

    mf_pair, mf_cnt = final_pairs.most_common(1)[0]
    print(f"\nMOST COMMON FINAL: {mf_pair[0]} vs {mf_pair[1]}  ({mf_cnt/runs*100:.1f}% of runs)")
    sf1 = most_common_sf_pairs["sf1"]
    sf2 = most_common_sf_pairs["sf2"]
    print(f"MOST COMMON SF1:   {sf1[0]} vs {sf1[1]}  ({sf1[2]*100:.1f}%)")
    print(f"MOST COMMON SF2:   {sf2[0]} vs {sf2[1]}  ({sf2[2]*100:.1f}%)")
    print(f"\nRuntime: {total_time:.1f}s  |  Cache: {len(_predict_cache)} matchups\n")


def _play_bracket_r16(
    r16_matchups: list[tuple[str, str]],
    forced: dict[int, str],
) -> dict:
    """
    Play the bracket from R16 (M89–M96) to the final (M104).

    r16_matchups: 8 (team1, team2) pairs in bracket order —
                  QF pairing is sequential: M97 = W89 vs W90, M98 = W91 vs W92, …
    forced:       match_num → winner for matches already played in reality;
                  those results are used as-is instead of being simulated.
    """
    results: dict[int, str] = {}
    in_match: dict[int, tuple[str, str]] = {}

    for i, (t1, t2) in enumerate(r16_matchups):
        m = 89 + i
        in_match[m] = (t1, t2)
        results[m] = forced[m] if m in forced else _knockout_match(t1, t2)

    for i, (a, b) in enumerate([(89, 90), (91, 92), (93, 94), (95, 96)]):
        m = 97 + i
        t1, t2 = results[a], results[b]
        in_match[m] = (t1, t2)
        results[m] = forced[m] if m in forced else _knockout_match(t1, t2)

    for i, (a, b) in enumerate([(97, 98), (99, 100)]):
        m = 101 + i
        t1, t2 = results[a], results[b]
        in_match[m] = (t1, t2)
        results[m] = forced[m] if m in forced else _knockout_match(t1, t2)

    t1, t2 = results[101], results[102]
    in_match[104] = (t1, t2)
    results[104] = forced[104] if 104 in forced else _knockout_match(t1, t2)

    return {"results": results, "in_match": in_match}


def run_ko_only_r16(bracket_path: str, runs: int = RUNS, seed: int = 42) -> None:
    """
    Simulate from the real R16 bracket onward.

    bracket_path format:
      {"r16": [{"match":89,"team1":"X","team2":"Y","winner":"X"|null}, ...]}
    Matches with a non-null "winner" are locked to that result in every run.
    """
    if runs < 1:
        raise ValueError("runs must be positive")
    random.seed(seed)
    np.random.seed(seed)
    _predict_cache.clear()
    bracket_file = Path(bracket_path)
    if not bracket_file.exists():
        print(f"ERROR: Bracket file not found: {bracket_path}")
        return

    raw = json.loads(bracket_file.read_text())
    entries = raw.get("r16", [])
    if len(entries) != 8:
        print(f"ERROR: Expected 8 R16 matches, got {len(entries)}")
        return

    r16_matchups: list[tuple[str, str]] = []
    forced: dict[int, str] = {}
    for i, e in enumerate(entries):
        t1 = e.get("team1", "").strip()
        t2 = e.get("team2", "").strip()
        if not t1 or not t2:
            print(f"ERROR: Match {e.get('match','?')} missing team1 or team2")
            return
        r16_matchups.append((t1, t2))
        w = (e.get("winner") or "").strip()
        if w:
            if w not in (t1, t2):
                print(f"ERROR: Match {e.get('match','?')} winner '{w}' is not one of {t1}/{t2}")
                return
            forced[89 + i] = w

    from config import SNAPSHOT_DATE
    for entry in entries:
        if entry.get("winner") and (not entry.get("date") or entry["date"] >= SNAPSHOT_DATE):
            raise ValueError("Locked results must be dated before the snapshot cutoff")
    if len({t for pair in r16_matchups for t in pair}) != 16:
        raise ValueError("The R16 bracket must contain 16 distinct teams")
    all_teams = sorted(set(t for pair in r16_matchups for t in pair))
    print(f"KO-only simulation from R16 ({runs:,} runs) from: {bracket_path}")
    print(f"R16 field ({len(all_teams)} teams): {', '.join(all_teams)}")
    if forced:
        locked = ", ".join(f"M{m}→{w}" for m, w in sorted(forced.items()))
        print(f"Locked results: {locked}")
    print(f"\nLoading tactical stats …")
    _load_team_stats(all_teams)

    ko_counts: dict[str, dict[str, int]] = {
        t: {"r32": 0, "r16": 0, "qf": 0, "sf": 0, "final": 0, "champion": 0}
        for t in all_teams
    }
    final_pairs: Counter = Counter()
    sf1_counter: Counter = Counter()
    sf2_counter: Counter = Counter()

    t0 = time.time()
    for run_i in range(1, runs + 1):
        br    = _play_bracket_r16(r16_matchups, forced)
        res   = br["results"]
        in_m  = br["in_match"]

        r16_all   = set(t for pair in r16_matchups for t in pair)
        qf_all    = set(t for m in range(97, 101) for t in in_m[m])
        sf_all    = set(t for m in (101, 102)     for t in in_m[m])
        finalists = list(in_m[104])
        champion  = res[104]

        for stage, ts in [("r32", r16_all), ("r16", r16_all), ("qf", qf_all),
                          ("sf", sf_all), ("final", set(finalists)), ("champion", {champion})]:
            for team in ts:
                if team in ko_counts:
                    ko_counts[team][stage] += 1

        final_pairs[tuple(sorted(finalists))] += 1
        sf1_counter[tuple(sorted(in_m[101]))] += 1
        sf2_counter[tuple(sorted(in_m[102]))] += 1

        if run_i % 1000 == 0:
            elapsed = time.time() - t0
            eta     = (elapsed / run_i) * (runs - run_i)
            print(f"  {run_i:>6,}/{runs:,}  {elapsed:.0f}s elapsed  ETA {eta:.0f}s  "
                  f"cache {len(_predict_cache)}")

    total_time = time.time() - t0

    knockout_progression = {
        t: {s: round(ko_counts[t][s] / runs, 4)
            for s in ("r32", "r16", "qf", "sf", "final", "champion")}
        for t in all_teams
    }
    group_advancement = {
        t: {"rate": 1.0, "as_winner": 0.0, "as_runner_up": 0.0, "as_third": 0.0}
        for t in all_teams
    }
    champion_probs = sorted(
        [{"team": t, "probability": knockout_progression[t]["champion"]}
         for t in all_teams],
        key=lambda x: -x["probability"],
    )
    most_likely_final = list(final_pairs.most_common(1)[0][0])

    def _sf_entry(ctr: Counter) -> list:
        pair, cnt = ctr.most_common(1)[0]
        return [pair[0], pair[1], round(cnt / runs, 4)]

    most_common_sf_pairs = {
        "sf1": _sf_entry(sf1_counter),
        "sf2": _sf_entry(sf2_counter),
    }

    arg_in = "Argentina" in all_teams
    argentina_path = {
        "won_group":  1.0 if arg_in else 0.0,
        "runner_up":  0.0,
        "eliminated": 0.0 if arg_in else 1.0,
        "r32_opponents": {},
    }

    output = {
        "runs":                   runs,
        "simulation_type":        "ko_only_from_r16_bracket",
        "snapshot_date":          SNAPSHOT_DATE,
        "seed":                   seed,
        "cutoff_convention":      "match date < snapshot_date",
        "bracket_source":         str(Path(bracket_path).relative_to(_REPO_ROOT)) if Path(bracket_path).is_absolute() and Path(bracket_path).is_relative_to(_REPO_ROOT) else bracket_path,
        "group_advancement":      group_advancement,
        "knockout_progression":   knockout_progression,
        "most_likely_final":      most_likely_final,
        "champion_probabilities": champion_probs,
        "argentina_path":         argentina_path,
        "most_common_sf_pairs":   most_common_sf_pairs,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved → {RESULTS_PATH}  ({total_time:.1f}s total)")

    _print_summary_header(runs)
    for e in champion_probs[:15]:
        bar = "█" * round(e["probability"] * 200)
        print(f"  {e['team']:<28} {bar:<20}  {e['probability']*100:5.2f}%")

    mf_pair, mf_cnt = final_pairs.most_common(1)[0]
    print(f"\nMOST COMMON FINAL: {mf_pair[0]} vs {mf_pair[1]}  ({mf_cnt/runs*100:.1f}% of runs)")
    sf1 = most_common_sf_pairs["sf1"]
    sf2 = most_common_sf_pairs["sf2"]
    print(f"MOST COMMON SF1:   {sf1[0]} vs {sf1[1]}  ({sf1[2]*100:.1f}%)")
    print(f"MOST COMMON SF2:   {sf2[0]} vs {sf2[1]}  ({sf2[2]*100:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="World Cup tournament simulations")
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--from-bracket")
    modes.add_argument("--from-r16-bracket")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.from_r16_bracket:
        run_ko_only_r16(args.from_r16_bracket, args.runs, args.seed)
    elif args.from_bracket:
        run_ko_only(args.from_bracket, args.runs)
    else:
        run(args.runs, sample_only=args.sample)


if __name__ == "__main__":
    main()
