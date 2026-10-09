#!/usr/bin/env python3
"""
build_squad_values.py

Builds squad_values.json from actual called-up WC 2026 squads.
Output is strictly filtered to the 48 WC 2026 tournament teams — no extras.

Data sources:
  - FIFA official squad PDF:      data/fifa_squads_2026.pdf
  - Transfermarkt player values:  data/tm_players.csv.gz

Never modifies squad_values_backup.json.
Requires user confirmation before writing squad_values.json.

Usage:
    cd /Users/mahmoodalgbory/wc_predictor
    python3 scripts/build_squad_values.py
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import re
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

import pdfplumber

# ── PATHS ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
FIFA_PDF    = REPO_ROOT / "data" / "fifa_squads_2026.pdf"
TM_CSV      = REPO_ROOT / "data" / "tm_players.csv.gz"
OUTPUT_FILE = REPO_ROOT / "data" / "squad_values.json"
BACKUP_FILE = REPO_ROOT / "data" / "squad_values_backup.json"

TOURNAMENT_START = date(2026, 6, 11)

# ── THE 48 WC 2026 TEAMS (canonical names used throughout this project) ────────
WC48: frozenset[str] = frozenset({
    # Group A
    "Mexico", "South Africa", "South Korea", "Czech Republic",
    # Group B
    "Canada", "Bosnia-Herzegovina", "Qatar", "Switzerland",
    # Group C
    "Brazil", "Morocco", "Haiti", "Scotland",
    # Group D
    "United States", "Paraguay", "Australia", "Turkey",
    # Group E
    "Germany", "Curaçao", "Ivory Coast", "Ecuador",
    # Group F
    "Netherlands", "Japan", "Sweden", "Tunisia",
    # Group G
    "Belgium", "Egypt", "Iran", "New Zealand",
    # Group H
    "Spain", "Cape Verde", "Saudi Arabia", "Uruguay",
    # Group I
    "France", "Senegal", "Norway", "Iraq",
    # Group J
    "Argentina", "Algeria", "Austria", "Jordan",
    # Group K
    "Portugal", "Congo DR", "Uzbekistan", "Colombia",
    # Group L
    "England", "Croatia", "Ghana", "Panama",
})
assert len(WC48) == 48, f"WC48 has {len(WC48)} teams, expected 48"

# ── FIFA PDF → project canonical name ─────────────────────────────────────────
TEAM_NAME_MAP: dict[str, str] = {
    "Korea Republic":           "South Korea",
    "IR Iran":                  "Iran",
    "United States":            "United States",
    "USA":                      "United States",
    "Côte d'Ivoire":            "Ivory Coast",
    "Cote d'Ivoire":            "Ivory Coast",
    "Côte D'Ivoire":            "Ivory Coast",   # PDF capitalises D'
    "Cote D'Ivoire":            "Ivory Coast",
    "DR Congo":                 "Congo DR",
    "Bosnia & Herzegovina":     "Bosnia-Herzegovina",
    "Bosnia and Herzegovina":   "Bosnia-Herzegovina",
    "Bosnia And Herzegovina":   "Bosnia-Herzegovina",  # PDF title-cases 'And'
    "Cabo Verde":               "Cape Verde",
    "Türkiye":                  "Turkey",
    "Curacao":                  "Curaçao",
    "New Zealand":              "New Zealand",
    "Saudi Arabia":             "Saudi Arabia",
    "Czech Republic":           "Czech Republic",
    "Czechia":                  "Czech Republic",  # FIFA PDF uses 'Czechia'
}

# ── FIFA position codes → squad_values categories ─────────────────────────────
POSITION_MAP: dict[str, str] = {
    "GK": "goalkeeper",
    "DF": "defense",
    "MF": "midfield",
    "FW": "attack",
}

# ── MANUAL VALUES for known stars / poor TM name-match risk ───────────────────
# Values in EUR (not millions). Use June 2026 estimates.
MANUAL_VALUES: dict[str, float] = {
    # Argentina
    "Lionel Messi":        35_000_000,   # 38yo at tournament
    "Julian Alvarez":      90_000_000,
    "Rodrigo De Paul":     28_000_000,

    # France
    "Kylian Mbappe":      180_000_000,
    "Kylian Mbappé":      180_000_000,
    "Antoine Griezmann":   18_000_000,   # 35yo
    "Ousmane Dembele":     60_000_000,
    "Ousmane Dembélé":     60_000_000,

    # Spain
    "Lamine Yamal":       130_000_000,
    "Pedri":              100_000_000,
    "Gavi":                80_000_000,
    "Rodri":              120_000_000,
    "Nico Williams":       80_000_000,

    # England
    "Jude Bellingham":    160_000_000,
    "Bukayo Saka":        130_000_000,
    "Phil Foden":         120_000_000,
    "Harry Kane":          80_000_000,
    "Declan Rice":         90_000_000,

    # Germany
    "Florian Wirtz":      110_000_000,
    "Jamal Musiala":      130_000_000,

    # Brazil
    "Vinicius Junior":    180_000_000,
    "Vinicius Jr":        180_000_000,
    "Rodrygo":             80_000_000,
    "Raphinha":            65_000_000,
    "Endrick":             60_000_000,

    # Portugal
    "Cristiano Ronaldo":   12_000_000,   # 41yo
    "Bruno Fernandes":     55_000_000,
    "Bernardo Silva":      60_000_000,
    "Ruben Dias":          70_000_000,
    "Diogo Costa":         50_000_000,

    # Netherlands
    "Virgil Van Dijk":     25_000_000,
    "Cody Gakpo":          70_000_000,
    "Xavi Simons":        100_000_000,

    # Belgium
    "Kevin De Bruyne":     22_000_000,   # 35yo
    "Romelu Lukaku":       15_000_000,   # 32yo
    "Jeremy Doku":         80_000_000,

    # Morocco
    "Achraf Hakimi":       65_000_000,
    "Sofyan Amrabat":      30_000_000,
    "Hakim Ziyech":        12_000_000,
    "Azzedine Ounahi":     25_000_000,
    "Brahim Diaz":         40_000_000,
    "Brahim Díaz":         40_000_000,

    # Argentina others
    "Enzo Fernandez":      80_000_000,
    "Enzo Fernández":      80_000_000,
    "Alexis Mac Allister": 75_000_000,

    # Norway
    "Erling Haaland":     200_000_000,
    "Martin Odegaard":     90_000_000,
    "Martin Ødegaard":     90_000_000,

    # Saudi Arabia
    "Mohamed Salah":       35_000_000,   # 34yo — playing in Saudi

    # Brazil single-name players (TM uses shirt name as primary key)
    "Alisson":             35_000_000,
    "Alisson Becker":      35_000_000,
    "Casemiro":            15_000_000,   # 34yo
    "Marquinhos":          25_000_000,   # 30yo
    "Raphinha":            65_000_000,
    "Endrick":             60_000_000,
    "Neymar":              10_000_000,   # returning from injury
    "Rodrygo":             80_000_000,
    "Gabriel Martinelli":  55_000_000,

    # General aging stars
    "Luka Modric":          8_000_000,   # 40yo
    "Thomas Muller":        6_000_000,   # retiring
    "Thomas Müller":        6_000_000,

    # Ivory Coast stars
    "Nicolas Pepe":        15_000_000,
    "Sebastien Haller":    20_000_000,
    "Franck Kessie":       18_000_000,
    "Franck Kessié":       18_000_000,
    "Simon Adingra":       35_000_000,
    "Oumar Diakite":       12_000_000,
}


# ── PARSE FIFA PDF ─────────────────────────────────────────────────────────────

def parse_fifa_pdf(pdf_path: Path) -> dict[str, list[dict]]:
    """
    Parse the official FIFA WC 2026 squad list PDF.
    Returns {canonical_team_name: [player_dict, ...], ...}

    PDF format (one team per page):
        SQUAD LIST
        FIFA World Cup 2026™
        11 June 2026 – 19 July 2026
        TeamName (TRI)
        # POS PLAYER NAME FIRST NAME(S) LAST NAME(S) NAME ON SHIRT DOB CLUB HEIGHT
        1 GK SURNAME FirstName Full SURNAME SHIRT DD/MM/YYYY Club Name (COD) 185
        ...
    """
    squads: dict[str, list[dict]] = {}

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text:
                continue

            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
            current_team: str | None = None
            players: list[dict] = []

            for line in lines:
                # ── Team header: "TeamName (TRI)" ──────────────────────────
                # Must appear before any player rows on this page
                if current_team is None:
                    team_m = re.match(r"^(.+?)\s+\(([A-Z]{3})\)\s*$", line)
                    if team_m:
                        raw = team_m.group(1).strip()
                        canonical = TEAM_NAME_MAP.get(raw, raw)
                        if canonical in WC48:
                            current_team = canonical
                            players = []
                    continue

                # ── Player row: "N GK|DF|MF|FW PLAYER_NAME FirstName ..." ──
                # FIFA PDF column order:
                #   # | POS | PLAYER NAME (all-caps) | FIRST NAME(S) | LAST NAME(S) |
                #   NAME ON SHIRT | DOB | CLUB (COD) | HEIGHT
                # PLAYER NAME is the canonical display name (shirt name).
                # It can be 1 token ("MESSI", "RAPHINHA") or compound ("VINICIUS JUNIOR").
                player_m = re.match(r"^(\d{1,2})\s+(GK|DF|MF|FW)\s+(.+)$", line)
                if not player_m:
                    continue

                pos_code = player_m.group(2)
                rest     = player_m.group(3)

                # DOB anchor: DD/MM/YYYY
                dob_m = re.search(r"(\d{2}/\d{2}/\d{4})", rest)
                dob_str: str | None = dob_m.group(1) if dob_m else None
                before_dob = rest[: dob_m.start()].strip() if dob_m else rest

                # Extract club: text after DOB ending with "(COD) HEIGHT"
                club: str | None = None
                if dob_m:
                    after_dob = rest[dob_m.end():].strip()
                    club_m = re.match(r"^(.+?)\s+\([A-Z]{3}\)(?:\s+\d+)?$", after_dob)
                    if club_m:
                        club = club_m.group(1).strip()

                # ── Extract PLAYER NAME (all-caps leading tokens) ──────────
                # Scan tokens: all-caps tokens at the start = PLAYER NAME,
                # first mixed-case token marks start of FIRST NAME column.
                tokens = before_dob.split()
                shirt_tokens: list[str] = []
                first_name: str = ""
                for tok in tokens:
                    # A token belongs to PLAYER NAME if it's all upper-case
                    # (allowing digits, hyphens, dots: "JR.", "D'")
                    clean = re.sub(r"[^A-Za-z]", "", tok)
                    if clean and clean == clean.upper():
                        shirt_tokens.append(tok)
                    else:
                        first_name = tok
                        break

                if not shirt_tokens:
                    # Fallback — use first token as surname
                    shirt_tokens = [tokens[0]] if tokens else ["UNKNOWN"]
                    first_name   = tokens[1] if len(tokens) > 1 else ""

                shirt_name = " ".join(t.title() for t in shirt_tokens)

                # Build display name:
                # - Compound shirt name (2+ tokens): use shirt_name directly
                #   e.g. "VINICIUS JUNIOR" → "Vinicius Junior"
                #        "GABRIEL MAGALHAES" → "Gabriel Magalhaes"
                # - Single token: prepend first name for European-style names
                #   e.g. "MESSI" + "Lionel" → "Lionel Messi"
                #   but also store shirt_name alone as alt for nickname lookup
                if len(shirt_tokens) >= 2:
                    display_name = shirt_name          # "Vinicius Junior"
                    alt_name     = shirt_name          # same
                else:
                    display_name = f"{first_name} {shirt_name}".strip() if first_name else shirt_name
                    alt_name     = shirt_name          # e.g. "Raphinha", "Alisson"

                players.append({
                    "name":     display_name,
                    "alt_name": alt_name,   # tried second in TM lookup
                    "position": POSITION_MAP.get(pos_code, "midfield"),
                    "dob":      dob_str,
                    "club":     club or "Unknown",
                })

            if current_team and players:
                squads[current_team] = players

    return squads


# ── LOAD TRANSFERMARKT VALUES ──────────────────────────────────────────────────

def load_tm_values(csv_gz: Path) -> dict[str, float]:
    """Returns {player_name: market_value_eur}."""
    values: dict[str, float] = {}
    try:
        with gzip.open(str(csv_gz), "rt", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get("name") or row.get("player_name") or "").strip()
                raw_val = (row.get("market_value_in_eur") or row.get("market_value") or "").strip()
                if name and raw_val:
                    try:
                        values[name] = float(raw_val)
                    except ValueError:
                        pass
    except Exception as exc:
        print(f"[warn] TM CSV load error: {exc} — MANUAL_VALUES only")
    return values


def get_player_value(
    display_name: str,
    tm_values: dict[str, float],
    alt_name: str = "",
) -> float:
    """
    Look up market value trying multiple name variants:
    1. MANUAL_VALUES fuzzy match (85%+) on display_name and alt_name
    2. TM exact match (display_name, then alt_name)
    3. TM fuzzy match ≥ 0.82 across both names
    4. Fallback €5M

    alt_name is the raw shirt name (e.g. "Raphinha", "Alisson") and helps
    when display_name is the Western "First Surname" form that TM doesn't use.
    """
    candidates = list({display_name, alt_name} - {""})  # deduplicate

    # 1. Manual values (try all candidates)
    for name in candidates:
        lower = name.lower()
        for manual_name, val in MANUAL_VALUES.items():
            if SequenceMatcher(None, lower, manual_name.lower()).ratio() >= 0.85:
                return float(val)

    # 2. TM exact
    for name in candidates:
        if name in tm_values:
            return tm_values[name]

    # 3. TM fuzzy — run once, check both candidates
    best_ratio = 0.0
    best_val: float | None = None
    for tm_name, val in tm_values.items():
        tm_lower = tm_name.lower()
        for name in candidates:
            r = SequenceMatcher(None, name.lower(), tm_lower).ratio()
            if r > best_ratio:
                best_ratio = r
                best_val = val
    if best_ratio >= 0.82 and best_val is not None:
        return best_val

    # 4. Fallback
    return 5_000_000.0


# ── COMPUTE PER-TEAM SQUAD VALUES ──────────────────────────────────────────────

def compute_squad_values(players: list[dict], tm_values: dict[str, float]) -> dict:
    """
    Given a list of 26 called-up players (with name/position/dob/club),
    compute all fields for squad_values.json.
    """
    if not players:
        return {}

    enriched = [
        {**p, "value_eur": get_player_value(p["name"], tm_values, p.get("alt_name", ""))}
        for p in players
    ]
    enriched.sort(key=lambda x: x["value_eur"], reverse=True)

    total_eur  = sum(p["value_eur"] for p in enriched)
    xi         = enriched[:11]
    bench      = enriched[11:]
    xi_eur     = sum(p["value_eur"] for p in xi)
    bench_eur  = sum(p["value_eur"] for p in bench)

    # Positional values (all 26 players)
    pos_vals: dict[str, float] = {
        "goalkeeper": 0.0, "defense": 0.0, "midfield": 0.0, "attack": 0.0
    }
    for p in enriched:
        cat = p.get("position", "midfield")
        pos_vals[cat] = pos_vals.get(cat, 0.0) + p["value_eur"]

    # Average age at tournament start
    ages: list[float] = []
    for p in enriched:
        if p.get("dob"):
            try:
                d, m, y = p["dob"].split("/")
                born = date(int(y), int(m), int(d))
                ages.append((TOURNAMENT_START - born).days / 365.25)
            except Exception:
                pass
    avg_age = round(sum(ages) / len(ages), 1) if ages else 27.0

    # Top-5 league percentage
    top5_keywords = [
        "premier league", "la liga", "bundesliga", "serie a",
        "ligue 1", "laliga", "fa premier",
    ]
    top5_count = sum(
        1 for p in enriched
        if any(kw in (p.get("club") or "").lower() for kw in top5_keywords)
    )
    top5_pct = round(top5_count / len(enriched), 2)

    # Star concentration: top-3 value / XI value
    top3_eur   = sum(p["value_eur"] for p in enriched[:3])
    star_conc  = round(top3_eur / xi_eur, 2) if xi_eur > 0 else 0.30

    M = 1_000_000  # EUR → millions
    return {
        "total_value_eur_millions": round(total_eur / M, 1),
        "starting_xi_value":        round(xi_eur   / M, 1),
        "bench_value":              round(bench_eur / M, 1),
        "positional_values": {
            k: round(v / M, 1) for k, v in pos_vals.items()
        },
        "avg_age":            avg_age,
        "top5_league_pct":    top5_pct,
        "star_concentration": star_conc,
    }


# ── DIFF PRINTER ───────────────────────────────────────────────────────────────

def print_diff(old: dict, new: dict) -> None:
    print("\nSQUAD VALUE CORRECTIONS — called-up squads vs old estimates")
    print("=" * 68)

    changed: list[tuple[str, float, float]] = []
    for team in sorted(new):
        old_total = old.get(team, {}).get("total_value_eur_millions", 0.0)
        new_total = new[team].get("total_value_eur_millions", 0.0)
        delta     = new_total - old_total
        if abs(delta) >= 20:
            changed.append((team, old_total, new_total))

    for team, old_v, new_v in sorted(changed, key=lambda x: x[2] - x[1]):
        arrow = "⬆" if new_v > old_v else "⬇"
        flag  = "  ⚠️  MAJOR" if abs(new_v - old_v) > 200 else ""
        print(f"  {arrow} {team:<22} €{old_v:.0f}M  →  €{new_v:.0f}M  ({new_v-old_v:+.0f}M){flag}")

        # Show positional breakdown when large swing
        if abs(new_v - old_v) >= 50:
            old_pos = old.get(team, {}).get("positional_values", {})
            new_pos = new[team].get("positional_values", {})
            for pos in ("goalkeeper", "defense", "midfield", "attack"):
                op = old_pos.get(pos, 0); np_ = new_pos.get(pos, 0)
                if abs(np_ - op) >= 20:
                    print(f"       {pos:<12} €{op:.0f}M → €{np_:.0f}M  ({np_-op:+.0f}M)")

    unchanged = [t for t in new if t not in {x[0] for x in changed}]
    print(f"\n  UNCHANGED ({len(unchanged)} teams): "
          f"{', '.join(sorted(unchanged)[:10])}{'...' if len(unchanged) > 10 else ''}")
    print(f"  CHANGED (Δ≥€20M): {len(changed)} teams")


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== WC 2026 Squad Value Builder ===\n")

    # ── Load backup ──────────────────────────────────────────────────────────
    with open(BACKUP_FILE) as f:
        old_values: dict = json.load(f)
    print(f"Backup loaded — {len(old_values)} teams (will be filtered to 48)\n")

    # ── Parse PDF ────────────────────────────────────────────────────────────
    print(f"Parsing FIFA squad PDF ({FIFA_PDF.name})...")
    squads = parse_fifa_pdf(FIFA_PDF)
    print(f"  {len(squads)} teams parsed (expecting 48)")
    for team, players in sorted(squads.items())[:4]:
        print(f"  {team}: {len(players)} players — "
              f"e.g. {players[0]['name']} ({players[0]['position']})")

    missing_pdf = sorted(WC48 - set(squads))
    extra_pdf   = sorted(set(squads) - WC48)
    if missing_pdf:
        print(f"\n  ⚠️  Not found in PDF ({len(missing_pdf)}): {missing_pdf}")
    if extra_pdf:
        print(f"  ⚠️  Extra (not WC48): {extra_pdf}")

    if len(squads) < 40:
        print(f"\n❌  Only {len(squads)} teams — PDF parsing issue. Aborting.")
        return

    # ── Load TM values ───────────────────────────────────────────────────────
    print(f"\nLoading Transfermarkt values ({TM_CSV.name})...")
    tm_values = load_tm_values(TM_CSV)
    print(f"  {len(tm_values):,} player values loaded")

    # ── Compute per-team values ──────────────────────────────────────────────
    print(f"\nComputing squad values...")

    # Start from only WC48 teams from the backup (strips extra 4 right here)
    new_values: dict = {t: v for t, v in old_values.items() if t in WC48}
    computed, fallback = 0, 0

    for team in sorted(WC48):
        players = squads.get(team)
        if not players or len(players) < 20:
            fallback += 1
            print(f"  [fallback] {team}: "
                  f"{'not in PDF' if not players else f'only {len(players)} players'}")
            continue
        result = compute_squad_values(players, tm_values)
        if result:
            new_values[team] = result
            computed += 1

    print(f"\n  {computed} teams computed from called-up squads")
    print(f"  {fallback} teams kept from backup")
    print(f"  Output will contain {len(new_values)} teams (exactly WC48)")

    # ── Spot-check key teams ─────────────────────────────────────────────────
    print("\nSpot-check (key teams):")
    for team in ["Argentina", "Belgium", "Spain", "France", "England", "Brazil"]:
        if team in new_values:
            old_v = old_values.get(team, {}).get("total_value_eur_millions", "?")
            new_v = new_values[team]["total_value_eur_millions"]
            print(f"  {team:<14} €{old_v}M  →  €{new_v}M")

    # ── Show diff ────────────────────────────────────────────────────────────
    print_diff({t: v for t, v in old_values.items() if t in WC48}, new_values)

    # ── Sanity guards — abort if values look insane ──────────────────────────
    guards = [
        ("Argentina", 400,  900),
        ("Belgium",   250,  600),
        ("Spain",     700, 1300),
        ("France",    900, 1500),
        ("England",   900, 1600),
        ("Brazil",    600, 1200),
    ]
    guard_fail = False
    for team, lo, hi in guards:
        val = new_values.get(team, {}).get("total_value_eur_millions", 0)
        if not (lo <= val <= hi):
            print(f"\n  ⚠️  GUARD FAIL — {team}: €{val}M not in [{lo}, {hi}]")
            guard_fail = True

    if guard_fail:
        print("\n⚠️  One or more sanity guards failed — review before applying.")

    # ── Confirm ──────────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"Ready to write {len(new_values)} teams (WC48 only) to {OUTPUT_FILE.name}")
    confirm = input("Apply? (y/n): ").strip().lower()
    if confirm != "y":
        print("Aborted — no files modified.")
        return

    with open(OUTPUT_FILE, "w") as f:
        json.dump(new_values, f, indent=2)

    print(f"\n✓ {OUTPUT_FILE.name} updated ({len(new_values)} teams)")
    print(f"✓ Backup preserved at {BACKUP_FILE.name}")
    print(f"✓ {computed} teams from called-up squads, {fallback} from backup")


if __name__ == "__main__":
    main()
