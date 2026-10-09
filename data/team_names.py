"""Common aliases mapped to this project's squad-data keys."""

ALIASES = {
    "USA": "United States",
    "Bosnia": "Bosnia-Herzegovina",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "DR Congo": "Congo DR",
    "Curacao": "Curaçao",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Cote d'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
}

def canonical_team(name: str) -> str:
    return ALIASES.get(name.strip(), name.strip())
