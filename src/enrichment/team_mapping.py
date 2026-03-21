"""Team name mapping between Betfair runner/event names and stats API names.

Betfair uses names like "Sydney FC", "Man Utd", "Western Sydney".
Football-data.org uses "Sydney FC", "Manchester United FC", "Western Sydney Wanderers FC".
Squiggle uses short AFL names like "Sydney", "Collingwood".

Strategy:
  1. Check hard-coded alias dict (covers known mismatches)
  2. Normalize and fuzzy-match via difflib.SequenceMatcher
  3. Cache successful mappings permanently
  4. Return None on no confident match — caller falls back to mid_price
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

_FUZZY_THRESHOLD = 0.75

# ---------------------------------------------------------------------------
# Hard-coded aliases: Betfair name (lowercase) -> canonical stats API name
# ---------------------------------------------------------------------------

_FOOTBALL_ALIASES: dict[str, str] = {
    # A-League
    "western sydney": "Western Sydney Wanderers FC",
    "western sydney wanderers": "Western Sydney Wanderers FC",
    "melbourne city": "Melbourne City FC",
    "melbourne victory": "Melbourne Victory FC",
    "sydney fc": "Sydney FC",
    "newcastle jets": "Newcastle Jets FC",
    "perth glory": "Perth Glory FC",
    "central coast": "Central Coast Mariners FC",
    "central coast mariners": "Central Coast Mariners FC",
    "wellington phoenix": "Wellington Phoenix FC",
    "macarthur fc": "Macarthur FC",
    "macarthur": "Macarthur FC",
    "brisbane roar": "Brisbane Roar FC",
    "adelaide united": "Adelaide United FC",
    "auckland fc": "Auckland FC",
    # EPL
    "man utd": "Manchester United FC",
    "manchester utd": "Manchester United FC",
    "man city": "Manchester City FC",
    "manchester city": "Manchester City FC",
    "spurs": "Tottenham Hotspur FC",
    "tottenham": "Tottenham Hotspur FC",
    "wolves": "Wolverhampton Wanderers FC",
    "wolverhampton": "Wolverhampton Wanderers FC",
    "newcastle": "Newcastle United FC",
    "newcastle utd": "Newcastle United FC",
    "west ham": "West Ham United FC",
    "brighton": "Brighton & Hove Albion FC",
    "nott'm forest": "Nottingham Forest FC",
    "nottingham forest": "Nottingham Forest FC",
    "leicester": "Leicester City FC",
    "ipswich": "Ipswich Town FC",
    # Bundesliga
    "bayern": "FC Bayern München",
    "bayern munich": "FC Bayern München",
    "dortmund": "Borussia Dortmund",
    "borussia dortmund": "Borussia Dortmund",
    "leverkusen": "Bayer 04 Leverkusen",
    "bayer leverkusen": "Bayer 04 Leverkusen",
    "rb leipzig": "RB Leipzig",
    "leipzig": "RB Leipzig",
    # La Liga
    "atletico madrid": "Club Atlético de Madrid",
    "atletico": "Club Atlético de Madrid",
    "real madrid": "Real Madrid CF",
    "barcelona": "FC Barcelona",
    "barca": "FC Barcelona",
    # Serie A
    "ac milan": "AC Milan",
    "inter": "FC Internazionale Milano",
    "inter milan": "FC Internazionale Milano",
    "juventus": "Juventus FC",
    "napoli": "SSC Napoli",
    "roma": "AS Roma",
    "lazio": "SS Lazio",
    # Ligue 1
    "psg": "Paris Saint-Germain FC",
    "paris saint-germain": "Paris Saint-Germain FC",
    "marseille": "Olympique de Marseille",
    "lyon": "Olympique Lyonnais",
}

_AFL_ALIASES: dict[str, str] = {
    "sydney swans": "Sydney",
    "sydney": "Sydney",
    "collingwood": "Collingwood",
    "collingwood magpies": "Collingwood",
    "carlton": "Carlton",
    "carlton blues": "Carlton",
    "brisbane": "Brisbane Lions",
    "brisbane lions": "Brisbane Lions",
    "melbourne": "Melbourne",
    "melbourne demons": "Melbourne",
    "geelong": "Geelong",
    "geelong cats": "Geelong",
    "western bulldogs": "Western Bulldogs",
    "bulldogs": "Western Bulldogs",
    "gws": "GWS",
    "gws giants": "GWS",
    "greater western sydney": "GWS",
    "greater western sydney giants": "GWS",
    "hawthorn": "Hawthorn",
    "hawthorn hawks": "Hawthorn",
    "essendon": "Essendon",
    "essendon bombers": "Essendon",
    "richmond": "Richmond",
    "richmond tigers": "Richmond",
    "fremantle": "Fremantle",
    "fremantle dockers": "Fremantle",
    "port adelaide": "Port Adelaide",
    "port adelaide power": "Port Adelaide",
    "west coast": "West Coast",
    "west coast eagles": "West Coast",
    "north melbourne": "North Melbourne",
    "north melbourne kangaroos": "North Melbourne",
    "gold coast": "Gold Coast",
    "gold coast suns": "Gold Coast",
    "adelaide": "Adelaide",
    "adelaide crows": "Adelaide",
    "st kilda": "St Kilda",
    "st kilda saints": "St Kilda",
}

# Runtime cache for successful fuzzy matches: (betfair_name_lower, sport) -> canonical
_resolved_cache: dict[tuple[str, str], str] = {}


def _normalize(name: str) -> str:
    """Normalize a team name for fuzzy comparison."""
    n = name.lower().strip()
    # Strip common suffixes that vary between sources
    for suffix in (" fc", " afc", " sc"):
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
    return n


def resolve_team(betfair_name: str, sport: str = "football") -> str | None:
    """Map a Betfair runner/event name to the canonical stats API team name.

    Returns None if no confident match is found (caller falls back to mid_price).
    """
    key_lower = betfair_name.lower().strip()
    cache_key = (key_lower, sport)

    # 1. Check runtime cache
    if cache_key in _resolved_cache:
        return _resolved_cache[cache_key]

    # 2. Check hard-coded aliases
    aliases = _AFL_ALIASES if sport == "afl" else _FOOTBALL_ALIASES
    if key_lower in aliases:
        result = aliases[key_lower]
        _resolved_cache[cache_key] = result
        return result

    # 3. Fuzzy match against alias values
    normalized = _normalize(betfair_name)
    best_score = 0.0
    best_match: str | None = None

    for alias_key, canonical in aliases.items():
        score = SequenceMatcher(None, normalized, _normalize(alias_key)).ratio()
        if score > best_score:
            best_score = score
            best_match = canonical

    if best_score >= _FUZZY_THRESHOLD and best_match is not None:
        _resolved_cache[cache_key] = best_match
        logger.debug(
            "Fuzzy-matched Betfair name %r -> %r (score=%.2f)",
            betfair_name, best_match, best_score,
        )
        return best_match

    logger.warning(
        "Team mapping failed for %r (sport=%s, best_score=%.2f) — falling back to mid_price.",
        betfair_name, sport, best_score,
    )
    return None


def parse_teams_from_event(event_name: str) -> tuple[str, str] | None:
    """Parse 'Home Team v Away Team' from a Betfair event name.

    Returns (home, away) tuple or None if the format doesn't match.
    """
    parts = event_name.split(" v ")
    if len(parts) == 2:
        home = parts[0].strip()
        away = parts[1].strip()
        if home and away:
            return home, away
    return None
