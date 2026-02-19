"""
LLM-powered fuzzy matching — Stage 2 of the pipeline.

Handles abbreviations, medleys, and cover detection using OpenAI.
Includes exponential backoff with jitter and an in-memory result cache.
"""

import hashlib
import json
import logging
import random
import time
from typing import Dict, List, Optional

from openai import OpenAI

from config import (
    BACKOFF_BASE,
    BACKOFF_MAX,
    LLM_MODEL,
    LLM_TEMPERATURE,
    MAX_RETRIES,
    RATE_LIMIT_DELAY,
)
from matching import normalize, validate_match

logger = logging.getLogger(__name__)

# ── In-memory cache ──────────────────────────────────────
# Keyed on normalized track name so "Tokyo (Acoustic)" and
# "Tokyo (Live)" both hit the same cache entry.
_match_cache: Dict[str, List[Dict]] = {}


def _cache_key(track_name: str) -> str:
    """Produce a stable cache key from a track name."""
    return hashlib.sha256(normalize(track_name).encode()).hexdigest()


def clear_cache() -> None:
    """Clear the LLM result cache (useful between runs or in tests)."""
    _match_cache.clear()


def _backoff_sleep(attempt: int) -> None:
    """Sleep with exponential backoff + jitter."""
    delay = min(BACKOFF_BASE ** attempt + random.uniform(0, 1), BACKOFF_MAX)
    logger.info("Retrying in %.1fs (attempt %d)...", delay, attempt + 1)
    time.sleep(delay)


# ── Prompt construction ──────────────────────────────────

SYSTEM_PROMPT = """You are a music catalog matching expert at a major music publisher.
Your job is to match live performance setlist track names against our internal song catalog.

RULES:
1. A setlist track may be an abbreviation, variation, or alternate version of a catalog song.
   Example: "Tokyo (Acoustic)" could match "Midnight in Tokyo".
2. A setlist track with "/" likely indicates a MEDLEY — it may match MULTIPLE catalog songs.
   Example: "Desert Rain / Ocean Avenue" should match both "Desert Rain" AND "Ocean Avenue".
3. If a track is a well-known song by ANOTHER artist (a cover), it is NOT CONTROLLED by us.
   Example: "Yesterday" is by The Beatles — flag as not controlled.
   Example: "Bhemn Rhpsdy" is "Bohemian Rhapsody" by Queen — flag as not controlled.
4. Only match if you are genuinely confident. Do NOT force-match vaguely similar titles.
5. If unsure, set confidence to "Review" rather than guessing wrong.

CONFIDENCE LEVELS:
- "High" — You are confident this is a match (abbreviation, suffix removed, clear variation).
- "Review" — Possible match but needs human review.
- "None" — No match found or song is not controlled (cover).

Return ONLY valid JSON. No other text."""


def _build_user_prompt(track_name: str, catalog: List[Dict]) -> str:
    """Build the user prompt for the LLM call."""
    catalog_list = "\n".join(
        f"- {s['catalog_id']}: \"{s['title']}\" (Writers: {s['writers']})"
        for s in catalog
    )

    # Detect medley
    is_medley = "/" in track_name
    medley_instruction = ""
    if is_medley:
        parts = [p.strip() for p in track_name.split("/")]
        medley_instruction = (
            f"\nIMPORTANT: This is a MEDLEY containing {len(parts)} songs: {parts}.\n"
            f"You MUST return a SEPARATE match entry for EACH part of the medley.\n"
            f"Return {len(parts)} objects in the \"matches\" array — one per song."
        )

    return (
        f'Match this setlist track against our catalog:\n\n'
        f'SETLIST TRACK: "{track_name}"\n'
        f'{medley_instruction}\n\n'
        f'OUR CATALOG:\n{catalog_list}\n\n'
        f'Return JSON with this exact structure:\n'
        f'{{"matches": [{{"catalog_id": "CAT-XXX or None", '
        f'"confidence": "High/Review/None", '
        f'"reasoning": "brief explanation"}}]}}\n\n'
        f'If this is a medley, include one entry per song in the medley.\n'
        f'If no match or it\'s a cover, return: '
        f'{{"matches": [{{"catalog_id": "None", "confidence": "None", '
        f'"reasoning": "explanation"}}]}}'
    )


# ── Core LLM call ────────────────────────────────────────

def _parse_llm_response(raw: str) -> List[Dict]:
    """
    Parse the LLM JSON response into a list of match dicts.

    Handles several response shapes the LLM might produce:
    - {"matches": [...]}
    - {"results": [...]} or {"data": [...]}
    - A bare list [...]
    - A single dict with "catalog_id"
    """
    parsed = json.loads(raw)

    if isinstance(parsed, list):
        return parsed

    if isinstance(parsed, dict):
        for key in ("matches", "results", "data"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
        if "catalog_id" in parsed:
            return [parsed]
        # Last resort: look for any list value
        for val in parsed.values():
            if isinstance(val, list):
                return val

    return []


def llm_fuzzy_match(
    track_name: str,
    catalog: List[Dict],
    client: OpenAI,
    max_retries: Optional[int] = None,
) -> List[Dict]:
    """
    Use the LLM to match a setlist track against the catalog.

    Features:
    - In-memory cache: same track across shows triggers only one API call.
    - Exponential backoff with jitter on failures.
    - Response validation against our actual catalog IDs.

    Returns a list of validated match dicts.
    """
    if max_retries is None:
        max_retries = MAX_RETRIES

    # Check cache first
    key = _cache_key(track_name)
    if key in _match_cache:
        logger.info("Cache hit for '%s'", track_name)
        return _match_cache[key]

    catalog_ids = {s["catalog_id"] for s in catalog}
    user_prompt = _build_user_prompt(track_name, catalog)

    # Rate limit: pause between consecutive LLM calls to avoid 429 errors
    if RATE_LIMIT_DELAY > 0:
        time.sleep(RATE_LIMIT_DELAY)

    last_error: Optional[str] = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=LLM_TEMPERATURE,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content.strip()
            matches = _parse_llm_response(raw)

            if not matches:
                matches = [{"catalog_id": "None", "confidence": "None",
                            "reasoning": "Unparseable response"}]

            validated = [validate_match(m, catalog_ids) for m in matches]

            # Store in cache
            _match_cache[key] = validated
            return validated

        except json.JSONDecodeError as exc:
            last_error = f"Invalid JSON from LLM: {exc}"
            logger.warning("Attempt %d: %s", attempt + 1, last_error)
        except Exception as exc:
            last_error = f"API error: {exc}"
            logger.warning("Attempt %d: %s", attempt + 1, last_error)

        if attempt < max_retries:
            _backoff_sleep(attempt)

    # All retries exhausted
    logger.error(
        "LLM matching failed after %d attempts: %s", max_retries + 1, last_error
    )
    fallback = [{"catalog_id": "None", "confidence": "None"}]
    _match_cache[key] = fallback
    return fallback
