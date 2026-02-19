"""
Deterministic matching logic — pure functions with no I/O.

Stage 1 of the two-stage pipeline: free, instant, and 100% reliable.
Handles exact matches (case-insensitive) and suffix-stripped matches.
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from config import VALID_CONFIDENCE

logger = logging.getLogger(__name__)


def normalize(text: str) -> str:
    """
    Normalize a string for comparison.

    Strips common live performance suffixes like (Acoustic),
    (Extended Jam), (Live), etc.  Then lowercases and collapses
    whitespace.
    """
    # Remove parenthetical suffixes
    text = re.sub(r'\s*\(.*?\)\s*', ' ', text)
    # Lowercase + strip
    text = text.lower().strip()
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text


def deterministic_match(
    track_name: str,
    catalog: List[Dict],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to match a setlist track to the catalog using code logic.

    Matching stages:
      1a. Exact match (case-insensitive)
      1b. Normalized match (strip suffixes like "(Acoustic)")

    Returns (catalog_id, confidence) or (None, None) if no match.
    """
    normalized_track = normalize(track_name)

    for song in catalog:
        # Stage 1a: Exact match (case-insensitive)
        if track_name.lower().strip() == song["title"].lower().strip():
            return song["catalog_id"], "Exact"

        # Stage 1b: Normalized match (strip suffixes)
        if normalized_track == normalize(song["title"]):
            return song["catalog_id"], "Exact"

    return None, None


def validate_match(
    match: Dict,
    catalog_ids: Set[str],
) -> Dict:
    """
    Validate a single match result from the LLM.

    Ensures catalog_id exists in our catalog (or is "None")
    and confidence is a valid level.  Corrects invalid values
    rather than silently passing them through.
    """

    catalog_id = str(match.get("catalog_id", "None")).strip()
    confidence = str(match.get("confidence", "None")).strip()

    # Validate catalog_id
    if catalog_id != "None" and catalog_id not in catalog_ids:
        logger.warning(
            "LLM returned invalid catalog_id '%s', setting to None", catalog_id
        )
        catalog_id = "None"
        confidence = "None"

    # Validate confidence level
    if confidence not in VALID_CONFIDENCE:
        logger.warning(
            "LLM returned invalid confidence '%s', setting to Review", confidence
        )
        confidence = "Review"

    return {"catalog_id": catalog_id, "confidence": confidence}
