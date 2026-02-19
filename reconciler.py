"""
Pipeline orchestration — wires ingestion, matching, and LLM together.
"""

import logging
import os
from typing import Dict, List, Optional

from openai import OpenAI

from matching import deterministic_match
from llm_matching import llm_fuzzy_match

logger = logging.getLogger(__name__)


def reconcile(
    tracks: List[Dict],
    catalog: List[Dict],
    client: Optional[OpenAI] = None,
) -> List[Dict]:
    """
    Run the full two-stage matching pipeline.

    Stage 1: Deterministic matching (free, instant)
    Stage 2: LLM fuzzy matching (only for unmatched tracks)

    Args:
        tracks: Flat list of track dicts from flatten_setlists().
        catalog: List of catalog song dicts.
        client: OpenAI client (None disables LLM matching).

    Returns:
        List of result dicts ready for CSV output.
    """
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            client = OpenAI(api_key=api_key)
        else:
            logger.warning(
                "OPENAI_API_KEY not set — only deterministic matches will be attempted."
            )

    results: List[Dict] = []
    deterministic_hits = 0
    llm_calls = 0

    for track in tracks:
        track_name = track["setlist_track_name"]

        # Medleys (contains "/") go straight to LLM
        if "/" in track_name:
            if client is None:
                logger.info('[MEDLEY] "%s" → skipped (no API key)', track_name)
                results.append(_result_row(track, "None", "None", catalog))
                continue

            logger.info('[MEDLEY] "%s" → sending to LLM', track_name)
            llm_calls += 1
            matches = llm_fuzzy_match(track_name, catalog, client)
            logger.info(
                "  Medley returned %d matches: %s",
                len(matches),
                [m.get("catalog_id") for m in matches],
            )
            for match in matches:
                results.append(_result_row(
                    track,
                    match.get("catalog_id", "None"),
                    match.get("confidence", "None"),
                    catalog,
                ))
            continue

        # Stage 1: Deterministic match
        catalog_id, confidence = deterministic_match(track_name, catalog)

        if catalog_id:
            deterministic_hits += 1
            logger.info('[EXACT]  "%s" → %s', track_name, catalog_id)
            results.append(_result_row(track, catalog_id, confidence, catalog))
            continue

        # Stage 2: LLM fuzzy match
        if client is None:
            logger.info('[SKIP]   "%s" → no API key', track_name)
            results.append(_result_row(track, "None", "None", catalog))
            continue

        llm_calls += 1
        logger.info('[LLM]    "%s" → sending to LLM', track_name)
        matches = llm_fuzzy_match(track_name, catalog, client)

        for match in matches:
            results.append(_result_row(
                track,
                match.get("catalog_id", "None"),
                match.get("confidence", "None"),
                catalog,
            ))

    logger.info("Match summary: deterministic=%d, llm_calls=%d, total_rows=%d",
                deterministic_hits, llm_calls, len(results))
    return results


def _result_row(
    track: dict,
    catalog_id: str,
    confidence: str,
    catalog: List[Dict],
) -> dict:
    """Build a single result row dict."""
    # Look up matched title for human-readable output
    matched_title = ""
    if catalog_id and catalog_id != "None":
        for song in catalog:
            if song["catalog_id"] == catalog_id:
                matched_title = song["title"]
                break

    return {
        "show_date": track["show_date"],
        "venue_name": track["venue_name"],
        "setlist_track_name": track["setlist_track_name"],
        "matched_catalog_id": catalog_id,
        "matched_catalog_title": matched_title,
        "match_confidence": confidence,
    }
