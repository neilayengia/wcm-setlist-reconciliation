"""
Data ingestion: fetch tour setlists from API and load the song catalog.

All I/O and external calls live here so the matching logic stays pure.
"""

import csv
import json
import logging
from typing import Dict, List, Set

import requests as http_requests

from config import DATA_DIR, DEFAULT_API_URL

logger = logging.getLogger(__name__)


# ── Tour data ─────────────────────────────────────────────

def fetch_tour_data(api_url: str = DEFAULT_API_URL) -> dict:
    """
    Fetch tour setlist data from the API endpoint.

    Falls back to the local JSON file if the API is unavailable.
    Validates the response structure before returning.

    Raises:
        FileNotFoundError: If both the API and local file are unavailable.
        ValueError: If the data doesn't contain the expected structure.
    """
    data = None

    if api_url:
        try:
            logger.info("Fetching tour data from API: %s", api_url)
            response = http_requests.get(api_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            logger.info("API response status: %s", data.get("status", "unknown"))
        except Exception as exc:
            logger.warning("API request failed: %s — falling back to local file", exc)

    # Fallback to local file
    if data is None:
        local_path = DATA_DIR / "tour_setlist.json"
        if not local_path.exists():
            raise FileNotFoundError(
                f"Tour data unavailable: API failed and local file not found at {local_path}"
            )
        logger.info("Loading tour data from local file: %s", local_path)
        with open(local_path, "r") as f:
            data = json.load(f)

    # Validate structure
    _validate_tour_data(data)
    return data


def _validate_tour_data(data: dict) -> None:
    """Ensure the tour JSON has the expected nested structure."""
    if not isinstance(data, dict):
        raise ValueError("Tour data must be a JSON object")
    if "data" not in data:
        raise ValueError("Tour data missing top-level 'data' key")
    if "shows" not in data["data"]:
        raise ValueError("Tour data missing 'data.shows' key")
    shows = data["data"]["shows"]
    if not isinstance(shows, list) or not shows:
        raise ValueError("'data.shows' must be a non-empty list")
    for i, show in enumerate(shows):
        for key in ("date", "venue", "city", "setlist"):
            if key not in show:
                raise ValueError(f"Show {i} missing required key '{key}'")
        if not isinstance(show["setlist"], list):
            raise ValueError(f"Show {i} 'setlist' must be a list")


# ── Catalog ───────────────────────────────────────────────

REQUIRED_CATALOG_COLUMNS: Set[str] = {"catalog_id", "title", "writers", "controlled_percentage"}


def load_catalog() -> List[Dict]:
    """
    Load the internal song catalog from CSV.

    Returns a list of dicts with catalog_id, title, writers,
    controlled_percentage.

    Raises:
        FileNotFoundError: If the catalog CSV does not exist.
        ValueError: If required columns are missing or the file is empty.
    """
    file_path = DATA_DIR / "catalog.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Catalog file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8-sig") as f:
        raw_lines = f.readlines()

    if not raw_lines:
        raise ValueError("Catalog file is empty")

    # Strip outer quotes that wrap each row (non-standard CSV format)
    # e.g. "CAT-001,Neon Dreams,Alex Park,100" → CAT-001,Neon Dreams,Alex Park,100
    cleaned_lines = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue
        cleaned_lines.append(stripped.strip("'\""))

    if not cleaned_lines:
        raise ValueError("Catalog file contains no data rows")

    # Use csv.DictReader for robust parsing (handles commas in field values)
    reader = csv.DictReader(cleaned_lines)
    headers = reader.fieldnames or []

    # Validate required columns
    missing = REQUIRED_CATALOG_COLUMNS - set(headers)
    if missing:
        raise ValueError(f"Catalog CSV missing required columns: {missing}")

    catalog: List[Dict] = []
    for row in reader:
        entry = {k: v.strip() for k, v in row.items()}
        catalog.append(entry)

    logger.info("Loaded catalog: %d songs", len(catalog))
    return catalog


# ── Flatten ───────────────────────────────────────────────

def flatten_setlists(tour_data: Dict) -> List[Dict]:
    """
    Convert nested JSON (shows → setlist arrays) into flat
    track rows ready for matching.

    Each row = one track performed at one show.
    """
    tracks: List[Dict] = []
    shows = tour_data["data"]["shows"]

    for show in shows:
        for track_name in show["setlist"]:
            tracks.append({
                "show_date": show["date"],
                "venue_name": show["venue"],
                "city": show["city"],
                "setlist_track_name": track_name,
            })

    logger.info("Flattened %d track entries across %d shows", len(tracks), len(shows))
    return tracks
