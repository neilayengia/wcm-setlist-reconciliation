"""
Output module — writes reconciliation results to CSV.
"""

import csv
import logging
from typing import Dict, List

from config import OUTPUT_DIR

logger = logging.getLogger(__name__)

FIELDNAMES = [
    "show_date",
    "venue_name",
    "setlist_track_name",
    "matched_catalog_id",
    "matched_catalog_title",
    "match_confidence",
]


def write_csv(results: List[Dict]) -> str:
    """
    Write the reconciliation results to CSV.

    Returns the path to the output file.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "matched_setlists.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)

    logger.info("Output saved to: %s", output_path)
    return str(output_path)
