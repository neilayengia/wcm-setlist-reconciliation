"""
WCM Setlist API Reconciliation Agent
======================================
Matches messy live tour setlists against our internal song catalog
using a two-stage approach: deterministic pre-processing first,
then LLM-powered fuzzy matching for remaining tracks.

Usage:
    python main.py
"""

import logging
import sys

from config import setup_logging, validate_config, ConfigurationError
from ingestion import fetch_tour_data, load_catalog, flatten_setlists
from reconciler import reconcile
from output import write_csv

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()

    logger.info("=" * 55)
    logger.info("  WCM Setlist Reconciliation Agent")
    logger.info("=" * 55)

    # Validate configuration
    try:
        validate_config()
    except ConfigurationError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    # Step 1: Ingest data
    logger.info("[1] Loading data...")
    try:
        tour_data = fetch_tour_data()
        catalog = load_catalog()
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Data ingestion failed: %s", exc)
        sys.exit(1)

    # Step 2: Flatten setlists
    logger.info("[2] Flattening setlists...")
    tracks = flatten_setlists(tour_data)

    # Step 3: Run reconciliation pipeline
    logger.info("[3] Matching tracks...")
    results = reconcile(tracks, catalog)

    # Step 4: Output CSV
    logger.info("[4] Writing output...")
    write_csv(results)

    # Step 5: Display results
    logger.info("")
    logger.info("=" * 55)
    logger.info("  RESULTS")
    logger.info("=" * 55)
    logger.info("  %-35s %-15s %-25s %s", "Track", "Catalog ID", "Matched Title", "Confidence")
    logger.info("  %s %s %s %s", "-" * 35, "-" * 15, "-" * 25, "-" * 12)
    for r in results:
        logger.info(
            "  %-35s %-15s %-25s %s",
            r["setlist_track_name"],
            r["matched_catalog_id"],
            r.get("matched_catalog_title", ""),
            r["match_confidence"],
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
