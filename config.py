"""
Configuration and environment setup for the WCM Setlist Agent.

Centralizes paths, constants, logging configuration, and startup
validation so that problems are caught early and reported clearly.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = BASE_DIR / "logs"

# ── API ───────────────────────────────────────────────────
DEFAULT_API_URL = (
    "https://gist.githubusercontent.com/neilayengia/"
    "2fda9f8f8baef4a2562abc86042b7f91/raw/tour_setlist.json"
)

# ── Matching constants ────────────────────────────────────
VALID_CONFIDENCE = {"Exact", "High", "Review", "None"}

# ── Retry / backoff ──────────────────────────────────────
MAX_RETRIES = 3
BACKOFF_BASE = 2          # seconds – exponential base
BACKOFF_MAX = 30          # seconds – cap per retry

# ── LLM ──────────────────────────────────────────────────
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.0

# ── Rate limiting ────────────────────────────────────────
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", "0.1"))  # seconds between LLM calls

# ── Logging ───────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "false").lower() == "true"
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024   # 10 MB
LOG_FILE_BACKUP_COUNT = 5


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure root logger with console output and optional file rotation.

    Set LOG_TO_FILE=true in .env to enable file logging to logs/agent.log
    with automatic rotation at 10 MB (5 backups kept).
    """
    handlers = [logging.StreamHandler(sys.stdout)]

    if LOG_TO_FILE:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_DIR / "agent.log",
            maxBytes=LOG_FILE_MAX_BYTES,
            backupCount=LOG_FILE_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=handlers,
    )


class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""


def validate_config() -> None:
    """
    Check that all required files and env vars exist.

    Raises ConfigurationError with a clear message if anything
    is missing, so the pipeline fails fast instead of halfway
    through processing.
    """
    errors: List[str] = []

    # Required data files
    catalog_path = DATA_DIR / "catalog.csv"
    if not catalog_path.exists():
        errors.append(f"Catalog file not found: {catalog_path}")

    # API key (warn, don't fail – deterministic mode still works)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.getLogger(__name__).warning(
            "OPENAI_API_KEY not set — LLM matching will be disabled. "
            "Only deterministic matches will be attempted."
        )

    if errors:
        raise ConfigurationError(
            "Configuration errors:\n  • " + "\n  • ".join(errors)
        )
