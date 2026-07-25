"""Single source of truth for paths, season range, and project constants.

No other module hardcodes a path or a season range. Advancing the project to a new
completed season is a one-line change to ``SEASONS``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

# --- Seasons -----------------------------------------------------------------

FIRST_SEASON: int = 2014
LAST_SEASON: int = 2025
"""Last completed season. Bump when the following season finishes."""

SEASONS: range = range(FIRST_SEASON, LAST_SEASON + 1)

# --- Paths -------------------------------------------------------------------

DATA_DIR: Path = PROJECT_ROOT / "data"
CACHE_DIR: Path = PROJECT_ROOT / "cache"
GOLD_DIR: Path = PROJECT_ROOT / "gold"

DB_PATH: Path = DATA_DIR / "cfb.sqlite"
FEATURE_STORE_DIR: Path = DATA_DIR / "features"

# --- External API ------------------------------------------------------------

CFBD_BASE_URL: str = "https://api.collegefootballdata.com"


def get_api_key() -> str:
    """Return the CollegeFootballData API key from the environment.

    Read at call time, not import time, so the package imports cleanly without a key
    (tests, linting, and Phase 0 need no credentials).

    Returns:
        The API key string.

    Raises:
        RuntimeError: If ``CFBD_API_KEY`` is unset or empty.
    """
    key = os.environ.get("CFBD_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "CFBD_API_KEY is not set. Copy .env.example to .env and add your key "
            "from https://collegefootballdata.com/key"
        )
    return key


def ensure_dirs() -> None:
    """Create the gitignored working directories if they do not already exist."""
    for directory in (DATA_DIR, CACHE_DIR, FEATURE_STORE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
