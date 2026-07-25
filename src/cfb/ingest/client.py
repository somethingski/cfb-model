"""Throttled, cached HTTP client for the CollegeFootballData API.

The on-disk cache is the idempotency mechanism for the whole ingestion phase: every
successful response is written to ``cache/{endpoint}/{params}.json`` and read back before
any network call, so a second backfill run reproduces the first without touching the API.

The API key is read from the environment, sent only in an ``Authorization`` header, and
never written to a cache path, a log line, or an exception message.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

from cfb import config

log = logging.getLogger(__name__)

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._=-]+")


class CFBDError(RuntimeError):
    """Raised when the API cannot satisfy a request after retries."""


def cache_slug(params: dict[str, Any]) -> str:
    """Build a deterministic, human-readable cache filename stem from query params.

    Readable over hashed: when a season's row count looks wrong, the first question is
    always "which request produced this?", and ``year=2020__week=3.json`` answers it
    without a lookup table. Params are ints and short lowercase strings, so collisions
    are not a practical concern and nothing sensitive can reach the filename — the API
    key is never a query param.

    Args:
        params: Query parameters for the request.

    Returns:
        A filesystem-safe stem, or ``"all"`` when there are no parameters.
    """
    if not params:
        return "all"
    parts = [f"{key}={params[key]}" for key in sorted(params)]
    return _SAFE_CHARS.sub("-", "__".join(parts))


class CFBDClient:
    """Fetches CFBD endpoints through a disk cache, a throttle, and a retry policy.

    Attributes:
        network_calls: Count of requests actually sent over the network this run. Exit
            criterion 1 requires this to be zero on a second, fully-cached backfill, so
            it is exposed rather than merely logged.
        cache_hits: Count of requests served from disk this run.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        base_url: str = config.CFBD_BASE_URL,
        min_interval_s: float = 1.0,
        max_retries: int = 5,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Initialise the client.

        Args:
            cache_dir: Root of the response cache. Defaults to ``config.CACHE_DIR``.
            base_url: API base URL.
            min_interval_s: Minimum seconds between two network calls (~1 req/sec).
            max_retries: Attempts after the first before giving up on a retryable status.
            session: Injected for testing; a real ``requests.Session`` by default.
            sleep: Injected for testing so throttling costs nothing in unit tests.
        """
        self.cache_dir = cache_dir if cache_dir is not None else config.CACHE_DIR
        self.base_url = base_url.rstrip("/")
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self._session = session if session is not None else requests.Session()
        self._sleep = sleep
        self._api_key: str | None = None
        self._last_call_at: float | None = None
        self.network_calls = 0
        self.cache_hits = 0

    def cache_path(self, endpoint: str, params: dict[str, Any]) -> Path:
        """Return the cache file for an endpoint/params pair.

        Args:
            endpoint: API path such as ``"/games/teams"``.
            params: Query parameters.

        Returns:
            Path under ``cache_dir``; nested endpoints flatten to one directory
            (``/games/teams`` -> ``games_teams``).
        """
        folder = endpoint.strip("/").replace("/", "_")
        return self.cache_dir / folder / f"{cache_slug(params)}.json"

    def get(self, endpoint: str, **params: Any) -> Any:
        """Fetch an endpoint, serving from cache when possible.

        Args:
            endpoint: API path such as ``"/games"``.
            **params: Query parameters. ``None`` values are dropped.

        Returns:
            The decoded JSON payload.

        Raises:
            CFBDError: On a non-retryable status, on exhausted retries, or when the API
                returns a validation error in an otherwise-200 body.
        """
        params = {key: value for key, value in params.items() if value is not None}
        path = self.cache_path(endpoint, params)
        if path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text())

        payload = self._fetch(endpoint, params)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write via a temp file so an interrupted run cannot leave a truncated cache
        # entry that a later run would happily treat as a complete response.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(path)
        return payload

    def _fetch(self, endpoint: str, params: dict[str, Any]) -> Any:
        """Send the request with throttling and exponential backoff."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        last_status: int | None = None

        for attempt in range(self.max_retries + 1):
            self._throttle()
            self.network_calls += 1
            response = self._session.get(url, params=params, headers=self._headers(), timeout=60)
            status = response.status_code

            if status == 200:
                return self._decode(response, endpoint, params)
            if status in (401, 403):
                raise CFBDError(
                    f"CFBD rejected the credentials for {endpoint} (HTTP {status}). "
                    "Check CFBD_API_KEY in .env."
                )
            if status not in RETRY_STATUS:
                raise CFBDError(f"CFBD returned HTTP {status} for {endpoint} {params}")

            last_status = status
            if attempt < self.max_retries:
                backoff = 2.0**attempt
                log.warning(
                    "HTTP %s on %s %s; retrying in %.0fs (attempt %d/%d)",
                    status,
                    endpoint,
                    params,
                    backoff,
                    attempt + 1,
                    self.max_retries,
                )
                self._sleep(backoff)

        raise CFBDError(
            f"CFBD returned HTTP {last_status} for {endpoint} {params} after "
            f"{self.max_retries} retries; stopping rather than ingesting partial data."
        )

    @staticmethod
    def _decode(response: requests.Response, endpoint: str, params: dict[str, Any]) -> Any:
        """Decode a 200 body, treating an embedded validation error as a failure.

        CFBD reports bad query parameters as a 200 with a ``message`` body. Caching that
        would poison the run with an empty result that looks like a legitimate answer.
        """
        payload = response.json()
        if isinstance(payload, dict) and "message" in payload:
            raise CFBDError(f"CFBD rejected {endpoint} {params}: {payload['message']}")
        return payload

    def _headers(self) -> dict[str, str]:
        """Build request headers, resolving the API key on first network use.

        Resolved lazily so that a fully-cached run — and every unit test — works without
        a key present.
        """
        if self._api_key is None:
            self._api_key = config.get_api_key()
        return {"Authorization": f"Bearer {self._api_key}", "Accept": "application/json"}

    def _throttle(self) -> None:
        """Sleep as needed to keep at least ``min_interval_s`` between network calls."""
        if self._last_call_at is not None:
            elapsed = time.monotonic() - self._last_call_at
            if elapsed < self.min_interval_s:
                self._sleep(self.min_interval_s - elapsed)
        self._last_call_at = time.monotonic()
