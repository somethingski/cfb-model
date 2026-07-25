"""The disk cache is the idempotency mechanism, so it is tested without a network.

Exit criterion 1 (a second backfill hits the API zero times) and exit criterion 4 (no raw
API key in code, cache filenames, or logs) both reduce to properties of this client.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cfb.ingest.client import CFBDClient, CFBDError, cache_slug

SENTINEL_KEY = "SENTINEL-SECRET-KEY-do-not-leak"


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        return self._payload


class FakeSession:
    """Records every call it receives and replays a queue of responses."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: dict[str, Any], headers: dict[str, str], timeout: int):
        self.calls.append({"url": url, "params": params, "headers": headers, "timeout": timeout})
        if not self._responses:
            raise AssertionError("FakeSession ran out of responses: an extra network call")
        return self._responses.pop(0)


@pytest.fixture
def api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("CFBD_API_KEY", SENTINEL_KEY)
    return SENTINEL_KEY


def make_client(tmp_path: Path, responses: list[FakeResponse]) -> tuple[CFBDClient, FakeSession]:
    session = FakeSession(responses)
    client = CFBDClient(cache_dir=tmp_path, session=session, sleep=lambda _: None)
    return client, session


def test_second_identical_request_never_touches_the_network(tmp_path: Path, api_key: str) -> None:
    client, session = make_client(tmp_path, [FakeResponse(200, [{"id": 1}])])

    first = client.get("/games", year=2023)
    second = client.get("/games", year=2023)

    assert first == second == [{"id": 1}]
    assert client.network_calls == 1
    assert client.cache_hits == 1
    assert len(session.calls) == 1


def test_different_params_are_cached_separately(tmp_path: Path, api_key: str) -> None:
    client, _ = make_client(
        tmp_path, [FakeResponse(200, [{"id": 1}]), FakeResponse(200, [{"id": 2}])]
    )

    assert client.get("/games", year=2023) == [{"id": 1}]
    assert client.get("/games", year=2024) == [{"id": 2}]
    assert client.network_calls == 2
    assert client.cache_hits == 0


def test_cached_run_needs_no_api_key(
    tmp_path: Path, api_key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fully-cached rerun must work even with no credentials present."""
    client, _ = make_client(tmp_path, [FakeResponse(200, [{"id": 1}])])
    client.get("/games", year=2023)

    monkeypatch.setenv("CFBD_API_KEY", "")
    fresh = CFBDClient(cache_dir=tmp_path, session=FakeSession([]), sleep=lambda _: None)
    assert fresh.get("/games", year=2023) == [{"id": 1}]
    assert fresh.network_calls == 0


def test_api_key_travels_in_the_header_and_nowhere_else(tmp_path: Path, api_key: str) -> None:
    client, session = make_client(tmp_path, [FakeResponse(200, [{"id": 1}])])
    client.get("/games/teams", year=2023, week=1)

    call = session.calls[0]
    assert call["headers"]["Authorization"] == f"Bearer {SENTINEL_KEY}"
    assert SENTINEL_KEY not in call["url"]
    assert SENTINEL_KEY not in json.dumps(call["params"])

    written = list(tmp_path.rglob("*.json"))
    assert written, "expected the response to be cached"
    for path in written:
        assert SENTINEL_KEY not in str(path)
        assert SENTINEL_KEY not in path.read_text()


def test_cache_slug_is_readable_and_order_independent() -> None:
    assert cache_slug({"year": 2023, "week": 1}) == "week=1__year=2023"
    assert cache_slug({"week": 1, "year": 2023}) == "week=1__year=2023"
    assert cache_slug({}) == "all"


def test_retries_then_succeeds_on_a_retryable_status(tmp_path: Path, api_key: str) -> None:
    slept: list[float] = []
    session = FakeSession(
        [FakeResponse(429, None), FakeResponse(503, None), FakeResponse(200, [{"id": 7}])]
    )
    # Throttling off, so the recorded sleeps are the backoff schedule and nothing else.
    client = CFBDClient(cache_dir=tmp_path, session=session, sleep=slept.append, min_interval_s=0)

    assert client.get("/lines", year=2023) == [{"id": 7}]
    assert client.network_calls == 3
    assert slept == [1.0, 2.0]


def test_network_calls_are_throttled(tmp_path: Path, api_key: str) -> None:
    slept: list[float] = []
    client = CFBDClient(
        cache_dir=tmp_path,
        session=FakeSession([FakeResponse(200, []), FakeResponse(200, [])]),
        sleep=slept.append,
        min_interval_s=5.0,
    )

    client.get("/games", year=2023)
    client.get("/games", year=2024)

    assert len(slept) == 1, "the first call should not wait, the second should"
    assert 4.9 < slept[0] <= 5.0


def test_gives_up_after_max_retries(tmp_path: Path, api_key: str) -> None:
    session = FakeSession([FakeResponse(503, None) for _ in range(4)])
    client = CFBDClient(cache_dir=tmp_path, session=session, sleep=lambda _: None, max_retries=3)

    with pytest.raises(CFBDError, match="503"):
        client.get("/lines", year=2023)
    assert not list(tmp_path.rglob("*.json")), "a failed request must not be cached"


def test_validation_error_in_a_200_body_raises_and_is_not_cached(
    tmp_path: Path, api_key: str
) -> None:
    """CFBD reports bad params as a 200 with a message body; caching that poisons the run."""
    client, _ = make_client(
        tmp_path, [FakeResponse(200, {"message": "Validation Failed", "details": {}})]
    )

    with pytest.raises(CFBDError, match="Validation Failed"):
        client.get("/games/teams", year=2023)
    assert not list(tmp_path.rglob("*.json"))


def test_bad_credentials_fail_immediately_without_echoing_the_key(
    tmp_path: Path, api_key: str
) -> None:
    client, _ = make_client(tmp_path, [FakeResponse(401, None)])

    with pytest.raises(CFBDError) as excinfo:
        client.get("/games", year=2023)
    assert SENTINEL_KEY not in str(excinfo.value)
    assert client.network_calls == 1


def test_non_retryable_status_raises(tmp_path: Path, api_key: str) -> None:
    client, _ = make_client(tmp_path, [FakeResponse(404, None)])
    with pytest.raises(CFBDError, match="404"):
        client.get("/nope", year=2023)
