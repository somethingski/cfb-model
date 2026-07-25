"""Phase 0 smoke tests: the package imports and its constants are sane.

Leakage-sensitive tests begin in Phase 1; there is no data yet.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cfb import config


def test_season_range_is_the_documented_span() -> None:
    assert config.FIRST_SEASON == 2014
    assert config.LAST_SEASON == 2025
    assert list(config.SEASONS)[0] == 2014
    assert list(config.SEASONS)[-1] == 2025
    assert len(config.SEASONS) == 12


def test_paths_are_constructible_and_under_project_root() -> None:
    for path in (
        config.DATA_DIR,
        config.CACHE_DIR,
        config.GOLD_DIR,
        config.DB_PATH,
        config.FEATURE_STORE_DIR,
    ):
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert config.PROJECT_ROOT in path.parents


def test_project_root_is_the_repo_root() -> None:
    assert (config.PROJECT_ROOT / "pyproject.toml").exists()
    assert (config.PROJECT_ROOT / "CLAUDE.md").exists()


def test_get_api_key_raises_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing key must fail loudly, not return an empty string into a request."""
    monkeypatch.setenv("CFBD_API_KEY", "")
    with pytest.raises(RuntimeError, match="CFBD_API_KEY"):
        config.get_api_key()


def test_get_api_key_returns_stripped_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CFBD_API_KEY", "  test-key  ")
    assert config.get_api_key() == "test-key"
