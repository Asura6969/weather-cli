"""Tests for :mod:`src.cache` and its integration with :mod:`src.api_client`.

All filesystem writes are confined to pytest's ``tmp_path`` fixture (a
per-test temporary directory under the pytest tmp root) so the host
system's real user directories — in particular ``~/.cache`` — are never
touched. Every test constructs :class:`WeatherCache` with an explicit
``cache_dir=tmp_path`` for this reason.

All HTTP interactions continue to be mocked with ``respx``.
"""

from __future__ import annotations

import json
import logging
import os
import stat
from pathlib import Path
from typing import Any, Dict, Iterator

import httpx
import pytest
import respx

from src.api_client import (
    OPEN_METEO_GEOCODING_URL,
    OPEN_METEO_URL,
    CurrentWeather,
    Location,
    WeatherClient,
)
from src.cache import DEFAULT_TTL_SECONDS, WeatherCache, default_cache_dir
from src.exceptions import CacheError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_location() -> Location:
    return Location(
        latitude=52.52437,
        longitude=13.41053,
        name="Berlin, Land Berlin, Germany",
    )


@pytest.fixture
def sample_weather() -> CurrentWeather:
    return CurrentWeather(
        latitude=52.52,
        longitude=13.42,
        temperature=13.4,
        windspeed=10.6,
        winddirection=292.0,
        weathercode=3,
        time="2023-05-18T12:00",
        timezone="Europe/Berlin",
    )


@pytest.fixture
def geocoding_payload() -> Dict[str, Any]:
    return {
        "results": [
            {
                "id": 2950159,
                "name": "Berlin",
                "latitude": 52.52437,
                "longitude": 13.41053,
                "country": "Germany",
                "admin1": "Land Berlin",
                "timezone": "Europe/Berlin",
            }
        ],
    }


@pytest.fixture
def weather_payload() -> Dict[str, Any]:
    return {
        "latitude": 52.52,
        "longitude": 13.42,
        "timezone": "Europe/Berlin",
        "current_weather": {
            "temperature": 13.4,
            "windspeed": 10.6,
            "winddirection": 292.0,
            "weathercode": 3,
            "time": "2023-05-18T12:00",
        },
    }


@pytest.fixture(autouse=True)
def _isolate_home(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Sandbox the process' HOME so the production default path is safe.

    Even though every test explicitly passes ``cache_dir=tmp_path``, this
    autouse fixture redirects ``$HOME`` to a throwaway directory — so any
    accidental call to :func:`default_cache_dir` during the suite still
    cannot reach the real user's home directory.
    """
    fake_home: Path = tmp_path_factory.mktemp("fake-home")
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))  # Windows parity
    yield


# ---------------------------------------------------------------------------
# Construction & configuration
# ---------------------------------------------------------------------------


def test_default_ttl_is_fifteen_minutes() -> None:
    assert DEFAULT_TTL_SECONDS == 15 * 60


def test_cache_rejects_non_positive_ttl(tmp_path: Path) -> None:
    with pytest.raises(CacheError):
        WeatherCache(cache_dir=tmp_path, ttl_seconds=0)
    with pytest.raises(CacheError):
        WeatherCache(cache_dir=tmp_path, ttl_seconds=-5)


def test_default_cache_dir_is_under_user_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The production default resolves relative to ``~`` — but only the
    sandboxed home set by the autouse fixture, never the real one."""
    monkeypatch.setenv("HOME", str(tmp_path))
    assert default_cache_dir() == tmp_path / ".cache" / "weather-cli"


# ---------------------------------------------------------------------------
# Storage & retrieval round-trip
# ---------------------------------------------------------------------------


def test_set_then_get_returns_stored_pair(
    tmp_path: Path,
    sample_location: Location,
    sample_weather: CurrentWeather,
) -> None:
    cache = WeatherCache(cache_dir=tmp_path)
    cache.set("Berlin", sample_location, sample_weather)

    hit = cache.get("Berlin")
    assert hit is not None
    location, weather = hit
    assert location == sample_location
    assert weather == sample_weather


def test_get_returns_none_for_unknown_city(tmp_path: Path) -> None:
    cache = WeatherCache(cache_dir=tmp_path)
    assert cache.get("Atlantis") is None


def test_get_is_case_insensitive_and_whitespace_tolerant(
    tmp_path: Path,
    sample_location: Location,
    sample_weather: CurrentWeather,
) -> None:
    cache = WeatherCache(cache_dir=tmp_path)
    cache.set("Berlin", sample_location, sample_weather)

    assert cache.get("  BERLIN  ") is not None
    assert cache.get("berlin") is not None


def test_empty_city_is_a_noop(
    tmp_path: Path, sample_location: Location, sample_weather: CurrentWeather
) -> None:
    cache = WeatherCache(cache_dir=tmp_path)
    cache.set("   ", sample_location, sample_weather)
    assert cache.get("   ") is None
    # No files were created for empty keys.
    assert list(tmp_path.iterdir()) == []


def test_invalidate_removes_entry(
    tmp_path: Path,
    sample_location: Location,
    sample_weather: CurrentWeather,
) -> None:
    cache = WeatherCache(cache_dir=tmp_path)
    cache.set("Berlin", sample_location, sample_weather)
    assert cache.get("Berlin") is not None

    cache.invalidate("Berlin")
    assert cache.get("Berlin") is None


# ---------------------------------------------------------------------------
# TTL / expiry
# ---------------------------------------------------------------------------


def test_entry_within_ttl_is_fresh(
    tmp_path: Path,
    sample_location: Location,
    sample_weather: CurrentWeather,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = WeatherCache(cache_dir=tmp_path, ttl_seconds=15 * 60)
    base: float = 1_700_000_000.0
    monkeypatch.setattr("src.cache.time.time", lambda: base)
    cache.set("Berlin", sample_location, sample_weather)

    # 14 minutes 59 seconds later — still fresh.
    monkeypatch.setattr("src.cache.time.time", lambda: base + 14 * 60 + 59)
    assert cache.get("Berlin") is not None


def test_entry_exactly_at_ttl_boundary_is_fresh(
    tmp_path: Path,
    sample_location: Location,
    sample_weather: CurrentWeather,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``age == ttl`` counts as fresh; only strictly older entries expire."""
    cache = WeatherCache(cache_dir=tmp_path, ttl_seconds=15 * 60)
    base: float = 1_700_000_000.0
    monkeypatch.setattr("src.cache.time.time", lambda: base)
    cache.set("Berlin", sample_location, sample_weather)

    monkeypatch.setattr("src.cache.time.time", lambda: base + 15 * 60)
    assert cache.get("Berlin") is not None


def test_entry_past_ttl_is_expired(
    tmp_path: Path,
    sample_location: Location,
    sample_weather: CurrentWeather,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = WeatherCache(cache_dir=tmp_path, ttl_seconds=15 * 60)
    base: float = 1_700_000_000.0
    monkeypatch.setattr("src.cache.time.time", lambda: base)
    cache.set("Berlin", sample_location, sample_weather)

    # 15 minutes and 1 second — expired.
    monkeypatch.setattr("src.cache.time.time", lambda: base + 15 * 60 + 1)
    assert cache.get("Berlin") is None


# ---------------------------------------------------------------------------
# Edge cases: corruption, permissions, odd filesystem state
# ---------------------------------------------------------------------------


def test_corrupted_json_payload_is_treated_as_miss(
    tmp_path: Path,
    sample_location: Location,
    sample_weather: CurrentWeather,
) -> None:
    cache = WeatherCache(cache_dir=tmp_path)
    cache.set("Berlin", sample_location, sample_weather)

    # Overwrite the persisted file with non-JSON garbage.
    entry: Path = next(tmp_path.glob("*.json"))
    entry.write_text("<not-json>", encoding="utf-8")

    assert cache.get("Berlin") is None
    # Corrupted entries are pruned so we don't re-pay the parse cost.
    assert not entry.exists()


def test_payload_missing_required_field_is_treated_as_miss(
    tmp_path: Path,
    sample_location: Location,
    sample_weather: CurrentWeather,
) -> None:
    cache = WeatherCache(cache_dir=tmp_path)
    cache.set("Berlin", sample_location, sample_weather)

    entry: Path = next(tmp_path.glob("*.json"))
    data = json.loads(entry.read_text(encoding="utf-8"))
    del data["weather"]["temperature"]
    entry.write_text(json.dumps(data), encoding="utf-8")

    assert cache.get("Berlin") is None


def test_payload_with_non_dict_root_is_treated_as_miss(
    tmp_path: Path,
    sample_location: Location,
    sample_weather: CurrentWeather,
) -> None:
    cache = WeatherCache(cache_dir=tmp_path)
    cache.set("Berlin", sample_location, sample_weather)

    entry: Path = next(tmp_path.glob("*.json"))
    entry.write_text(json.dumps(["unexpected", "list"]), encoding="utf-8")
    assert cache.get("Berlin") is None


@pytest.mark.skipif(os.name != "posix", reason="chmod semantics are POSIX-only")
@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses file permissions")
def test_permission_denied_on_read_is_treated_as_miss(
    tmp_path: Path,
    sample_location: Location,
    sample_weather: CurrentWeather,
) -> None:
    cache = WeatherCache(cache_dir=tmp_path)
    cache.set("Berlin", sample_location, sample_weather)
    entry: Path = next(tmp_path.glob("*.json"))

    original_mode: int = entry.stat().st_mode
    os.chmod(entry, 0)
    try:
        assert cache.get("Berlin") is None
    finally:
        os.chmod(entry, stat.S_IMODE(original_mode))


@pytest.mark.skipif(os.name != "posix", reason="chmod semantics are POSIX-only")
@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses file permissions")
def test_permission_denied_on_write_is_silent(
    tmp_path: Path,
    sample_location: Location,
    sample_weather: CurrentWeather,
) -> None:
    """A read-only cache directory must not raise from ``set()``."""
    read_only: Path = tmp_path / "ro-cache"
    read_only.mkdir()
    original_mode: int = read_only.stat().st_mode
    os.chmod(read_only, stat.S_IRUSR | stat.S_IXUSR)  # no write bit
    try:
        cache = WeatherCache(cache_dir=read_only)
        # Should not raise — caching is best-effort.
        cache.set("Berlin", sample_location, sample_weather)
        assert cache.get("Berlin") is None
    finally:
        os.chmod(read_only, stat.S_IMODE(original_mode))


def test_set_when_cache_dir_missing_creates_it(
    tmp_path: Path,
    sample_location: Location,
    sample_weather: CurrentWeather,
) -> None:
    nested: Path = tmp_path / "does" / "not" / "exist" / "yet"
    cache = WeatherCache(cache_dir=nested)
    cache.set("Berlin", sample_location, sample_weather)
    assert nested.is_dir()
    assert cache.get("Berlin") is not None


# ---------------------------------------------------------------------------
# Integration with WeatherClient.fetch_weather_for_city
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_cache_miss_triggers_network_and_populates_cache(
    tmp_path: Path,
    geocoding_payload: Dict[str, Any],
    weather_payload: Dict[str, Any],
) -> None:
    geo_route = respx.get(OPEN_METEO_GEOCODING_URL).mock(
        return_value=httpx.Response(200, json=geocoding_payload)
    )
    wx_route = respx.get(OPEN_METEO_URL).mock(
        return_value=httpx.Response(200, json=weather_payload)
    )

    cache = WeatherCache(cache_dir=tmp_path)
    async with WeatherClient(cache=cache) as client:
        location, weather = await client.fetch_weather_for_city("Berlin")

    assert geo_route.call_count == 1
    assert wx_route.call_count == 1
    assert location.name == "Berlin, Land Berlin, Germany"
    assert weather.temperature == 13.4
    # Written to cache for next time.
    assert cache.get("Berlin") == (location, weather)


@pytest.mark.asyncio
@respx.mock
async def test_cache_hit_bypasses_all_http_requests(
    tmp_path: Path,
    sample_location: Location,
    sample_weather: CurrentWeather,
) -> None:
    geo_route = respx.get(OPEN_METEO_GEOCODING_URL).mock(
        return_value=httpx.Response(500, text="should not be called")
    )
    wx_route = respx.get(OPEN_METEO_URL).mock(
        return_value=httpx.Response(500, text="should not be called")
    )

    cache = WeatherCache(cache_dir=tmp_path)
    cache.set("Berlin", sample_location, sample_weather)

    async with WeatherClient(cache=cache) as client:
        location, weather = await client.fetch_weather_for_city("Berlin")

    assert not geo_route.called
    assert not wx_route.called
    assert location == sample_location
    assert weather == sample_weather


@pytest.mark.asyncio
@respx.mock
async def test_expired_cache_falls_back_to_network(
    tmp_path: Path,
    sample_location: Location,
    sample_weather: CurrentWeather,
    geocoding_payload: Dict[str, Any],
    weather_payload: Dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = WeatherCache(cache_dir=tmp_path, ttl_seconds=15 * 60)
    base: float = 1_700_000_000.0
    monkeypatch.setattr("src.cache.time.time", lambda: base)
    cache.set("Berlin", sample_location, sample_weather)

    # Jump past the TTL.
    monkeypatch.setattr("src.cache.time.time", lambda: base + 15 * 60 + 1)

    geo_route = respx.get(OPEN_METEO_GEOCODING_URL).mock(
        return_value=httpx.Response(200, json=geocoding_payload)
    )
    wx_route = respx.get(OPEN_METEO_URL).mock(
        return_value=httpx.Response(200, json=weather_payload)
    )

    async with WeatherClient(cache=cache) as client:
        await client.fetch_weather_for_city("Berlin")

    assert geo_route.called
    assert wx_route.called


@pytest.mark.asyncio
@respx.mock
async def test_corrupted_cache_falls_back_to_network(
    tmp_path: Path,
    sample_location: Location,
    sample_weather: CurrentWeather,
    geocoding_payload: Dict[str, Any],
    weather_payload: Dict[str, Any],
) -> None:
    cache = WeatherCache(cache_dir=tmp_path)
    cache.set("Berlin", sample_location, sample_weather)
    entry: Path = next(tmp_path.glob("*.json"))
    entry.write_text("{not valid json", encoding="utf-8")

    geo_route = respx.get(OPEN_METEO_GEOCODING_URL).mock(
        return_value=httpx.Response(200, json=geocoding_payload)
    )
    wx_route = respx.get(OPEN_METEO_URL).mock(
        return_value=httpx.Response(200, json=weather_payload)
    )

    async with WeatherClient(cache=cache) as client:
        await client.fetch_weather_for_city("Berlin")

    assert geo_route.called
    assert wx_route.called


@pytest.mark.asyncio
@respx.mock
async def test_fetch_without_cache_argument_still_works(
    geocoding_payload: Dict[str, Any],
    weather_payload: Dict[str, Any],
) -> None:
    """The cache is an opt-in feature; omitting it must not change behaviour."""
    respx.get(OPEN_METEO_GEOCODING_URL).mock(
        return_value=httpx.Response(200, json=geocoding_payload)
    )
    respx.get(OPEN_METEO_URL).mock(
        return_value=httpx.Response(200, json=weather_payload)
    )

    async with WeatherClient() as client:
        assert client.cache is None
        location, weather = await client.fetch_weather_for_city("Berlin")

    assert location.latitude == pytest.approx(52.52437)
    assert weather.temperature == 13.4


def test_weather_client_cache_defaults_to_none() -> None:
    """Existing callers that never pass ``cache=`` must be unaffected."""
    client = WeatherClient()
    assert client.cache is None


def test_weather_client_cache_exposed_via_property(tmp_path: Path) -> None:
    cache = WeatherCache(cache_dir=tmp_path)
    client = WeatherClient(cache=cache)
    assert client.cache is cache


def test_weather_client_cache_returns_exact_instance(tmp_path: Path) -> None:
    """``WeatherClient.cache`` must return the *same* object (identity, not
    equality) passed to the constructor — no wrapping, copying, or
    substitution is permitted when caching is active."""
    cache = WeatherCache(cache_dir=tmp_path, ttl_seconds=42)
    client = WeatherClient(cache=cache)

    retrieved = client.cache
    assert retrieved is cache, (
        f"Expected client.cache to be the exact same object as the one "
        f"passed in (id={id(cache)}); got id={id(retrieved)}"
    )
    # The instance-level configuration must come through untouched.
    assert retrieved is not None
    assert retrieved.cache_dir == tmp_path
    assert retrieved.ttl_seconds == 42


# ---------------------------------------------------------------------------
# Diagnostic logging
# ---------------------------------------------------------------------------


def test_corrupted_entry_emits_debug_log(
    tmp_path: Path,
    sample_location: Location,
    sample_weather: CurrentWeather,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cache = WeatherCache(cache_dir=tmp_path)
    cache.set("Berlin", sample_location, sample_weather)
    entry: Path = next(tmp_path.glob("*.json"))
    entry.write_text("{not-valid-json", encoding="utf-8")

    with caplog.at_level(logging.DEBUG, logger="src.cache"):
        assert cache.get("Berlin") is None

    corrupt_records = [
        r
        for r in caplog.records
        if r.name == "src.cache"
        and r.levelno == logging.DEBUG
        and "corrupted" in r.getMessage().lower()
    ]
    assert corrupt_records, (
        f"Expected a DEBUG log mentioning corruption, got: "
        f"{[r.getMessage() for r in caplog.records]}"
    )
    message: str = corrupt_records[0].getMessage()
    assert "Berlin" in message
    # The underlying exception class must be named so operators can
    # distinguish e.g. JSONDecodeError from KeyError without re-running.
    assert "JSONDecodeError" in message, (
        f"Expected exception class 'JSONDecodeError' in log message, got: "
        f"{message!r}"
    )
    # The filesystem path must NOT appear — only the opaque hash key.
    assert (
        str(tmp_path) not in message
    ), f"Log message leaks the cache directory path: {message!r}"


def test_expired_entry_emits_debug_log(
    tmp_path: Path,
    sample_location: Location,
    sample_weather: CurrentWeather,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = WeatherCache(cache_dir=tmp_path, ttl_seconds=15 * 60)
    base: float = 1_700_000_000.0
    monkeypatch.setattr("src.cache.time.time", lambda: base)
    cache.set("Berlin", sample_location, sample_weather)

    # Advance past the TTL.
    monkeypatch.setattr("src.cache.time.time", lambda: base + 15 * 60 + 1)

    with caplog.at_level(logging.DEBUG, logger="src.cache"):
        assert cache.get("Berlin") is None

    expired_records = [
        r
        for r in caplog.records
        if r.name == "src.cache"
        and r.levelno == logging.DEBUG
        and "expired" in r.getMessage().lower()
    ]
    assert expired_records, (
        f"Expected a DEBUG log mentioning expiration, got: "
        f"{[r.getMessage() for r in caplog.records]}"
    )
    message: str = expired_records[0].getMessage()
    assert "Berlin" in message
    # Expiration is not an exception path, but the log must still carry
    # the actionable diagnostic values: concrete age and the TTL so
    # operators can see how far past the threshold an entry was.
    assert (
        "age=901.0s" in message
    ), f"Expected precise age diagnostic in log message, got: {message!r}"
    assert (
        "ttl=900s" in message
    ), f"Expected ttl diagnostic in log message, got: {message!r}"
    # The filesystem path must NOT appear — only the opaque hash key.
    assert (
        str(tmp_path) not in message
    ), f"Log message leaks the cache directory path: {message!r}"


@pytest.mark.skipif(os.name != "posix", reason="chmod semantics are POSIX-only")
@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses file permissions")
def test_permission_denied_emits_debug_log(
    tmp_path: Path,
    sample_location: Location,
    sample_weather: CurrentWeather,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cache = WeatherCache(cache_dir=tmp_path)
    cache.set("Berlin", sample_location, sample_weather)
    entry: Path = next(tmp_path.glob("*.json"))

    original_mode: int = entry.stat().st_mode
    os.chmod(entry, 0)
    try:
        with caplog.at_level(logging.DEBUG, logger="src.cache"):
            assert cache.get("Berlin") is None
    finally:
        os.chmod(entry, stat.S_IMODE(original_mode))

    denied_records = [
        r
        for r in caplog.records
        if r.name == "src.cache"
        and r.levelno == logging.DEBUG
        and "cache read failed" in r.getMessage().lower()
    ]
    assert denied_records, (
        f"Expected a DEBUG log for permission-denied read, got: "
        f"{[r.getMessage() for r in caplog.records]}"
    )
    message: str = denied_records[0].getMessage()
    assert "Berlin" in message
    # The underlying exception class must be named so operators can
    # distinguish a permission denial from a generic I/O error.
    assert "PermissionError" in message, (
        f"Expected exception class 'PermissionError' in log message, got: "
        f"{message!r}"
    )
    # The filesystem path must NOT appear — only the opaque hash key.
    assert (
        str(tmp_path) not in message
    ), f"Log message leaks the cache directory path: {message!r}"
