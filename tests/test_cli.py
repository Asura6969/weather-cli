"""Tests for :mod:`src.cli`.

All HTTP interactions are mocked with ``respx``; no real network requests
are issued by this test suite. The cache is isolated to a per-test temp
directory so nothing is written under the real home directory.
"""

from pathlib import Path
from typing import Any, Dict, Iterator

import httpx
import pytest
import respx
from rich.console import Console

from src import cli
from src.api_client import (
    OPEN_METEO_GEOCODING_URL,
    OPEN_METEO_URL,
    CurrentWeather,
    Location,
)
from src.exceptions import APIError, CityNotFoundError


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Redirect the default cache dir to a per-test temp directory."""
    monkeypatch.setattr("src.cache.default_cache_dir", lambda: tmp_path / "cache")
    yield tmp_path / "cache"


@pytest.fixture
def geocode_payload() -> Dict[str, Any]:
    return {
        "results": [
            {
                "name": "Berlin",
                "latitude": 52.52,
                "longitude": 13.42,
                "admin1": "Land Berlin",
                "country": "Germany",
            }
        ]
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


@respx.mock
def test_main_success_renders_table_and_returns_zero(
    capsys: pytest.CaptureFixture[str],
    geocode_payload: Dict[str, Any],
    weather_payload: Dict[str, Any],
) -> None:
    geo_route = respx.get(OPEN_METEO_GEOCODING_URL).mock(
        return_value=httpx.Response(200, json=geocode_payload)
    )
    wx_route = respx.get(OPEN_METEO_URL).mock(
        return_value=httpx.Response(200, json=weather_payload)
    )

    exit_code: int = cli.main(["Berlin"])

    assert exit_code == 0
    assert geo_route.called
    assert wx_route.called

    captured = capsys.readouterr()
    assert "Berlin, Land Berlin, Germany" in captured.out
    assert "13.4" in captured.out
    assert "10.6" in captured.out
    assert "Overcast" in captured.out
    # Errors go to stderr, stdout holds the table.
    assert captured.err == ""


@respx.mock
def test_main_city_not_found_renders_error_and_returns_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    respx.get(OPEN_METEO_GEOCODING_URL).mock(
        return_value=httpx.Response(200, json={"results": []})
    )

    exit_code: int = cli.main(["Atlantis"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "City not found" in captured.err
    assert "Atlantis" in captured.err


@respx.mock
def test_main_network_error_renders_error_and_returns_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    respx.get(OPEN_METEO_GEOCODING_URL).mock(
        side_effect=httpx.ConnectError("connection refused")
    )

    exit_code: int = cli.main(["Berlin"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Weather service error" in captured.err


@respx.mock
def test_main_timeout_renders_error_and_returns_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An ``httpx`` timeout must surface as a formatted panel, not a traceback.

    ``httpx.ReadTimeout`` ⊂ ``httpx.TimeoutException`` ⊂ ``httpx.RequestError``,
    which ``WeatherClient._get_json`` wraps in ``APIError`` — so ``main`` should
    catch it and exit ``1`` with a rendered error on stderr.
    """
    respx.get(OPEN_METEO_GEOCODING_URL).mock(
        side_effect=httpx.ReadTimeout("read timed out")
    )

    exit_code: int = cli.main(["Berlin"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Weather service error" in captured.err
    assert "Traceback" not in captured.err


def test_httpx_network_exceptions_are_request_errors() -> None:
    """Guard the assumption that timeout/connect errors are ``RequestError``.

    ``WeatherClient._get_json`` catches ``httpx.RequestError`` to wrap all
    network failures in :class:`~src.exceptions.APIError`. If a future
    ``httpx`` release were to move these out of that hierarchy, timeouts
    would leak as raw tracebacks — this test fails loudly in that case.
    """
    assert issubclass(httpx.TimeoutException, httpx.RequestError)
    assert issubclass(httpx.ConnectTimeout, httpx.RequestError)
    assert issubclass(httpx.ReadTimeout, httpx.RequestError)
    assert issubclass(httpx.ConnectError, httpx.RequestError)


@respx.mock
def test_main_populates_cache_on_miss(
    isolated_cache: Path,
    geocode_payload: Dict[str, Any],
    weather_payload: Dict[str, Any],
) -> None:
    """After a successful run the cache directory contains one entry."""
    respx.get(OPEN_METEO_GEOCODING_URL).mock(
        return_value=httpx.Response(200, json=geocode_payload)
    )
    respx.get(OPEN_METEO_URL).mock(
        return_value=httpx.Response(200, json=weather_payload)
    )

    assert not isolated_cache.exists()
    assert cli.main(["Berlin"]) == 0
    entries = list(isolated_cache.glob("*.json"))
    assert len(entries) == 1


@respx.mock
def test_main_uses_cache_on_second_invocation(
    geocode_payload: Dict[str, Any],
    weather_payload: Dict[str, Any],
) -> None:
    """A second run for the same city must not hit the network."""
    geo_route = respx.get(OPEN_METEO_GEOCODING_URL).mock(
        return_value=httpx.Response(200, json=geocode_payload)
    )
    wx_route = respx.get(OPEN_METEO_URL).mock(
        return_value=httpx.Response(200, json=weather_payload)
    )

    assert cli.main(["Berlin"]) == 0
    assert geo_route.call_count == 1
    assert wx_route.call_count == 1

    assert cli.main(["Berlin"]) == 0
    # Still one call each — the second run was served from cache.
    assert geo_route.call_count == 1
    assert wx_route.call_count == 1


def test_build_parser_requires_city_positional() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    ns = parser.parse_args(["Tokyo"])
    assert ns.city == "Tokyo"


def test_describe_weather_code_known_and_unknown() -> None:
    label, icon = cli._describe_weather_code(0)
    assert label == "Clear sky"
    assert icon

    label, icon = cli._describe_weather_code(-1)
    assert "Unknown" in label
    assert icon


def test_temperature_style_thresholds() -> None:
    assert cli._temperature_style(-5.0) == "bold bright_blue"
    assert cli._temperature_style(5.0) == "cyan"
    assert cli._temperature_style(15.0) == "green"
    assert cli._temperature_style(25.0) == "yellow"
    assert cli._temperature_style(35.0) == "bold red"


def test_wind_style_thresholds() -> None:
    assert cli._wind_style(5.0) == "green"
    assert cli._wind_style(25.0) == "yellow"
    assert cli._wind_style(60.0) == "bold red"


@pytest.mark.parametrize(
    "degrees, expected",
    [
        (0.0, "N"),
        (11.24, "N"),
        (11.26, "NNE"),
        (45.0, "NE"),
        (90.0, "E"),
        (180.0, "S"),
        (270.0, "W"),
        (292.0, "WNW"),
        (348.75, "N"),
        (360.0, "N"),
        (405.0, "NE"),
        (-90.0, "W"),
    ],
)
def test_compass_point(degrees: float, expected: str) -> None:
    assert cli._compass_point(degrees) == expected


def test_render_error_headings() -> None:
    console = Console(record=True, stderr=True, width=120)

    cli._render_error(console, CityNotFoundError("no such city"))
    cli._render_error(console, APIError("boom", status_code=500))

    output: str = console.export_text()
    assert "City not found" in output
    assert "no such city" in output
    assert "Weather service error" in output
    assert "boom" in output


def test_render_weather_contains_all_metrics() -> None:
    console = Console(record=True, width=120)
    location = Location(latitude=52.52, longitude=13.42, name="Berlin, Germany")
    weather = CurrentWeather(
        latitude=52.52,
        longitude=13.42,
        temperature=13.4,
        windspeed=10.6,
        winddirection=292.0,
        weathercode=3,
        time="2023-05-18T12:00",
        timezone="Europe/Berlin",
    )

    cli._render_weather(console, location, weather)
    output: str = console.export_text()

    assert "Berlin, Germany" in output
    assert "13.4" in output
    assert "10.6" in output
    assert "292° WNW" in output
    assert "Overcast" in output
    assert "2023-05-18T12:00" in output
