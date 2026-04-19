"""Tests for :mod:`src.api_client`.

All HTTP interactions are mocked with ``respx``; no real network requests
are issued by this test suite.
"""

from typing import Any, Dict

import httpx
import pytest
import respx

from src.api_client import OPEN_METEO_URL, CurrentWeather, WeatherClient
from src.exceptions import APIError, LocationNotFoundError


@pytest.fixture
def sample_payload() -> Dict[str, Any]:
    """A representative Open-Meteo ``current_weather`` response body."""
    return {
        "latitude": 52.52,
        "longitude": 13.42,
        "generationtime_ms": 0.2,
        "utc_offset_seconds": 7200,
        "timezone": "Europe/Berlin",
        "timezone_abbreviation": "CEST",
        "elevation": 38.0,
        "current_weather": {
            "temperature": 13.4,
            "windspeed": 10.6,
            "winddirection": 292.0,
            "weathercode": 3,
            "time": "2023-05-18T12:00",
        },
    }


@pytest.mark.asyncio
@respx.mock
async def test_fetch_current_weather_returns_parsed_result(
    sample_payload: Dict[str, Any],
) -> None:
    route = respx.get(OPEN_METEO_URL).mock(
        return_value=httpx.Response(200, json=sample_payload)
    )

    async with WeatherClient() as client:
        result = await client.fetch_current_weather(52.52, 13.42)

    assert route.called
    request = route.calls.last.request
    assert request.url.params["latitude"] == "52.52"
    assert request.url.params["longitude"] == "13.42"
    assert request.url.params["current_weather"] == "true"

    assert result == CurrentWeather(
        latitude=52.52,
        longitude=13.42,
        temperature=13.4,
        windspeed=10.6,
        winddirection=292.0,
        weathercode=3,
        time="2023-05-18T12:00",
        timezone="Europe/Berlin",
    )


@pytest.mark.asyncio
@respx.mock
async def test_fetch_current_weather_without_context_manager(
    sample_payload: Dict[str, Any],
) -> None:
    """Calling the client outside an ``async with`` block still works."""
    respx.get(OPEN_METEO_URL).mock(
        return_value=httpx.Response(200, json=sample_payload)
    )

    client = WeatherClient()
    result = await client.fetch_current_weather(52.52, 13.42)

    assert result.temperature == 13.4


@pytest.mark.asyncio
@respx.mock
async def test_invalid_coordinates_raise_location_not_found() -> None:
    respx.get(OPEN_METEO_URL).mock(
        return_value=httpx.Response(
            400,
            json={
                "error": True,
                "reason": "Latitude must be in range of -90 to 90.",
            },
        )
    )

    async with WeatherClient() as client:
        with pytest.raises(LocationNotFoundError) as excinfo:
            await client.fetch_current_weather(999.0, 0.0)

    assert "Latitude must be in range" in str(excinfo.value)
    # Must also be catchable as APIError so broad handlers still work.
    assert isinstance(excinfo.value, APIError)
    assert excinfo.value.status_code == 400


@pytest.mark.asyncio
@respx.mock
async def test_invalid_coordinates_without_reason_uses_fallback_message() -> None:
    """HTTP 400 with no ``reason`` field falls back to a generic message."""
    respx.get(OPEN_METEO_URL).mock(
        return_value=httpx.Response(400, json={"error": True}),
    )

    async with WeatherClient() as client:
        with pytest.raises(LocationNotFoundError) as excinfo:
            await client.fetch_current_weather(999.0, 0.0)

    assert "Invalid coordinates for weather lookup" in str(excinfo.value)
    # Must also be catchable as APIError so broad handlers still work.
    assert isinstance(excinfo.value, APIError)
    assert excinfo.value.status_code == 400


@pytest.mark.asyncio
@respx.mock
async def test_server_error_raises_api_error_with_status_code() -> None:
    respx.get(OPEN_METEO_URL).mock(
        return_value=httpx.Response(503, text="Service Unavailable"),
    )

    async with WeatherClient() as client:
        with pytest.raises(APIError) as excinfo:
            await client.fetch_current_weather(52.52, 13.42)

    assert excinfo.value.status_code == 503
    assert "503" in str(excinfo.value)


@pytest.mark.asyncio
@respx.mock
async def test_network_error_is_wrapped_as_api_error() -> None:
    respx.get(OPEN_METEO_URL).mock(side_effect=httpx.ConnectError("boom"))

    async with WeatherClient() as client:
        with pytest.raises(APIError) as excinfo:
            await client.fetch_current_weather(52.52, 13.42)

    assert "Network error" in str(excinfo.value)
    assert excinfo.value.status_code is None
    # Preserve the original exception chain for debuggability.
    assert isinstance(excinfo.value.__cause__, httpx.ConnectError)


@pytest.mark.asyncio
@respx.mock
async def test_malformed_json_body_raises_api_error() -> None:
    respx.get(OPEN_METEO_URL).mock(
        return_value=httpx.Response(200, text="<html>not json</html>")
    )

    async with WeatherClient() as client:
        with pytest.raises(APIError, match="invalid JSON"):
            await client.fetch_current_weather(52.52, 13.42)


@pytest.mark.asyncio
@respx.mock
async def test_missing_current_weather_section_raises_api_error() -> None:
    respx.get(OPEN_METEO_URL).mock(
        return_value=httpx.Response(
            200, json={"latitude": 1.0, "longitude": 2.0, "timezone": "GMT"}
        )
    )

    async with WeatherClient() as client:
        with pytest.raises(APIError, match="current_weather"):
            await client.fetch_current_weather(1.0, 2.0)


@pytest.mark.asyncio
@respx.mock
async def test_malformed_current_weather_fields_raise_api_error() -> None:
    """Payload has ``current_weather`` but fields are missing/invalid."""
    respx.get(OPEN_METEO_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "latitude": 52.52,
                "longitude": 13.42,
                "timezone": "GMT",
                "current_weather": {
                    "temperature": "not-a-number",
                    "windspeed": 1.0,
                    "winddirection": 1.0,
                    "weathercode": 1,
                    "time": "2023-01-01T00:00",
                },
            },
        )
    )

    async with WeatherClient() as client:
        with pytest.raises(APIError, match="malformed field"):
            await client.fetch_current_weather(52.52, 13.42)
