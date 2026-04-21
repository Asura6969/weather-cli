"""Tests for :mod:`src.api_client`.

All HTTP interactions are mocked with ``respx``; no real network requests
are issued by this test suite.
"""

from typing import Any, Dict

import httpx
import pytest
import respx

from src.api_client import (
    IP_GEOLOCATION_URL,
    OPEN_METEO_GEOCODING_URL,
    OPEN_METEO_URL,
    CurrentWeather,
    Location,
    WeatherClient,
)
from src.exceptions import (
    APIError,
    CityNotFoundError,
    GeolocationError,
    LocationNotFoundError,
)


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


# ---------------------------------------------------------------------------
# Geocoding tests
# ---------------------------------------------------------------------------


@pytest.fixture
def geocoding_payload() -> Dict[str, Any]:
    """A representative Open-Meteo geocoding response for ``Berlin``."""
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
        "generationtime_ms": 0.5,
    }


@pytest.mark.asyncio
@respx.mock
async def test_geocode_city_returns_parsed_location(
    geocoding_payload: Dict[str, Any],
) -> None:
    route = respx.get(OPEN_METEO_GEOCODING_URL).mock(
        return_value=httpx.Response(200, json=geocoding_payload)
    )

    async with WeatherClient() as client:
        location = await client.geocode_city("Berlin")

    assert route.called
    request = route.calls.last.request
    assert request.url.params["name"] == "Berlin"
    assert request.url.params["count"] == "1"

    assert location == Location(
        latitude=52.52437,
        longitude=13.41053,
        name="Berlin, Land Berlin, Germany",
    )
    # display_name is an alias used for terminal rendering.
    assert location.display_name == "Berlin, Land Berlin, Germany"


@pytest.mark.asyncio
@respx.mock
async def test_geocode_city_feeds_into_fetch_current_weather(
    geocoding_payload: Dict[str, Any],
    sample_payload: Dict[str, Any],
) -> None:
    """Coordinates from geocoding can be handed to the weather endpoint."""
    respx.get(OPEN_METEO_GEOCODING_URL).mock(
        return_value=httpx.Response(200, json=geocoding_payload)
    )
    weather_route = respx.get(OPEN_METEO_URL).mock(
        return_value=httpx.Response(200, json=sample_payload)
    )

    async with WeatherClient() as client:
        location = await client.geocode_city("Berlin")
        weather = await client.fetch_current_weather(
            location.latitude, location.longitude
        )

    assert weather_route.called
    assert weather.temperature == 13.4


@pytest.mark.asyncio
@respx.mock
async def test_geocode_city_empty_results_raises_city_not_found() -> None:
    """Zero results from the geocoding API raises ``CityNotFoundError``."""
    respx.get(OPEN_METEO_GEOCODING_URL).mock(
        return_value=httpx.Response(200, json={"generationtime_ms": 0.1}),
    )

    async with WeatherClient() as client:
        with pytest.raises(CityNotFoundError) as excinfo:
            await client.geocode_city("Xyznowhere")

    assert "Xyznowhere" in str(excinfo.value)
    # Must also be catchable as APIError for broad handlers.
    assert isinstance(excinfo.value, APIError)


@pytest.mark.asyncio
@respx.mock
async def test_geocode_city_empty_results_list_raises_city_not_found() -> None:
    """An explicit empty ``results`` list also raises ``CityNotFoundError``."""
    respx.get(OPEN_METEO_GEOCODING_URL).mock(
        return_value=httpx.Response(200, json={"results": []}),
    )

    async with WeatherClient() as client:
        with pytest.raises(CityNotFoundError):
            await client.geocode_city("Nowhereville")


@pytest.mark.asyncio
async def test_geocode_city_empty_name_raises_city_not_found() -> None:
    """An empty or whitespace-only name never touches the network."""
    async with WeatherClient() as client:
        with pytest.raises(CityNotFoundError):
            await client.geocode_city("   ")


@pytest.mark.asyncio
@respx.mock
async def test_geocode_city_malformed_json_raises_api_error() -> None:
    respx.get(OPEN_METEO_GEOCODING_URL).mock(
        return_value=httpx.Response(200, text="<html>not json</html>"),
    )

    async with WeatherClient() as client:
        with pytest.raises(APIError, match="invalid JSON"):
            await client.geocode_city("Berlin")


@pytest.mark.asyncio
@respx.mock
async def test_geocode_city_malformed_result_entry_raises_api_error() -> None:
    """A result entry missing required fields raises ``APIError``."""
    respx.get(OPEN_METEO_GEOCODING_URL).mock(
        return_value=httpx.Response(
            200,
            json={"results": [{"name": "Berlin"}]},  # missing lat/long
        ),
    )

    async with WeatherClient() as client:
        with pytest.raises(APIError, match="missing or malformed field"):
            await client.geocode_city("Berlin")


@pytest.mark.asyncio
@respx.mock
async def test_geocode_city_non_dict_result_entry_raises_api_error() -> None:
    """A ``results`` list whose first entry is not a dict raises ``APIError``."""
    respx.get(OPEN_METEO_GEOCODING_URL).mock(
        return_value=httpx.Response(
            200,
            json={"results": ["not-a-dict"]},
        ),
    )

    async with WeatherClient() as client:
        with pytest.raises(APIError, match="malformed result entry"):
            await client.geocode_city("Berlin")
        # Must not be the narrower CityNotFoundError - results is non-empty.
        # (Separate assertion for clarity; see test above for empty case.)


@pytest.mark.asyncio
@respx.mock
async def test_geocode_city_server_error_raises_api_error() -> None:
    respx.get(OPEN_METEO_GEOCODING_URL).mock(
        return_value=httpx.Response(503, text="Service Unavailable"),
    )

    async with WeatherClient() as client:
        with pytest.raises(APIError) as excinfo:
            await client.geocode_city("Berlin")

    assert excinfo.value.status_code == 503


@pytest.mark.asyncio
@respx.mock
async def test_geocode_city_network_error_is_wrapped_as_api_error() -> None:
    respx.get(OPEN_METEO_GEOCODING_URL).mock(
        side_effect=httpx.ConnectTimeout("timeout"),
    )

    async with WeatherClient() as client:
        with pytest.raises(APIError) as excinfo:
            await client.geocode_city("Berlin")

    assert "Network error" in str(excinfo.value)
    assert excinfo.value.status_code is None
    assert isinstance(excinfo.value.__cause__, httpx.ConnectTimeout)


# ---------------------------------------------------------------------------
# IP geolocation tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_detect_current_city_returns_city_from_payload() -> None:
    route = respx.get(IP_GEOLOCATION_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "ip": "203.0.113.7",
                "city": "Berlin",
                "region": "Land Berlin",
                "country_name": "Germany",
            },
        )
    )

    async with WeatherClient() as client:
        city: str = await client.detect_current_city()

    assert route.called
    assert city == "Berlin"


@pytest.mark.asyncio
@respx.mock
async def test_detect_current_city_strips_whitespace() -> None:
    respx.get(IP_GEOLOCATION_URL).mock(
        return_value=httpx.Response(200, json={"city": "  Berlin  "}),
    )

    async with WeatherClient() as client:
        assert await client.detect_current_city() == "Berlin"


@pytest.mark.asyncio
@respx.mock
async def test_detect_current_city_connect_error_raises_geolocation_error() -> None:
    """A DNS/connect failure surfaces the 'unreachable' wording, not 'timed out'."""
    respx.get(IP_GEOLOCATION_URL).mock(
        side_effect=httpx.ConnectError("connection refused"),
    )

    async with WeatherClient() as client:
        with pytest.raises(GeolocationError) as excinfo:
            await client.detect_current_city()

    message: str = str(excinfo.value)
    assert "Could not reach IP geolocation service" in message
    assert "timed out" not in message
    assert isinstance(excinfo.value.__cause__, httpx.ConnectError)


@pytest.mark.asyncio
@respx.mock
async def test_detect_current_city_timeout_raises_geolocation_error() -> None:
    """Timeouts surface a distinct message from generic connectivity errors."""
    respx.get(IP_GEOLOCATION_URL).mock(
        side_effect=httpx.ReadTimeout("read timed out"),
    )

    async with WeatherClient() as client:
        with pytest.raises(GeolocationError) as excinfo:
            await client.detect_current_city()

    assert isinstance(excinfo.value.__cause__, httpx.ReadTimeout)
    # Timeout wording is specific ("timed out"), not the generic
    # "could not reach" message used for DNS/connect errors.
    message: str = str(excinfo.value)
    assert "timed out" in message
    assert "Could not reach" not in message


@pytest.mark.asyncio
@respx.mock
async def test_detect_current_city_connect_timeout_uses_timeout_wording() -> None:
    """``ConnectTimeout`` is a ``TimeoutException``, so the timeout path wins."""
    respx.get(IP_GEOLOCATION_URL).mock(
        side_effect=httpx.ConnectTimeout("connect timed out"),
    )

    async with WeatherClient() as client:
        with pytest.raises(GeolocationError) as excinfo:
            await client.detect_current_city()

    assert "timed out" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, httpx.ConnectTimeout)


@pytest.mark.asyncio
@respx.mock
async def test_detect_current_city_http_error_raises_geolocation_error() -> None:
    respx.get(IP_GEOLOCATION_URL).mock(
        return_value=httpx.Response(503, text="Service Unavailable"),
    )

    async with WeatherClient() as client:
        with pytest.raises(GeolocationError, match="HTTP 503"):
            await client.detect_current_city()


@pytest.mark.asyncio
@respx.mock
async def test_detect_current_city_invalid_json_raises_geolocation_error() -> None:
    respx.get(IP_GEOLOCATION_URL).mock(
        return_value=httpx.Response(200, text="<html>not json</html>"),
    )

    async with WeatherClient() as client:
        with pytest.raises(GeolocationError, match="invalid JSON"):
            await client.detect_current_city()


@pytest.mark.asyncio
@respx.mock
async def test_detect_current_city_missing_city_field_raises() -> None:
    respx.get(IP_GEOLOCATION_URL).mock(
        return_value=httpx.Response(200, json={"ip": "203.0.113.7"}),
    )

    async with WeatherClient() as client:
        with pytest.raises(GeolocationError, match="did not include a city"):
            await client.detect_current_city()


@pytest.mark.asyncio
@respx.mock
async def test_detect_current_city_empty_city_field_raises() -> None:
    respx.get(IP_GEOLOCATION_URL).mock(
        return_value=httpx.Response(200, json={"city": "   "}),
    )

    async with WeatherClient() as client:
        with pytest.raises(GeolocationError, match="did not include a city"):
            await client.detect_current_city()


@pytest.mark.asyncio
@respx.mock
async def test_detect_current_city_non_dict_payload_raises() -> None:
    respx.get(IP_GEOLOCATION_URL).mock(
        return_value=httpx.Response(200, json=["not", "a", "dict"]),
    )

    async with WeatherClient() as client:
        with pytest.raises(GeolocationError, match="unexpected payload"):
            await client.detect_current_city()
