"""Async client for the Open-Meteo weather and geocoding APIs."""

from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import TracebackType
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Type

import httpx

from src.exceptions import APIError, CityNotFoundError, LocationNotFoundError

OPEN_METEO_URL: str = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODING_URL: str = "https://geocoding-api.open-meteo.com/v1/search"
DEFAULT_TIMEOUT: float = 10.0


@dataclass(frozen=True)
class Location:
    """A geocoded location resolved from a user-supplied city name."""

    latitude: float
    longitude: float
    name: str

    @property
    def display_name(self) -> str:
        """Alias for :attr:`name` for rendering in the terminal UI."""
        return self.name


@dataclass(frozen=True)
class CurrentWeather:
    """Current-weather snapshot returned by :class:`WeatherClient`."""

    latitude: float
    longitude: float
    temperature: float
    windspeed: float
    winddirection: float
    weathercode: int
    time: str
    timezone: str


class WeatherClient:
    """Async client that fetches current weather from Open-Meteo.

    The client is designed to be used as an async context manager so that the
    underlying :class:`httpx.AsyncClient` connection pool is reused across
    calls::

        async with WeatherClient() as client:
            weather = await client.fetch_current_weather(52.52, 13.42)

    Calling :meth:`fetch_current_weather` outside of a context manager is also
    supported; a transient ``httpx.AsyncClient`` will be created and closed
    for that single request.
    """

    def __init__(
        self,
        base_url: str = OPEN_METEO_URL,
        timeout: float = DEFAULT_TIMEOUT,
        client: Optional[httpx.AsyncClient] = None,
        geocoding_url: str = OPEN_METEO_GEOCODING_URL,
    ) -> None:
        self._base_url: str = base_url
        self._geocoding_url: str = geocoding_url
        self._timeout: float = timeout
        self._client: Optional[httpx.AsyncClient] = client
        self._owns_client: bool = client is None

    async def __aenter__(self) -> "WeatherClient":
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
            self._owns_client = True
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying ``httpx.AsyncClient`` if owned here."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch_current_weather(
        self, latitude: float, longitude: float
    ) -> CurrentWeather:
        """Fetch the current weather at ``(latitude, longitude)``.

        Raises:
            LocationNotFoundError: If the API rejects the coordinates as
                invalid (HTTP 400). The error body's ``reason`` field is
                used as the message when present; otherwise a generic
                fallback message is used. ``LocationNotFoundError`` is a
                subclass of ``APIError``.
            APIError: For any other failure - network errors, non-2xx
                responses, or payloads that cannot be parsed.
        """
        params: Dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
            "current_weather": "true",
        }

        def _weather_status_handler(response: httpx.Response) -> None:
            if response.status_code == 400:
                reason: Optional[str] = _extract_error_reason(response)
                raise LocationNotFoundError(
                    reason or "Invalid coordinates for weather lookup",
                    status_code=400,
                )

        payload: Dict[str, Any] = await self._get_json(
            self._base_url,
            params,
            api_name="Weather",
            status_handler=_weather_status_handler,
        )
        return _parse_current_weather(payload)

    @asynccontextmanager
    async def _request_client(self) -> AsyncIterator[httpx.AsyncClient]:
        """Yield an ``httpx.AsyncClient`` scoped to a single request.

        If the instance owns a persistent client (via ``async with
        WeatherClient()``), that client is yielded and left open. Otherwise
        a transient client is created and closed when the ``async with``
        block exits, regardless of success or exception.
        """
        if self._client is not None:
            yield self._client
            return
        transient: httpx.AsyncClient = httpx.AsyncClient(timeout=self._timeout)
        try:
            yield transient
        finally:
            await transient.aclose()

    async def geocode_city(self, city: str) -> Location:
        """Resolve a ``city`` name to a :class:`Location`.

        Uses the Open-Meteo Geocoding API. The returned :class:`Location`
        carries ``latitude`` and ``longitude`` that can be fed directly
        into :meth:`fetch_current_weather`, plus a fully qualified ``name``
        (e.g. ``"Berlin, Land Berlin, Germany"``) suitable for display in
        the terminal UI.

        Raises:
            CityNotFoundError: If the geocoding service returns zero
                results for ``city`` or a structurally valid payload with
                no usable match. Subclass of ``APIError``.
            APIError: For any other failure - network errors, non-2xx
                responses, timeouts, or payloads that cannot be parsed.
        """
        if not city or not city.strip():
            raise CityNotFoundError("City name must be a non-empty string")

        params: Dict[str, Any] = {
            "name": city,
            "count": 1,
            "format": "json",
        }

        payload: Dict[str, Any] = await self._get_json(
            self._geocoding_url,
            params,
            api_name="Geocoding",
        )
        return _parse_geocoding_result(payload, city)

    async def _get_json(
        self,
        url: str,
        params: Dict[str, Any],
        api_name: str,
        status_handler: Optional[Callable[[httpx.Response], None]] = None,
    ) -> Dict[str, Any]:
        """Perform a GET against ``url`` and return the validated JSON dict.

        Wraps all boundary failures (network errors, non-2xx responses,
        malformed JSON, non-dict payloads) in :class:`APIError` so callers
        never see raw ``httpx`` or ``ValueError`` exceptions. ``api_name`` is
        interpolated into error messages (e.g. ``"Weather API returned..."``).

        If ``status_handler`` is provided, it is invoked after the response
        is received and before the generic ``>= 400`` check; it may raise a
        more specific :class:`APIError` subclass for known status codes
        (e.g. mapping HTTP 400 to :class:`LocationNotFoundError`). If it
        returns normally, generic status handling proceeds.
        """
        async with self._request_client() as client:
            try:
                response: httpx.Response = await client.get(url, params=params)
            except httpx.RequestError as exc:
                raise APIError(
                    f"Network error while contacting {api_name.lower()} API: {exc}"
                ) from exc

            if status_handler is not None:
                status_handler(response)

            if response.status_code >= 400:
                detail: str = (
                    _extract_error_reason(response) or response.reason_phrase or ""
                )
                message: str = (
                    f"{api_name} API returned HTTP {response.status_code}"
                    f"{': ' + detail if detail else ''}"
                )
                raise APIError(message, status_code=response.status_code)

            try:
                payload: Any = response.json()
            except ValueError as exc:
                raise APIError(f"{api_name} API returned invalid JSON") from exc

            if not isinstance(payload, dict):
                raise APIError(f"{api_name} API returned an unexpected JSON structure")

            return payload


def _extract_error_reason(response: httpx.Response) -> Optional[str]:
    """Return the ``reason`` field from an Open-Meteo error body, if any."""
    try:
        data: Any = response.json()
    except ValueError:
        return None
    if isinstance(data, dict):
        reason: Any = data.get("reason")
        if isinstance(reason, str) and reason:
            return reason
    return None


def _parse_current_weather(payload: Dict[str, Any]) -> CurrentWeather:
    current: Any = payload.get("current_weather")
    if not isinstance(current, dict):
        raise APIError("Weather API response missing 'current_weather' section")
    try:
        return CurrentWeather(
            latitude=float(payload["latitude"]),
            longitude=float(payload["longitude"]),
            temperature=float(current["temperature"]),
            windspeed=float(current["windspeed"]),
            winddirection=float(current["winddirection"]),
            weathercode=int(current["weathercode"]),
            time=str(current["time"]),
            timezone=str(payload.get("timezone", "GMT")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise APIError(
            f"Weather API response missing or malformed field: {exc}"
        ) from exc


def _parse_geocoding_result(payload: Dict[str, Any], city: str) -> Location:
    """Parse the first result in an Open-Meteo geocoding payload."""
    results: Any = payload.get("results")
    if not isinstance(results, list) or not results:
        raise CityNotFoundError(f"No geocoding results for city: {city!r}")

    first: Any = results[0]
    if not isinstance(first, dict):
        raise APIError("Geocoding API returned a malformed result entry")

    try:
        latitude: float = float(first["latitude"])
        longitude: float = float(first["longitude"])
        name_field: str = str(first["name"])
    except (KeyError, TypeError, ValueError) as exc:
        raise APIError(
            f"Geocoding API response missing or malformed field: {exc}"
        ) from exc

    admin1: Any = first.get("admin1")
    country: Any = first.get("country")
    parts: List[str] = [name_field]
    if isinstance(admin1, str) and admin1:
        parts.append(admin1)
    if isinstance(country, str) and country:
        parts.append(country)
    full_name: str = ", ".join(parts)

    return Location(latitude=latitude, longitude=longitude, name=full_name)
