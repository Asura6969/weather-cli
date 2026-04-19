"""Async client for the Open-Meteo weather API."""

from dataclasses import dataclass
from types import TracebackType
from typing import Any, Dict, Optional, Type

import httpx

from src.exceptions import APIError, LocationNotFoundError

OPEN_METEO_URL: str = "https://api.open-meteo.com/v1/forecast"
DEFAULT_TIMEOUT: float = 10.0


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
    ) -> None:
        self._base_url: str = base_url
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
        client: httpx.AsyncClient
        close_after: bool
        if self._client is not None:
            client = self._client
            close_after = False
        else:
            client = httpx.AsyncClient(timeout=self._timeout)
            close_after = True

        params: Dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
            "current_weather": "true",
        }

        try:
            try:
                response: httpx.Response = await client.get(
                    self._base_url, params=params
                )
            except httpx.RequestError as exc:
                raise APIError(
                    f"Network error while contacting weather API: {exc}"
                ) from exc

            if response.status_code == 400:
                reason: Optional[str] = _extract_error_reason(response)
                raise LocationNotFoundError(
                    reason or "Invalid coordinates for weather lookup",
                    status_code=400,
                )

            if response.status_code >= 400:
                detail: str = (
                    _extract_error_reason(response) or response.reason_phrase or ""
                )
                message: str = (
                    f"Weather API returned HTTP {response.status_code}"
                    f"{': ' + detail if detail else ''}"
                )
                raise APIError(message, status_code=response.status_code)

            try:
                payload: Any = response.json()
            except ValueError as exc:
                raise APIError("Weather API returned invalid JSON") from exc

            if not isinstance(payload, dict):
                raise APIError("Weather API returned an unexpected JSON structure")

            return _parse_current_weather(payload)
        finally:
            if close_after:
                await client.aclose()


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
