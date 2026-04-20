"""Custom exceptions for the weather-cli application."""

from typing import Optional


class WeatherCLIError(Exception):
    """Base class for all weather-cli exceptions."""


class APIError(WeatherCLIError):
    """Raised when the weather API is unreachable or fails.

    Attributes:
        status_code: The HTTP status code returned by the API, or ``None``
            if the request never produced a response (e.g. a network error).
    """

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code: Optional[int] = status_code


class LocationNotFoundError(APIError):
    """Raised when the requested coordinates are rejected by the API.

    This is a subclass of :class:`APIError` so callers that broadly catch
    ``APIError`` still handle bad-coordinate failures.
    """


class CityNotFoundError(APIError):
    """Raised when a city name cannot be resolved by the geocoding API.

    This is a subclass of :class:`APIError` so callers that broadly catch
    ``APIError`` still handle unknown-city failures. It is raised both when
    the geocoding service returns zero results and when the response payload
    is structurally valid but contains no usable match.
    """


class CacheError(WeatherCLIError):
    """Raised for unrecoverable cache subsystem failures.

    The cache layer is designed to be best-effort: routine issues such as
    missing files, permission denials, or corrupted entries are handled
    silently (treated as a miss) so that the caller falls back to a live
    network request. ``CacheError`` is reserved for programmer errors
    (e.g. invalid construction parameters) where silent degradation would
    mask a real bug.
    """
