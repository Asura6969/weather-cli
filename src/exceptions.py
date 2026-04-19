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
