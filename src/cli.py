"""Command-line entry point for the weather-cli application.

Wires together the backend components — :class:`~src.cache.WeatherCache`
for local persistence and :class:`~src.api_client.WeatherClient` for live
Open-Meteo lookups — behind a single ``city`` positional argument, and
renders the result with :mod:`rich`. Domain-specific failures (unknown
city, invalid coordinates, network errors) are caught and displayed as
formatted error panels rather than raw tracebacks.

Run with::

    python -m src.cli Berlin
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Dict, Optional, Sequence, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.api_client import CurrentWeather, Location, WeatherClient
from src.cache import WeatherCache
from src.exceptions import (
    APIError,
    CityNotFoundError,
    LocationNotFoundError,
    WeatherCLIError,
)

#: Maps WMO weather interpretation codes to a short label and an emoji
#: glyph used as the visual indicator in the terminal table. Only the
#: codes Open-Meteo's ``current_weather`` endpoint documents are listed;
#: anything else falls back to a generic entry.
WEATHER_CODES: Dict[int, Tuple[str, str]] = {
    0: ("Clear sky", "☀️"),
    1: ("Mainly clear", "🌤️"),
    2: ("Partly cloudy", "⛅"),
    3: ("Overcast", "☁️"),
    45: ("Fog", "🌫️"),
    48: ("Depositing rime fog", "🌫️"),
    51: ("Light drizzle", "🌦️"),
    53: ("Moderate drizzle", "🌦️"),
    55: ("Dense drizzle", "🌧️"),
    56: ("Light freezing drizzle", "🌧️"),
    57: ("Dense freezing drizzle", "🌧️"),
    61: ("Slight rain", "🌧️"),
    63: ("Moderate rain", "🌧️"),
    65: ("Heavy rain", "🌧️"),
    66: ("Light freezing rain", "🌧️"),
    67: ("Heavy freezing rain", "🌧️"),
    71: ("Slight snow fall", "🌨️"),
    73: ("Moderate snow fall", "🌨️"),
    75: ("Heavy snow fall", "❄️"),
    77: ("Snow grains", "❄️"),
    80: ("Slight rain showers", "🌦️"),
    81: ("Moderate rain showers", "🌧️"),
    82: ("Violent rain showers", "⛈️"),
    85: ("Slight snow showers", "🌨️"),
    86: ("Heavy snow showers", "❄️"),
    95: ("Thunderstorm", "⛈️"),
    96: ("Thunderstorm with slight hail", "⛈️"),
    99: ("Thunderstorm with heavy hail", "⛈️"),
}


def _describe_weather_code(code: int) -> Tuple[str, str]:
    """Return ``(label, icon)`` for a WMO weather ``code``."""
    return WEATHER_CODES.get(code, (f"Unknown (code {code})", "❔"))


def _temperature_style(temperature: float) -> str:
    """Return a ``rich`` colour style appropriate for ``temperature`` °C."""
    if temperature <= 0.0:
        return "bold bright_blue"
    if temperature < 10.0:
        return "cyan"
    if temperature < 20.0:
        return "green"
    if temperature < 30.0:
        return "yellow"
    return "bold red"


def _wind_style(windspeed: float) -> str:
    """Return a ``rich`` colour style appropriate for ``windspeed`` km/h."""
    if windspeed < 20.0:
        return "green"
    if windspeed < 40.0:
        return "yellow"
    return "bold red"


#: 16-point compass rose, indexed clockwise from north in 22.5° steps.
_COMPASS_POINTS: Tuple[str, ...] = (
    "N", "NNE", "NE", "ENE",
    "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW",
    "W", "WNW", "NW", "NNW",
)  # fmt: skip


def _compass_point(degrees: float) -> str:
    """Return the 16-point compass cardinal for a bearing in ``degrees``.

    ``degrees`` is normalised into ``[0, 360)`` so negative or >360 values
    (which Open-Meteo shouldn't emit, but cost nothing to handle) still
    resolve correctly. 292° → ``"WNW"``.
    """
    index: int = round((degrees % 360.0) / 22.5) % 16
    return _COMPASS_POINTS[index]


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the ``weather-cli`` entry point."""
    parser = argparse.ArgumentParser(
        prog="weather-cli",
        description="Fetch and display the current weather for a city.",
    )
    parser.add_argument(
        "city",
        help="Name of the city to look up (e.g. 'Berlin' or 'New York').",
    )
    return parser


async def _fetch(city: str) -> Tuple[Location, CurrentWeather]:
    """Resolve ``city`` to a ``(Location, CurrentWeather)`` pair.

    Consults the local :class:`~src.cache.WeatherCache` first; on a miss,
    geocodes the city, fetches the current weather, and writes the result
    back to the cache. The cache-then-network flow is delegated to
    :meth:`WeatherClient.fetch_weather_for_city`.
    """
    cache: WeatherCache = WeatherCache()
    async with WeatherClient(cache=cache) as client:
        return await client.fetch_weather_for_city(city)


def _render_weather(
    console: Console, location: Location, weather: CurrentWeather
) -> None:
    """Render ``location`` and ``weather`` as a styled ``rich`` table."""
    label, icon = _describe_weather_code(weather.weathercode)

    table: Table = Table(
        title=f"{icon}  Current Weather — {location.display_name}",
        title_style="bold bright_white",
        show_header=True,
        header_style="bold magenta",
        expand=False,
    )
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row(
        "Location",
        Text(location.display_name, style="bold bright_white"),
    )
    table.add_row(
        "Coordinates",
        f"{weather.latitude:.4f}, {weather.longitude:.4f}",
    )
    table.add_row(
        "Conditions",
        Text(f"{icon}  {label}", style="bold"),
    )
    table.add_row(
        "Temperature",
        Text(
            f"{weather.temperature:.1f} °C",
            style=_temperature_style(weather.temperature),
        ),
    )
    table.add_row(
        "Wind speed",
        Text(
            f"{weather.windspeed:.1f} km/h",
            style=_wind_style(weather.windspeed),
        ),
    )
    table.add_row(
        "Wind direction",
        f"{weather.winddirection:.0f}° {_compass_point(weather.winddirection)}",
    )
    table.add_row("Observed at", f"{weather.time} ({weather.timezone})")

    console.print(table)


def _render_error(console: Console, exc: WeatherCLIError) -> None:
    """Render a domain-specific ``exc`` as a formatted error panel."""
    if isinstance(exc, CityNotFoundError):
        heading: str = "City not found"
    elif isinstance(exc, LocationNotFoundError):
        heading = "Invalid location"
    elif isinstance(exc, APIError):
        heading = "Weather service error"
    else:
        heading = "Error"

    body: Text = Text(str(exc) or heading, style="bold red")
    panel: Panel = Panel(
        body,
        title=f"[bold red]✗ {heading}[/]",
        border_style="red",
        expand=False,
    )
    console.print(panel)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for ``python -m src.cli``.

    Returns a process exit code: ``0`` on success, ``1`` on any handled
    :class:`~src.exceptions.WeatherCLIError`.
    """
    parser: argparse.ArgumentParser = build_parser()
    args: argparse.Namespace = parser.parse_args(argv)

    console: Console = Console()
    err_console: Console = Console(stderr=True)

    try:
        location, weather = asyncio.run(_fetch(args.city))
    except WeatherCLIError as exc:
        _render_error(err_console, exc)
        return 1

    _render_weather(console, location, weather)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
