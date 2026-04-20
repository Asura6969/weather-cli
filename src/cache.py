"""File-based cache for weather data keyed by city name.

The cache is intentionally best-effort. Any I/O failure (permission denied,
corrupted payload, missing directory) is treated as a cache miss, so the
caller falls back to a live network request rather than halting execution.
Entries older than :data:`DEFAULT_TTL_SECONDS` are considered stale and are
not returned to the caller.

Typical usage::

    cache = WeatherCache()
    hit = cache.get("Berlin")
    if hit is None:
        location, weather = await client.geocode_and_fetch("Berlin")
        cache.set("Berlin", location, weather)
    else:
        location, weather = hit
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.api_client import CurrentWeather, Location
from src.exceptions import CacheError

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS: int = 15 * 60
"""Fifteen-minute cache time-to-live, per product requirement."""


def default_cache_dir() -> Path:
    """Return the user-level default cache directory.

    Resolves to ``~/.cache/weather-cli`` on POSIX systems. Callers that need
    isolation (notably the test suite) should pass an explicit ``cache_dir``
    to :class:`WeatherCache` so no files are written under the real home
    directory.
    """
    return Path.home() / ".cache" / "weather-cli"


def _cache_key(city: str) -> str:
    """Return a filesystem-safe, case-insensitive key for ``city``."""
    normalized: str = city.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class WeatherCache:
    """A best-effort, file-backed cache of geocode + weather pairs.

    Each entry is stored as a single JSON file under ``cache_dir`` named
    after a SHA-256 digest of the normalised city name. The payload carries
    a monotonic-ish ``stored_at`` wall-clock timestamp used to enforce the
    TTL on read.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        if ttl_seconds <= 0:
            raise CacheError("ttl_seconds must be a positive integer")
        self._cache_dir: Path = (
            Path(cache_dir) if cache_dir is not None else default_cache_dir()
        )
        self._ttl: int = ttl_seconds

    @property
    def cache_dir(self) -> Path:
        """The directory this cache persists entries to."""
        return self._cache_dir

    @property
    def ttl_seconds(self) -> int:
        """The maximum age, in seconds, of an entry considered fresh."""
        return self._ttl

    def _path_for(self, city: str) -> Path:
        return self._cache_dir / f"{_cache_key(city)}.json"

    def get(self, city: str) -> Optional[Tuple[Location, CurrentWeather]]:
        """Return a cached ``(location, weather)`` pair for ``city`` if fresh.

        Returns ``None`` on any of: missing file, permission denied,
        unreadable/corrupted payload, or an entry older than the configured
        TTL. Corrupted entries are removed opportunistically so subsequent
        reads don't keep paying the parse cost.
        """
        if not city or not city.strip():
            return None

        key: str = _cache_key(city)
        path: Path = self._cache_dir / f"{key}.json"
        try:
            raw: str = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except (OSError, PermissionError) as exc:
            # Permission denied, I/O error, etc. — treat as a miss.
            # Log the hash key instead of the full path to avoid leaking
            # the host's directory layout into diagnostic output. Use
            # ``strerror``/``errno`` rather than ``str(exc)`` because
            # ``OSError.__str__`` embeds the offending filename.
            logger.debug(
                "Cache read failed for city=%r key=%s: %s (errno=%s: %s)",
                city,
                key,
                type(exc).__name__,
                exc.errno,
                exc.strerror,
            )
            return None

        try:
            data: Any = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("cache payload is not a JSON object")
            stored_at: float = float(data["stored_at"])
            age: float = time.time() - stored_at
            if age > self._ttl:
                logger.debug(
                    "Cache entry expired for city=%r key=%s " "(age=%.1fs, ttl=%ds)",
                    city,
                    key,
                    age,
                    self._ttl,
                )
                return None
            location_data: Any = data["location"]
            weather_data: Any = data["weather"]
            if not isinstance(location_data, dict) or not isinstance(
                weather_data, dict
            ):
                raise ValueError("cache payload has malformed sub-objects")
            location: Location = Location(
                latitude=float(location_data["latitude"]),
                longitude=float(location_data["longitude"]),
                name=str(location_data["name"]),
            )
            weather: CurrentWeather = CurrentWeather(
                latitude=float(weather_data["latitude"]),
                longitude=float(weather_data["longitude"]),
                temperature=float(weather_data["temperature"]),
                windspeed=float(weather_data["windspeed"]),
                winddirection=float(weather_data["winddirection"]),
                weathercode=int(weather_data["weathercode"]),
                time=str(weather_data["time"]),
                timezone=str(weather_data["timezone"]),
            )
            return location, weather
        except (ValueError, KeyError, TypeError) as exc:
            # Corrupted cache entry — remove it and report a miss.
            # Logged with the hash key (not the path) to avoid leaking
            # the host's directory layout into diagnostic output.
            logger.debug(
                "Cache entry corrupted for city=%r key=%s: %s: %s",
                city,
                key,
                type(exc).__name__,
                exc,
            )
            self._safe_unlink(path)
            return None

    def set(self, city: str, location: Location, weather: CurrentWeather) -> None:
        """Persist ``(location, weather)`` under ``city``.

        Failures (permission denied, disk full, etc.) are swallowed: the
        cache is a pure optimisation, and a failed write must never break
        the enclosing weather lookup.
        """
        if not city or not city.strip():
            return

        payload: Dict[str, Any] = {
            "stored_at": time.time(),
            "city": city,
            "location": asdict(location),
            "weather": asdict(weather),
        }
        target: Path = self._path_for(city)

        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            return

        tmp_path: Optional[str] = None
        try:
            fd, tmp_path = tempfile.mkstemp(
                prefix=".weather-cache-", suffix=".tmp", dir=str(self._cache_dir)
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            # Atomic rename avoids half-written files on crash.
            os.replace(tmp_path, target)
            tmp_path = None
        except (OSError, PermissionError, TypeError, ValueError):
            # Any failure during write: silently give up on caching.
            return
        finally:
            if tmp_path is not None:
                self._safe_unlink(Path(tmp_path))

    def invalidate(self, city: str) -> None:
        """Remove any cached entry for ``city`` (a no-op if none exists)."""
        if not city or not city.strip():
            return
        self._safe_unlink(self._path_for(city))

    @staticmethod
    def _safe_unlink(path: Path) -> None:
        try:
            path.unlink()
        except (FileNotFoundError, OSError, PermissionError):
            pass
