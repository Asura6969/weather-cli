# CLAUDE.md - Developer Guidelines for weather-cli

## Project Overview
`weather-cli` is a fast, terminal-based weather application built in Python. It uses the Open-Meteo API (which requires no API keys) to fetch weather data and `rich` to print beautiful terminal outputs.

## Workflows
- **Setup environment:** `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
- **Run tests:** `pytest tests/ -v`
- **Run linter/formatter:** We use `black` for formatting and `flake8` for linting. Run `black src tests && flake8 src tests`.
- **Run CLI locally:** `python -m src.cli`

## Code Style & Conventions
- **Type Hints:** All function signatures and class properties must have strict Python type hints. 
- **Async:** Use `asyncio` and `httpx.AsyncClient` for all network requests.
- **Error Handling:** Create custom exceptions in `src/exceptions.py` (e.g., `APIError`, `LocationNotFoundError`). Do not silently swallow exceptions; catch them and display a user-friendly error message via `rich.console`.
- **Testing:** All API calls must be mocked using `respx` in tests. Never make live network requests in the test suite.

## Git Workflow
- Write concise, descriptive commit messages starting with conventional commits (e.g., `feat:`, `fix:`, `test:`, `refactor:`).