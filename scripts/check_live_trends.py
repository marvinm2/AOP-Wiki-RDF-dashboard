#!/usr/bin/env python3
"""Check that every Historical Trends plot renders on a deployed dashboard (#157).

The in-process download audit starts its own copy of the app, so it cannot see
what the *deployed* process is serving. On 2026-10-04 the live dashboard served
"data unavailable" for 24 of 48 trend plots for ~15 hours while /health stayed
green. This script asks the deployed app directly.

Plot names are read from the /trends page itself (its lazy-load placeholders),
so the check follows whatever the page currently shows.

Usage:
    python scripts/check_live_trends.py [BASE_URL]

Exits 1 if any trend plot fails to render, listing the failures.
"""

import re
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

DEFAULT_BASE_URL = "https://aopwiki-dashboard.vhp4safety.nl"
PLOT_NAME_RE = re.compile(r'class="lazy-plot"[^>]*data-plot-name="([\w-]+)"')


def trend_plot_names(base_url: str) -> list:
    page = requests.get(f"{base_url}/trends", timeout=60)
    page.raise_for_status()
    return sorted(set(PLOT_NAME_RE.findall(page.text)))


def check_plot(base_url: str, name: str):
    """Return None if the plot renders, else a short reason."""
    try:
        response = requests.get(f"{base_url}/api/plot/{name}", timeout=180)
        payload = response.json()
    except (requests.RequestException, ValueError) as e:
        return f"request failed: {e}"
    if response.status_code != 200 or not payload.get("success") or not (payload.get("html") or "").strip():
        return f"HTTP {response.status_code}: {payload.get('error', 'empty plot')}"
    return None


def main() -> int:
    base_url = (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE_URL).rstrip("/")
    names = trend_plot_names(base_url)
    if not names:
        print(f"No trend plots found on {base_url}/trends — page layout changed?")
        return 1

    # A few at a time: each request may trigger a recompute on the server.
    with ThreadPoolExecutor(max_workers=3) as pool:
        failures = {n: r for n, r in zip(names, pool.map(lambda n: check_plot(base_url, n), names)) if r}

    print(f"{len(names) - len(failures)}/{len(names)} trend plots render on {base_url}")
    for name, reason in sorted(failures.items()):
        print(f"  FAIL {name}: {reason}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
