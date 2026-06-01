"""Privacy-respecting, server-side usage analytics for the dashboard.

Records lightweight events when plots are viewed or downloaded so we can report
which visualisations are actually used, per version and format. Deliberately
minimal and privacy-preserving:

* No cookies, no client-side trackers, no third-party services.
* No IP addresses, user agents, sessions, or any personal data is stored.
* Only: a timestamp, the UTC day, the event kind, and the (plot, version,
  format, route) that was requested.

Events are written to a small SQLite database (WAL mode, short-lived connections)
so it tolerates the multi-worker gunicorn deployment. Writes never raise into the
request path — analytics failures are logged and swallowed. The whole feature is
gated behind ``Config.ENABLE_USAGE_ANALYTICS``.

Author:
    Marvin Martens
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time

from config import Config

logger = logging.getLogger(__name__)

_init_lock = threading.Lock()
_initialized = False

VALID_EVENTS = ("page", "download")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(Config.USAGE_DB_PATH, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db() -> None:
    """Create the events table and indexes once (idempotent, thread-safe)."""
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        os.makedirs(os.path.dirname(os.path.abspath(Config.USAGE_DB_PATH)), exist_ok=True)
        with _connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_events (
                    id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts      REAL NOT NULL,
                    day     TEXT NOT NULL,
                    event   TEXT NOT NULL,
                    plot    TEXT,
                    version TEXT,
                    fmt     TEXT,
                    route   TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_plot ON usage_events(plot)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_day ON usage_events(day)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_event ON usage_events(event)")
        _initialized = True


def record_event(event: str, *, plot: str | None = None, version: str | None = None,
                 fmt: str | None = None, route: str | None = None) -> None:
    """Record a single usage event. Never raises into the caller."""
    if not Config.ENABLE_USAGE_ANALYTICS:
        return
    if event not in VALID_EVENTS:
        return
    try:
        init_db()
        now = time.time()
        day = time.strftime("%Y-%m-%d", time.gmtime(now))
        with _connect() as conn:
            conn.execute(
                "INSERT INTO usage_events (ts, day, event, plot, version, fmt, route) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (now, day, event, plot, version, fmt, route),
            )
    except Exception as exc:  # analytics must never break a request
        logger.warning("usage analytics write failed: %s", exc)


def get_summary(limit: int = 25) -> dict:
    """Return aggregated usage counts for the /api/usage endpoint."""
    if not Config.ENABLE_USAGE_ANALYTICS:
        return {"enabled": False}
    try:
        init_db()
        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            total = conn.execute("SELECT COUNT(*) AS n FROM usage_events").fetchone()["n"]
            by_event = {
                r["event"]: r["n"]
                for r in conn.execute(
                    "SELECT event, COUNT(*) AS n FROM usage_events GROUP BY event"
                )
            }
            by_format = {
                (r["fmt"] or "n/a"): r["n"]
                for r in conn.execute(
                    "SELECT fmt, COUNT(*) AS n FROM usage_events "
                    "WHERE event='download' GROUP BY fmt"
                )
            }
            top_downloads = [
                {"plot": r["plot"], "count": r["n"]}
                for r in conn.execute(
                    "SELECT plot, COUNT(*) AS n FROM usage_events "
                    "WHERE event='download' AND plot IS NOT NULL "
                    "GROUP BY plot ORDER BY n DESC LIMIT ?",
                    (limit,),
                )
            ]
            page_views = [
                {"page": r["plot"], "count": r["n"]}
                for r in conn.execute(
                    "SELECT plot, COUNT(*) AS n FROM usage_events "
                    "WHERE event='page' AND plot IS NOT NULL "
                    "GROUP BY plot ORDER BY n DESC LIMIT ?",
                    (limit,),
                )
            ]
            by_version = [
                {"version": r["version"], "count": r["n"]}
                for r in conn.execute(
                    "SELECT version, COUNT(*) AS n FROM usage_events "
                    "WHERE version IS NOT NULL GROUP BY version ORDER BY n DESC LIMIT ?",
                    (limit,),
                )
            ]
            by_day = [
                {"day": r["day"], "count": r["n"]}
                for r in conn.execute(
                    "SELECT day, COUNT(*) AS n FROM usage_events "
                    "GROUP BY day ORDER BY day DESC LIMIT 90"
                )
            ]
        return {
            "enabled": True,
            "total_events": total,
            "by_event": by_event,
            "by_format": by_format,
            "top_downloads": top_downloads,
            "page_views": page_views,
            "by_version": by_version,
            "by_day": by_day,
        }
    except Exception as exc:
        logger.error("usage analytics summary failed: %s", exc)
        return {"enabled": True, "error": str(exc)}
