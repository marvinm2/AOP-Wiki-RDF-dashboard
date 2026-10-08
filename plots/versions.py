"""Which AOP-Wiki versions the endpoint holds (#169).

One query answers both "what is the latest version" and "list all versions".
It looks only at graphs that contain AOPs, which lets Virtuoso use its type
index instead of scanning every triple: about 0.65 s, against 12 s for the old
``GRAPH ?g { ?s ?p ?o }`` latest-version scan. There is no ``LIMIT``, so the
list keeps working past 100 named graphs.

The answer is memoized per process for ``ttl`` seconds. A failed or empty
answer never replaces a good one: the last known list keeps being served and the
query is retried on the next call.
"""

import logging
import threading
import time
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

GRAPH_PREFIX = "http://aopwiki.org/graph/"

VERSIONS_QUERY = f"""
SELECT DISTINCT ?g
WHERE {{
    GRAPH ?g {{ ?aop a <http://aopkb.org/aop_ontology#AdverseOutcomePathway> }}
    FILTER(STRSTARTS(STR(?g), "{GRAPH_PREFIX}"))
}}
ORDER BY DESC(?g)
"""


class VersionRegistry:
    """Memoized, newest-first list of version strings (``YYYY-MM-DD``)."""

    def __init__(self, run_query: Callable[[str], list], ttl: float = 600.0,
                 clock: Callable[[], float] = time.monotonic):
        self._run_query = run_query
        self._ttl = ttl
        self._clock = clock
        self._lock = threading.Lock()
        self._versions: List[str] = []
        self._fetched_at: Optional[float] = None

    def _fresh(self) -> bool:
        return (self._fetched_at is not None
                and self._clock() - self._fetched_at < self._ttl)

    def versions(self) -> List[str]:
        """All versions, newest first; ``[]`` only if none was ever fetched."""
        if self._fresh():
            return list(self._versions)
        with self._lock:
            if not self._fresh():
                self._refresh_locked()
            return list(self._versions)

    def latest(self) -> Optional[str]:
        versions = self.versions()
        return versions[0] if versions else None

    def invalidate(self) -> None:
        with self._lock:
            self._fetched_at = None

    def _refresh_locked(self) -> None:
        try:
            rows = self._run_query(VERSIONS_QUERY)
        except Exception as e:  # keep serving the last good list
            logger.error(f"Version list query failed: {e}")
            rows = []
        versions = sorted(
            {
                uri.rsplit("/", 1)[-1]
                for uri in (r.get("g", {}).get("value", "") for r in rows)
                if uri.startswith(GRAPH_PREFIX)
            },
            reverse=True,
        )
        if versions:
            if versions != self._versions:
                logger.info(f"Version registry: {len(versions)} versions, latest {versions[0]}")
            self._versions = versions
            self._fetched_at = self._clock()
        elif self._versions:
            logger.warning("Version list query returned nothing; keeping the last known list")
        # else: nothing known yet, leave _fetched_at unset so the next call retries
