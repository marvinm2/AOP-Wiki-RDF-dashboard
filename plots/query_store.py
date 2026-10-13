"""A persistent store of SPARQL results, keyed by query and graph fingerprint (#179).

Every plot reads the endpoint through ``run_sparql_query_with_retry``, and the
named graphs it reads are immutable snapshots. So a query's answer is fixed by
the query text and the content of the graphs it touches. This module stores
that answer on disk under ``(sha1(query), fingerprints of those graphs)``:

- A build job (``scripts/build_query_store.py``) runs every plot for every
  version and writes each result it fetches. Historical results are fetched
  once; a new quarter only adds the queries that touch it, plus the queries
  over all graphs.
- Web workers read results from the store instead of querying Virtuoso, and
  fall back to the endpoint on a miss.

Which graphs a query touches is read from its text: graph IRIs written in it
(``GRAPH <…/2024-01-01>``, or ``GRAPH ?g`` pinned by ``FILTER(?g = <…>)``). A
query with an unpinned ``GRAPH ?g``, or none at all, depends on every graph,
so any change to any graph moves its key.

A graph's fingerprint is its triple count plus the VoID ``pav:createdOn`` and
``void:triples`` it was loaded with, so reloading or regenerating a graph moves
it. ``manifest.json`` records the fingerprints of the last complete build. It is
written last, and readers key on it, so a reader never sees a half-built
state: results for new fingerprints exist before the manifest names them.

Every file is written to a temporary name and renamed into place, so a killed
build leaves complete files or none. Only one build runs at a time
(``BuildLock``: an atomic ``mkdir``, taken over once stale).

Pure standard library: the build script and the tests use it without the
plotting code.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import re
import shutil
import socket
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, List, Optional

logger = logging.getLogger(__name__)

GRAPH_PREFIX = "http://aopwiki.org/graph/"
ALL_GRAPHS = "*"
FORMAT = 1  # bump to invalidate every stored result

_GRAPH_IRI = re.compile(r"<(" + re.escape(GRAPH_PREFIX) + r"\d{4}-\d{2}-\d{2})>")
_GRAPH_VAR = re.compile(r"\bGRAPH\s+\?(\w+)", re.IGNORECASE)
_ANY_GRAPH = re.compile(r"\bGRAPH\s+[<?]", re.IGNORECASE)  # the keyword, not ".../graph/..."

# One query fingerprints every graph: its triple count, and the creation date
# and declared size of the files loaded into it (from their VoID).
FINGERPRINT_QUERY = f"""
SELECT ?g (SAMPLE(?n) AS ?triples)
       (GROUP_CONCAT(DISTINCT STR(?created); separator=" ") AS ?created_on)
       (GROUP_CONCAT(DISTINCT STR(?declared); separator=" ") AS ?declared)
WHERE {{
  {{
    SELECT ?g (COUNT(*) AS ?n)
    WHERE {{ GRAPH ?g {{ ?s ?p ?o }} FILTER(STRSTARTS(STR(?g), "{GRAPH_PREFIX}")) }}
    GROUP BY ?g
  }}
  OPTIONAL {{ GRAPH ?g {{ ?file <http://purl.org/pav/createdOn> ?created }} }}
  OPTIONAL {{ GRAPH ?g {{ ?file2 <http://rdfs.org/ns/void#triples> ?declared }} }}
}}
GROUP BY ?g
ORDER BY ?g
"""


def graph_dependencies(query: str) -> Optional[FrozenSet[str]]:
    """The graph IRIs a query reads, or None if it may read any graph.

    A ``GRAPH ?var`` counts as pinned only by an equality filter on that
    variable (``FILTER(?var = <…/YYYY-MM-DD>)``); anything else, including a
    query with no ``GRAPH`` at all (Virtuoso's default graph is the union of
    all graphs), is taken to depend on every graph.
    """
    iris = set(_GRAPH_IRI.findall(query))
    for var in set(_GRAPH_VAR.findall(query)):
        pinned = re.findall(
            r"FILTER\s*\(\s*\?" + re.escape(var) + r"\s*=\s*<(" + re.escape(GRAPH_PREFIX)
            + r"\d{4}-\d{2}-\d{2})>\s*\)", query, re.IGNORECASE)
        if not pinned:
            return None
    if not iris or not _ANY_GRAPH.search(query):
        return None
    return frozenset(iris)


def fingerprints_from_rows(rows: List[dict]) -> Dict[str, str]:
    """Graph IRI -> fingerprint, from the bindings of ``FINGERPRINT_QUERY``."""
    def value(row, name):
        return row.get(name, {}).get("value", "")

    def words(row, name):
        return " ".join(sorted(value(row, name).split()))

    return {
        value(r, "g"): f"{value(r, 'triples')}|{words(r, 'created_on')}|{words(r, 'declared')}"
        for r in rows if value(r, "g").startswith(GRAPH_PREFIX)
    }


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _atomic_write(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


@dataclass(frozen=True)
class Manifest:
    """Fingerprints of the graphs the last complete build covered."""

    fingerprints: Dict[str, str]
    built_at: float = 0.0
    stats: Optional[dict] = None

    @property
    def all_graphs(self) -> str:
        """Fingerprint of the whole set of graphs, for queries that read any graph."""
        return _sha1(json.dumps(sorted(self.fingerprints.items())))

    def dependency_key(self, deps: Optional[FrozenSet[str]]) -> Optional[str]:
        """The fingerprint part of a result's key, or None if a graph is unknown."""
        if deps is None:
            return f"{ALL_GRAPHS}={self.all_graphs}"
        parts = []
        for graph in sorted(deps):
            fp = self.fingerprints.get(graph)
            if fp is None:
                return None
            parts.append(f"{graph}={fp}")
        return "\n".join(parts)

    def to_json(self) -> bytes:
        return json.dumps({"format": FORMAT, "built_at": self.built_at,
                           "fingerprints": self.fingerprints, "stats": self.stats or {}},
                          indent=1, sort_keys=True).encode()


class QueryStore:
    """Results on disk under ``root``.

    Args:
        root: store directory (``results/``, ``manifest.json``, ``.lock/``).
        manifest_ttl: seconds a reader trusts its loaded manifest before
            checking the file again.
    """

    def __init__(self, root: str, manifest_ttl: float = 30.0,
                 clock: Callable[[], float] = time.monotonic):
        self.root = root
        self._manifest_ttl = manifest_ttl
        self._clock = clock
        self._manifest: Optional[Manifest] = None
        self._manifest_mtime: Optional[float] = None
        self._checked_at: Optional[float] = None
        self._override: Optional[Manifest] = None
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.failures = 0
        self.touched: set = set()  # keys read or written, for prune()
        # When a list, every query answered or stored is appended to it as
        # (sha1(query), its graphs or None), so a build can tell later whether a
        # plot's queries are all stored without running it.
        self.recorder: Optional[list] = None

    # --- manifest ---------------------------------------------------------
    @property
    def manifest_path(self) -> str:
        return os.path.join(self.root, "manifest.json")

    def manifest(self) -> Optional[Manifest]:
        """The manifest readers key on: the last complete build, re-read when it changes."""
        if self._override is not None:
            return self._override
        now = self._clock()
        if self._checked_at is not None and now - self._checked_at < self._manifest_ttl:
            return self._manifest
        with self._lock:
            self._checked_at = now
            try:
                mtime = os.stat(self.manifest_path).st_mtime
            except OSError:
                return self._manifest
            if mtime != self._manifest_mtime:
                try:
                    with open(self.manifest_path, "rb") as fh:
                        data = json.load(fh)
                    if data.get("format") == FORMAT:
                        self._manifest = Manifest(data["fingerprints"], data.get("built_at", 0.0),
                                                  data.get("stats"))
                        logger.info(f"Query store manifest: {len(self._manifest.fingerprints)} graphs")
                    else:
                        self._manifest = None
                    self._manifest_mtime = mtime
                except (OSError, ValueError, KeyError) as e:
                    logger.error(f"Unreadable query store manifest, keeping the previous one: {e}")
            return self._manifest

    def use_fingerprints(self, fingerprints: Dict[str, str]) -> Manifest:
        """Key on these fingerprints instead of the manifest (the build job does)."""
        self._override = Manifest(dict(fingerprints), time.time())
        return self._override

    def publish(self, stats: Optional[dict] = None) -> Manifest:
        """Write the manifest for the fingerprints in use. Call last, after every result."""
        manifest = Manifest(self._override.fingerprints, time.time(), stats)
        _atomic_write(self.manifest_path, manifest.to_json())
        return manifest

    # --- results ----------------------------------------------------------
    @staticmethod
    def ref(query: str) -> tuple:
        """What a result's key is made of besides fingerprints: (sha1(query), its graphs)."""
        deps = graph_dependencies(query)
        return _sha1(query), (sorted(deps) if deps is not None else None)

    def key_for(self, query_hash: str, graphs: Optional[List[str]]) -> Optional[str]:
        """Key of a query's result at the current fingerprints; None if a graph is unknown."""
        manifest = self.manifest()
        if manifest is None:
            return None
        deps_key = manifest.dependency_key(frozenset(graphs) if graphs is not None else None)
        if deps_key is None:
            return None
        return _sha1(f"{FORMAT}\n{query_hash}\n{deps_key}")

    def key(self, query: str) -> Optional[str]:
        return self.key_for(*self.ref(query))

    def has(self, key: str) -> bool:
        return os.path.exists(self._path(key))

    def _record(self, query: str, key: str) -> None:
        self.touched.add(key)
        if self.recorder is not None:
            self.recorder.append(self.ref(query))

    def _path(self, key: str) -> str:
        return os.path.join(self.root, "results", key[:2], f"{key}.json.gz")

    def get(self, query: str) -> Optional[list]:
        """The stored bindings for ``query``, or None on a miss."""
        key = self.key(query)
        if key is None:
            self.misses += 1
            return None
        try:
            with gzip.open(self._path(key), "rb") as fh:
                entry = json.load(fh)
        except FileNotFoundError:
            self.misses += 1
            return None
        except (OSError, ValueError, EOFError) as e:
            logger.error(f"Unreadable query store entry {key}: {e}")
            self.misses += 1
            return None
        self.hits += 1
        self._record(query, key)
        return entry["bindings"]

    def put(self, query: str, bindings: list) -> None:
        key = self.key(query)
        if key is None:
            return
        deps = graph_dependencies(query)
        entry = {"format": FORMAT, "query": query,
                 "graphs": sorted(deps) if deps is not None else ALL_GRAPHS,
                 "stored_at": time.time(), "bindings": bindings}
        _atomic_write(self._path(key), gzip.compress(json.dumps(entry).encode(), compresslevel=6, mtime=0))
        self.writes += 1
        self._record(query, key)

    # --- housekeeping -----------------------------------------------------
    def prune(self, keep: set, older_than: float = 7 * 86400) -> int:
        """Delete results not in ``keep`` that are older than ``older_than`` seconds.

        Results a previous manifest pointed at stay for a while, so a reader
        still on that manifest keeps hitting them.
        """
        removed = 0
        cutoff = time.time() - older_than
        results = os.path.join(self.root, "results")
        for dirpath, _, files in os.walk(results):
            for name in files:
                key = name.split(".", 1)[0]
                path = os.path.join(dirpath, name)
                if key in keep:
                    continue
                try:
                    if os.stat(path).st_mtime < cutoff:
                        os.unlink(path)
                        removed += 1
                except OSError:
                    pass
        return removed


class BuildLock:
    """Single-writer lock: an atomic ``mkdir`` of ``<root>/.lock``.

    A lock older than ``stale_after`` seconds is taken over (a build that was
    killed never released it). Gluster's ``flock`` is not relied on.
    """

    def __init__(self, root: str, stale_after: float = 6 * 3600):
        self.path = os.path.join(root, ".lock")
        self.stale_after = stale_after

    def acquire(self) -> bool:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        for _ in range(2):
            try:
                os.mkdir(self.path)
            except FileExistsError:
                try:
                    age = time.time() - os.stat(self.path).st_mtime
                except OSError:
                    continue
                if age < self.stale_after:
                    return False
                logger.warning(f"Taking over a stale build lock ({age / 3600:.1f} h old)")
                shutil.rmtree(self.path, ignore_errors=True)
                continue
            info = {"host": socket.gethostname(), "pid": os.getpid(), "since": time.time()}
            with open(os.path.join(self.path, "owner.json"), "w") as fh:
                json.dump(info, fh)
            return True
        return False

    def release(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError(f"Another build holds {self.path}")
        return self

    def __exit__(self, *exc):
        self.release()
        return False


# --- process-wide store --------------------------------------------------
_store: Optional[QueryStore] = None
_writable = False


def configure(root: Optional[str], writable: bool = False, **kwargs) -> Optional[QueryStore]:
    """Install the store the SPARQL runner consults (None switches it off)."""
    global _store, _writable
    _store = QueryStore(root, **kwargs) if root else None
    _writable = bool(root) and writable
    return _store


def active_store() -> Optional[QueryStore]:
    return _store


def is_writable() -> bool:
    return _writable
