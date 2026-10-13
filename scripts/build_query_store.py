#!/usr/bin/env python3
"""Fill the persistent SPARQL result store by running every plot (#179).

Runs each trend task once and each snapshot plot for every version, with the
store open for writing: a query already stored at the current graph
fingerprints is answered from disk, anything else goes to the endpoint and is
stored. ``manifest.json`` is published last, so web workers switch to the new
fingerprints only once every result is in place.

A plot run whose queries were all stored is remembered in a marker under
``runs/``, keyed by the plot, its version and a hash of the plotting code. A
later build skips that run as long as every query it made is still stored at
the current fingerprints, so a rerun with nothing new issues only the
fingerprint query. Change the code, or reload a graph, and the affected runs
happen again.

Usage:
    python scripts/build_query_store.py --store DIR            # what is missing
    python scripts/build_query_store.py --store DIR --force    # rerun every plot
    python scripts/build_query_store.py --store DIR --versions 2026-10-01 2026-07-01

Exits 0 when every plot ran without a failed query, 1 when some did (their
results are kept and the manifest is still published), 2 when another build
holds the lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger("build_query_store")

# Files whose content decides which queries the plots issue.
CODE_GLOBS = ("plots/*.py", "*.csv", "static/data/*.json")


def code_hash() -> str:
    digest = hashlib.sha1()
    for pattern in CODE_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            digest.update(path.relative_to(ROOT).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def setting_variants(spec):
    """Every combination of the settings a page offers for this plot (defaults included).

    Mirrors the controls in templates/latest.html: the coverage scope/view
    toggles, the overlap slider (0.30-0.60 in steps of 0.02), and the ontology
    branch and domain selectors. ``min_shared_kes`` has no control and keeps its
    default.
    """
    import itertools
    import plots

    choices = {
        "scope": ["all", "apical", "ao"],
        "view": ["absolute", "percentage"],
        "min_jaccard": [f"{0.30 + 0.02 * i:.2f}" for i in range(16)],
        "branch": plots.available_ontology_branches(),
        "domain": plots.available_domains(),
    }
    names = [p for p in spec.params if p in choices]
    return [dict(zip(names, combo)) for combo in itertools.product(*(choices[n] for n in names))]


def plan_runs(registry, versions, only=None):
    """(run name, version, callable) for every trend task and snapshot plot x version x setting."""
    import plots

    runs = [("versions", None, lambda: plots.get_all_versions())]
    for task, spec in registry.startup_tasks().items():
        if spec.kind == "trend" and (not only or task in only):
            runs.append((f"trend:{task}", None, plots.bind_producer(spec)))
    for spec in registry.snapshot_specs():
        if only and spec.id not in only:
            continue
        func = plots.bind_producer(spec)
        for settings in setting_variants(spec):
            label = f"snapshot:{spec.id}" + "".join(f" {k}={v}" for k, v in settings.items())
            for version in versions:
                runs.append((label, version, lambda f=func, v=version, kw=settings: f(v, **kw)))
    if not only:
        runs.append(("network", None, lambda: plots.get_or_compute_network()))
    return runs


def marker_path(store_root: str, code: str, name: str, version) -> str:
    key = hashlib.sha1(f"{code}\n{name}\n{version}".encode()).hexdigest()
    return os.path.join(store_root, "runs", key[:2], f"{key}.json")


def stored_keys(store, marker: str):
    """Keys of the queries a marked run made, if all are stored now; else None."""
    try:
        with open(marker) as fh:
            refs = json.load(fh)["queries"]
    except (OSError, ValueError, KeyError):
        return None
    keys = []
    for query_hash, graphs in refs:
        key = store.key_for(query_hash, graphs)
        if key is None or not store.has(key):
            return None
        keys.append(key)
    return keys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--store", default=os.getenv("QUERY_STORE_DIR"), required=not os.getenv("QUERY_STORE_DIR"))
    parser.add_argument("--force", action="store_true", help="rerun every plot, ignoring markers")
    parser.add_argument("--versions", nargs="*", help="snapshot versions to build (default: all)")
    parser.add_argument("--only", nargs="*", help="trend task names / snapshot plot ids to build")
    parser.add_argument("--no-prune", action="store_true", help="keep results no run referenced")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("plots", "plots.shared", "plots.latest_plots", "plots.trends_plots", "plots.network"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from plots import query_store, registry
    from plots.shared import run_sparql_query_with_retry

    lock = query_store.BuildLock(args.store)
    if not lock.acquire():
        logger.error(f"Another build holds {lock.path}; exiting")
        return 2
    started = time.time()
    try:
        # Fingerprint with the store switched off, so the answer is live.
        query_store.configure(None)
        fingerprints = query_store.fingerprints_from_rows(
            run_sparql_query_with_retry(query_store.FINGERPRINT_QUERY))
        if not fingerprints:
            logger.error("The fingerprint query found no graphs; not building")
            return 1
        store = query_store.configure(args.store, writable=True)
        store.use_fingerprints(fingerprints)
        logger.info(f"{len(fingerprints)} graphs fingerprinted")

        import plots
        versions = args.versions or [v["version"] for v in plots.get_all_versions()]
        code = code_hash()
        runs = plan_runs(registry, versions, set(args.only) if args.only else None)

        ran = skipped = 0
        failed = []
        for name, version, run in runs:
            marker = marker_path(args.store, code, name, version)
            if not args.force:
                keys = stored_keys(store, marker)
                if keys is not None:
                    store.touched.update(keys)
                    skipped += 1
                    continue
            store.recorder, failures_before = [], store.failures
            t0 = time.time()
            try:
                run()
            except Exception as e:  # producers normally catch their own errors
                logger.error(f"{name} {version or ''} raised: {e}")
                store.failures += 1
            refs, store.recorder = store.recorder, None
            ran += 1
            label = f"{name} {version or ''}".strip()
            if store.failures > failures_before:
                failed.append(label)
                logger.warning(f"{label}: a query failed; not marked as complete")
                continue
            query_store._atomic_write(marker, json.dumps({"queries": refs}).encode())
            logger.info(f"{label}: {len(refs)} queries in {time.time() - t0:.1f}s")

        stats = {"runs": len(runs), "ran": ran, "skipped": skipped, "failed": failed,
                 "hits": store.hits, "writes": store.writes, "versions": len(versions),
                 "seconds": round(time.time() - started, 1), "code": code}
        store.publish(stats)
        if not args.no_prune and not args.only and not args.versions:
            removed = store.prune(store.touched)
            stats["pruned"] = removed
        logger.info(f"Published manifest: {json.dumps(stats)}")
        return 1 if failed else 0
    finally:
        lock.release()


if __name__ == "__main__":
    sys.exit(main())
