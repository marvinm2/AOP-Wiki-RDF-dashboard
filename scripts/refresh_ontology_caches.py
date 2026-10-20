#!/usr/bin/env python3
"""Rebuild the three committed ontology caches and keep only real changes
(multi-endpoint #22).

The caches are built offline from Ubergraph and the AOP-Wiki endpoint and
committed (the dashboard never queries Ubergraph at request time):

    static/data/organ_system_cache.json    scripts/build_organ_system_cache.py
    static/data/domain_lens_cache.json     scripts/build_domain_lens_cache.py
    static/data/ontology_branch_cache.json scripts/build_ontology_branch_cache.py

They go stale as new quarters bring new terms. This runs each builder into a
temporary file and replaces the committed cache only when its content changed;
``generated_at`` and ``stats.elapsed_s`` differ on every run and are ignored.
A summary (Markdown, for the PR body) is written to ``--summary``.

Usage:
    python scripts/refresh_ontology_caches.py [--summary caches.md]

Exit code 0 either way; the summary's first line says whether anything changed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHES = {
    "organ systems": ("static/data/organ_system_cache.json", "scripts/build_organ_system_cache.py"),
    "domain lens": ("static/data/domain_lens_cache.json", "scripts/build_domain_lens_cache.py"),
    "ontology branches": ("static/data/ontology_branch_cache.json", "scripts/build_ontology_branch_cache.py"),
}
VOLATILE = ("generated_at",)
VOLATILE_STATS = ("elapsed_s",)


def comparable(cache: dict) -> dict:
    """The cache without the fields that change on every build."""
    out = {k: v for k, v in cache.items() if k not in VOLATILE}
    if isinstance(out.get("stats"), dict):
        out["stats"] = {k: v for k, v in out["stats"].items() if k not in VOLATILE_STATS}
    return out


def stat_changes(old: dict, new: dict) -> list[str]:
    """'key: old -> new' for every numeric stat that moved."""
    a = comparable(old).get("stats") or {}
    b = comparable(new).get("stats") or {}
    return [f"{k}: {a.get(k)} -> {b.get(k)}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]


def refresh(name: str, cache_rel: str, builder_rel: str, workdir: Path) -> dict:
    """Build one cache; replace the committed file if its content changed."""
    target = ROOT / cache_rel
    fresh = workdir / target.name
    subprocess.run([sys.executable, str(ROOT / builder_rel), "--out", str(fresh)],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    old = json.loads(target.read_text()) if target.exists() else {}
    new = json.loads(fresh.read_text())
    changed = comparable(old) != comparable(new)
    if changed:
        target.write_text(fresh.read_text())
    return {"name": name, "file": cache_rel, "changed": changed,
            "was": old.get("generated_at"), "now": new.get("generated_at"),
            "stats": stat_changes(old, new)}


def summary(results: list[dict]) -> str:
    changed = [r for r in results if r["changed"]]
    lines = [f"{len(changed)} of {len(results)} ontology caches changed." if changed
             else "No ontology cache changed.", ""]
    for r in results:
        lines.append(f"- **{r['name']}** (`{r['file']}`): "
                     + (f"rebuilt (was {r['was']})" if r["changed"] else "unchanged"))
        lines += [f"  - {s}" for s in r["stats"]] if r["changed"] else []
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--summary", type=Path, help="write the Markdown summary here")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        results = [refresh(name, *paths, Path(tmp)) for name, paths in CACHES.items()]
    text = summary(results)
    if args.summary:
        args.summary.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
