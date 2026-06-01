"""Export per-version plot bundles (CSV + PNG + SVG + manifest) for archival.

Produces, for each quarterly version, a folder of every snapshot plot rendered at
that version plus the underlying CSV data and a manifest — the artifact archived
to the per-version plot-data Zenodo record (see ../ZENODO.md). Trend plots (whole
history) are exported once into a separate `trends/` bundle.

It drives the dashboard in-process through ``app.test_client()`` so it reuses the
exact route + cache-key logic the browser uses (no reimplementation). Importing
the app runs the normal startup against the configured SPARQL endpoint.

Usage
-----
    # Latest version + trends bundle (the usual quarterly refresh)
    python scripts/export_all_versions.py --out exports

    # A specific version, or every version, and zip each bundle
    python scripts/export_all_versions.py 2026-04-01 --out exports --zip
    python scripts/export_all_versions.py --all --out exports --zip
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.test_all_downloads import LATEST_PLOTS, TREND_PLOTS  # single source of truth

GRAPH_BASE = "http://aopwiki.org/graph/"
FORMATS = ("csv", "png", "svg")


def _write(path: Path, data: bytes) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return len(data)


def export_version(client, version: str, out_dir: Path, formats=FORMATS) -> dict:
    """Export every snapshot (latest_*) plot at `version`. Returns a manifest dict."""
    vdir = out_dir / version
    files: list[dict] = []
    for plot in LATEST_PLOTS:
        # View first so the version-specific cache keys are populated.
        client.get(f"/api/plot/{plot}?version={version}")
        for fmt in formats:
            resp = client.get(f"/download/{plot}?format={fmt}&version={version}")
            if resp.status_code == 200 and resp.get_data():
                n = _write(vdir / f"{plot}.{fmt}", resp.get_data())
                files.append({"plot": plot, "format": fmt, "bytes": n})
    manifest = {
        "version": version,
        "graph_uri": f"{GRAPH_BASE}{version}",
        "endpoint": os.environ.get("SPARQL_ENDPOINT", "http://localhost:8890/sparql"),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "scope": "latest-snapshot plots at this version",
        "files": files,
        "methodology": "methodology.json (SPARQL query per plot, in this bundle)",
    }
    (vdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # Ship the query provenance alongside the data.
    meth = _ROOT / "static" / "data" / "methodology_notes.json"
    if meth.exists():
        shutil.copyfile(meth, vdir / "methodology.json")
    return manifest


def export_trends(client, out_dir: Path, formats=FORMATS) -> dict:
    """Export the whole-history trend plots once into a `trends/` bundle."""
    tdir = out_dir / "trends"
    files: list[dict] = []
    for plot in TREND_PLOTS:
        for fmt in formats:
            resp = client.get(f"/download/trend/{plot}?format={fmt}")
            if resp.status_code == 200 and resp.get_data():
                n = _write(tdir / f"{plot}.{fmt}", resp.get_data())
                files.append({"plot": plot, "format": fmt, "bytes": n})
    manifest = {
        "scope": "historical trend plots (full version history)",
        "endpoint": os.environ.get("SPARQL_ENDPOINT", "http://localhost:8890/sparql"),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    (tdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _zip_dir(src: Path, dst_zip: Path) -> None:
    with zipfile.ZipFile(dst_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(src.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("versions", nargs="*", help="Version dates to export (default: latest).")
    parser.add_argument("--all", action="store_true", help="Export every available version.")
    parser.add_argument("--out", default="exports", help="Output directory (default: exports).")
    parser.add_argument("--formats", default="csv,png,svg", help="Comma-separated formats.")
    parser.add_argument("--no-trends", action="store_true", help="Skip the trends bundle.")
    parser.add_argument("--zip", action="store_true", help="Zip each bundle.")
    args = parser.parse_args()

    os.environ.setdefault("SPARQL_ENDPOINT", "https://aopwiki-multirdf.vhp4safety.nl/sparql")
    os.environ.setdefault("SPARQL_PUBLIC_ENDPOINT", os.environ["SPARQL_ENDPOINT"])
    formats = tuple(f.strip() for f in args.formats.split(",") if f.strip())

    import app as dashboard_app  # triggers startup
    client = dashboard_app.app.test_client()
    all_versions = [v["version"] for v in dashboard_app.get_all_versions()]
    if not all_versions:
        print("No versions available from the endpoint.", file=sys.stderr)
        return 1

    if args.all:
        targets = all_versions
    elif args.versions:
        targets = args.versions
    else:
        targets = [all_versions[0]]  # latest

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for v in targets:
        print(f"Exporting version {v} ...")
        m = export_version(client, v, out_dir, formats)
        print(f"  {len(m['files'])} files")
        if args.zip:
            _zip_dir(out_dir / v, out_dir / f"aopwiki-plots-{v}.zip")

    if not args.no_trends:
        print("Exporting trends bundle ...")
        m = export_trends(client, out_dir, formats)
        print(f"  {len(m['files'])} files")
        if args.zip:
            _zip_dir(out_dir / "trends", out_dir / "aopwiki-plots-trends.zip")

    print(f"Done. Bundles in {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
