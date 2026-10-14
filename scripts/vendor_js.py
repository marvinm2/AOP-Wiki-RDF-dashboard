#!/usr/bin/env python3
"""Vendor the dashboard's third-party JavaScript into static/vendor/ (#175).

Each library is downloaded once from its official CDN into
``static/vendor/<name>@<version>/`` and recorded in
``static/vendor/manifest.json`` with its sha384 Subresource Integrity hash.
Templates emit the ``<script>`` tags from that manifest
(``templates/macros/vendor.html``), so pages make no third-party requests and a
tampered file is refused by the browser.

To upgrade a library: change its entry below, run this script, commit the new
directory and manifest, and delete the old directory.

Usage:
    python scripts/vendor_js.py
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "static" / "vendor"

# name -> (version, download URL). Plotly: the cartesian partial bundle covers
# every trace type the plots use (bar, scatter/line/area, box, histogram,
# heatmap/imshow, pie) at about a third of the full bundle's size; keep its
# version in step with what plotly.py (requirements.txt) emits.
LIBRARIES = {
    "plotly-cartesian": ("2.32.0", "https://cdn.plot.ly/plotly-cartesian-2.32.0.min.js"),
    "cytoscape": ("3.33.1", "https://unpkg.com/cytoscape@3.33.1/dist/cytoscape.min.js"),
    # D3 for the views Plotly has no form for (version scrubber #182, ...).
    "d3": ("7.9.0", "https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"),
}


def sri(data: bytes) -> str:
    return "sha384-" + base64.b64encode(hashlib.sha384(data).digest()).decode()


def main() -> int:
    manifest = {}
    for name, (version, url) in LIBRARIES.items():
        filename = url.rsplit("/", 1)[-1]
        target = VENDOR / f"{name}@{version}" / filename
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=60) as response:
                target.write_bytes(response.read())
            print(f"downloaded {url} -> {target.relative_to(ROOT)}")
        data = target.read_bytes()
        manifest[name] = {
            "version": version,
            "path": str(target.relative_to(ROOT / "static")),
            "integrity": sri(data),
            "bytes": len(data),
            "source": url,
        }
    (VENDOR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {(VENDOR / 'manifest.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
