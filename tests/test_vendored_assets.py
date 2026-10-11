"""Third-party JavaScript is vendored with Subresource Integrity (#175)."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"


def test_manifest_hashes_match_the_vendored_files():
    manifest = json.loads((STATIC / "vendor" / "manifest.json").read_text())
    assert {"plotly-cartesian", "cytoscape"} <= set(manifest)
    for name, lib in manifest.items():
        data = (STATIC / lib["path"]).read_bytes()
        digest = "sha384-" + base64.b64encode(hashlib.sha384(data).digest()).decode()
        assert lib["integrity"] == digest, name


def test_no_page_loads_scripts_or_styles_from_a_cdn():
    pattern = re.compile(r'<(?:script|link)[^>]+(?:src|href)="https?://', re.I)
    offenders = [p.name for p in (ROOT / "templates").rglob("*.html") if pattern.search(p.read_text())]
    assert offenders == []


def test_fallback_plots_do_not_pull_plotly_from_a_cdn():
    assert 'include_plotlyjs="cdn"' not in (ROOT / "plots" / "shared.py").read_text()
