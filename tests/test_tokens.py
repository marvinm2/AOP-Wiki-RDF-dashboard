"""static/data/tokens.json follows BRAND_COLORS, and so does main.css (#181)."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from plots.shared import BRAND_COLORS

ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location("build_tokens", ROOT / "scripts" / "build_tokens.py")
build_tokens = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_tokens)


def test_tokens_json_is_up_to_date():
    assert build_tokens.TOKENS.read_text() == build_tokens.render(), \
        "run python scripts/build_tokens.py"


def test_css_brand_variables_match_brand_colors():
    css = (ROOT / "static" / "css" / "main.css").read_text()
    declared = dict(re.findall(r"--color-([a-z-]+):\s*(#[0-9A-Fa-f]{6})", css))
    named = {k: v for k, v in BRAND_COLORS.items() if isinstance(v, str)}
    checked = {k for k in named if k.replace("_", "-") in declared}
    assert {"primary", "magenta", "blue", "orange", "teal"} <= checked
    for key in checked:
        assert declared[key.replace("_", "-")].upper() == named[key].upper(), key


def test_no_brand_palette_copy_in_page_scripts():
    # Page code reads the palette from tokens.json, not from its own list.
    js = (ROOT / "static" / "js" / "network-graph.js").read_text()
    assert "'#29235C', '#E6007E'" not in js
