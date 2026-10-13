#!/usr/bin/env python3
"""Write static/data/tokens.json from BRAND_COLORS (#181).

``plots/shared.py`` BRAND_COLORS is the one source of the dashboard's colours.
Browser code that needs them (the network page's community colours) reads
this file instead of keeping its own copy. ``tests/test_tokens.py`` fails when
the file is out of date, and when a brand variable in static/css/main.css
disagrees with it.

Usage:
    python scripts/build_tokens.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOKENS = ROOT / "static" / "data" / "tokens.json"


def tokens() -> dict:
    sys.path.insert(0, str(ROOT))
    from plots.shared import BRAND_COLORS
    return {
        "_comment": "Generated from plots/shared.py BRAND_COLORS by scripts/build_tokens.py. Do not edit.",
        "color": BRAND_COLORS,
    }


def render() -> str:
    return json.dumps(tokens(), indent=2) + "\n"


def main() -> None:
    TOKENS.write_text(render())
    print(f"wrote {TOKENS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
