"""The query-store build covers every setting the Snapshot page offers (#179).

A setting the build does not pre-render still works, but goes to the endpoint.
These tests keep scripts/build_query_store.py in step with the page's controls.
Offline.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_query_store as build  # noqa: E402
from plots import registry  # noqa: E402

PAGE = (ROOT / "templates" / "latest.html").read_text()


def _values(spec_id, param):
    return [v[param] for v in build.setting_variants(registry.BY_ID[spec_id])]


def test_overlap_slider_steps():
    slider = re.search(r'id="aop-aop-threshold-slider"[^>]*', PAGE).group(0)
    lo, hi, step = (float(re.search(f'{a}="([^"]+)"', slider).group(1)) for a in ("min", "max", "step"))
    expected = [round(lo + step * i, 2) for i in range(int(round((hi - lo) / step)) + 1)]
    assert sorted({float(v) for v in _values("latest_aop_aop_overlap", "min_jaccard")}) == expected


def test_coverage_toggles():
    scopes = re.findall(r'data-toggle-scope="([^"]+)"', PAGE)
    assert set(_values("latest_organ_coverage_unified", "scope")) == set(scopes)
    assert set(_values("latest_organ_coverage_pie", "scope")) == set(scopes)
    views = set(re.findall(r'data-toggle-view="([^"]+)"', PAGE)) or {"absolute", "percentage"}
    assert set(_values("latest_organ_coverage_unified", "view")) == views


def test_every_tuning_param_has_choices():
    """A new tuning param must get its choices in setting_variants()."""
    for spec in registry.snapshot_specs():
        variants = build.setting_variants(spec)
        if set(spec.params) & build.ON_DEMAND:
            assert variants == [], spec.id   # one entity at a time, never stored
            continue
        assert variants, spec.id
        missing = set(spec.params) - set(variants[0]) - {"min_shared_kes"}
        assert not missing, (spec.id, missing)


def test_plan_runs_each_version_for_each_setting():
    runs = build.plan_runs(registry, ["2026-10-01", "2024-01-01"])
    names = [(name, version) for name, version, _ in runs]
    assert len(names) == len(set(names))
    overlap = [n for n, v in names if n.startswith("snapshot:latest_aop_aop_overlap")]
    assert len(overlap) == 2 * len(_values("latest_aop_aop_overlap", "min_jaccard"))
