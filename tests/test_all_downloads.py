"""End-to-end audit of every plot download the dashboard exposes.

This suite re-checks that *every* plot can actually be downloaded as CSV, PNG and
SVG, exercising the same routes and ``?version=`` semantics the browser uses. It
deliberately drives the app in-process via ``app.test_client()`` so the full
startup (parallel plot computation against the live SPARQL endpoint) runs exactly
once.

Two ways to run it:

* ``python tests/test_all_downloads.py`` — prints a PASS/FAIL/EMPTY table for the
  whole download surface and exits non-zero if anything is broken. This is the
  operator-facing "re-check each plot download" report.
* ``pytest tests/test_all_downloads.py -m slow`` — same checks as parametrized
  ``@pytest.mark.slow`` tests for CI (gated behind the live endpoint).

Cache-key model this audit relies on (see ``plots/latest_plots.py`` and
``plots/shared.py``):

* Trend plots write **bare** data/figure cache keys at startup
  (``aop_network_density`` etc.), downloadable via ``/download/trend/<key>``.
* Most ``latest_*`` plots write a **bare** key that is overwritten per requested
  version (the ``?version=`` arg then only affects the filename, not selection).
* Five ``latest_*`` plots write a **version-suffixed key only** — no bare alias:
  ``latest_ke_by_bio_level``, ``latest_taxonomic_groups``,
  ``latest_entity_by_oecd_status``, ``latest_ke_reuse``,
  ``latest_ke_reuse_distribution``. They are *not* precomputed at startup, so a
  download must be preceded by a view (``/api/plot/<name>?version=``) carrying the
  same version — which is what the version selector does.

Author:
    Marvin Martens
"""
from __future__ import annotations

import io
import os
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest

# Make the dashboard package importable when pytest is invoked from the repo root.
_DASHBOARD_ROOT = Path(__file__).resolve().parent.parent
if str(_DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_ROOT))

LIVE_ENDPOINT = "https://aopwiki-multirdf.vhp4safety.nl/sparql"

# --- Plot inventory (authoritative, sourced from app.py + the plot modules) ----

# latest_* plots registered in app.get_plot's ``latest_plots_with_version`` map.
LATEST_PLOTS = [
    "latest_entity_counts",
    "latest_ke_components",
    "latest_aop_connectivity",
    "latest_avg_per_aop",
    "latest_process_usage",
    "latest_object_usage",
    "latest_aop_completeness",
    "latest_aop_completeness_by_status",
    "latest_ke_completeness_by_status",
    "latest_ker_completeness_by_status",
    "latest_ontology_usage",
    "latest_ke_annotation_depth",
    "latest_ke_by_bio_level",
    "latest_taxonomic_groups",
    "latest_entity_by_oecd_status",
    "latest_ke_reuse",
    "latest_ke_reuse_distribution",
    "latest_top_ontology_terms",
    "latest_author_contributions",
    "latest_ontology_diversity",
    "latest_aop_completeness_unique",
    "latest_organ_coverage",
    "latest_organ_coverage_percentage",
    "latest_organ_coverage_apical",
    "latest_organ_coverage_ao_only",
    "latest_organ_coverage_unified",
    "latest_organ_coverage_pie",
    "latest_multi_organ_aops",
    "latest_life_stage",
    "latest_ke_mmo_coverage",
    "latest_aop_aop_overlap",
]

# Bare data/figure cache keys written by the trend plots at startup. Downloaded
# through the generic /download/trend/<key> route.
TREND_PLOTS = [
    "main_graph_absolute",
    "main_graph_delta",
    "aop_entity_counts_absolute",
    "aop_entity_counts_delta",
    "entity_birth_death",
    "entity_cumulative_removed",
    "oecd_status_distribution_absolute",
    "oecd_status_distribution_percentage",
    "stressor_coverage_growth_absolute",
    "stressor_coverage_growth_delta",
    "aops_per_stressor_distribution_absolute",
    "aops_per_stressor_distribution_percentage",
    "ke_mmo_coverage_absolute",
    "ke_mmo_coverage_percentage",
    "ke_migration_map",
    "average_components_per_aop_absolute",
    "average_components_per_aop_delta",
    "aop_network_density",
    "aop_authors_absolute",
    "aop_authors_delta",
    "aops_created_over_time",
    "aop_creation_vs_modification_timeline",
    "ke_component_annotations_absolute",
    "ke_component_annotations_delta",
    "ke_components_percentage_absolute",
    "ke_components_percentage_delta",
    "unique_ke_components_absolute",
    "unique_ke_components_delta",
    "biological_process_annotations_absolute",
    "biological_process_annotations_delta",
    "biological_object_annotations_absolute",
    "biological_object_annotations_delta",
    "aop_property_presence_absolute",
    "aop_property_presence_percentage",
    "ke_property_presence_absolute",
    "ke_property_presence_percentage",
    "ker_property_presence_absolute",
    "ker_property_presence_percentage",
    "stressor_property_presence_absolute",
    "stressor_property_presence_percentage",
    "kes_by_kec_count_absolute",
    "kes_by_kec_count_delta",
    "entity_completeness_trends",
    "aop_completeness_boxplot",
    "aop_completeness_boxplot_all",
    "oecd_completeness_trend",
    "ontology_term_growth_absolute",
    "ontology_term_growth_delta",
    "organ_coverage_absolute",
    "organ_coverage_percentage",
]

# Explicitly-registered routes (bare-key lookups) worth hitting directly to catch
# the "explicit route ignores version / bare key missing" 404 class.
EXPLICIT_ROUTES = [
    "/download/latest_entity_counts",
    "/download/latest_ke_components",
    "/download/latest_aop_completeness",
    "/download/main_graph_absolute",
    "/download/main_graph_delta",
    "/download/trend/ke_property_presence_absolute",
    "/download/trend/ker_property_presence_percentage",
    "/download/trend/stressor_property_presence_absolute",
]

PNG_MIN_BYTES = 2000
SVG_MIN_BYTES = 500
ZIP_MIN_ENTRIES = 1


@dataclass
class CheckResult:
    """Outcome of a single download check."""

    target: str
    fmt: str
    status: str  # PASS | FAIL | EMPTY
    detail: str = ""
    allow_empty: bool = False  # EMPTY tolerated (e.g. sparse historical slice)

    @property
    def ok(self) -> bool:
        return self.status == "PASS"


@dataclass
class AuditReport:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.results.append(result)

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results
                if r.status == "FAIL" or (r.status == "EMPTY" and not r.allow_empty)]

    def render(self) -> str:
        lines = []
        width = max((len(r.target) for r in self.results), default=20) + 2
        for r in self.results:
            flag = {"PASS": "✓", "EMPTY": "∅", "FAIL": "✗"}[r.status]
            lines.append(f"  {flag} {r.target:<{width}} {r.fmt:<5} {r.detail}")
        passed = sum(1 for r in self.results if r.status == "PASS")
        lines.append("")
        lines.append(f"  {passed}/{len(self.results)} checks passed, "
                     f"{len(self.failures)} need attention")
        return "\n".join(lines)


# --- CSV / image validators ----------------------------------------------------

def _validate_csv(body: bytes, expect_version: str | None = None) -> CheckResult:
    text = body.decode("utf-8", errors="replace")
    if not text.startswith("# AOP-Wiki RDF Dashboard Export"):
        return CheckResult("", "csv", "FAIL", "missing metadata header")
    data_lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    if len(data_lines) < 2:  # column row + >=1 data row
        return CheckResult("", "csv", "EMPTY", f"{len(data_lines)} data line(s)")
    detail = f"{len(data_lines) - 1} rows"
    if expect_version:
        db_line = next((ln for ln in text.splitlines()
                        if ln.startswith("# Database Version:")), None)
        if db_line and expect_version not in db_line:
            return CheckResult("", "csv", "FAIL",
                               f"version mismatch: asked {expect_version}, got "
                               f"'{db_line.split(':', 1)[1].strip()}'")
    return CheckResult("", "csv", "PASS", detail)


def _validate_image(body: bytes, fmt: str) -> CheckResult:
    floor = PNG_MIN_BYTES if fmt == "png" else SVG_MIN_BYTES
    if len(body) < floor:
        return CheckResult("", fmt, "FAIL", f"{len(body)} bytes (< {floor})")
    return CheckResult("", fmt, "PASS", f"{len(body)} bytes")


def _check(client, url: str, fmt: str, target: str,
           expect_version: str | None = None) -> CheckResult:
    resp = client.get(url)
    if resp.status_code != 200:
        body = resp.get_data(as_text=True)[:120].replace("\n", " ")
        r = CheckResult(target, fmt, "FAIL", f"HTTP {resp.status_code}: {body}")
        return r
    if fmt == "csv":
        r = _validate_csv(resp.get_data(), expect_version)
    elif fmt in ("png", "svg"):
        r = _validate_image(resp.get_data(), fmt)
    else:  # json (network graph)
        r = CheckResult("", fmt, "PASS" if resp.get_data() else "EMPTY",
                        f"{len(resp.get_data())} bytes")
    r.target, r.fmt = target, fmt
    return r


# --- The audit -----------------------------------------------------------------

def run_audit(client, *, latest: str, historical: str,
              formats=("csv", "png", "svg")) -> AuditReport:
    """Exercise the whole download surface and return a structured report."""
    report = AuditReport()

    # 1) Latest-version: every latest_* plot, all formats. View first (populates
    #    version-suffixed cache keys), then download with the same version.
    for plot in LATEST_PLOTS:
        client.get(f"/api/plot/{plot}?version={latest}")
        for fmt in formats:
            report.add(_check(client, f"/download/{plot}?format={fmt}&version={latest}",
                              fmt, f"{plot}@latest"))

    # 2) One historical version: CSV only, asserting the returned data actually
    #    corresponds to the requested version (catches version-ignored plots).
    for plot in LATEST_PLOTS:
        client.get(f"/api/plot/{plot}?version={historical}")
        r = _check(client, f"/download/{plot}?format=csv&version={historical}",
                   "csv", f"{plot}@{historical}", expect_version=historical)
        r.allow_empty = True  # older snapshots legitimately lack some annotations
        report.add(r)

    # 3) The five version-suffixed-only plots via the /download/latest/<suffix>
    #    route the template actually uses for them.
    for plot in ("latest_ke_by_bio_level", "latest_taxonomic_groups",
                 "latest_entity_by_oecd_status", "latest_ke_reuse",
                 "latest_ke_reuse_distribution", "latest_top_ontology_terms",
                 "latest_author_contributions"):
        suffix = plot[len("latest_"):]
        client.get(f"/api/plot/{plot}?version={latest}")
        report.add(_check(client,
                          f"/download/latest/{suffix}?format=csv&version={latest}",
                          "csv", f"/download/latest/{suffix}"))

    # 4) Trend plots (precomputed at startup), all formats.
    for plot in TREND_PLOTS:
        for fmt in formats:
            report.add(_check(client, f"/download/trend/{plot}?format={fmt}",
                              fmt, f"trend/{plot}"))

    # 5) Snapshot-range filter on a couple of representative trend CSVs (#44).
    for plot in ("aop_entity_counts_absolute", "aop_network_density"):
        report.add(_check(
            client,
            f"/download/trend/{plot}?format=csv&start=2020-01-01&end=2022-01-01",
            "csv", f"trend/{plot}[range]"))

    # 6) Explicit routes (bare-key lookups).
    for url in EXPLICIT_ROUTES:
        report.add(_check(client, f"{url}?format=csv", "csv", url))

    # 7) Network downloads.
    report.add(_check(client, "/download/network/metrics?format=csv", "csv",
                      "network/metrics"))
    report.add(_check(client, "/download/network/graph", "json", "network/graph"))

    # 8) Bulk ZIP — open it and assert it carries entries.
    resp = client.get("/download/bulk?category=all&formats=csv,png,svg")
    if resp.status_code != 200:
        report.add(CheckResult("bulk[all]", "zip", "FAIL", f"HTTP {resp.status_code}"))
    else:
        try:
            zf = zipfile.ZipFile(io.BytesIO(resp.get_data()))
            n = len(zf.namelist())
            status = "PASS" if n >= ZIP_MIN_ENTRIES else "EMPTY"
            report.add(CheckResult("bulk[all]", "zip", status, f"{n} entries"))
        except zipfile.BadZipFile:
            report.add(CheckResult("bulk[all]", "zip", "FAIL", "not a valid zip"))

    return report


# --- Shared client construction ------------------------------------------------

def _build_client():
    os.environ.setdefault("SPARQL_ENDPOINT", LIVE_ENDPOINT)
    os.environ.setdefault("SPARQL_PUBLIC_ENDPOINT", os.environ["SPARQL_ENDPOINT"])
    import app as dashboard_app  # noqa: E402 — import triggers startup compute
    client = dashboard_app.app.test_client()
    versions = [v["version"] for v in
                dashboard_app.get_all_versions()]  # newest first
    return client, versions


def _pick_versions(versions: list[str]) -> tuple[str, str]:
    latest = versions[0]
    historical = "2020-04-01" if "2020-04-01" in versions else versions[-1]
    return latest, historical


# --- pytest integration --------------------------------------------------------

@pytest.fixture(scope="session")
def audit_report():
    client, versions = _build_client()
    latest, historical = _pick_versions(versions)
    return run_audit(client, latest=latest, historical=historical)


@pytest.mark.slow
def test_no_download_failures(audit_report):
    """Every plot download returns valid, non-empty content."""
    failures = audit_report.failures
    assert not failures, "Broken downloads:\n" + "\n".join(
        f"  {r.status} {r.target} [{r.fmt}] — {r.detail}" for r in failures)


# --- CLI report ----------------------------------------------------------------

def main() -> int:
    print("Importing dashboard app (startup computes plots against the live "
          "endpoint, ~60s)...")
    client, versions = _build_client()
    latest, historical = _pick_versions(versions)
    print(f"Latest version: {latest} | historical probe: {historical} | "
          f"{len(versions)} versions total\n")
    report = run_audit(client, latest=latest, historical=historical)
    print(report.render())
    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
