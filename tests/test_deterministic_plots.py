"""Plots render the same bytes in every process (#220).

Python randomises set iteration order per process (PYTHONHASHSEED), and the
parallel per-version queries finish in any order. Before #220 that leaked into
four plots: row order in the completeness boxplots, the overlap network's
layout, and the last digits of the completeness correlations. A content-hashed
artifact store (#179) would see each rebuild as a change.

Each case renders a plot from fixed stub data in two subprocesses with
different hash seeds and compares the output. Offline.
"""

import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

STUBS = textwrap.dedent('''
    import re, sys
    from concurrent.futures import Future
    import plots
    from plots import latest_plots, trends_plots

    G = "http://aopwiki.org/graph/"
    AOP = "https://identifiers.org/aop/"
    KE = "https://identifiers.org/aop.events/"
    TITLE = "http://purl.org/dc/elements/1.1/title"
    CREATOR = "http://purl.org/dc/elements/1.1/creator"
    # 30 AOPs over 40 KEs, with overlapping KE sets and mixed property presence.
    MEMBERS = {a: sorted({(a * 7 + i * 3) % 40 for i in range(2 + a % 5)}) for a in range(1, 31)}

    def latest_answer(query):
        if "SELECT DISTINCT ?aop ?p" in query:
            rows = []
            for a in MEMBERS:
                rows.append({"aop": {"value": f"{AOP}{a}"}, "p": {"value": "http://aopkb.org/aop_ontology#has_key_event"}})
                if a % 3:
                    rows.append({"aop": {"value": f"{AOP}{a}"}, "p": {"value": CREATOR}})
                if a % 4:
                    rows.append({"aop": {"value": f"{AOP}{a}"}, "p": {"value": "http://aopkb.org/aop_ontology#LifeStageContext"}})
            return rows
        if "has_key_event ?ke" in query:
            return [{"aop": {"value": f"{AOP}{a}"}, "ke": {"value": f"{KE}{k}"}}
                    for a, kes in MEMBERS.items() for k in kes]
        if "VALUES ?aop" in query:
            return [{"aop": {"value": u}, "title": {"value": "AOP " + u.rsplit("/", 1)[-1]}}
                    for u in re.findall(r"<([^>]+)>", query.split("VALUES ?aop", 1)[1].split("}", 1)[0])]
        return []

    VERSIONS = ["2024-01-01", "2024-04-01", "2024-07-01"]

    def network(version):
        return [{"aop": {"value": f"{AOP}{a}"},
                 "kes": {"value": "|".join(f"{KE}{k}" for k in kes)}, "kers": {"value": ""}}
                for a, kes in MEMBERS.items()]

    def box_version(v, *args):
        return {"version": v["version"],
                "totals": [{"entity_type": {"value": t}, "total": {"value": "30"}} for t in ("AOP", "KE", "KER")],
                "presence": [{"entity_type": {"value": "AOP"}, "p": {"value": CREATOR}, "count": {"value": "20"}}],
                "network": network(v["version"])}

    def box_entity_props(v, *args):
        rows = [{"entity": {"value": f"{AOP}{a}"}, "entity_type": {"value": "AOP"}, "prop_count": {"value": str(1 + a % 4)}}
                for a in MEMBERS]
        rows += [{"entity": {"value": f"{KE}{k}"}, "entity_type": {"value": "KE"}, "prop_count": {"value": str(1 + k % 3)}}
                 for k in range(40)]
        return {"version": v["version"], "entity_props": rows, "network": network(v["version"])}

    class Pool:
        """Runs each task at submit time; as_completed below picks the order."""
        def __init__(self, *a, **k): self.futures = []
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def submit(self, fn, *args):
            f = Future(); f.set_result(fn(*args)); self.futures.append(f); return f

    latest_plots.run_sparql_query = latest_answer
    trends_plots.get_all_versions = lambda: [{"version": v} for v in VERSIONS]
    trends_plots._query_boxplot_version = box_version
    trends_plots._query_boxplot_entity_props = box_entity_props
    trends_plots.ThreadPoolExecutor = Pool
    # Finishing order varies run to run; tie it to the hash seed so it varies here too.
    trends_plots.as_completed = lambda futures: sorted(futures, key=lambda f: hash(futures[f]))

    UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
    case = sys.argv[1]
    if case == "boxplot":
        out = trends_plots.plot_aop_completeness_boxplot()
        out = "".join(out) + trends_plots._plot_data_cache.get("aop_completeness_boxplot").to_csv()
    elif case == "overlap":
        out = plots.plot_latest_aop_aop_overlap("2026-10-01", min_jaccard=0.3, min_shared_kes=1)
    else:
        out = plots.plot_latest_completeness_correlation("2026-10-01")
    sys.stdout.write(UUID.sub("U", out))
''')


def _render(case, seed):
    env = {**os.environ, "PYTHONHASHSEED": str(seed)}
    proc = subprocess.run([sys.executable, "-c", STUBS, case], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc.stdout


@pytest.mark.parametrize("case", ["boxplot", "overlap", "correlation"])
def test_same_output_under_different_hash_seeds(case):
    first = _render(case, 1)
    assert "plotly" in first and "Data Unavailable" not in first
    assert _render(case, 2) == first
    assert _render(case, 3) == first
