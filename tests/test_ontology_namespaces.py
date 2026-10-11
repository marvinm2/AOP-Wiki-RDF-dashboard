"""Every ontology view buckets term namespaces with one map (#155).

Before #155 six views carried five different hand-written bucket lists, and the
methodology note of one of them disclosed a list its code did not use. These
tests pin the plots and the disclosed queries to ``plots.shared``'s single map.
Offline: no endpoint is contacted.
"""

import json
import pathlib
import re

import pytest

from plots.shared import (
    ONTOLOGY_NAMESPACES,
    ONTOLOGY_OTHER,
    ontology_bucket,
    ontology_bucket_sparql,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTES = json.loads((ROOT / "static" / "data" / "methodology_notes.json").read_text())


@pytest.mark.parametrize("label,prefix", ONTOLOGY_NAMESPACES)
def test_python_rule_buckets_each_namespace(label, prefix):
    assert ontology_bucket(prefix + "0000001") == label


def test_unknown_namespace_is_other():
    assert ontology_bucket("http://purl.obolibrary.org/obo/PATO_0000001") == ONTOLOGY_OTHER


def test_sparql_rule_lists_namespaces_in_python_order():
    """SPARQL and Python must check prefixes in the same order to agree."""
    expr = ontology_bucket_sparql("t")
    found = re.findall(r'STRSTARTS\(STR\(\?t\), "([^"]+)"\), "([^"]+)"', expr)
    assert found == [(p, l) for l, p in ONTOLOGY_NAMESPACES]
    assert expr.rstrip(")").endswith(f'"{ONTOLOGY_OTHER}"')
    assert expr.count("(") == expr.count(")")


def test_no_hand_written_bucket_lists_in_plots():
    """Bucketing goes through ontology_bucket_sparql, never a local IF chain."""
    offenders = [
        f"{path.name}:{n}"
        for path in (ROOT / "plots").glob("*.py")
        if path.name != "shared.py"
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if "IF(STRSTARTS" in line
    ]
    assert offenders == []


# (note key, where the query lives, bucketed variable)
DISCLOSED = [
    ("latest_ontology_usage", ("queries", 1, "query"), "term"),
    ("latest_process_usage", ("queries", 1, "query"), "process"),
    ("latest_object_usage", ("queries", 1, "query"), "object"),
    ("bio_annotations", ("sparql",), "process"),
]


def _query(key, path):
    node = NOTES[key]
    for step in path:
        node = node[step]
    return node


@pytest.mark.parametrize("key,path,var", DISCLOSED)
def test_disclosed_query_uses_the_shared_map(key, path, var):
    query = _query(key, path)
    assert ontology_bucket_sparql(var) in query
    assert f"FILTER(isIRI(?{var}))" in query


def test_every_note_with_a_bucket_list_is_checked():
    """A new note that buckets namespaces must be added to DISCLOSED."""
    bucketing = {k for k, e in NOTES.items() if "IF(STRSTARTS" in json.dumps(e)}
    assert bucketing == {k for k, _, _ in DISCLOSED}


@pytest.mark.parametrize("key", ["latest_ontology_usage", "latest_top_ontology_terms"])
def test_action_slot_not_queried(key):
    """aopo:hasAction holds literal directions only — no ontology terms."""
    assert "hasAction" not in json.dumps(NOTES[key]["queries"])
