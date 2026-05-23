"""Synthetic plot fixture for LINT-04: function issues more SPARQL calls than disclosed.

This file is parsed via ``ast.parse`` by ``scripts/lint_methodology.py`` —
it is never imported or executed. The ``from plots.shared import …`` line
exists only so the AST scan finds the canonical helper name.
"""

from plots.shared import run_sparql_query_with_retry  # noqa: F401 — AST anchor only


def plot_latest_over_disclosed():
    """A plot that issues three SPARQL calls but the methodology JSON discloses one."""
    a = run_sparql_query_with_retry("SELECT (COUNT(?aop) AS ?c) WHERE { ?aop a <http://aopkb.org/aop_ontology#AdverseOutcomePathway> } LIMIT 1")
    b = run_sparql_query_with_retry("SELECT (COUNT(?ke) AS ?c) WHERE { ?ke a <http://aopkb.org/aop_ontology#KeyEvent> } LIMIT 1")
    c = run_sparql_query_with_retry("SELECT (COUNT(?ker) AS ?c) WHERE { ?ker a <http://aopkb.org/aop_ontology#KeyEventRelationship> } LIMIT 1")
    return a, b, c
