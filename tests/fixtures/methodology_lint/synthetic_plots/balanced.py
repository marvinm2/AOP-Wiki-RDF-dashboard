"""Synthetic plot fixture for LINT-04: function issues exactly one SPARQL call.

The methodology JSON discloses one SPARQL string for ``latest_balanced``.
Disclosure count and call count match (1 == 1), so LINT-04 must pass.
"""

from plots.shared import run_sparql_query_with_retry  # noqa: F401 — AST anchor only


def plot_latest_balanced():
    """A plot that issues exactly one SPARQL call, matching its disclosure."""
    return run_sparql_query_with_retry(
        "SELECT (COUNT(?aop) AS ?c) WHERE { ?aop a <http://aopkb.org/aop_ontology#AdverseOutcomePathway> } LIMIT 1"
    )
