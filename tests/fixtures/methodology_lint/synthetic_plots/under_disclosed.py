"""Synthetic plot fixture for LINT-04: function has fewer SPARQL calls than disclosed.

The methodology JSON discloses one SPARQL string for ``latest_under_disclosed``,
but the plot function below issues zero — a legitimate scenario where the
plot reads from a precomputed pickle or cache. LINT-04 must NOT warn here
because ``code_count (0) <= disclosed_count (1)`` (see RESEARCH Open Q2).
"""


def plot_latest_under_disclosed():
    """A plot that reads from a cache and makes no live SPARQL calls."""
    # Pretend we load data from a precomputed source rather than querying.
    return {"cached": True}
