"""Shared pytest fixtures for the dashboard test suite.

No imports from the production `plots` package — keep this collection-only and
fast. The methodology lint suite hits the live SPARQL endpoint via
`@pytest.mark.slow` tests; pure-static tests must not require network.
"""

import pathlib

import pytest


@pytest.fixture
def endpoint_url() -> str:
    """Live multi-version SPARQL endpoint used by `@pytest.mark.slow` tests."""
    return "https://aopwiki-multirdf.vhp4safety.nl/sparql"


@pytest.fixture
def fixtures_dir() -> pathlib.Path:
    """Directory containing the frozen methodology-lint fixtures."""
    return pathlib.Path(__file__).parent / "fixtures" / "methodology_lint"
