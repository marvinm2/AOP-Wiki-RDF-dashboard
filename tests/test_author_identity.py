"""Regression tests for dc:creator first-author normalisation (#144).

dc:creator in the AOP-Wiki RDF is free text — a name, a co-author list, or a
workgroup byline, often trailed by affiliations, emails, provenance framing and
reference-number superscripts. `_first_author_identity` collapses each string to
a single first-author (or workgroup) identity so the many spellings of one
contributor merge into one bar on the Top Contributors chart.

These tests pin the behaviour on real-world shapes drawn from the 2026-07-01
graph and, crucially, guard the cases that must NOT over-merge or over-strip.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_DASHBOARD_ROOT = Path(__file__).resolve().parent.parent
if str(_DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_ROOT))

from plots.latest_plots import _first_author_identity  # noqa: E402


def key(raw: str) -> str:
    return _first_author_identity(raw)[1]


def display(raw: str) -> str:
    return _first_author_identity(raw)[0]


# Every variant here is the same contributor ("You Song") and must share a key.
YOU_SONG_VARIANTS = [
    "You Song\n",
    "You Song\n\nNorwegian Institute for Water Research (NIVA), Oslo, Norway\n",
    "You Song1, Jorke H. Kamstra2, Matej Oresic3,4\n\n1 Norwegian Institute",
    "You Song, Li Xie, Knut Erik Tollefsen\n\nNIVA",
    "You Song & co-authors\n\nNIVA",
    "You Song1, Knut Erik Tollefsen1,2\n1Norwegian Institute",
    "You Songa, b, *, Li Xiea, b, c, YeonKyeong Leeb,d, Knut Erik Tollefsena",
    "Simon Schmid 1,2, You Song 1, and Knut Erik Tollefsen 1,2,3",  # NOT You Song — first author is Simon Schmid
]


def test_you_song_variants_merge():
    keys = [key(v) for v in YOU_SONG_VARIANTS[:-1]]
    assert set(keys) == {"you song"}, keys


def test_first_author_is_not_a_later_coauthor():
    # "Simon Schmid 1,2, You Song 1, ..." must key on Simon Schmid, not You Song.
    assert key(YOU_SONG_VARIANTS[-1]) == "simon schmid"


@pytest.mark.parametrize("raw,expected_key", [
    ("Kellie Fay\n", "kellie fay"),
    ("Carlie A. LaLone, U.S. Environmental Protection Agency (x@epa.gov)\n", "carlie a lalone"),
    ("Dan Villeneuve, US EPA Mid-Continent Ecology Division (villeneuve.dan@epa.gov)\n", "dan villeneuve"),
    ("Michelle Angrish, Brian Chorley, U.S. EPA\n", "michelle angrish"),
    ("Of the content populated in the AOP-Wiki: John R. Frisch and Travis Karschnik, GDIT;", "john r frisch"),
    ("\nOf the originating work:\n\nBrooke Bowe\n\n", "brooke bowe"),
    ("\nAnna Lanzoni\n\nMartina Panzarea\n\n", "anna lanzoni"),
    ("CHRISTINE L. RUSSOM (1), DANIEL L. VILLENEUVE* (2)", "christine l russom"),
])
def test_first_author_extraction(raw, expected_key):
    assert key(raw) == expected_key


def test_allcaps_byline_is_titlecased_for_display():
    assert display("CHRISTINE L. RUSSOM (1), DANIEL L. VILLENEUVE* (2)") == "Christine L. Russom"


def test_workgroup_byline_keeps_group_name():
    raw = ("Cancer AOP Workgroup. National Health and Environmental Effects "
           "Research Laboratory, Office of Research and Development,")
    assert display(raw) == "Cancer AOP Workgroup"


def test_short_surname_not_over_stripped():
    # "Dianke Yu" is not followed by a lone-letter affiliation list, so the
    # affiliation-letter rule must NOT truncate it to "Dianke Y".
    assert display("Dianke Yu\n\nDepartment of Toxicology") == "Dianke Yu"


def test_empty_and_garbage_return_blank():
    assert _first_author_identity("") == ("", "")
    assert _first_author_identity("   \n\n  ") == ("", "")


def test_html_entities_and_nbsp_are_cleaned():
    d, k = _first_author_identity("You&nbsp;Song&nbsp;&amp;&nbsp;co-authors")
    assert k == "you song"
    assert "&" not in d and "\xa0" not in d
