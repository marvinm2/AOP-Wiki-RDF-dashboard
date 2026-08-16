"""Offline tests for the domain lens (#149).

Two things are worth guarding without an endpoint: the placeholder rule that
decides whether a free-text field counts as filled, and the shape and content of
the committed domain vocabulary. Both are where a silent regression would move
published numbers — the placeholder rule feeds the Assessment tier, and the
cache decides which AOPs are in a domain at all.

No endpoint, no Ubergraph.
"""

import pytest

from plots.domain_plots import (
    DEFAULT_DOMAIN,
    _load_domain_lens_cache,
    _resolve_domain,
    available_domains,
    domain_term_iris,
    is_placeholder,
)
from plots.shared import get_properties_for_entity


class TestPlaceholderRule:
    """Presence queries count `&nbsp;` as a filled field; this rule is the fix."""

    @pytest.mark.parametrize("value", [
        "", "   ", "\n", "&nbsp;", "&nbsp;\n", "&nbsp;\n\n&nbsp;\n",
        "&nbsp; &nbsp;&nbsp;\n", "\xa0", "-", "--", "TBD\n", "tbd", "N/A\n",
        "N/A.", "n.a.", "?", "x", "TBA", "to be determined", ".", "...",
        "<p>&nbsp;</p>", "<p></p>",
    ])
    def test_formatting_artefacts_are_placeholders(self, value):
        assert is_placeholder(value)

    @pytest.mark.parametrize("value", [
        "No Data.\n",
        "None identified\n",
        "There are no known inconsistencies.\n",
        "no inconsistencies\n",
        "high\n",
        "See details below.\n",
        "TAIR:AT5G64350",
    ])
    def test_curator_answers_are_not_placeholders(self, value):
        """A statement about the science is content, however terse.

        Stripping "None identified" would silently rewrite a curator's answer
        into a missing field, which is a judgement the dashboard does not get to
        make. The rule is typographic only.
        """
        assert not is_placeholder(value)

    def test_long_text_short_circuits(self):
        assert not is_placeholder("Considerations for applications. " * 10)

    def test_missing_value_is_a_placeholder(self):
        assert is_placeholder(None)


class TestDomainResolution:
    def test_unknown_domain_falls_back_to_the_default(self):
        assert _resolve_domain("Pancreatic") == DEFAULT_DOMAIN
        assert _resolve_domain(None) == DEFAULT_DOMAIN

    def test_known_domain_is_kept(self):
        for name in available_domains():
            assert _resolve_domain(name) == name


class TestCacheContract:
    """Guard the shape the builder writes and the plots read."""

    def cache(self):
        cache = _load_domain_lens_cache()
        if not cache:
            pytest.skip("domain_lens_cache.json not built in this checkout")
        return cache

    def test_envelope(self):
        cache = self.cache()
        assert {"generated_at", "source", "domains", "stats"} <= set(cache)
        assert DEFAULT_DOMAIN in cache["domains"]
        cv = cache["domains"][DEFAULT_DOMAIN]
        assert {"roots", "closure_size", "terms", "terms_by_namespace"} <= set(cv)
        sample = next(iter(cv["terms"].values()))
        assert {"label", "roots", "snapshots"} <= set(sample)

    def test_cardiovascular_roots_are_the_validated_five(self):
        """The preset the JRC comparison was measured on (28/41 precision).

        Changing it is allowed — but it invalidates the precision and recall
        figures published in the methodology note, so it has to be deliberate.
        """
        cv = self.cache()["domains"][DEFAULT_DOMAIN]
        assert [r["id"] for r in cv["roots"]] == [
            "UBERON_0001009", "GO_0003013", "GO_0072359",
            "MP_0005385", "HP_0001626",
        ]

    def test_every_domain_has_anatomy_and_process_or_phenotype_roots(self):
        """Anatomy alone under-counts.

        On 2026-07-01 the cardiovascular UBERON branch alone reaches 34 of the
        41 AOPs; the other 7 are annotated only with GO processes or MP/HP
        phenotypes.
        """
        for name, data in self.cache()["domains"].items():
            namespaces = {r["id"].split("_", 1)[0] for r in data["roots"]}
            assert "UBERON" in namespaces, f"{name} has no anatomy root"
            assert namespaces & {"GO", "MP", "HP"}, \
                f"{name} has no process or phenotype root"

    def test_no_foreign_namespace_terms(self):
        """Each root contributes only its own namespace family.

        Ubergraph mixes FBbt (Drosophila), PCL and the mouse brain atlas into
        UBERON branches. AOP-Wiki annotates with none of them, and an accidental
        widening would pull unrelated terms into a domain's scope.
        """
        allowed = ("UBERON_", "CL_", "GO_", "MP_", "HP_")
        for name, data in self.cache()["domains"].items():
            for short_id in data["terms"]:
                assert short_id.startswith(allowed), \
                    f"{name} carries a foreign-namespace term: {short_id}"
                assert not short_id.startswith("UBERON_6"), \
                    f"{name} carries a Drosophila-derived term: {short_id}"

    def test_every_domain_resolves_to_at_least_one_in_use_term(self):
        for name, data in self.cache()["domains"].items():
            assert data["terms"], f"{name} matches no term AOP-Wiki has ever used"

    def test_term_iris_are_obo_iris(self):
        iris = domain_term_iris(DEFAULT_DOMAIN)
        assert iris
        assert all(i.startswith("http://purl.obolibrary.org/obo/") for i in iris)


class TestPropertyScoping:
    """The tier view scores each property only on entities it can appear on."""

    def test_ke_does_not_inherit_ker_only_properties(self):
        """`applies_to` is pipe-separated, so "KE" must not match "KER".

        has_upstream_key_event / has_downstream_key_event apply to KERs alone;
        scoring them against Key Events added two permanently-zero properties to
        the KE Content tier and dropped it by ~21 points.
        """
        ke_labels = {
            p["label"]
            for tier in get_properties_for_entity("KE").values()
            for p in tier
        }
        assert "Upstream Key Event" not in ke_labels
        assert "Downstream Key Event" not in ke_labels

    def test_ker_keeps_its_own_content_properties(self):
        ker_labels = {
            p["label"]
            for tier in get_properties_for_entity("KER").values()
            for p in tier
        }
        assert {"Upstream Key Event", "Downstream Key Event"} <= ker_labels
