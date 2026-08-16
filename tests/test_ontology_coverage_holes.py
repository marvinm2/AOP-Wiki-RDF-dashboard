"""Offline tests for the ontology coverage-holes rule (#151).

The plot reports sub-branches of an ontology in which nothing is annotated. The
rule that matters is the roll-up: only the TOPMOST unannotated node on a path is
a hole, so the report names "arterial system" once rather than each of the 564
terms beneath it.

These tests run against fixture branch data — no endpoint, no Ubergraph, no
cache file.
"""

import pytest

from plots.latest_plots import find_ontology_coverage_holes


def branch(candidates, used_ancestors, branch_size=100):
    """Assemble a cache-shaped branch entry from terse fixture input."""
    return {
        "branch_size": branch_size,
        "candidates": {
            short: {
                "label": label,
                "subtree_size": size,
                "depth": depth,
                "parents": parents,
            }
            for short, (label, size, depth, parents) in candidates.items()
        },
        "used_term_ancestors": used_ancestors,
        "used_terms_in_branch": sorted(used_ancestors),
    }


class TestHoleRule:
    def test_unannotated_child_of_an_annotated_parent_is_a_hole(self):
        data = branch(
            candidates={
                "UBERON_0000948": ("heart", 50, 1, []),
                "UBERON_0000946": ("cardiac valve", 34, 2, ["UBERON_0000948"]),
            },
            # The heart is annotated, so it is touched; the valve is not.
            used_ancestors={"UBERON_0000948": ["UBERON_0000948"]},
        )
        holes = find_ontology_coverage_holes(data, {"UBERON_0000948"})
        assert [h["Ontology ID"] for h in holes] == ["UBERON:0000946"]
        assert holes[0]["Term"] == "cardiac valve"
        assert holes[0]["Unused Terms"] == 34

    def test_only_the_topmost_unannotated_node_is_reported(self):
        """The roll-up. Reporting every descendant would bury the useful row."""
        data = branch(
            candidates={
                "UBERON_0000948": ("heart", 50, 1, []),
                "UBERON_0004572": ("arterial system", 564, 2, ["UBERON_0000948"]),
                "UBERON_0004571": ("systemic arterial system", 176, 3, ["UBERON_0004572"]),
            },
            used_ancestors={"UBERON_0000948": ["UBERON_0000948"]},
        )
        holes = find_ontology_coverage_holes(data, {"UBERON_0000948"})
        # systemic arterial system sits under an already-reported hole.
        assert [h["Ontology ID"] for h in holes] == ["UBERON:0004572"]

    def test_an_annotated_descendant_suppresses_its_whole_ancestor_chain(self):
        data = branch(
            candidates={
                "UBERON_0000948": ("heart", 50, 1, []),
                "UBERON_0004151": ("cardiac chamber", 138, 2, ["UBERON_0000948"]),
                "UBERON_0002082": ("cardiac ventricle", 40, 3, ["UBERON_0004151"]),
            },
            # A term deep under cardiac chamber is annotated, so neither the
            # chamber nor the heart above it can be a hole.
            used_ancestors={
                "UBERON_0002082": ["UBERON_0000948", "UBERON_0004151", "UBERON_0002082"],
            },
        )
        holes = find_ontology_coverage_holes(data, {"UBERON_0002082"})
        assert holes == []

    def test_multi_parent_node_is_a_hole_only_when_reachable_from_a_touched_parent(self):
        """The DAG case a treemap could not represent.

        `lymph vasculature` hangs off two parents. It should surface once,
        because at least one of them is annotated.
        """
        data = branch(
            candidates={
                "UBERON_0000948": ("heart", 50, 1, []),
                "UBERON_0006558": ("lymphatic part of lymphoid system", 64, 1, []),
                "UBERON_0004536": (
                    "lymph vasculature", 56, 2,
                    ["UBERON_0000948", "UBERON_0006558"],
                ),
            },
            used_ancestors={"UBERON_0000948": ["UBERON_0000948"]},
        )
        holes = find_ontology_coverage_holes(data, {"UBERON_0000948"})
        ids = [h["Ontology ID"] for h in holes]
        assert ids.count("UBERON:0004536") == 1, "a multi-parent hole must not duplicate"
        assert "UBERON:0006558" in ids

    def test_ranked_by_subtree_size(self):
        data = branch(
            candidates={
                "UBERON_0000948": ("heart", 50, 1, []),
                "UBERON_0004572": ("arterial system", 564, 2, ["UBERON_0000948"]),
                "UBERON_0004582": ("venous system", 459, 2, ["UBERON_0000948"]),
                "UBERON_0005983": ("heart layer", 108, 2, ["UBERON_0000948"]),
            },
            used_ancestors={"UBERON_0000948": ["UBERON_0000948"]},
        )
        holes = find_ontology_coverage_holes(data, {"UBERON_0000948"})
        assert [h["Unused Terms"] for h in holes] == [564, 459, 108]

    def test_an_annotated_candidate_is_never_itself_a_hole(self):
        data = branch(
            candidates={
                "UBERON_0000948": ("heart", 50, 1, []),
                "UBERON_0001981": ("blood vessel", 20, 2, ["UBERON_0000948"]),
            },
            used_ancestors={
                "UBERON_0000948": ["UBERON_0000948"],
                "UBERON_0001981": ["UBERON_0000948", "UBERON_0001981"],
            },
        )
        holes = find_ontology_coverage_holes(data, {"UBERON_0000948", "UBERON_0001981"})
        assert holes == []

    def test_nothing_annotated_anywhere_reports_nothing(self):
        """With no annotation at all there is no evidence of a *gap* — the
        whole branch is simply unannotated, which the used-fraction says."""
        data = branch(
            candidates={
                "UBERON_0000948": ("heart", 50, 1, []),
                "UBERON_0004572": ("arterial system", 564, 2, ["UBERON_0000948"]),
            },
            used_ancestors={},
        )
        assert find_ontology_coverage_holes(data, set()) == []

    def test_empty_branch_data_is_survivable(self):
        assert find_ontology_coverage_holes({}, {"UBERON_0000948"}) == []


class TestValvulopathyRegression:
    """The finding that motivated the issue.

    A JRC expert panel read 80 candidate AOPs and concluded no coherent
    valvulopathy AOP exists. The same conclusion must fall out of the graph:
    cardiac valve reported, heart not.
    """

    DATA = branch(
        candidates={
            "UBERON_0000948": ("heart", 50, 1, []),
            "UBERON_0000946": ("cardiac valve", 34, 2, ["UBERON_0000948"]),
            "UBERON_0004151": ("cardiac chamber", 138, 2, ["UBERON_0000948"]),
        },
        used_ancestors={"UBERON_0000948": ["UBERON_0000948"]},
        branch_size=1781,
    )

    def test_cardiac_valve_is_reported_unused(self):
        ids = [h["Ontology ID"] for h in find_ontology_coverage_holes(
            self.DATA, {"UBERON_0000948"})]
        assert "UBERON:0000946" in ids

    def test_heart_is_not_reported(self):
        ids = [h["Ontology ID"] for h in find_ontology_coverage_holes(
            self.DATA, {"UBERON_0000948"})]
        assert "UBERON:0000948" not in ids

    def test_annotating_the_valve_closes_the_hole(self):
        """If a curator annotates a Key Event with cardiac valve, the gap goes."""
        data = branch(
            candidates=dict(
                UBERON_0000948=("heart", 50, 1, []),
                UBERON_0000946=("cardiac valve", 34, 2, ["UBERON_0000948"]),
            ),
            used_ancestors={
                "UBERON_0000948": ["UBERON_0000948"],
                "UBERON_0000946": ["UBERON_0000948", "UBERON_0000946"],
            },
        )
        ids = [h["Ontology ID"] for h in find_ontology_coverage_holes(
            data, {"UBERON_0000948", "UBERON_0000946"})]
        assert "UBERON:0000946" not in ids


class TestCacheContract:
    """Guard the shape the builder writes and the plot reads."""

    def test_real_cache_has_the_expected_envelope(self):
        from plots.latest_plots import _load_ontology_branch_cache

        cache = _load_ontology_branch_cache()
        if not cache:
            pytest.skip("ontology_branch_cache.json not built in this checkout")
        assert {"generated_at", "source", "branches", "stats"} <= set(cache)
        assert "Cardiovascular" in cache["branches"]
        cv = cache["branches"]["Cardiovascular"]
        assert {"roots", "branch_size", "candidates", "used_term_ancestors"} <= set(cv)
        sample = next(iter(cv["candidates"].values()))
        assert {"label", "subtree_size", "depth", "parents"} <= set(sample)

    def test_no_foreign_namespace_terms_in_the_cache(self):
        """Branches are UBERON/CL only.

        Ubergraph mixes FBbt (Drosophila), PCL and the mouse brain atlas into
        UBERON branches — 88% of the raw nervous-system closure. Those would be
        reported as AOP-Wiki coverage gaps, which is meaningless.
        """
        from plots.latest_plots import _load_ontology_branch_cache

        cache = _load_ontology_branch_cache()
        if not cache:
            pytest.skip("ontology_branch_cache.json not built in this checkout")
        for name, data in cache["branches"].items():
            for short_id in data["candidates"]:
                assert short_id.startswith(("UBERON_", "CL_")), \
                    f"{name} carries a foreign-namespace term: {short_id}"
