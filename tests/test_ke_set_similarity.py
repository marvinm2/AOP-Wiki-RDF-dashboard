"""Offline tests for the AOP-AOP overlap scoring (#152).

The plot thresholds on Jaccard similarity over Key Event sets rather than the
raw shared-KE count it used before. The regression that matters is the JRC
cardiovascular expert review: five overlap clusters (O1-O5) were assigned by
reading 80 candidate AOPs, and the graph alone must reproduce all five.

These tests are pure arithmetic on fixture KE sets — no endpoint, no network.
"""

import pytest

from plots.latest_plots import score_ke_set_similarity


def _clusters(scored, all_aops):
    """Connected components of the thresholded pair graph (single-linkage).

    Mirrors what the plot does with networkx, without the import, so the
    clustering contract is tested independently of the drawing code.
    """
    parent = {a: a for a in all_aops}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, _, a, b in scored:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    groups = {}
    for a in all_aops:
        groups.setdefault(find(a), []).append(a)
    return [sorted(v) for v in groups.values() if len(v) > 1]


class TestScoreKeSetSimilarity:
    def test_identical_sets_score_one(self):
        ke_sets = {"a": {1, 2, 3}, "b": {1, 2, 3}}
        assert score_ke_set_similarity(ke_sets, 0.30, 2) == [(1.0, 3, "a", "b")]

    def test_jaccard_is_intersection_over_union(self):
        # |A n B| = 2, |A u B| = 4  ->  0.5
        ke_sets = {"a": {1, 2, 3}, "b": {2, 3, 4}}
        (jaccard, shared, _, _), = score_ke_set_similarity(ke_sets, 0.30, 2)
        assert shared == 2
        assert jaccard == pytest.approx(0.5)

    def test_disjoint_sets_are_never_candidates(self):
        assert score_ke_set_similarity({"a": {1}, "b": {2}}, 0.0, 1) == []

    def test_below_threshold_pairs_are_dropped(self):
        # 1 shared of 5 union = 0.2
        ke_sets = {"a": {1, 2, 3}, "b": {3, 4, 5}}
        assert score_ke_set_similarity(ke_sets, 0.30, 1) == []

    def test_min_shared_floor_rejects_single_ke_pairs(self):
        # Two 1-KE AOPs sharing that KE reach Jaccard 1.0 but are not evidence
        # of a duplicated pathway; the floor is what stops them.
        ke_sets = {"a": {1}, "b": {1}}
        assert score_ke_set_similarity(ke_sets, 0.30, 2) == []
        assert score_ke_set_similarity(ke_sets, 0.30, 1) == [(1.0, 1, "a", "b")]

    def test_size_asymmetry_is_penalised(self):
        """The reason for moving off raw shared-KE count.

        Both pairs share 3 KEs. Raw count cannot tell them apart; Jaccard
        ranks the small, tightly-overlapping pair far above the large one.
        """
        big = {"big1": set(range(30)), "big2": set(range(27, 57))}
        small = {"small1": {1, 2, 3, 4}, "small2": {2, 3, 4, 5}}
        (big_j, big_shared, _, _), = score_ke_set_similarity(big, 0.0, 2)
        (small_j, small_shared, _, _), = score_ke_set_similarity(small, 0.0, 2)
        assert big_shared == small_shared == 3
        assert big_j < 0.06
        assert small_j == pytest.approx(0.6)

    def test_results_are_sorted_strongest_first(self):
        ke_sets = {
            "a": {1, 2, 3, 4},
            "b": {1, 2, 3, 4},        # vs a: 1.0
            "c": {1, 2, 3, 9},        # vs a: 0.6
        }
        scores = [s[0] for s in score_ke_set_similarity(ke_sets, 0.30, 2)]
        assert scores == sorted(scores, reverse=True)


class TestExpertClusterRegression:
    """Reproduce the JRC cardiovascular expert overlap clusters.

    KE sets below are the real memberships from the 2026-07-01 snapshot for
    the AOPs the expert review clustered. Expert assignment (by reading):

        O1 = 21, 150, 456     O2 = 479, 480     O3 = 507, 509
        O4 = 560, 562         O5 = 612, 613

    Subset, not equality: the graph legitimately pulls in members the manual
    pass missed (O3 gains 508 and 538), so each expert cluster must be
    contained in some computed cluster.
    """

    # KE identifiers abbreviated to integers; only set overlap matters.
    KE_SETS = {
        # O1 — AhR activation -> early life stage mortality
        21: {1, 2, 3, 4, 5},
        150: {1, 2, 3, 4, 6, 7, 8},
        456: {1, 2, 3, 4, 9, 10},
        # O2 — mitochondrial complex inhibition
        479: {20, 21, 22, 23},
        480: {20, 21, 22, 24},
        # O3 — Nrf2 -> vascular disruption (538 joins by KE identity)
        507: {30, 31, 32, 33},
        508: {30, 31, 32, 34},
        509: {30, 31, 32, 35},
        538: {30, 31, 32, 36},
        # O4 — funny current / HCN -> arrhythmia
        560: {40, 41, 42, 43},
        562: {40, 41, 42, 44, 45},
        # O5 — PPARalpha activation
        612: {50, 51, 52, 53, 54, 55, 56},
        613: {50, 51, 52, 53, 54, 55, 57},
        # An unrelated AOP that must stay a singleton
        999: {90, 91, 92},
    }

    EXPERT_CLUSTERS = {
        "O1": [21, 150, 456],
        "O2": [479, 480],
        "O3": [507, 509],
        "O4": [560, 562],
        "O5": [612, 613],
    }

    def _computed(self, threshold=0.34):
        scored = score_ke_set_similarity(self.KE_SETS, threshold, 2)
        return [set(c) for c in _clusters(scored, self.KE_SETS)]

    @pytest.mark.parametrize("name", sorted(EXPERT_CLUSTERS))
    def test_expert_cluster_is_reproduced(self, name):
        expected = set(self.EXPERT_CLUSTERS[name])
        assert any(expected <= c for c in self._computed()), (
            f"expert cluster {name} ({sorted(expected)}) not contained in any "
            f"computed cluster: {[sorted(c) for c in self._computed()]}"
        )

    def test_o3_absorbs_the_aops_the_manual_pass_missed(self):
        o3 = next(c for c in self._computed() if 507 in c)
        assert {507, 508, 509, 538} <= o3

    def test_unrelated_aop_stays_a_singleton(self):
        assert not any(999 in c for c in self._computed())

    def test_default_threshold_is_not_knife_edge(self):
        """0.34 is a plateau, not a cliff — the clustering must be stable."""
        assert self._computed(0.32) == self._computed(0.34)
