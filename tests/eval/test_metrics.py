"""Tests for search quality evaluation metrics.

Comprehensive test coverage for precision, recall, reciprocal rank, and nDCG metrics.
"""

import pytest

from .metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank


class TestPrecisionAtK:
    """Tests for precision@k metric."""

    def test_perfect_precision(self) -> None:
        """All top-k results are relevant."""
        retrieved = [1, 2, 3, 4, 5]
        relevant = {1, 2, 3, 4, 5}
        assert precision_at_k(retrieved, relevant, k=5) == 1.0

    def test_partial_precision(self) -> None:
        """Some top-k results are relevant."""
        retrieved = [1, 2, 3, 4, 5]
        relevant = {1, 3, 5}
        assert precision_at_k(retrieved, relevant, k=5) == 0.6

    def test_zero_precision(self) -> None:
        """No top-k results are relevant."""
        retrieved = [1, 2, 3]
        relevant = {4, 5, 6}
        assert precision_at_k(retrieved, relevant, k=3) == 0.0

    def test_k_smaller_than_results(self) -> None:
        """K is smaller than total results."""
        retrieved = [1, 2, 3, 4, 5]
        relevant = {3, 4, 5}
        # Top-2: [1, 2], none relevant
        assert precision_at_k(retrieved, relevant, k=2) == 0.0
        # Top-3: [1, 2, 3], one relevant
        assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(1 / 3)

    def test_k_larger_than_results(self) -> None:
        """K is larger than total results."""
        retrieved = [1, 2]
        relevant = {1, 2, 3}
        assert precision_at_k(retrieved, relevant, k=10) == 1.0

    def test_empty_retrieved(self) -> None:
        """No results retrieved."""
        retrieved: list[int] = []
        relevant = {1, 2, 3}
        assert precision_at_k(retrieved, relevant, k=5) == 0.0

    def test_empty_relevant(self) -> None:
        """No relevant items exist."""
        retrieved = [1, 2, 3]
        relevant: set[int] = set()
        assert precision_at_k(retrieved, relevant, k=3) == 0.0

    def test_k_zero(self) -> None:
        """K is zero."""
        retrieved = [1, 2, 3]
        relevant = {1, 2}
        assert precision_at_k(retrieved, relevant, k=0) == 0.0


class TestRecallAtK:
    """Tests for recall@k metric."""

    def test_perfect_recall(self) -> None:
        """All relevant items found in top-k."""
        retrieved = [1, 2, 3, 4, 5]
        relevant = {1, 3, 5}
        assert recall_at_k(retrieved, relevant, k=5) == 1.0

    def test_partial_recall(self) -> None:
        """Some relevant items found in top-k."""
        retrieved = [1, 2, 3]
        relevant = {1, 2, 3, 4, 5}
        assert recall_at_k(retrieved, relevant, k=3) == 0.6

    def test_zero_recall(self) -> None:
        """No relevant items found in top-k."""
        retrieved = [1, 2, 3]
        relevant = {4, 5, 6}
        assert recall_at_k(retrieved, relevant, k=3) == 0.0

    def test_k_smaller_than_results(self) -> None:
        """K is smaller than total results."""
        retrieved = [1, 2, 3, 4, 5]
        relevant = {3, 4, 5}
        # Top-2: [1, 2], zero out of three relevant found
        assert recall_at_k(retrieved, relevant, k=2) == 0.0
        # Top-3: [1, 2, 3], one out of three relevant found
        assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(1 / 3)

    def test_k_larger_than_relevant(self) -> None:
        """K is larger than number of relevant items."""
        retrieved = [1, 2, 3, 4, 5]
        relevant = {1, 2}
        assert recall_at_k(retrieved, relevant, k=10) == 1.0

    def test_empty_retrieved(self) -> None:
        """No results retrieved."""
        retrieved: list[int] = []
        relevant = {1, 2, 3}
        assert recall_at_k(retrieved, relevant, k=5) == 0.0

    def test_empty_relevant(self) -> None:
        """No relevant items exist (vacuous truth)."""
        retrieved = [1, 2, 3]
        relevant: set[int] = set()
        assert recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_k_zero(self) -> None:
        """K is zero."""
        retrieved = [1, 2, 3]
        relevant = {1, 2}
        assert recall_at_k(retrieved, relevant, k=0) == 0.0


class TestReciprocalRank:
    """Tests for reciprocal rank (RR) metric."""

    def test_first_result_relevant(self) -> None:
        """First result is relevant (perfect RR)."""
        retrieved = [1, 2, 3]
        relevant = {1}
        assert reciprocal_rank(retrieved, relevant) == 1.0

    def test_second_result_relevant(self) -> None:
        """Second result is relevant."""
        retrieved = [1, 2, 3]
        relevant = {2}
        assert reciprocal_rank(retrieved, relevant) == 0.5

    def test_third_result_relevant(self) -> None:
        """Third result is relevant."""
        retrieved = [1, 2, 3]
        relevant = {3}
        assert reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 3)

    def test_no_relevant_found(self) -> None:
        """No relevant results (worst RR)."""
        retrieved = [1, 2, 3]
        relevant = {4, 5}
        assert reciprocal_rank(retrieved, relevant) == 0.0

    def test_multiple_relevant_uses_first(self) -> None:
        """Multiple relevant items, uses first occurrence."""
        retrieved = [1, 2, 3, 4, 5]
        relevant = {2, 4}
        # First relevant is at position 2 (index 1)
        assert reciprocal_rank(retrieved, relevant) == 0.5

    def test_empty_retrieved(self) -> None:
        """No results retrieved."""
        retrieved: list[int] = []
        relevant = {1, 2}
        assert reciprocal_rank(retrieved, relevant) == 0.0

    def test_empty_relevant(self) -> None:
        """No relevant items exist."""
        retrieved = [1, 2, 3]
        relevant: set[int] = set()
        assert reciprocal_rank(retrieved, relevant) == 0.0

    def test_tenth_position(self) -> None:
        """Relevant result at 10th position."""
        retrieved = list(range(1, 20))
        relevant = {10}
        assert reciprocal_rank(retrieved, relevant) == 0.1


class TestNDCGAtK:
    """Tests for Normalized Discounted Cumulative Gain (nDCG@k) metric."""

    def test_perfect_ranking(self) -> None:
        """Perfect ranking (ideal order)."""
        retrieved = [1, 2, 3]
        relevance = {1: 1.0, 2: 0.5, 3: 0.0}
        # Relevance decreases monotonically (ideal)
        assert ndcg_at_k(retrieved, relevance, k=3) == pytest.approx(1.0)

    def test_worst_ranking(self) -> None:
        """Worst ranking (reversed order)."""
        retrieved = [3, 2, 1]
        relevance = {1: 1.0, 2: 0.5, 3: 0.0}
        # Relevance increases (worst order)
        score = ndcg_at_k(retrieved, relevance, k=3)
        assert 0.0 < score < 1.0  # Not ideal, but not zero

    def test_binary_relevance(self) -> None:
        """Binary relevance (0 or 1)."""
        retrieved = [1, 2, 3, 4]
        relevance = {1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0}
        # Two relevant at top
        score = ndcg_at_k(retrieved, relevance, k=4)
        assert score == pytest.approx(1.0)

    def test_graded_relevance(self) -> None:
        """Graded relevance (multiple levels)."""
        retrieved = [1, 2, 3]
        relevance = {1: 1.0, 2: 0.7, 3: 0.3}
        # Good but not perfect order
        score = ndcg_at_k(retrieved, relevance, k=3)
        assert score == pytest.approx(1.0)

    def test_missing_relevance_labels(self) -> None:
        """Some retrieved items missing relevance labels."""
        retrieved = [1, 2, 3, 4]
        relevance = {1: 1.0, 3: 0.5}  # Missing 2, 4
        # Unlabeled items treated as 0.0 relevance
        score = ndcg_at_k(retrieved, relevance, k=4)
        assert 0.0 < score < 1.0

    def test_k_smaller_than_results(self) -> None:
        """K is smaller than total results."""
        retrieved = [1, 2, 3, 4, 5]
        relevance = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.2}
        # Only consider top-3
        score = ndcg_at_k(retrieved, relevance, k=3)
        assert score == pytest.approx(1.0)

    def test_no_relevant_items(self) -> None:
        """No relevant items exist."""
        retrieved = [1, 2, 3]
        relevance: dict[int, float] = {}
        assert ndcg_at_k(retrieved, relevance, k=3) == 0.0

    def test_all_zero_relevance(self) -> None:
        """All items have zero relevance."""
        retrieved = [1, 2, 3]
        relevance = {1: 0.0, 2: 0.0, 3: 0.0}
        assert ndcg_at_k(retrieved, relevance, k=3) == 0.0

    def test_single_relevant_item(self) -> None:
        """Single relevant item at different positions."""
        relevance = {1: 1.0, 2: 0.0, 3: 0.0}

        # Relevant at first position
        retrieved = [1, 2, 3]
        assert ndcg_at_k(retrieved, relevance, k=3) == pytest.approx(1.0)

        # Relevant at second position
        retrieved = [2, 1, 3]
        score = ndcg_at_k(retrieved, relevance, k=3)
        assert 0.0 < score < 1.0

    def test_k_zero(self) -> None:
        """K is zero."""
        retrieved = [1, 2, 3]
        relevance = {1: 1.0, 2: 0.5}
        # Empty lists for both actual and ideal
        assert ndcg_at_k(retrieved, relevance, k=0) == 0.0

    def test_empty_retrieved(self) -> None:
        """No results retrieved."""
        retrieved: list[int] = []
        relevance = {1: 1.0, 2: 0.5}
        assert ndcg_at_k(retrieved, relevance, k=5) == 0.0
