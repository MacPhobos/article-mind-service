"""Unit tests for heuristic reranking module.

Tests cover all 6 signals independently and in combination.
Uses neutral baseline (50-word chunks, late chunk index) to isolate signals.
"""

from datetime import datetime, timedelta, timezone

import pytest

from article_mind_service.search.heuristic_reranker import (
    BOOST_EARLY_CHUNK,
    BOOST_LONG_CHUNK,
    BOOST_RECENCY,
    BOOST_TITLE_EXACT,
    BOOST_TITLE_PARTIAL,
    BOOST_WORD_DENSITY,
    DOMAIN_AUTHORITY,
    PENALTY_SHORT_CHUNK,
    heuristic_rerank,
)


def make_neutral_content(word_count=50):
    """Generate neutral content that doesn't match common queries."""
    return " ".join([f"word{i}" for i in range(word_count)])


def make_result(chunk_id="article_1_chunk_10", rrf_score=0.5, content=None, **metadata_fields):
    """Helper to create test result with neutral defaults."""
    if content is None:
        content = make_neutral_content(50)
    return {
        "chunk_id": chunk_id,
        "rrf_score": rrf_score,
        "content": content,
        "metadata": {
            "article_id": 1,
            "chunk_index": 10,  # Late chunk, no early boost
            "word_count": 50,   # Medium length, no quality signal
            **metadata_fields,
        },
    }


class TestTitleMatchSignal:
    """Tests for title matching signal (Signal 1)."""

    def test_exact_title_match(self):
        """Exact query match in title should boost score."""
        results = [make_result()]
        metadata = {1: {"title": "Python Programming Guide"}}

        reranked = heuristic_rerank(results, "Python Programming", metadata)

        expected = 0.5 + BOOST_TITLE_EXACT
        assert reranked[0]["heuristic_score"] == pytest.approx(expected, abs=0.01)

    def test_partial_title_match(self):
        """Partial word matches in title should boost score."""
        results = [make_result()]
        metadata = {1: {"title": "Python Tutorial"}}

        reranked = heuristic_rerank(results, "Python Guide Tutorial", metadata)

        # 2 words match (python, tutorial)
        expected = 0.5 + (2 * BOOST_TITLE_PARTIAL)
        assert reranked[0]["heuristic_score"] == pytest.approx(expected, abs=0.01)

    def test_partial_title_match_capped_at_three_words(self):
        """Partial title match should cap at 3 words."""
        results = [make_result()]
        metadata = {1: {"title": "One Two Three Four Five"}}

        reranked = heuristic_rerank(results, "one two three four five", metadata)

        # Max 3 words boost
        expected = 0.5 + (3 * BOOST_TITLE_PARTIAL)
        assert reranked[0]["heuristic_score"] == pytest.approx(expected, abs=0.01)


class TestWordDensitySignal:
    """Tests for query word density signal (Signal 2)."""

    def test_high_word_density(self):
        """High query word density should boost score."""
        results = [make_result(
            content="Python is great for programming. Python syntax is clean."
        )]

        reranked = heuristic_rerank(results, "Python programming", {})

        # 100% density (both words present)
        expected = 0.5 + BOOST_WORD_DENSITY
        assert reranked[0]["heuristic_score"] == pytest.approx(expected, abs=0.01)

    def test_partial_word_density(self):
        """Partial word density should proportionally boost score."""
        results = [make_result(content="Python is great")]

        reranked = heuristic_rerank(results, "Python Java", {})

        # 50% density (1 of 2 words)
        expected = 0.5 + (0.5 * BOOST_WORD_DENSITY)
        assert reranked[0]["heuristic_score"] == pytest.approx(expected, abs=0.01)


class TestEarlyChunkSignal:
    """Tests for early chunk position signal (Signal 3)."""

    def test_first_chunk_boost(self):
        """First chunk should get maximum early chunk boost."""
        results = [make_result(chunk_index=0)]

        reranked = heuristic_rerank(results, "test", {})

        expected = 0.5 + BOOST_EARLY_CHUNK
        assert reranked[0]["heuristic_score"] == pytest.approx(expected, abs=0.01)

    def test_second_chunk_boost(self):
        """Second chunk should get reduced early chunk boost."""
        results = [make_result(chunk_index=1)]

        reranked = heuristic_rerank(results, "test", {})

        expected = 0.5 + (BOOST_EARLY_CHUNK * (1.0 - 1 / 3.0))
        assert reranked[0]["heuristic_score"] == pytest.approx(expected, abs=0.01)

    def test_third_chunk_boost(self):
        """Third chunk should get minimal early chunk boost."""
        results = [make_result(chunk_index=2)]

        reranked = heuristic_rerank(results, "test", {})

        expected = 0.5 + (BOOST_EARLY_CHUNK * (1.0 - 2 / 3.0))
        assert reranked[0]["heuristic_score"] == pytest.approx(expected, abs=0.01)

    def test_later_chunk_no_boost(self):
        """Chunks beyond index 2 should not get early chunk boost."""
        results = [make_result(chunk_index=5)]

        reranked = heuristic_rerank(results, "test", {})

        assert reranked[0]["heuristic_score"] == pytest.approx(0.5, abs=0.01)


class TestRecencySignal:
    """Tests for recency signal (Signal 4)."""

    def test_recent_article_boost(self):
        """Articles published within 30 days should get full recency boost."""
        now = datetime.now(timezone.utc)
        recent_date = now - timedelta(days=15)

        results = [make_result()]
        metadata = {1: {"published_date": recent_date}}

        reranked = heuristic_rerank(results, "test", metadata)

        expected = 0.5 + BOOST_RECENCY
        assert reranked[0]["heuristic_score"] == pytest.approx(expected, abs=0.01)

    def test_medium_recent_article_boost(self):
        """Articles 30-90 days old should get half recency boost."""
        now = datetime.now(timezone.utc)
        medium_date = now - timedelta(days=60)

        results = [make_result()]
        metadata = {1: {"published_date": medium_date}}

        reranked = heuristic_rerank(results, "test", metadata)

        expected = 0.5 + (BOOST_RECENCY * 0.5)
        assert reranked[0]["heuristic_score"] == pytest.approx(expected, abs=0.01)

    def test_old_article_no_boost(self):
        """Articles older than 90 days should not get recency boost."""
        now = datetime.now(timezone.utc)
        old_date = now - timedelta(days=180)

        results = [make_result()]
        metadata = {1: {"published_date": old_date}}

        reranked = heuristic_rerank(results, "test", metadata)

        assert reranked[0]["heuristic_score"] == pytest.approx(0.5, abs=0.01)


class TestDomainAuthoritySignal:
    """Tests for domain authority signal (Signal 5)."""

    def test_arxiv_authority_boost(self):
        """ArXiv URLs should get domain authority boost."""
        results = [make_result(source_url="https://arxiv.org/abs/1234.5678")]

        reranked = heuristic_rerank(results, "test", {})

        expected = 0.5 + DOMAIN_AUTHORITY["arxiv.org"]
        assert reranked[0]["heuristic_score"] == pytest.approx(expected, abs=0.01)

    def test_github_authority_boost(self):
        """GitHub URLs should get domain authority boost."""
        results = [make_result(source_url="https://github.com/user/repo")]

        reranked = heuristic_rerank(results, "test", {})

        expected = 0.5 + DOMAIN_AUTHORITY["github.com"]
        assert reranked[0]["heuristic_score"] == pytest.approx(expected, abs=0.01)

    def test_source_url_from_metadata(self):
        """Domain authority should work with source_url in article metadata."""
        results = [make_result()]
        metadata = {1: {"source_url": "https://docs.python.org/3/tutorial/"}}

        reranked = heuristic_rerank(results, "test", metadata)

        expected = 0.5 + DOMAIN_AUTHORITY["docs.python.org"]
        assert reranked[0]["heuristic_score"] == pytest.approx(expected, abs=0.01)


class TestChunkQualitySignal:
    """Tests for chunk quality signal (Signal 6)."""

    def test_long_chunk_boost(self):
        """Chunks with >100 words should get quality boost."""
        results = [make_result(word_count=150, content=make_neutral_content(150))]

        reranked = heuristic_rerank(results, "test", {})

        expected = 0.5 + BOOST_LONG_CHUNK
        assert reranked[0]["heuristic_score"] == pytest.approx(expected, abs=0.01)

    def test_short_chunk_penalty(self):
        """Chunks with <30 words should get quality penalty."""
        results = [make_result(word_count=20, content=make_neutral_content(20))]

        reranked = heuristic_rerank(results, "test", {})

        expected = 0.5 + PENALTY_SHORT_CHUNK
        assert reranked[0]["heuristic_score"] == pytest.approx(expected, abs=0.01)

    def test_medium_chunk_no_boost_or_penalty(self):
        """Chunks with 30-100 words should not get boost or penalty."""
        results = [make_result(word_count=50, content=make_neutral_content(50))]

        reranked = heuristic_rerank(results, "test", {})

        assert reranked[0]["heuristic_score"] == pytest.approx(0.5, abs=0.01)


class TestCombinedSignals:
    """Tests for combined signal interactions."""

    def test_all_signals_combined(self):
        """All signals should combine additively."""
        now = datetime.now(timezone.utc)
        recent_date = now - timedelta(days=10)

        results = [make_result(
            chunk_index=0,
            word_count=120,
            content="Python programming is great. " * 20,
            source_url="https://arxiv.org/abs/1234",
        )]
        metadata = {
            1: {
                "title": "Python Programming Tutorial",
                "published_date": recent_date,
                "source_url": "https://arxiv.org/abs/1234",
            }
        }

        reranked = heuristic_rerank(results, "Python programming", metadata)

        # Expected boosts:
        # - Title exact match: +0.15
        # - Word density (100%): +0.10
        # - Early chunk (index 0): +0.03
        # - Recency (<30 days): +0.03
        # - Domain authority (arxiv): +0.05
        # - Long chunk (>100 words): +0.02
        expected = 0.5 + 0.15 + 0.10 + 0.03 + 0.03 + 0.05 + 0.02
        assert reranked[0]["heuristic_score"] == pytest.approx(expected, abs=0.01)

    def test_score_clamping_at_one(self):
        """Scores should be clamped at 1.0."""
        now = datetime.now(timezone.utc)
        recent_date = now - timedelta(days=10)

        results = [make_result(
            rrf_score=0.95,  # High base score
            chunk_index=0,
            word_count=120,
            content="Python programming is great. " * 20,
            source_url="https://arxiv.org/abs/1234",
        )]
        metadata = {
            1: {
                "title": "Python Programming Tutorial",
                "published_date": recent_date,
            }
        }

        reranked = heuristic_rerank(results, "Python programming", metadata)

        assert reranked[0]["heuristic_score"] == 1.0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_results(self):
        """Empty results should return empty list."""
        reranked = heuristic_rerank([], "test", {})
        assert reranked == []

    def test_missing_metadata(self):
        """Missing article metadata should not crash."""
        results = [make_result()]
        reranked = heuristic_rerank(results, "test", None)
        assert len(reranked) == 1

    def test_missing_chunk_metadata(self):
        """Missing chunk metadata should not crash."""
        results = [{
            "chunk_id": "article_1_chunk_0",
            "rrf_score": 0.5,
            "content": make_neutral_content(50),
        }]
        reranked = heuristic_rerank(results, "test", {})
        assert len(reranked) == 1

    def test_sorting_order(self):
        """Results should be sorted by heuristic_score descending."""
        results = [
            make_result(chunk_id="low", rrf_score=0.3, chunk_index=10),  # No boost
            make_result(chunk_id="high", rrf_score=0.5, chunk_index=0),  # Early boost
        ]

        reranked = heuristic_rerank(results, "test", {})

        # First result: 0.3 (no boost)
        # Second result: 0.5 + 0.03 = 0.53 (early chunk boost)
        assert reranked[0]["chunk_id"] == "high"
        assert reranked[1]["chunk_id"] == "low"
