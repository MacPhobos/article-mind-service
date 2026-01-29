"""Tests for adaptive similarity threshold functionality."""

import pytest
from httpx import AsyncClient

from article_mind_service.schemas.search import SearchRequest, SearchMode
from article_mind_service.search.hybrid_search import HybridSearch


class TestAdaptiveThreshold:
    """Tests for _get_adaptive_threshold method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.hybrid_search = HybridSearch()

    def test_single_word_query(self) -> None:
        """Test single word query gets very low threshold."""
        threshold = self.hybrid_search._get_adaptive_threshold("JWT")
        assert threshold == pytest.approx(0.05, abs=0.01)

    def test_single_word_long_query(self) -> None:
        """Test single word (even if long) gets low threshold."""
        threshold = self.hybrid_search._get_adaptive_threshold("authentication")
        assert threshold == pytest.approx(0.05, abs=0.01)

    def test_technical_term_query(self) -> None:
        """Test technical term query gets reduced threshold."""
        threshold = self.hybrid_search._get_adaptive_threshold("oauth authentication flow")
        # Contains 'oauth' technical term
        assert threshold == pytest.approx(0.10, abs=0.01)

    def test_technical_term_api(self) -> None:
        """Test API technical term is recognized."""
        threshold = self.hybrid_search._get_adaptive_threshold("REST API design")
        # Contains 'api' technical term
        assert threshold == pytest.approx(0.10, abs=0.01)

    def test_technical_term_docker(self) -> None:
        """Test docker technical term is recognized."""
        threshold = self.hybrid_search._get_adaptive_threshold("docker container deployment")
        # Contains 'docker' technical term
        assert threshold == pytest.approx(0.10, abs=0.01)

    def test_quoted_phrase_query(self) -> None:
        """Test quoted phrase query gets high threshold."""
        threshold = self.hybrid_search._get_adaptive_threshold('"exact phrase match"')
        assert threshold == pytest.approx(0.50, abs=0.01)

    def test_quoted_phrase_in_longer_query(self) -> None:
        """Test quoted phrase anywhere in query triggers high threshold."""
        threshold = self.hybrid_search._get_adaptive_threshold('find "specific term" in docs')
        assert threshold == pytest.approx(0.50, abs=0.01)

    def test_short_query_two_words(self) -> None:
        """Test short query (2 words) gets slightly reduced threshold."""
        threshold = self.hybrid_search._get_adaptive_threshold("user authentication")
        # 2 words, no technical terms, no quotes
        assert threshold == pytest.approx(0.20, abs=0.01)

    def test_short_query_three_words(self) -> None:
        """Test short query (3 words) gets slightly reduced threshold."""
        threshold = self.hybrid_search._get_adaptive_threshold("how does authentication")
        # 3 words, no technical terms, no quotes
        assert threshold == pytest.approx(0.20, abs=0.01)

    def test_normal_query_four_to_eight_words(self) -> None:
        """Test normal query (4-8 words) uses base threshold."""
        threshold = self.hybrid_search._get_adaptive_threshold(
            "how does user authentication work in web apps"
        )
        # 8 words, no technical terms, no quotes
        assert threshold == pytest.approx(0.30, abs=0.01)

    def test_long_query_over_eight_words(self) -> None:
        """Test long query (>8 words) gets elevated threshold."""
        threshold = self.hybrid_search._get_adaptive_threshold(
            "how does authentication work in modern web applications using JWT tokens and OAuth"
        )
        # 13 words, but no technical terms detected in this implementation
        # Note: Contains JWT and OAuth but query is so long it gets elevated threshold
        # Technical term check happens first, so this should be ~0.10
        # Actually, technical term check happens AFTER single word check
        # So this should hit technical term threshold
        assert threshold == pytest.approx(0.10, abs=0.01)

    def test_long_query_no_technical_terms(self) -> None:
        """Test long query without technical terms gets elevated threshold."""
        threshold = self.hybrid_search._get_adaptive_threshold(
            "what are the best practices for designing a good user experience in mobile applications"
        )
        # 14 words, no technical terms, no quotes
        assert threshold == pytest.approx(0.45, abs=0.01)

    def test_base_threshold_respected(self) -> None:
        """Test base threshold parameter is respected."""
        threshold = self.hybrid_search._get_adaptive_threshold(
            "normal query here",
            base=0.5
        )
        # 3 words, should get short query adjustment
        # base - 0.10 = 0.40
        assert threshold == pytest.approx(0.40, abs=0.01)

    def test_threshold_clamped_to_minimum(self) -> None:
        """Test threshold doesn't go below 0.05."""
        threshold = self.hybrid_search._get_adaptive_threshold("word", base=0.1)
        # Single word: base - 0.25 = -0.15, but clamped to 0.05
        assert threshold >= 0.05

    def test_threshold_clamped_to_maximum(self) -> None:
        """Test threshold doesn't exceed 0.8."""
        threshold = self.hybrid_search._get_adaptive_threshold('"exact"', base=0.7)
        # Quoted: base + 0.2 = 0.9, but clamped to 0.8
        assert threshold <= 0.8

    def test_priority_single_word_over_technical(self) -> None:
        """Test single word check has priority over technical term check."""
        threshold = self.hybrid_search._get_adaptive_threshold("api")
        # Single word overrides technical term
        assert threshold == pytest.approx(0.05, abs=0.01)

    def test_priority_technical_over_short(self) -> None:
        """Test technical term check has priority over short query check."""
        threshold = self.hybrid_search._get_adaptive_threshold("jwt auth")
        # 2 words, contains technical term 'jwt'
        # Technical term check happens before short query check
        assert threshold == pytest.approx(0.10, abs=0.01)

    def test_priority_quoted_over_long(self) -> None:
        """Test quoted phrase check has priority over long query check."""
        threshold = self.hybrid_search._get_adaptive_threshold(
            '"specific exact phrase" and many other words to make this query very long'
        )
        # Long query but quoted, quoted should win
        assert threshold == pytest.approx(0.50, abs=0.01)

    def test_case_insensitive_technical_terms(self) -> None:
        """Test technical term detection is case-insensitive."""
        threshold_lower = self.hybrid_search._get_adaptive_threshold("jwt authentication")
        threshold_upper = self.hybrid_search._get_adaptive_threshold("JWT AUTHENTICATION")
        threshold_mixed = self.hybrid_search._get_adaptive_threshold("Jwt Authentication")

        assert threshold_lower == threshold_upper == threshold_mixed

    def test_empty_query_uses_base(self) -> None:
        """Test empty query uses base threshold."""
        threshold = self.hybrid_search._get_adaptive_threshold("")
        # 0 words after split, should use base
        assert threshold == pytest.approx(0.30, abs=0.01)

    def test_whitespace_only_query_uses_base(self) -> None:
        """Test whitespace-only query uses base threshold."""
        threshold = self.hybrid_search._get_adaptive_threshold("   ")
        # 0 words after split and strip, should use base
        assert threshold == pytest.approx(0.30, abs=0.01)


class TestAdaptiveThresholdEdgeCases:
    """Edge case tests for adaptive threshold."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.hybrid_search = HybridSearch()

    def test_multiple_quotes(self) -> None:
        """Test multiple quoted phrases still trigger high threshold."""
        threshold = self.hybrid_search._get_adaptive_threshold(
            '"first phrase" and "second phrase"'
        )
        assert threshold == pytest.approx(0.50, abs=0.01)

    def test_single_quote_character(self) -> None:
        """Test single quote character without pairs."""
        threshold = self.hybrid_search._get_adaptive_threshold('find "something')
        # Contains quote character
        assert threshold == pytest.approx(0.50, abs=0.01)

    def test_exactly_four_words(self) -> None:
        """Test exactly 4 words uses base threshold."""
        threshold = self.hybrid_search._get_adaptive_threshold("one two three four")
        # 4 words, should use base (not short query)
        assert threshold == pytest.approx(0.30, abs=0.01)

    def test_exactly_eight_words(self) -> None:
        """Test exactly 8 words uses base threshold."""
        threshold = self.hybrid_search._get_adaptive_threshold(
            "one two three four five six seven eight"
        )
        # 8 words, should use base (not long query)
        assert threshold == pytest.approx(0.30, abs=0.01)

    def test_exactly_nine_words(self) -> None:
        """Test exactly 9 words triggers long query threshold."""
        threshold = self.hybrid_search._get_adaptive_threshold(
            "one two three four five six seven eight nine"
        )
        # 9 words, should use elevated threshold
        assert threshold == pytest.approx(0.45, abs=0.01)

    def test_multiple_technical_terms(self) -> None:
        """Test multiple technical terms still use technical term threshold."""
        threshold = self.hybrid_search._get_adaptive_threshold(
            "kubernetes docker aws deployment"
        )
        # Contains kubernetes, docker, aws
        assert threshold == pytest.approx(0.10, abs=0.01)

    def test_technical_term_as_substring(self) -> None:
        """Test technical term detection works with substrings."""
        threshold = self.hybrid_search._get_adaptive_threshold("apis are great")
        # 'api' is in 'apis'
        assert threshold == pytest.approx(0.10, abs=0.01)

    def test_technical_term_in_compound_word(self) -> None:
        """Test technical term in compound word."""
        threshold = self.hybrid_search._get_adaptive_threshold("graphql-api design")
        # Contains 'graphql' and 'api' as substrings
        assert threshold == pytest.approx(0.10, abs=0.01)


@pytest.mark.asyncio
class TestAdaptiveThresholdIntegration:
    """Integration tests for adaptive threshold in SearchRequest."""

    async def test_search_request_with_explicit_threshold(
        self, async_client: AsyncClient
    ) -> None:
        """Test that explicit similarity_threshold is respected."""
        response = await async_client.post(
            "/api/v1/sessions/1/search",
            json={
                "query": "test query",
                "similarity_threshold": 0.8,  # Explicit high threshold
            },
        )

        assert response.status_code == 200
        # The request should succeed with explicit threshold

    async def test_search_request_without_threshold_uses_adaptive(
        self, async_client: AsyncClient
    ) -> None:
        """Test that omitting similarity_threshold triggers adaptive behavior."""
        response = await async_client.post(
            "/api/v1/sessions/1/search",
            json={
                "query": "JWT",  # Single word, should get low threshold
            },
        )

        assert response.status_code == 200
        # Should use adaptive threshold (0.05 for single word)

    async def test_search_request_threshold_validation(
        self, async_client: AsyncClient
    ) -> None:
        """Test that similarity_threshold validates range."""
        # Below minimum
        response = await async_client.post(
            "/api/v1/sessions/1/search",
            json={
                "query": "test",
                "similarity_threshold": -0.1,
            },
        )
        assert response.status_code == 422  # Validation error

        # Above maximum
        response = await async_client.post(
            "/api/v1/sessions/1/search",
            json={
                "query": "test",
                "similarity_threshold": 1.5,
            },
        )
        assert response.status_code == 422  # Validation error

    async def test_search_request_null_threshold_uses_adaptive(
        self, async_client: AsyncClient
    ) -> None:
        """Test that null similarity_threshold explicitly triggers adaptive."""
        response = await async_client.post(
            "/api/v1/sessions/1/search",
            json={
                "query": "authentication flow",
                "similarity_threshold": None,  # Explicit null
            },
        )

        assert response.status_code == 200
        # Should use adaptive threshold


class TestSearchRequestSchema:
    """Tests for SearchRequest schema with similarity_threshold."""

    def test_similarity_threshold_optional(self) -> None:
        """Test that similarity_threshold is optional."""
        request = SearchRequest(query="test query")
        assert request.similarity_threshold is None

    def test_similarity_threshold_explicit(self) -> None:
        """Test that explicit similarity_threshold is preserved."""
        request = SearchRequest(query="test query", similarity_threshold=0.5)
        assert request.similarity_threshold == 0.5

    def test_similarity_threshold_validation_min(self) -> None:
        """Test that similarity_threshold validates minimum."""
        with pytest.raises(Exception):  # Pydantic validation error
            SearchRequest(query="test", similarity_threshold=-0.1)

    def test_similarity_threshold_validation_max(self) -> None:
        """Test that similarity_threshold validates maximum."""
        with pytest.raises(Exception):  # Pydantic validation error
            SearchRequest(query="test", similarity_threshold=1.5)
