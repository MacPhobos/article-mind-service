"""Unit tests for query expansion module."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from article_mind_service.chat.llm_providers import LLMResponse
from article_mind_service.chat.query_expansion import HyDEExpander, NoOpExpander


class TestNoOpExpander:
    """Tests for NoOpExpander (pass-through expander)."""

    @pytest.mark.asyncio
    async def test_returns_original_query(self):
        """Test that NoOpExpander returns the original query unchanged."""
        expander = NoOpExpander()
        query = "How does JWT authentication work?"

        result = await expander.expand(query)

        assert result == query

    @pytest.mark.asyncio
    async def test_preserves_special_characters(self):
        """Test that special characters are preserved."""
        expander = NoOpExpander()
        query = "What is OAuth 2.0 & JWT? (with examples!)"

        result = await expander.expand(query)

        assert result == query


class TestHyDEExpander:
    """Tests for HyDEExpander (hypothetical document generation)."""

    @pytest.mark.asyncio
    async def test_generates_hypothetical_document(self):
        """Test that HyDE generates a hypothetical answer."""
        # Mock LLM provider
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(
            return_value=LLMResponse(
                content="JWT (JSON Web Token) is an authentication method that uses signed tokens to verify user identity. It consists of three parts: header, payload, and signature.",
                tokens_input=50,
                tokens_output=30,
                model="gpt-4o-mini",
                provider="openai",
            )
        )

        expander = HyDEExpander(mock_provider)
        query = "How does JWT work?"

        result = await expander.expand(query)

        # Verify LLM was called
        mock_provider.generate.assert_called_once()

        # Verify result is different from query (expanded)
        assert result != query
        assert "JWT" in result or "token" in result.lower()
        assert len(result) > len(query)

    @pytest.mark.asyncio
    async def test_uses_low_temperature(self):
        """Test that HyDE uses low temperature for factual output."""
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(
            return_value=LLMResponse(
                content="Test response",
                tokens_input=10,
                tokens_output=5,
                model="gpt-4o-mini",
                provider="openai",
            )
        )

        expander = HyDEExpander(mock_provider)
        await expander.expand("test query")

        # Extract call args
        call_args = mock_provider.generate.call_args

        # Verify temperature is low (0.3)
        assert call_args.kwargs["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_limits_output_tokens(self):
        """Test that HyDE limits output to 200 tokens."""
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(
            return_value=LLMResponse(
                content="Test response",
                tokens_input=10,
                tokens_output=5,
                model="gpt-4o-mini",
                provider="openai",
            )
        )

        expander = HyDEExpander(mock_provider)
        await expander.expand("test query")

        # Extract call args
        call_args = mock_provider.generate.call_args

        # Verify max_tokens is 200
        assert call_args.kwargs["max_tokens"] == 200

    @pytest.mark.asyncio
    async def test_fallback_on_llm_error(self):
        """Test that HyDE falls back to original query on LLM error."""
        # Mock LLM provider that raises exception
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(
            side_effect=Exception("LLM API error")
        )

        expander = HyDEExpander(mock_provider)
        query = "How does JWT work?"

        # Should not raise exception
        result = await expander.expand(query)

        # Should fall back to original query
        assert result == query

    @pytest.mark.asyncio
    async def test_strips_whitespace(self):
        """Test that HyDE strips leading/trailing whitespace from response."""
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(
            return_value=LLMResponse(
                content="  \n  JWT is a token-based authentication method.  \n  ",
                tokens_input=10,
                tokens_output=15,
                model="gpt-4o-mini",
                provider="openai",
            )
        )

        expander = HyDEExpander(mock_provider)
        result = await expander.expand("What is JWT?")

        # Verify whitespace is stripped
        assert not result.startswith(" ")
        assert not result.startswith("\n")
        assert not result.endswith(" ")
        assert not result.endswith("\n")

    @pytest.mark.asyncio
    async def test_passes_empty_context(self):
        """Test that HyDE doesn't use context chunks for generation."""
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(
            return_value=LLMResponse(
                content="Test response",
                tokens_input=10,
                tokens_output=5,
                model="gpt-4o-mini",
                provider="openai",
            )
        )

        expander = HyDEExpander(mock_provider)
        await expander.expand("test query")

        # Extract call args
        call_args = mock_provider.generate.call_args

        # Verify context_chunks is empty
        assert call_args.kwargs["context_chunks"] == []

    @pytest.mark.asyncio
    async def test_includes_query_in_prompt(self):
        """Test that HyDE includes the original query in the prompt."""
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(
            return_value=LLMResponse(
                content="Test response",
                tokens_input=10,
                tokens_output=5,
                model="gpt-4o-mini",
                provider="openai",
            )
        )

        expander = HyDEExpander(mock_provider)
        query = "How does OAuth 2.0 work?"
        await expander.expand(query)

        # Extract call args
        call_args = mock_provider.generate.call_args
        user_message = call_args.kwargs["user_message"]

        # Verify query is in the prompt
        assert query in user_message
