"""Unit tests for LLM providers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from article_mind_service.chat.providers import (
    AnthropicProvider,
    OpenAIProvider,
    get_llm_provider,
)


class TestOpenAIProvider:
    """Tests for OpenAI provider."""

    @pytest.fixture
    def provider(self):
        return OpenAIProvider(api_key="test-key", model="gpt-4o-mini")

    @pytest.mark.asyncio
    async def test_generate_success(self, provider):
        """Test successful generation."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        result = await provider.generate(
            system_prompt="You are helpful.",
            user_message="Test question?",
            context_chunks=["Context 1"],
        )

        assert result.content == "Test response"
        assert result.tokens_input == 100
        assert result.tokens_output == 50
        assert result.provider == "openai"
        assert result.total_tokens == 150

    @pytest.mark.asyncio
    async def test_provider_name(self, provider):
        """Test provider name property."""
        assert provider.provider_name == "openai"


class TestAnthropicProvider:
    """Tests for Anthropic provider."""

    @pytest.fixture
    def provider(self):
        return AnthropicProvider(api_key="test-key")

    @pytest.mark.asyncio
    async def test_generate_success(self, provider):
        """Test successful generation."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Claude response")]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        result = await provider.generate(
            system_prompt="You are helpful.",
            user_message="Test question?",
            context_chunks=["Context 1"],
        )

        assert result.content == "Claude response"
        assert result.tokens_input == 100
        assert result.tokens_output == 50
        assert result.provider == "anthropic"

    @pytest.mark.asyncio
    async def test_provider_name(self, provider):
        """Test provider name property."""
        assert provider.provider_name == "anthropic"


class TestProviderFactory:
    """Tests for provider factory."""

    @pytest.mark.asyncio
    async def test_get_openai_provider(self):
        """Test getting OpenAI provider."""
        with patch("article_mind_service.chat.providers.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"
            mock_settings.llm_provider = "openai"
            mock_settings.llm_model = "gpt-4o-mini"

            provider = await get_llm_provider(provider_override="openai")
            assert isinstance(provider, OpenAIProvider)

    @pytest.mark.asyncio
    async def test_get_anthropic_provider(self):
        """Test getting Anthropic provider."""
        with patch("article_mind_service.chat.providers.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.llm_provider = "anthropic"
            mock_settings.llm_model = "claude-sonnet-4-5-20241022"

            provider = await get_llm_provider(provider_override="anthropic")
            assert isinstance(provider, AnthropicProvider)

    @pytest.mark.asyncio
    async def test_invalid_provider(self):
        """Test invalid provider name."""
        with pytest.raises(ValueError):
            await get_llm_provider(provider_override="invalid")  # type: ignore[arg-type]
