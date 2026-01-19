"""Unit tests for embedding providers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from article_mind_service.embeddings import (
    EmbeddingError,
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
)


class TestOpenAIProvider:
    """Tests for OpenAI embedding provider."""

    @pytest.fixture
    def provider(self) -> OpenAIEmbeddingProvider:
        """Create provider with test API key."""
        return OpenAIEmbeddingProvider(api_key="test-key")

    def test_dimensions(self, provider: OpenAIEmbeddingProvider) -> None:
        """Test dimensions property returns 1536."""
        assert provider.dimensions == 1536

    def test_model_name(self, provider: OpenAIEmbeddingProvider) -> None:
        """Test model_name property."""
        assert provider.model_name == "text-embedding-3-small"

    def test_max_tokens(self, provider: OpenAIEmbeddingProvider) -> None:
        """Test max_tokens property returns 8192."""
        assert provider.max_tokens == 8192

    def test_init_without_api_key(self) -> None:
        """Test initialization without API key raises ValueError."""
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            OpenAIEmbeddingProvider(api_key="")

    @pytest.mark.asyncio
    async def test_embed_empty_list(self, provider: OpenAIEmbeddingProvider) -> None:
        """Test embedding empty list returns empty list."""
        result = await provider.embed([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_single_text(self, provider: OpenAIEmbeddingProvider) -> None:
        """Test embedding single text."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]

        with patch.object(
            provider.client.embeddings,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await provider.embed(["test text"])

            assert len(result) == 1
            assert len(result[0]) == 1536
            assert result[0][0] == pytest.approx(0.1)

    @pytest.mark.asyncio
    async def test_embed_multiple_texts(self, provider: OpenAIEmbeddingProvider) -> None:
        """Test embedding multiple texts."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536),
            MagicMock(embedding=[0.2] * 1536),
            MagicMock(embedding=[0.3] * 1536),
        ]

        with patch.object(
            provider.client.embeddings,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await provider.embed(["text1", "text2", "text3"])

            assert len(result) == 3
            assert all(len(emb) == 1536 for emb in result)

    @pytest.mark.asyncio
    async def test_embed_api_error(self, provider: OpenAIEmbeddingProvider) -> None:
        """Test API error raises EmbeddingError."""
        with patch.object(
            provider.client.embeddings,
            "create",
            new_callable=AsyncMock,
            side_effect=Exception("API Error"),
        ):
            with pytest.raises(EmbeddingError) as exc_info:
                await provider.embed(["test"])

            assert "OpenAI embedding failed" in str(exc_info.value)


class TestOllamaProvider:
    """Tests for Ollama embedding provider."""

    @pytest.fixture
    def provider(self) -> OllamaEmbeddingProvider:
        """Create provider with default settings."""
        return OllamaEmbeddingProvider()

    def test_dimensions(self, provider: OllamaEmbeddingProvider) -> None:
        """Test dimensions property returns 1024."""
        assert provider.dimensions == 1024

    def test_model_name(self, provider: OllamaEmbeddingProvider) -> None:
        """Test model_name property."""
        assert provider.model_name == "nomic-embed-text"

    def test_max_tokens(self, provider: OllamaEmbeddingProvider) -> None:
        """Test max_tokens property returns 8192."""
        assert provider.max_tokens == 8192

    def test_custom_model(self) -> None:
        """Test initialization with custom model."""
        provider = OllamaEmbeddingProvider(model="custom-model")
        assert provider.model_name == "custom-model"

    @pytest.mark.asyncio
    async def test_embed_empty_list(self, provider: OllamaEmbeddingProvider) -> None:
        """Test embedding empty list returns empty list."""
        result = await provider.embed([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_single_text(self, provider: OllamaEmbeddingProvider) -> None:
        """Test embedding single text."""
        mock_response = {"embedding": [0.1] * 1024}

        with patch.object(
            provider.client,
            "embeddings",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await provider.embed(["test text"])

            assert len(result) == 1
            assert len(result[0]) == 1024
            assert result[0][0] == pytest.approx(0.1)

    @pytest.mark.asyncio
    async def test_embed_api_error(self, provider: OllamaEmbeddingProvider) -> None:
        """Test API error raises EmbeddingError."""
        with patch.object(
            provider.client,
            "embeddings",
            new_callable=AsyncMock,
            side_effect=Exception("Ollama Error"),
        ):
            with pytest.raises(EmbeddingError) as exc_info:
                await provider.embed(["test"])

            assert "Ollama embedding failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_health_check_model_available(self, provider: OllamaEmbeddingProvider) -> None:
        """Test health check when model is available."""
        with patch.object(
            provider.client,
            "list",
            new_callable=AsyncMock,
            return_value={"models": [{"name": "nomic-embed-text"}]},
        ):
            result = await provider.health_check()
            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_model_not_found(self, provider: OllamaEmbeddingProvider) -> None:
        """Test health check when model not found."""
        with patch.object(
            provider.client,
            "list",
            new_callable=AsyncMock,
            return_value={"models": []},
        ):
            result = await provider.health_check()
            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_server_error(self, provider: OllamaEmbeddingProvider) -> None:
        """Test health check when server error occurs."""
        with patch.object(
            provider.client,
            "list",
            new_callable=AsyncMock,
            side_effect=Exception("Server Error"),
        ):
            result = await provider.health_check()
            assert result is False
