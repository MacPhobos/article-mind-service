"""OpenAI embedding provider implementation."""

from openai import AsyncOpenAI

from .base import EmbeddingProvider
from .exceptions import EmbeddingError


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI text-embedding-3-small provider.

    Model Details:
        - Dimensions: 1536
        - Max tokens: 8,192
        - Cost: $0.02 per 1M tokens
        - MTEB Score: 62.3%

    Design Decisions:

    1. Async Client:
       - Uses AsyncOpenAI for non-blocking I/O
       - Integrates with FastAPI's async ecosystem
       - No synchronous blocking in event loop

    2. Batch Processing:
       - OpenAI API supports batching (up to 2048 inputs)
       - Reduces API calls and latency
       - Implement chunked batching if > 2048 texts

    3. Error Handling:
       - Wraps all API errors in EmbeddingError
       - Preserves original exception with __cause__
       - No retry logic in provider (handled at pipeline level)

    Performance:
        - Time Complexity: O(n) where n = number of texts
        - API latency: ~100-500ms for batch of 100 texts
        - Recommended batch size: 100-500 texts

    Trade-offs:
        - Cost: $0.02/1M tokens vs $0.00 for local models
        - Speed: API latency vs local GPU inference
        - Quality: 62.3% MTEB (good, not best-in-class)
        - Simplicity: No GPU required, easy setup
    """

    MODEL = "text-embedding-3-small"
    DIMENSIONS = 1536
    MAX_TOKENS = 8192
    MAX_BATCH_SIZE = 2048

    def __init__(self, api_key: str):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (from environment).

        Raises:
            ValueError: If api_key is empty or None.
        """
        if not api_key:
            raise ValueError("OpenAI API key is required")

        self.client = AsyncOpenAI(api_key=api_key)

    @property
    def dimensions(self) -> int:
        """Return embedding dimensions (1536)."""
        return self.DIMENSIONS

    @property
    def model_name(self) -> str:
        """Return model name."""
        return self.MODEL

    @property
    def max_tokens(self) -> int:
        """Return max context length (8192 tokens)."""
        return self.MAX_TOKENS

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using OpenAI API.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors (1536 dimensions each).

        Raises:
            EmbeddingError: If API call fails.

        Performance:
            - Batch size 100: ~200ms
            - Batch size 500: ~500ms
            - Batch size 2048: ~2s
        """
        if not texts:
            return []

        try:
            # Process in batches if necessary
            all_embeddings: list[list[float]] = []
            for i in range(0, len(texts), self.MAX_BATCH_SIZE):
                batch = texts[i : i + self.MAX_BATCH_SIZE]
                response = await self.client.embeddings.create(
                    model=self.MODEL,
                    input=batch,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            return all_embeddings

        except Exception as e:
            raise EmbeddingError(f"OpenAI embedding failed: {e}") from e
