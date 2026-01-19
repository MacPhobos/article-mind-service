"""Abstract base class for embedding providers."""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    Design Decision: Use ABC over Protocol because we need shared
    implementation logic (retry handling, batching) in the base class.

    This abstraction allows easy switching between:
    - OpenAI text-embedding-3-small (cloud API)
    - Ollama nomic-embed-text (local)
    - Future providers (Voyage AI, Cohere, etc.)

    Time Complexity: Varies by provider
    - OpenAI: O(n) with batching (up to 2048 texts)
    - Ollama: O(n) sequential, but parallelizable with asyncio.gather

    Trade-offs:
    - Abstraction overhead: Minimal performance cost
    - Flexibility: Easy to swap providers via configuration
    - Testing: Simple to mock providers in tests
    """

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.
            Each vector is a list of floats with length=self.dimensions.

        Raises:
            EmbeddingError: If embedding generation fails.

        Performance:
            - Batch processing recommended for >10 texts
            - OpenAI: Max 2048 texts per batch
            - Ollama: Process in parallel with asyncio.gather
        """
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the dimensionality of embeddings produced by this provider.

        Returns:
            Number of dimensions in embedding vector.

        Examples:
            - OpenAI text-embedding-3-small: 1536
            - Ollama nomic-embed-text: 1024
            - Sentence Transformers all-mpnet-base-v2: 768
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier for this provider.

        Returns:
            Model name string for logging and debugging.

        Examples:
            - "text-embedding-3-small"
            - "nomic-embed-text"
        """
        pass

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        """Return the maximum context length in tokens.

        Returns:
            Maximum number of tokens that can be embedded in single text.

        Examples:
            - OpenAI text-embedding-3-small: 8192
            - Ollama nomic-embed-text: 8192
        """
        pass
