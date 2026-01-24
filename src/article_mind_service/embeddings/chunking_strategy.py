"""Chunking strategy abstraction for pluggable chunking algorithms.

Design Decision: Strategy Pattern for Chunking
===============================================

Problem: Need to support multiple chunking algorithms (fixed-size, semantic,
hybrid) and allow runtime switching via configuration.

Solution: Strategy pattern with standardized ChunkResult interface.

Design Rationale:
- Decouple chunking algorithm from embedding pipeline
- Enable A/B testing of chunking strategies
- Allow per-session or per-article strategy override
- Simplify testing (mock strategy implementations)

Alternative Approaches Rejected:
1. Inheritance-based (e.g., BaseChunker with subclasses)
   - Rejected: Harder to compose strategies (e.g., hybrid chunking)
2. Function-based (simple functions instead of classes)
   - Rejected: No state management (can't cache tokenizers, etc.)
3. Single chunker with if/else logic
   - Rejected: Violates Open/Closed Principle (not extensible)

Trade-offs:
- ✅ Flexibility: Easy to add new strategies without modifying pipeline
- ✅ Testability: Mock strategies for unit tests
- ✅ Configuration: Runtime strategy selection
- ❌ Abstraction overhead: Extra layer of indirection
- ❌ Complexity: More classes to understand

Extension Points:
- HybridChunkingStrategy (semantic + fixed-size fallback)
- ClusteringChunkingStrategy (group similar sentences)
- Per-article strategy override (override_chunking_strategy field)
"""

from dataclasses import dataclass
from typing import Any, Protocol

from article_mind_service.embeddings.chunker import TextChunker
from article_mind_service.embeddings.semantic_chunker import (
    SemanticChunk,
    SemanticChunker,
)


@dataclass
class ChunkResult:
    """Standardized chunk result across all strategies.

    All chunking strategies return this uniform structure,
    enabling seamless strategy switching in the embedding pipeline.

    Attributes:
        text: The chunk text content.
        chunk_index: Zero-based chunk index in document.
        metadata: Additional metadata (article_id, source_url, chunk-specific metadata).
    """

    text: str
    chunk_index: int
    metadata: dict[str, Any]


class ChunkingStrategy(Protocol):
    """Protocol for chunking strategies.

    Design Decision: Protocol (Structural Typing) vs ABC (Nominal Typing)
    ====================================================================

    Rationale: Protocol enables structural subtyping (duck typing with type safety).
    - No need to inherit from base class
    - More flexible for testing (any class with chunk() method works)
    - Aligns with Python's duck typing philosophy

    Trade-off: Less explicit than ABC (no "is a" relationship in class hierarchy)

    All chunking strategies must implement this protocol.
    """

    async def chunk(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> list[ChunkResult]:
        """Chunk text and return standardized results.

        Args:
            text: The full text to chunk.
            metadata: Optional metadata to attach to all chunks.

        Returns:
            List of ChunkResult objects.
        """
        ...


class FixedSizeChunkingStrategy:
    """Traditional fixed-size chunking (current implementation).

    Uses RecursiveCharacterTextSplitter with token counting.

    Characteristics:
    - Fast (no embedding required)
    - Predictable chunk sizes (512 tokens ± overlap)
    - May split semantically related content

    When to Use:
    - Real-time ingestion (low latency required)
    - Short articles (<500 words)
    - Homogeneous content (no topic shifts)

    Performance:
    - 10KB text: ~50ms
    - 100KB text: ~500ms
    """

    def __init__(self, chunker: TextChunker):
        """Initialize fixed-size chunking strategy.

        Args:
            chunker: TextChunker instance with configured chunk_size/overlap.
        """
        self.chunker = chunker

    async def chunk(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> list[ChunkResult]:
        """Chunk using fixed-size splitter.

        Args:
            text: The full text to chunk.
            metadata: Optional metadata to attach to all chunks.

        Returns:
            List of ChunkResult objects.

        Note:
            This method is async for protocol consistency, but the
            underlying operation is synchronous (no I/O).
        """
        chunks = self.chunker.chunk_with_metadata(text, metadata or {})
        return [
            ChunkResult(
                text=c["text"],
                chunk_index=c.get("chunk_index", i),
                metadata=c,
            )
            for i, c in enumerate(chunks)
        ]


class SemanticChunkingStrategy:
    """Semantic similarity-based chunking.

    Uses sentence embeddings to find natural topic boundaries.

    Characteristics:
    - Slower (requires embedding all sentences)
    - Maintains semantic coherence
    - Variable chunk sizes (within min/max bounds)
    - Up to 70% better retrieval accuracy (research-backed)

    When to Use:
    - High-value content (research papers, technical docs)
    - Long-form articles with distinct topics
    - Quality over speed

    Performance:
    - 1K words (~50 sentences): 500ms - 2s
    - 5K words (~250 sentences): 2s - 10s
    - 10K words (~500 sentences): 5s - 20s
    """

    def __init__(self, chunker: SemanticChunker):
        """Initialize semantic chunking strategy.

        Args:
            chunker: SemanticChunker instance with embedding provider.
        """
        self.chunker = chunker

    async def chunk(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> list[ChunkResult]:
        """Chunk using semantic boundaries.

        Args:
            text: The full text to chunk.
            metadata: Optional metadata to attach to all chunks.

        Returns:
            List of ChunkResult objects.

        Raises:
            EmbeddingError: If sentence embedding generation fails.
        """
        chunks = await self.chunker.chunk(text, metadata)
        return [
            ChunkResult(
                text=c.text,
                chunk_index=i,
                metadata={**c.metadata, "chunk_index": i},
            )
            for i, c in enumerate(chunks)
        ]


# Future extension point: Hybrid strategy
# class HybridChunkingStrategy:
#     """Hybrid chunking: Semantic with fixed-size fallback.
#
#     Strategy:
#     1. Attempt semantic chunking
#     2. If semantic chunking fails (e.g., embedding timeout), fallback to fixed-size
#     3. If semantic chunks are too large/small, re-chunk with fixed-size
#
#     When to Use:
#     - Production systems requiring reliability
#     - Mixed content types (some benefit from semantic, some don't)
#     """
#     pass
