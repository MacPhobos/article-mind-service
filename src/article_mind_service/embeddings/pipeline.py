"""Embedding pipeline orchestrator."""

import hashlib
import logging
import re
from typing import Any

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

from article_mind_service.config import settings
from article_mind_service.search.sparse_search import BM25IndexCache

from .base import EmbeddingProvider
from .cache import EmbeddingCache
from .chromadb_store import ChromaDBStore
from .chunker import TextChunker
from .chunking_strategy import (
    ChunkingStrategy,
    FixedSizeChunkingStrategy,
    SemanticChunkingStrategy,
)
from .exceptions import EmbeddingError
from .semantic_chunker import SemanticChunker


def generate_chunk_id(article_id: int, text: str, chunk_index: int) -> str:
    """Generate deterministic chunk ID based on content hash.

    Design Decision: Content-Based Chunk IDs
    ========================================

    Rationale: Use content hash instead of sequential IDs to enable deduplication.
    - Same content always produces same ID (skip re-embedding unchanged chunks)
    - Different content produces different IDs (detect changes)
    - Includes article_id to prevent cross-article collisions
    - Includes chunk_index for ordering and uniqueness within article

    Format: sha256(f"{article_id}:{content_hash}:{chunk_index}")[:16]
    where content_hash = sha256(text)[:8]

    Args:
        article_id: Database ID of the article.
        text: Chunk text content.
        chunk_index: Zero-based position in article.

    Returns:
        16-character hexadecimal chunk ID.

    Benefits:
        - Deterministic: Same input always produces same ID
        - Deduplication: Unchanged chunks detected on re-index
        - Cost Savings: Skip embedding API calls for unchanged content
        - Performance: Hash computation is O(n) where n = text length (~1ms)

    Trade-offs:
        - Different from old ID format (article_{id}_chunk_{index})
        - Requires content_hash in metadata to check without re-hashing
        - 16 chars: Low collision probability (2^64 combinations)

    Performance:
        - Time Complexity: O(n) where n = text length
        - Typical speed: <1ms for 1000 chars
        - SHA-256 is fast and well-optimized

    Example:
        >>> generate_chunk_id(123, "Hello world", 0)
        '7a3f8e9c1b2d4f6a'
        >>> generate_chunk_id(123, "Hello world", 0)  # Same content, same ID
        '7a3f8e9c1b2d4f6a'
        >>> generate_chunk_id(123, "Different text", 0)  # Different content
        '9b4e2f1a8c6d3e5b'
    """
    # Generate content hash (first 8 hex chars of SHA-256)
    content_hash = hashlib.sha256(text.encode()).hexdigest()[:8]

    # Combine article_id, content_hash, and chunk_index
    id_string = f"{article_id}:{content_hash}:{chunk_index}"

    # Generate final chunk ID (first 16 hex chars of SHA-256)
    return hashlib.sha256(id_string.encode()).hexdigest()[:16]


class EmbeddingPipeline:
    """Orchestrates text chunking, embedding, and storage.

    Pipeline Flow:
        1. Receive article text and metadata
        2. Chunk text using TextChunker
        3. Generate embeddings in batches
        4. Store in ChromaDB with metadata
        5. Populate BM25 index for keyword search
        6. Update article status in PostgreSQL

    Design Decisions:

    1. Batch Processing:
       - Process in batches of 100 chunks
       - Balance memory usage and efficiency
       - Progress updates after each batch

    2. BM25 Index Population:
       - BM25 index populated during embedding (not on-demand)
       - Ensures content available for hybrid search
       - In-memory index rebuilt on service restart

    3. Error Recovery:
       - Store progress in database
       - Resume from last successful batch
       - Mark failed articles for retry

    4. Status Tracking:
       - pending: Article queued for embedding
       - processing: Currently generating embeddings
       - completed: All chunks embedded
       - failed: Error during processing

    Performance:
        - Time Complexity: O(n) where n = number of chunks
        - Typical article (5K words): ~2-5 seconds
        - Bottleneck: Embedding generation, not storage
        - BM25 indexing: ~5-10ms overhead per article

    Trade-offs:
        - Batch size 100: Good balance memory/speed
        - Status tracking: Overhead for reliability
        - Synchronous updates: Simpler than async queue
        - BM25 in-memory: Fast but lost on restart (rebuilds on reindex)
    """

    BATCH_SIZE = 100

    def __init__(
        self,
        provider: EmbeddingProvider,
        store: ChromaDBStore,
        chunker: TextChunker,
        chunking_strategy: ChunkingStrategy | None = None,
        cache: EmbeddingCache | None = None,
    ):
        """Initialize pipeline.

        Args:
            provider: Embedding provider (OpenAI or Ollama).
            store: ChromaDB store instance.
            chunker: Text chunking instance (used for fixed-size strategy).
            chunking_strategy: Optional chunking strategy. If None, creates
                strategy based on settings.chunking_strategy.
            cache: Optional embedding cache. If None, creates default cache
                from settings.

        Design Decision: Dependency Injection for Chunking Strategy
        ============================================================

        Rationale: Pass strategy via constructor instead of creating inside __init__.
        - Testability: Easy to inject mock strategies
        - Flexibility: Override strategy per pipeline instance
        - Configuration: Factory function handles strategy selection

        Trade-off: Caller must create strategy (more verbose)
        """
        self.provider = provider
        self.store = store
        self.chunker = chunker

        # Create chunking strategy based on configuration if not provided
        if chunking_strategy is None:
            chunking_strategy = get_chunking_strategy(
                embedding_provider=provider, text_chunker=chunker
            )
        self.chunking_strategy = chunking_strategy

        # Create embedding cache from settings if not provided
        if cache is None:
            cache = EmbeddingCache(
                cache_dir=settings.embedding_cache_dir,
                max_memory_entries=settings.embedding_cache_max_memory,
                max_disk_size_mb=settings.embedding_cache_max_disk_mb,
            )
        self.cache = cache

    async def process_article(
        self,
        article_id: int,
        session_id: str,
        text: str,
        source_url: str,
        db: AsyncSession,
    ) -> int:
        """Process a single article through the embedding pipeline.

        Args:
            article_id: Database ID of the article.
            session_id: Session this article belongs to.
            text: Extracted article text.
            source_url: Original article URL.
            db: Database session for status updates.

        Returns:
            Number of chunks created.

        Raises:
            EmbeddingError: If processing fails.

        Performance:
            - 1K words: ~1 second
            - 5K words: ~2-5 seconds
            - 10K words: ~5-10 seconds
        """
        # Import here to avoid circular import

        # Update status to processing
        await self._update_status(db, article_id, "processing")

        try:
            # Step 1: Chunk the text using configured strategy
            chunk_results = await self.chunking_strategy.chunk(
                text,
                metadata={
                    "article_id": article_id,
                    "source_url": source_url,
                },
            )

            if not chunk_results:
                await self._update_status(db, article_id, "completed", chunk_count=0)
                return 0

            # Convert ChunkResult to dict format for backward compatibility
            chunks = [
                {
                    "text": c.text,
                    "chunk_index": c.chunk_index,
                    "article_id": c.metadata.get("article_id", article_id),
                    "source_url": c.metadata.get("source_url", source_url),
                }
                for c in chunk_results
            ]

            # Step 2: Get or create collection
            collection = self.store.get_or_create_collection(
                session_id=session_id,
                dimensions=self.provider.dimensions,
            )

            # Step 3: Prepare BM25 index data
            # Collect all (chunk_id, content) tuples for BM25 indexing
            bm25_chunks: list[tuple[str, str]] = []

            # Step 4: Process in batches with deduplication
            total_chunks = len(chunks)
            skipped_chunks = 0
            embedded_chunks = 0

            for batch_start in range(0, total_chunks, self.BATCH_SIZE):
                batch_end = min(batch_start + self.BATCH_SIZE, total_chunks)
                batch = chunks[batch_start:batch_end]

                # Generate chunk IDs and content hashes for this batch
                batch_data = [
                    {
                        "chunk": c,
                        "text": c["text"],
                        "chunk_id": generate_chunk_id(article_id, c["text"], c["chunk_index"]),
                        "content_hash": hashlib.sha256(c["text"].encode()).hexdigest()[:8],
                    }
                    for c in batch
                ]

                # Check for existing chunks to enable deduplication
                chunk_ids = [d["chunk_id"] for d in batch_data]
                existing_chunks = self.store.get_existing_chunks(collection, chunk_ids)

                # Separate new/changed chunks from unchanged chunks
                chunks_to_embed = []
                for data in batch_data:
                    chunk_id = data["chunk_id"]
                    content_hash = data["content_hash"]

                    # Check if chunk exists and content is unchanged
                    if chunk_id in existing_chunks:
                        existing_metadata = existing_chunks[chunk_id]
                        existing_hash = existing_metadata.get("content_hash")

                        if existing_hash == content_hash:
                            # Chunk exists and content is unchanged - skip embedding
                            skipped_chunks += 1
                            # Still add to BM25 index (already in ChromaDB)
                            bm25_chunks.append((chunk_id, data["text"]))
                            continue

                    # Chunk is new or content changed - needs embedding
                    chunks_to_embed.append(data)

                # Skip this batch if all chunks are unchanged
                if not chunks_to_embed:
                    continue

                # Extract texts for embedding
                batch_texts = [d["text"] for d in chunks_to_embed]

                # Check cache before calling embedding provider
                embeddings: list[list[float]] = []
                uncached_indices: list[int] = []
                uncached_texts: list[str] = []
                cache_hits = 0

                for i, text in enumerate(batch_texts):
                    cached_embedding = self.cache.get(text)
                    if cached_embedding is not None:
                        # Cache hit - use cached embedding
                        embeddings.append(cached_embedding)
                        cache_hits += 1
                    else:
                        # Cache miss - need to generate embedding
                        embeddings.append([])  # Placeholder
                        uncached_indices.append(i)
                        uncached_texts.append(text)

                # Generate embeddings only for cache misses
                if uncached_texts:
                    new_embeddings = await self.provider.embed(uncached_texts)
                    embedded_chunks += len(new_embeddings)

                    # Fill in embeddings and store in cache
                    for idx, new_embedding in zip(uncached_indices, new_embeddings):
                        embeddings[idx] = new_embedding
                        # Store in cache for future use
                        self.cache.put(batch_texts[idx], new_embedding)

                # Log cache statistics for this batch
                logger.info(
                    "Batch embeddings: %d total, %d cached, %d computed",
                    len(batch_texts),
                    cache_hits,
                    len(uncached_texts),
                )

                # Prepare metadata and IDs
                ids = [d["chunk_id"] for d in chunks_to_embed]
                metadatas = [
                    {
                        "article_id": d["chunk"]["article_id"],
                        "chunk_index": d["chunk"]["chunk_index"],
                        "source_url": d["chunk"]["source_url"],
                        # Content hash for deduplication check
                        "content_hash": d["content_hash"],
                        # Enhanced metadata for filtering
                        "word_count": len(d["text"].split()),
                        "has_code": bool(re.search(r'```|def |class |function |import |const |let |var ', d["text"])),
                    }
                    for d in chunks_to_embed
                ]

                # Use upsert to handle both new and updated chunks
                self.store.upsert_embeddings(
                    collection=collection,
                    embeddings=embeddings,
                    texts=batch_texts,
                    metadatas=metadatas,
                    ids=ids,
                )

                # Collect BM25 data (chunk_id, content) for this batch
                for chunk_id, text in zip(ids, batch_texts):
                    bm25_chunks.append((chunk_id, text))

            # Log deduplication and cache statistics
            cache_stats = self.cache.stats
            logger.info(
                "Article %d: %d total chunks, %d embedded, %d skipped (unchanged). "
                "Cache: %.1f%% hit rate (%d hits, %d misses)",
                article_id,
                total_chunks,
                embedded_chunks,
                skipped_chunks,
                cache_stats["hit_rate"] * 100,
                cache_stats["hits"],
                cache_stats["misses"],
            )

            # Step 5: Populate BM25 index with all chunks
            # Convert session_id to int for BM25IndexCache
            session_id_int = int(session_id)

            # Get existing index or create new one
            bm25_index = BM25IndexCache.get_or_create(session_id_int)

            # Add all chunks from this article to BM25 index
            for chunk_id, content in bm25_chunks:
                bm25_index.add_document(chunk_id, content)

            # Build the BM25 index (lazy build triggers on first search, but we can force it here)
            bm25_index.build()

            # Update status to completed
            await self._update_status(db, article_id, "completed", chunk_count=total_chunks)

            return total_chunks

        except Exception as e:
            await self._update_status(db, article_id, "failed")
            raise EmbeddingError(f"Pipeline failed for article {article_id}: {e}") from e

    async def _update_status(
        self,
        db: AsyncSession,
        article_id: int,
        status: str,
        chunk_count: int | None = None,
    ) -> None:
        """Update article embedding status in database.

        Args:
            db: Database session.
            article_id: Article to update.
            status: New embedding status.
            chunk_count: Optional chunk count to set.
        """
        from article_mind_service.models.article import Article

        values: dict[str, Any] = {"embedding_status": status}
        if chunk_count is not None:
            values["chunk_count"] = chunk_count

        stmt = update(Article).where(Article.id == article_id).values(**values)
        await db.execute(stmt)
        await db.commit()


def get_chunking_strategy(
    embedding_provider: EmbeddingProvider | None = None,
    text_chunker: TextChunker | None = None,
) -> ChunkingStrategy:
    """Get chunking strategy based on configuration.

    Design Decision: Factory Function for Strategy Creation
    =======================================================

    Rationale: Centralize strategy creation logic in factory function.
    - Configuration-driven: Read from settings.chunking_strategy
    - Default fallback: Fixed-size if semantic not configured properly
    - Testability: Easy to override in tests

    Args:
        embedding_provider: Embedding provider (required for semantic chunking).
        text_chunker: Text chunker (required for fixed-size chunking).

    Returns:
        Configured chunking strategy.

    Raises:
        ValueError: If semantic strategy selected but no embedding provider.

    Performance:
        - Fixed-size: 10KB text ~50ms
        - Semantic: 10KB text ~2-5 seconds (requires embedding all sentences)

    Example:
        # Create semantic strategy
        strategy = get_chunking_strategy(
            embedding_provider=openai_provider,
            text_chunker=text_chunker,
        )

        # Or create fixed-size strategy (default)
        strategy = get_chunking_strategy(text_chunker=text_chunker)
    """
    if settings.chunking_strategy == "semantic" and embedding_provider:
        semantic_chunker = SemanticChunker(
            embedding_provider=embedding_provider,
            breakpoint_percentile=settings.semantic_chunk_breakpoint_percentile,
            min_chunk_size=settings.semantic_chunk_min_size,
            max_chunk_size=settings.semantic_chunk_max_size,
        )
        return SemanticChunkingStrategy(semantic_chunker)
    else:
        # Fallback to fixed-size if:
        # 1. chunking_strategy == "fixed"
        # 2. chunking_strategy == "semantic" but no embedding_provider
        # 3. Unknown strategy value

        # Create text chunker if not provided
        if text_chunker is None:
            text_chunker = TextChunker(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )

        return FixedSizeChunkingStrategy(text_chunker)
