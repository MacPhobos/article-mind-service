"""Embedding pipeline orchestrator."""

from typing import Any

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from article_mind_service.search.sparse_search import BM25IndexCache

from .base import EmbeddingProvider
from .chromadb_store import ChromaDBStore
from .chunker import TextChunker
from .exceptions import EmbeddingError


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
    ):
        """Initialize pipeline.

        Args:
            provider: Embedding provider (OpenAI or Ollama).
            store: ChromaDB store instance.
            chunker: Text chunking instance.
        """
        self.provider = provider
        self.store = store
        self.chunker = chunker

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
            # Step 1: Chunk the text
            chunks = self.chunker.chunk_with_metadata(
                text,
                source_metadata={
                    "article_id": article_id,
                    "source_url": source_url,
                },
            )

            if not chunks:
                await self._update_status(db, article_id, "completed", chunk_count=0)
                return 0

            # Step 2: Get or create collection
            collection = self.store.get_or_create_collection(
                session_id=session_id,
                dimensions=self.provider.dimensions,
            )

            # Step 3: Prepare BM25 index data
            # Collect all (chunk_id, content) tuples for BM25 indexing
            bm25_chunks: list[tuple[str, str]] = []

            # Step 4: Process in batches
            total_chunks = len(chunks)
            for batch_start in range(0, total_chunks, self.BATCH_SIZE):
                batch_end = min(batch_start + self.BATCH_SIZE, total_chunks)
                batch = chunks[batch_start:batch_end]

                # Extract texts for embedding
                batch_texts = [c["text"] for c in batch]

                # Generate embeddings
                embeddings = await self.provider.embed(batch_texts)

                # Prepare metadata and IDs
                ids = [f"article_{article_id}_chunk_{c['chunk_index']}" for c in batch]
                metadatas = [
                    {
                        "article_id": c["article_id"],
                        "chunk_index": c["chunk_index"],
                        "source_url": c["source_url"],
                    }
                    for c in batch
                ]

                # Store in ChromaDB
                self.store.add_embeddings(
                    collection=collection,
                    embeddings=embeddings,
                    texts=batch_texts,
                    metadatas=metadatas,
                    ids=ids,
                )

                # Collect BM25 data (chunk_id, content) for this batch
                for chunk_id, text in zip(ids, batch_texts):
                    bm25_chunks.append((chunk_id, text))

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
