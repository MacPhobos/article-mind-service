"""ChromaDB vector storage integration."""

from typing import Any

import chromadb

from .base import EmbeddingProvider
from .client import get_chromadb_client


class ChromaDBStore:
    """ChromaDB vector store for embedding storage.

    Design Decisions:

    1. Collection per Session:
       - Isolates embeddings by session for multi-tenant use
       - Collection name: session_{session_id}
       - Enables easy cleanup when session deleted

    2. Persistent Storage:
       - Uses DuckDB + Parquet backend
       - Path configurable via CHROMADB_PATH
       - Survives service restarts
       - Data stored on filesystem, not in-memory

    3. Metadata Strategy:
       - article_id: Reference back to PostgreSQL
       - chunk_index: Position in original document
       - source_url: Original article URL
       - Enables filtered retrieval

    4. Dimension Handling:
       - ChromaDB auto-detects dimensions from first insert
       - Must be consistent within collection
       - Store dimension info in metadata for validation

    Performance:
        - Time Complexity:
          - add: O(n) where n = number of embeddings
          - query: O(log n) with HNSW index
        - Suitable for: < 200K vectors
        - For 200K-10M: Consider FAISS with HNSW/IVF

    Trade-offs:
        - ChromaDB: High-level API, metadata, CRUD
        - FAISS: Low-level, GPU, billions of vectors
        - Scale ceiling: ~200K vectors with good performance
    """

    def __init__(
        self,
        persist_path: str = "./data/chromadb",
        embedding_provider: EmbeddingProvider | None = None,
    ):
        """Initialize ChromaDB store.

        Args:
            persist_path: DEPRECATED - ignored, uses singleton client.
            embedding_provider: Optional provider for query embedding.

        Note:
            Uses singleton ChromaDB client from get_chromadb_client().
            persist_path parameter is kept for backward compatibility but ignored.
            Client path is configured via settings.chroma_persist_directory.
        """
        # Use singleton client instead of creating new instance
        # This prevents client conflicts when same path is accessed from multiple modules
        self.client = get_chromadb_client()
        self.embedding_provider = embedding_provider

    def get_or_create_collection(
        self,
        session_id: str,
        dimensions: int,
    ) -> chromadb.Collection:
        """Get or create a collection for a session.

        Args:
            session_id: Unique session identifier.
            dimensions: Expected embedding dimensions.

        Returns:
            ChromaDB Collection instance.

        Design Decision: Collection per session.
        - Enables session isolation
        - Easy to delete all embeddings for a session
        - Metadata tracks dimensions for validation

        Note:
            If collection exists, dimensions must match stored metadata.
        """
        collection_name = f"session_{session_id}"

        # Create with metadata to track dimensions
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "dimensions": dimensions,
                "session_id": session_id,
            },
        )

    def add_embeddings(
        self,
        collection: chromadb.Collection,
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        """Add embeddings to collection.

        Args:
            collection: Target ChromaDB collection.
            embeddings: List of embedding vectors.
            texts: Original text chunks (stored for retrieval).
            metadatas: Metadata for each chunk.
            ids: Unique IDs for each chunk.

        Performance:
            - Batch size 100: ~50-100ms
            - Batch size 1000: ~500ms-1s
            - Recommended: 100-500 per batch

        Note:
            IDs must be unique within collection.
            Duplicate IDs will raise an error (use upsert_embeddings instead).
        """
        collection.add(
            embeddings=embeddings,  # type: ignore[arg-type]
            documents=texts,
            metadatas=metadatas,  # type: ignore[arg-type]
            ids=ids,
        )

    def upsert_embeddings(
        self,
        collection: chromadb.Collection,
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        """Upsert embeddings to collection (update existing or insert new).

        Design Decision: Upsert for Deduplication
        ==========================================

        Rationale: Use upsert instead of add to enable re-indexing without errors.
        - Updates existing chunks if content changed (new embedding)
        - Inserts new chunks if not present
        - Prevents duplicate ID errors on re-index

        Args:
            collection: Target ChromaDB collection.
            embeddings: List of embedding vectors.
            texts: Original text chunks (stored for retrieval).
            metadatas: Metadata for each chunk.
            ids: Unique IDs for each chunk.

        Performance:
            - Same as add_embeddings (~50-100ms per 100 chunks)
            - ChromaDB handles existence check internally
            - No performance penalty for upsert vs add

        Benefits:
            - Idempotent: Can re-run without errors
            - Deduplication: Skip unchanged chunks (when combined with existence check)
            - Flexibility: Handles both new and updated content

        Trade-offs:
            - Slightly more complex than add (but safer)
            - Always writes to DB (even if unchanged) unless we check first

        Example:
            # First time: inserts new chunks
            store.upsert_embeddings(collection, embeddings, texts, metadatas, ids)

            # Re-index with same content: updates existing (but could skip if unchanged)
            store.upsert_embeddings(collection, embeddings, texts, metadatas, ids)
        """
        collection.upsert(
            embeddings=embeddings,  # type: ignore[arg-type]
            documents=texts,
            metadatas=metadatas,  # type: ignore[arg-type]
            ids=ids,
        )

    def get_existing_chunks(
        self,
        collection: chromadb.Collection,
        chunk_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Get existing chunks by IDs to check for deduplication.

        Design Decision: Batch Existence Check
        ======================================

        Rationale: Check if chunks exist before embedding to skip unchanged content.
        - Reduces embedding API costs (skip unchanged chunks)
        - Enables smart re-indexing (only embed changed content)
        - Returns metadata to compare content_hash

        Args:
            collection: ChromaDB collection to query.
            chunk_ids: List of chunk IDs to check.

        Returns:
            Dict mapping chunk_id -> metadata for existing chunks.
            Only includes chunks that exist in collection.

        Performance:
            - Time Complexity: O(n) where n = number of IDs
            - Typical: 10-50ms for 100 IDs
            - ChromaDB get() is optimized for batch lookups

        Example:
            existing = store.get_existing_chunks(collection, ["chunk1", "chunk2"])
            # Returns: {"chunk1": {"content_hash": "abc123", ...}}
            # chunk2 not in result means it doesn't exist
        """
        try:
            result = collection.get(
                ids=chunk_ids,
                include=["metadatas"],
            )

            # Build dict of chunk_id -> metadata for existing chunks
            existing_chunks: dict[str, dict[str, Any]] = {}
            if result and result.get("ids") and result.get("metadatas"):
                for chunk_id, metadata in zip(result["ids"], result["metadatas"]):
                    existing_chunks[chunk_id] = metadata or {}

            return existing_chunks

        except Exception:
            # If collection is empty or error occurs, return empty dict
            return {}

    def query(
        self,
        collection: chromadb.Collection,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> Any:  # Returns chromadb QueryResult
        """Query collection for similar embeddings.

        Args:
            collection: ChromaDB collection to query.
            query_embedding: Embedding vector for query.
            n_results: Number of results to return.
            where: Optional metadata filter.

        Returns:
            Query results with ids, documents, distances, metadatas.

        Performance:
            - Time Complexity: O(log n) with HNSW index
            - Typical: 10-50ms for n_results=10

        Example:
            # Filter by article_id
            results = store.query(
                collection,
                query_embedding,
                n_results=5,
                where={"article_id": 123}
            )
        """
        return collection.query(
            query_embeddings=[query_embedding],  # type: ignore[arg-type]
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    def delete_collection(self, session_id: str) -> None:
        """Delete a session's collection.

        Args:
            session_id: Session to delete embeddings for.

        Note:
            Silently succeeds if collection doesn't exist.
            Use for cleanup when session is deleted.
        """
        collection_name = f"session_{session_id}"
        try:
            self.client.delete_collection(collection_name)
        except ValueError:
            # Collection doesn't exist, ignore
            pass

    def get_collection_stats(self, session_id: str) -> dict[str, Any]:
        """Get statistics for a session's collection.

        Args:
            session_id: Session to get stats for.

        Returns:
            Dict with count, dimensions, etc.

        Example:
            stats = store.get_collection_stats("abc123")
            # {"name": "session_abc123", "count": 150, "metadata": {...}}
        """
        collection_name = f"session_{session_id}"
        try:
            collection = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "count": collection.count(),
                "metadata": collection.metadata,
            }
        except ValueError:
            return {"name": collection_name, "count": 0, "exists": False}
