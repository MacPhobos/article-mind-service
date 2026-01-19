"""ChromaDB vector storage integration."""

from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from .base import EmbeddingProvider


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
            persist_path: Directory for persistent storage.
            embedding_provider: Optional provider for query embedding.

        Note:
            Creates directory if it doesn't exist.
            Safe to call multiple times with same path.
        """
        self.client = chromadb.PersistentClient(
            path=persist_path,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )
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
            Duplicate IDs will update existing embeddings.
        """
        collection.add(
            embeddings=embeddings,  # type: ignore[arg-type]
            documents=texts,
            metadatas=metadatas,  # type: ignore[arg-type]
            ids=ids,
        )

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
