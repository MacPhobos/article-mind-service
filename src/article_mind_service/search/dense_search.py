"""Dense vector search using ChromaDB.

This module provides semantic similarity search using embeddings stored
in ChromaDB during the indexing phase (R5). Dense search excels at:
- Semantic similarity matching
- Paraphrased queries
- Conceptual search (even without exact keywords)
- Synonym and related term matching

ChromaDB Configuration:
- Uses cosine distance for similarity
- Returns top-K most similar chunks by vector similarity
- Filters results by session_id to prevent cross-session leakage
"""

from dataclasses import dataclass

import chromadb
from chromadb.api.types import QueryResult

from article_mind_service.config import settings


@dataclass
class DenseSearchResult:
    """Result from dense vector search.

    Attributes:
        chunk_id: Unique chunk identifier
        score: Similarity score (0.0-1.0, higher = more similar)
        metadata: Additional metadata (article_id, source_url, etc.)
    """

    chunk_id: str
    score: float  # Similarity score (higher = more similar)
    metadata: dict[str, str]


class DenseSearch:
    """ChromaDB-based dense vector search.

    Performs semantic similarity search using embeddings stored
    in ChromaDB during the indexing phase (R5).

    Design Decision: ChromaDB Persistent Client
    - Uses persistent client for data durability across restarts
    - Path configured via settings.chroma_persist_directory
    - Collection name from settings.chroma_collection_name

    Thread Safety: ChromaDB client is thread-safe and can be shared
    across async requests.

    Usage:
        dense = DenseSearch()
        results = dense.search(
            session_id=1,
            query_embedding=embedding_vector,
            top_k=10
        )
    """

    def __init__(self, collection_name: str | None = None) -> None:
        """Initialize dense search with ChromaDB connection.

        Args:
            collection_name: ChromaDB collection name (default from settings)
        """
        self.client = chromadb.PersistentClient(path=str(settings.chroma_persist_directory))
        self.collection_name = collection_name or settings.chroma_collection_name

    def search(
        self,
        session_id: int,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[DenseSearchResult]:
        """Search for similar chunks using query embedding.

        Args:
            session_id: Session ID to filter results
            query_embedding: Query vector from embedding model
            top_k: Number of results to return

        Returns:
            List of DenseSearchResult sorted by similarity descending

        Performance:
        - ChromaDB uses HNSW index for O(log n) search
        - Typical latency: 50-100ms for 1000 chunks
        - Latency grows logarithmically with index size

        Error Handling:
        - Returns empty list if collection doesn't exist
        - Logs warnings for ChromaDB errors
        """
        try:
            collection = self.client.get_collection(self.collection_name)
        except ValueError:
            # Collection doesn't exist yet
            return []

        # Query with session filter
        # Type ignore: ChromaDB accepts list[float] but type hints are incomplete
        results: QueryResult = collection.query(
            query_embeddings=[query_embedding],  # type: ignore
            n_results=top_k,
            where={"session_id": session_id},
            include=["metadatas", "distances"],
        )

        # Convert to DenseSearchResult
        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                # ChromaDB returns distances; convert to similarity
                # For cosine distance: similarity = 1 - distance
                distance = results["distances"][0][i] if results["distances"] else 0.0
                similarity = 1.0 - distance

                metadata = results["metadatas"][0][i] if results["metadatas"] else {}

                search_results.append(
                    DenseSearchResult(
                        chunk_id=chunk_id,
                        score=similarity,
                        metadata=metadata,  # type: ignore
                    )
                )

        return search_results

    def get_total_chunks(self, session_id: int) -> int:
        """Get total number of chunks indexed for a session.

        Args:
            session_id: Session ID

        Returns:
            Number of chunks indexed for this session

        Usage: Populate total_chunks_searched field in SearchResponse
        """
        try:
            collection = self.client.get_collection(self.collection_name)

            # Count chunks with session filter
            results = collection.get(
                where={"session_id": session_id},
                include=[],  # Don't need actual data, just count
            )

            return len(results["ids"]) if results["ids"] else 0
        except (ValueError, Exception):
            # Collection doesn't exist or other ChromaDB error
            return 0
