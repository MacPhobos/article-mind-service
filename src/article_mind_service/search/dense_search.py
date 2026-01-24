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
from chromadb.errors import NotFoundError

from article_mind_service.config import settings
from article_mind_service.embeddings.client import get_chromadb_client


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

        Note:
            Uses singleton ChromaDB client from get_chromadb_client().
            This ensures consistent client settings across all modules.
        """
        # Use singleton client to prevent client conflicts
        self.client = get_chromadb_client()
        self.collection_name = collection_name or settings.chroma_collection_name

    def search(
        self,
        session_id: int,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, any] | None = None,
    ) -> list[DenseSearchResult]:
        """Search for similar chunks using query embedding.

        Args:
            session_id: Session ID to filter results
            query_embedding: Query vector from embedding model
            top_k: Number of results to return
            filters: Optional metadata filters (e.g., {"article_id": 42, "has_code": True})

        Returns:
            List of DenseSearchResult sorted by similarity descending

        Performance:
        - ChromaDB uses HNSW index for O(log n) search
        - Typical latency: 50-100ms for 1000 chunks
        - Latency grows logarithmically with index size
        - Filters applied at ChromaDB level (efficient)

        Error Handling:
        - Returns empty list if collection doesn't exist
        - Logs warnings for ChromaDB errors
        - Invalid filters ignored with warning

        Design Decision: Session-based collection naming
        - Uses session_{session_id} collection name to match indexing strategy
        - Each session has isolated collection for multi-tenant safety
        - Enables easy cleanup when session deleted

        Metadata Filtering:
        - Supports equality filters on any metadata field
        - ChromaDB applies filters before vector search (efficient)
        - Common filters: article_id, has_code, word_count range
        """
        # Use session-based collection naming (matches ChromaDBStore)
        collection_name = f"session_{session_id}"

        try:
            collection = self.client.get_collection(collection_name)
        except (ValueError, NotFoundError):
            # Collection doesn't exist yet
            return []

        # Build ChromaDB where clause from filters
        where_filter = None
        if filters:
            where_filter = self._build_where_clause(filters)

        # Query collection (no session filter needed since collection is per-session)
        # Type ignore: ChromaDB accepts list[float] but type hints are incomplete
        results: QueryResult = collection.query(
            query_embeddings=[query_embedding],  # type: ignore
            n_results=top_k,
            where=where_filter,  # Apply metadata filters
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

    def _build_where_clause(self, filters: dict[str, any]) -> dict[str, any]:
        """Build ChromaDB where clause from filter dictionary.

        Args:
            filters: Dictionary of metadata filters

        Returns:
            ChromaDB where clause dictionary

        Design Decision: Simple equality filters
        - Supports basic equality checks on metadata fields
        - ChromaDB syntax: {"field": {"$eq": value}} or {"field": value}
        - Future enhancement: Add range operators ($gt, $lt, $gte, $lte)

        Example:
            >>> _build_where_clause({"article_id": 42, "has_code": True})
            {"$and": [{"article_id": {"$eq": 42}}, {"has_code": {"$eq": True}}]}
        """
        if not filters:
            return {}

        # Convert simple filters to ChromaDB where clause
        # Single filter: {"field": value}
        # Multiple filters: {"$and": [{"field1": value1}, {"field2": value2}]}
        if len(filters) == 1:
            key, value = next(iter(filters.items()))
            return {key: {"$eq": value}}

        # Multiple filters - use $and operator
        return {
            "$and": [
                {key: {"$eq": value}}
                for key, value in filters.items()
            ]
        }

    def get_total_chunks(self, session_id: int) -> int:
        """Get total number of chunks indexed for a session.

        Args:
            session_id: Session ID

        Returns:
            Number of chunks indexed for this session

        Usage: Populate total_chunks_searched field in SearchResponse
        """
        # Use session-based collection naming
        collection_name = f"session_{session_id}"

        try:
            collection = self.client.get_collection(collection_name)

            # Count all chunks in session collection
            return collection.count()
        except (ValueError, NotFoundError, Exception):
            # Collection doesn't exist or other ChromaDB error
            return 0
