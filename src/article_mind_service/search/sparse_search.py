"""BM25 sparse keyword search implementation.

This module provides BM25-based sparse keyword search for exact matches,
technical terms, and keyword-heavy queries. BM25 complements dense semantic
search by excelling at:
- Exact keyword matches
- Technical terms and codes (e.g., JWT, UUID)
- Entity names and identifiers
- Phrases with specific terminology

The BM25Index maintains tokenized documents in memory for fast retrieval.
"""

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from rank_bm25 import BM25Okapi  # type: ignore
else:
    from rank_bm25 import BM25Okapi

TokenizedDoc = list[str]


@dataclass
class BM25Index:
    """BM25 index for a session's article chunks.

    Maintains tokenized documents and BM25 scoring model.
    Rebuilt when articles are added/removed from session.

    Design Decision: In-memory storage
    - BM25 indexes are lightweight (tokenized docs only)
    - Fast retrieval (<50ms for 1000 chunks)
    - Trade-off: Lost on restart, but easily rebuilt from DB

    Tokenization Strategy: Simple lowercase + alphanumeric split
    - Preserves technical terms (e.g., "API", "JWT")
    - Filters very short tokens (<2 chars)
    - No stemming (preserves exact matches)
    """

    session_id: int
    chunk_ids: list[str] = field(default_factory=list)
    chunk_contents: list[str] = field(default_factory=list)
    tokenized_docs: list[TokenizedDoc] = field(default_factory=list)
    _bm25: BM25Okapi | None = field(default=None, repr=False)

    def add_document(self, chunk_id: str, content: str) -> None:
        """Add a document chunk to the index.

        Args:
            chunk_id: Unique chunk identifier
            content: Text content to index
        """
        tokens = self._tokenize(content)
        self.chunk_ids.append(chunk_id)
        self.chunk_contents.append(content)
        self.tokenized_docs.append(tokens)
        self._invalidate_index()

    def remove_document(self, chunk_id: str) -> bool:
        """Remove a document chunk from the index.

        Args:
            chunk_id: Chunk identifier to remove

        Returns:
            True if document was found and removed, False otherwise
        """
        try:
            idx = self.chunk_ids.index(chunk_id)
            self.chunk_ids.pop(idx)
            self.chunk_contents.pop(idx)
            self.tokenized_docs.pop(idx)
            self._invalidate_index()
            return True
        except ValueError:
            return False

    def build(self) -> None:
        """Build or rebuild the BM25 index.

        Lazy build strategy: Index is built on first search call.
        Subsequent searches reuse cached index until invalidated.
        """
        if self.tokenized_docs:
            self._bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search the index and return ranked chunk IDs with scores.

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            List of (chunk_id, bm25_score) tuples, sorted by score descending

        Performance: O(n) where n is number of documents, typically <50ms for 1000 docs
        """
        if self._bm25 is None:
            self.build()

        if self._bm25 is None or not self.chunk_ids:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        # Get top-k indices sorted by score
        indexed_scores = [(i, score) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = [
            (self.chunk_ids[i], float(score))
            for i, score in indexed_scores[:top_k]
            if score > 0  # Only include non-zero scores
        ]

        return results

    def get_content(self, chunk_id: str) -> str | None:
        """Get content for a chunk by ID.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Chunk content if found, None otherwise
        """
        try:
            idx = self.chunk_ids.index(chunk_id)
            return self.chunk_contents[idx]
        except ValueError:
            return None

    def _tokenize(self, text: str) -> TokenizedDoc:
        """Tokenize text for BM25 indexing.

        Simple tokenization strategy:
        - Lowercase normalization
        - Split on non-alphanumeric characters
        - Filter tokens <2 characters
        - No stemming (preserves exact technical terms)

        Args:
            text: Text to tokenize

        Returns:
            List of tokens

        Design Decision: Why simple tokenization?
        - Preserves technical terms (API, JWT, UUID)
        - Fast (no complex NLP pipeline)
        - No false matches from aggressive stemming
        - Trade-off: Misses some paraphrases (handled by dense search)
        """
        text = text.lower()
        tokens = re.split(r"[^a-z0-9]+", text)
        return [t for t in tokens if len(t) > 1]

    def _invalidate_index(self) -> None:
        """Invalidate cached BM25 index (lazy rebuild on next search)."""
        self._bm25 = None

    def __len__(self) -> int:
        """Return number of indexed chunks."""
        return len(self.chunk_ids)


class BM25IndexCache:
    """Session-scoped BM25 index cache.

    Maintains one BM25 index per active session in memory.
    Indexes are built lazily on first search.

    Design Decision: Global class-level cache
    - Simple: No external cache service required
    - Fast: Sub-millisecond index lookups
    - Trade-off: Memory usage grows with active sessions
    - Future: Add LRU eviction for production with many sessions

    Thread Safety: Not thread-safe. If needed in production with
    multiple workers, consider using Redis or process-local caching.
    """

    _indexes: ClassVar[dict[int, BM25Index]] = {}

    @classmethod
    def get(cls, session_id: int) -> BM25Index | None:
        """Get existing index for session.

        Args:
            session_id: Session ID

        Returns:
            BM25Index if exists, None otherwise
        """
        return cls._indexes.get(session_id)

    @classmethod
    def get_or_create(cls, session_id: int) -> BM25Index:
        """Get existing index or create empty one for session.

        Args:
            session_id: Session ID

        Returns:
            BM25Index (existing or newly created)
        """
        if session_id not in cls._indexes:
            cls._indexes[session_id] = BM25Index(session_id=session_id)
        return cls._indexes[session_id]

    @classmethod
    def set(cls, session_id: int, index: BM25Index) -> None:
        """Store index for session.

        Args:
            session_id: Session ID
            index: BM25Index to cache
        """
        cls._indexes[session_id] = index

    @classmethod
    def invalidate(cls, session_id: int) -> None:
        """Remove cached index for session (force rebuild).

        Args:
            session_id: Session ID to invalidate
        """
        cls._indexes.pop(session_id, None)

    @classmethod
    def populate_from_chunks(
        cls,
        session_id: int,
        chunks: list[tuple[str, str]],  # [(chunk_id, content), ...]
    ) -> BM25Index:
        """Build index from list of chunks.

        Args:
            session_id: Session ID
            chunks: List of (chunk_id, content) tuples

        Returns:
            Built and cached BM25Index
        """
        index = BM25Index(session_id=session_id)
        for chunk_id, content in chunks:
            index.add_document(chunk_id, content)
        index.build()
        cls._indexes[session_id] = index
        return index


class SparseSearch:
    """BM25-based sparse keyword search.

    Manages BM25 indexes per session and provides search interface.

    Usage:
        sparse = SparseSearch()
        results = sparse.search(session_id=1, query="JWT authentication", top_k=10)
    """

    def __init__(self) -> None:
        """Initialize sparse search."""
        self.cache = BM25IndexCache

    def search(
        self,
        session_id: int,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Search session's BM25 index.

        Args:
            session_id: Session ID to search
            query: Search query string
            top_k: Number of results to return

        Returns:
            List of (chunk_id, score) tuples sorted by score descending
        """
        index = self.cache.get(session_id)
        if index is None:
            return []

        return index.search(query, top_k)

    def get_index(self, session_id: int) -> BM25Index | None:
        """Get BM25 index for session.

        Args:
            session_id: Session ID

        Returns:
            BM25Index if exists, None otherwise
        """
        return self.cache.get(session_id)

    def ensure_index(
        self,
        session_id: int,
        chunks: list[tuple[str, str]],
    ) -> BM25Index:
        """Ensure BM25 index exists for session, building if necessary.

        Args:
            session_id: Session ID
            chunks: List of (chunk_id, content) tuples

        Returns:
            Built BM25Index
        """
        return self.cache.populate_from_chunks(session_id, chunks)
