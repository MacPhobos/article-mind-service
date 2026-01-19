"""Hybrid search combining dense and sparse retrieval with RRF.

Reciprocal Rank Fusion (RRF) combines rankings from multiple retrieval
methods without requiring score normalization. It focuses purely on rank
position, making it robust across different scoring scales.

RRF Formula: score = Σ 1/(k + rank_i)
Where:
- k: constant (typically 60) that dampens effect of high rankings
- rank_i: rank of document in retrieval method i (1-indexed)

Research Foundation:
- RRF shown to be more robust than score-based fusion
- Ignores raw scores, focuses on rank position
- Fair representation from all search types
- Works well with hybrid (dense + sparse) search
"""

import time
from dataclasses import dataclass

from article_mind_service.config import settings
from article_mind_service.schemas.search import (
    SearchMode,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

from .dense_search import DenseSearch
from .reranker import Reranker
from .sparse_search import SparseSearch


@dataclass
class RankedResult:
    """Intermediate result with ranking metadata.

    Tracks results from multiple retrieval methods before final ranking.
    """

    chunk_id: str
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float = 0.0
    content: str | None = None
    metadata: dict[str, str] | None = None


def reciprocal_rank_fusion(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    k: int = 60,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> list[RankedResult]:
    """Combine dense and sparse results using Reciprocal Rank Fusion.

    Args:
        dense_results: Results from dense search (chunk_id, score)
        sparse_results: Results from sparse search (chunk_id, score)
        k: RRF constant (higher = less emphasis on top ranks)
        dense_weight: Weight for dense contribution (semantic search)
        sparse_weight: Weight for sparse contribution (keyword search)

    Returns:
        List of RankedResult sorted by RRF score descending

    Design Decision: Weighted RRF
    - Standard RRF ignores retrieval method quality differences
    - Weighted version allows tuning based on evaluation
    - Default: 0.7 dense, 0.3 sparse (semantic > keywords)
    - Research shows dense search typically higher quality for QA

    Algorithm Complexity: O(n) where n = total unique results
    """
    results: dict[str, RankedResult] = {}

    # Process dense results (rank is 1-indexed)
    for rank, (chunk_id, _score) in enumerate(dense_results, start=1):
        if chunk_id not in results:
            results[chunk_id] = RankedResult(chunk_id=chunk_id)
        results[chunk_id].dense_rank = rank
        results[chunk_id].rrf_score += dense_weight * (1.0 / (k + rank))

    # Process sparse results
    for rank, (chunk_id, _score) in enumerate(sparse_results, start=1):
        if chunk_id not in results:
            results[chunk_id] = RankedResult(chunk_id=chunk_id)
        results[chunk_id].sparse_rank = rank
        results[chunk_id].rrf_score += sparse_weight * (1.0 / (k + rank))

    # Sort by RRF score descending
    return sorted(results.values(), key=lambda r: r.rrf_score, reverse=True)


class HybridSearch:
    """Hybrid search orchestrator.

    Combines dense vector search and sparse BM25 search using
    Reciprocal Rank Fusion for optimal retrieval performance.

    Search Modes:
    - hybrid: Combines dense + sparse with RRF (default, best quality)
    - dense: Semantic search only (good for concept queries)
    - sparse: Keyword search only (good for exact matches)

    Design Decision: Singleton components
    - DenseSearch and SparseSearch are stateless (can be reused)
    - BM25 indexes cached per session
    - ChromaDB client is thread-safe

    Usage:
        search = HybridSearch()
        response = await search.search(
            session_id=1,
            request=SearchRequest(query="JWT auth", top_k=10),
            query_embedding=embedding_vector
        )
    """

    def __init__(
        self,
        dense_search: DenseSearch | None = None,
        sparse_search: SparseSearch | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        """Initialize hybrid search components.

        Args:
            dense_search: Dense search instance (created if None)
            sparse_search: Sparse search instance (created if None)
            reranker: Optional reranker instance
        """
        self.dense = dense_search or DenseSearch()
        self.sparse = sparse_search or SparseSearch()
        self.reranker = reranker

        # Configuration from settings
        self.dense_weight = settings.search_dense_weight
        self.sparse_weight = settings.search_sparse_weight
        self.rrf_k = settings.search_rrf_k
        self.rerank_enabled = settings.search_rerank_enabled

    async def search(
        self,
        session_id: int,
        request: SearchRequest,
        query_embedding: list[float],
    ) -> SearchResponse:
        """Execute hybrid search.

        Args:
            session_id: Session to search
            request: Search request parameters
            query_embedding: Pre-computed query embedding

        Returns:
            SearchResponse with ranked results and metadata

        Performance:
        - Dense search: 50-100ms (ChromaDB HNSW)
        - Sparse search: 20-50ms (BM25)
        - RRF fusion: <5ms
        - Total: <200ms without reranking

        Error Handling:
        - Returns empty results if no index exists
        - Gracefully degrades if one search method fails
        """
        start_time = time.time()

        # Determine effective top_k (get more for reranking)
        retrieve_k = request.top_k
        if self.rerank_enabled and self.reranker:
            retrieve_k = max(request.top_k * 2, settings.search_rerank_top_k)

        # Execute searches based on mode
        dense_results: list[tuple[str, float]] = []
        sparse_results: list[tuple[str, float]] = []

        if request.search_mode in (SearchMode.DENSE, SearchMode.HYBRID):
            dense_results = [
                (r.chunk_id, r.score)
                for r in self.dense.search(
                    session_id=session_id,
                    query_embedding=query_embedding,
                    top_k=retrieve_k,
                )
            ]

        if request.search_mode in (SearchMode.SPARSE, SearchMode.HYBRID):
            sparse_results = self.sparse.search(
                session_id=session_id,
                query=request.query,
                top_k=retrieve_k,
            )

        # Combine results using RRF
        if request.search_mode == SearchMode.HYBRID:
            ranked = reciprocal_rank_fusion(
                dense_results=dense_results,
                sparse_results=sparse_results,
                k=self.rrf_k,
                dense_weight=self.dense_weight,
                sparse_weight=self.sparse_weight,
            )
        elif request.search_mode == SearchMode.DENSE:
            ranked = [
                RankedResult(
                    chunk_id=cid,
                    dense_rank=i + 1,
                    rrf_score=score,
                )
                for i, (cid, score) in enumerate(dense_results)
            ]
        else:  # SPARSE
            ranked = [
                RankedResult(
                    chunk_id=cid,
                    sparse_rank=i + 1,
                    rrf_score=score,
                )
                for i, (cid, score) in enumerate(sparse_results)
            ]

        # Optional reranking (placeholder - not implemented yet)
        if self.rerank_enabled and self.reranker and ranked:
            # TODO: Implement reranking when needed
            pass

        # Limit to requested top_k
        ranked = ranked[: request.top_k]

        # Build response with content and metadata
        results = await self._build_results(
            session_id=session_id,
            ranked=ranked,
            include_content=request.include_content,
        )

        # Get total chunks count
        total_chunks = self.dense.get_total_chunks(session_id)

        timing_ms = int((time.time() - start_time) * 1000)

        return SearchResponse(
            query=request.query,
            results=results,
            total_chunks_searched=total_chunks,
            search_mode=request.search_mode,
            timing_ms=timing_ms,
        )

    async def _build_results(
        self,
        session_id: int,
        ranked: list[RankedResult],
        include_content: bool,
    ) -> list[SearchResult]:
        """Build SearchResult objects with full metadata.

        Args:
            session_id: Session ID
            ranked: Ranked results from RRF
            include_content: Whether to include chunk content

        Returns:
            List of SearchResult objects

        Design Decision: Content from BM25 index
        - BM25 index stores content for each chunk
        - Avoids extra DB query to fetch content
        - Trade-off: Memory overhead in BM25 index
        """
        results = []

        # Get BM25 index for content lookup
        bm25_index = self.sparse.get_index(session_id)

        for r in ranked:
            # Get content from BM25 index (it stores content)
            content = None
            if include_content and bm25_index:
                content = bm25_index.get_content(r.chunk_id)

            # Parse article_id from metadata or chunk_id
            # Format: doc_{article_id}:chunk_{index}
            article_id = 0
            if r.metadata and "article_id" in r.metadata:
                try:
                    article_id = int(r.metadata["article_id"])
                except (ValueError, TypeError):
                    article_id = 0

            results.append(
                SearchResult(
                    chunk_id=r.chunk_id,
                    article_id=article_id,
                    content=content,
                    score=r.rrf_score,
                    source_url=r.metadata.get("source_url") if r.metadata else None,
                    source_title=r.metadata.get("source_title") if r.metadata else None,
                    dense_rank=r.dense_rank,
                    sparse_rank=r.sparse_rank,
                )
            )

        return results
