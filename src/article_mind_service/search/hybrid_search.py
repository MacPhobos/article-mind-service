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

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from article_mind_service.config import settings
from article_mind_service.logging_config import get_logger
from article_mind_service.models.article import Article
from article_mind_service.schemas.search import (
    SearchMode,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

from .dense_search import DenseSearch
from .heuristic_reranker import heuristic_rerank
from .reranker import Reranker
from .sparse_search import SparseSearch

logger = get_logger(__name__)

# Technical terms that benefit from lower similarity thresholds
# These terms are precise and should match even with lower similarity
_TECHNICAL_TERMS = frozenset([
    "api", "oauth", "jwt", "ssl", "tls", "http", "tcp", "dns",
    "sql", "nosql", "graphql", "rest", "grpc", "websocket",
    "ml", "ai", "nlp", "llm", "rag", "embedding",
    "kubernetes", "docker", "aws", "gcp", "azure",
])


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

    async def _get_article_metadata(
        self,
        article_ids: list[int],
        db: AsyncSession | None = None,
    ) -> dict[int, dict]:
        """Fetch article metadata for heuristic reranking.

        Args:
            article_ids: List of article IDs to fetch metadata for
            db: Optional database session (if None, skips metadata fetching)

        Returns:
            Dict mapping article_id to metadata dict with keys:
            - title: Article title (str | None)
            - published_date: Publication date (datetime | None)
            - source_url: Original or canonical URL (str | None)

        Design Decision: Optional database dependency
        - Heuristic reranking works without metadata (degraded mode)
        - If db session not provided, returns empty dict
        - Search router can optionally pass db session for enhanced ranking
        - Rationale: Keep search module loosely coupled to database

        Performance:
        - Single batch query for all article IDs
        - Time: 5-20ms for 10-50 articles
        - No N+1 query problem
        """
        if db is None or not article_ids:
            return {}

        try:
            # Batch query for all article metadata
            query = select(
                Article.id,
                Article.title,
                Article.published_date,
                Article.original_url,
                Article.canonical_url,
            ).where(Article.id.in_(article_ids))

            result = await db.execute(query)
            rows = result.all()

            # Build metadata dict
            metadata_dict: dict[int, dict] = {}
            for row in rows:
                article_id, title, published_date, original_url, canonical_url = row
                metadata_dict[article_id] = {
                    "title": title,
                    "published_date": published_date,
                    "source_url": canonical_url or original_url,
                }

            return metadata_dict

        except Exception as e:
            # Log error but don't fail search
            logger.warning(
                "search.hybrid.metadata_fetch_failed",
                error=str(e),
                article_count=len(article_ids),
            )
            return {}

    def _get_adaptive_threshold(self, query: str, base: float = 0.3) -> float:
        """Calculate adaptive similarity threshold based on query characteristics.

        Args:
            query: Search query string
            base: Base threshold (default: 0.3)

        Returns:
            Adjusted similarity threshold (0.05-0.8 range)

        Design Decision: Query-aware thresholds
        - Single word: Lower threshold (~0.05) to capture sparse mentions
        - Technical terms: Lower threshold (~0.10) for precise technical matches
        - Quoted phrases: Higher threshold (~0.50) for exact semantic matches
        - Short queries (2-3 words): Slightly reduced (~0.20)
        - Long queries (>8 words): Elevated (~0.45) to filter noise
        - Normal queries: Base threshold (0.30)

        Rationale:
        - Fixed thresholds cause short queries to return nothing
        - Long queries return noise with fixed thresholds
        - Technical terms are precise and benefit from lower thresholds
        - Quoted phrases indicate user wants exact matches

        Examples:
            >>> _get_adaptive_threshold("JWT")  # Single technical term
            0.05
            >>> _get_adaptive_threshold("authentication flow")  # Short query
            0.20
            >>> _get_adaptive_threshold('"exact phrase match"')  # Quoted
            0.50
            >>> _get_adaptive_threshold("how does authentication work in modern web apps")  # Long
            0.45
        """
        words = query.strip().split()
        query_lower = query.lower()

        # Empty query: Use base threshold
        if len(words) == 0:
            return base

        # Single word queries: Very permissive threshold
        if len(words) == 1:
            return max(0.05, base - 0.25)

        # Technical term queries: Lower threshold for precise terms
        if any(term in query_lower for term in _TECHNICAL_TERMS):
            return max(0.05, base - 0.20)

        # Quoted phrase queries: Higher threshold for exact matches
        if '"' in query:
            return min(0.8, base + 0.2)

        # Short queries (2-3 words): Slightly reduced threshold
        if len(words) <= 3:
            return max(0.10, base - 0.10)

        # Long queries (>8 words): Elevated threshold to reduce noise
        if len(words) > 8:
            return min(0.7, base + 0.15)

        # Normal queries: Use base threshold
        return base

    async def search(
        self,
        session_id: int,
        request: SearchRequest,
        query_embedding: list[float],
        db: AsyncSession | None = None,
    ) -> SearchResponse:
        """Execute hybrid search.

        Args:
            session_id: Session to search
            request: Search request parameters
            query_embedding: Pre-computed query embedding
            db: Optional database session for article metadata fetching
                (enables heuristic reranking with title/date signals)

        Returns:
            SearchResponse with ranked results and metadata

        Performance:
        - Dense search: 50-100ms (ChromaDB HNSW)
        - Sparse search: 20-50ms (BM25)
        - RRF fusion: <5ms
        - Heuristic reranking: <5ms (metadata fetch: 5-20ms if db provided)
        - Cross-encoder reranking: 200-500ms (if enabled)
        - Total: <200ms without cross-encoder, <700ms with cross-encoder

        Error Handling:
        - Returns empty results if no index exists
        - Gracefully degrades if one search method fails
        - Heuristic reranking works without db session (reduced signals)
        """
        start_time = time.time()

        # Compute adaptive similarity threshold if not explicitly provided
        threshold = request.similarity_threshold
        if threshold is None:
            threshold = self._get_adaptive_threshold(request.query)

        logger.info(
            "search.hybrid.start",
            session_id=session_id,
            query=request.query[:100],
            top_k=request.top_k,
            search_mode=request.search_mode.value,
            include_content=request.include_content,
            similarity_threshold=threshold,
        )

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
                    filters=request.filters,  # Pass filters to dense search
                    similarity_threshold=threshold,  # Pass adaptive threshold
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

        # Apply heuristic reranking (lightweight signal-based boosting)
        if ranked:
            # Get BM25 index for content lookup
            bm25_index = self.sparse.get_index(session_id)

            # Convert RankedResult to dict format for heuristic_rerank
            results_dicts = []
            for r in ranked:
                content = None
                if bm25_index:
                    content = bm25_index.get_content(r.chunk_id)

                # Parse metadata from chunk_id if not available
                # Format: article_{article_id}_chunk_{index}
                metadata = {}
                if "_chunk_" in r.chunk_id:
                    try:
                        parts = r.chunk_id.split("_")
                        article_id = int(parts[1])
                        chunk_index = int(parts[3])
                        metadata = {
                            "article_id": article_id,
                            "chunk_index": chunk_index,
                        }
                    except (IndexError, ValueError):
                        pass

                results_dicts.append({
                    "chunk_id": r.chunk_id,
                    "rrf_score": r.rrf_score,
                    "content": content or "",
                    "metadata": metadata,
                })

            # Extract unique article IDs for metadata fetching
            article_ids = list({
                r["metadata"].get("article_id")
                for r in results_dicts
                if r["metadata"].get("article_id") is not None
            })

            # Fetch article metadata for heuristic scoring
            article_metadata = await self._get_article_metadata(article_ids, db)

            # Apply heuristic reranking
            results_dicts = heuristic_rerank(
                results=results_dicts,
                query=request.query,
                article_metadata=article_metadata,
            )

            # Update RankedResult objects with heuristic scores
            for i, r in enumerate(ranked):
                # Find corresponding dict result
                matching = next(
                    (d for d in results_dicts if d["chunk_id"] == r.chunk_id),
                    None
                )
                if matching:
                    r.rrf_score = matching["heuristic_score"]
                    r.metadata = matching.get("metadata")
                    r.content = matching.get("content")

            # Re-sort by heuristic score
            ranked.sort(key=lambda r: r.rrf_score, reverse=True)

            logger.info(
                "search.hybrid.heuristic_reranking_complete",
                session_id=session_id,
                candidates_reranked=len(ranked),
                metadata_articles=len(article_metadata),
            )

        # Optional reranking: Re-score and re-sort using cross-encoder
        if self.rerank_enabled and self.reranker and ranked:
            # Initialize reranker if not already done
            if self.reranker is None:
                self.reranker = Reranker()

            # Extract content for reranking (need to fetch from BM25 index)
            bm25_index = self.sparse.get_index(session_id)
            documents = []
            for r in ranked:
                content = None
                if bm25_index:
                    content = bm25_index.get_content(r.chunk_id)
                documents.append(content or "")

            # Get reranker scores
            rerank_scores = await self.reranker.rerank(
                query=request.query,
                documents=documents,
            )

            # Normalize scores to [0, 1] range
            # Cross-encoder scores are logits (can be negative)
            # We need non-negative scores for the API schema
            if rerank_scores:
                min_score = min(rerank_scores)
                max_score = max(rerank_scores)
                score_range = max_score - min_score

                if score_range > 0:
                    # Min-max normalization
                    normalized_scores = [
                        (score - min_score) / score_range
                        for score in rerank_scores
                    ]
                else:
                    # All scores the same - use uniform distribution
                    normalized_scores = [0.5] * len(rerank_scores)

                # Update RRF scores with normalized reranker scores
                # Design Decision: Replace RRF score with reranker score
                # - Reranker provides more accurate relevance than RRF
                # - RRF is first-stage retrieval, reranker is second-stage refinement
                # - Scores normalized to [0, 1] for API contract compliance
                for i, r in enumerate(ranked):
                    r.rrf_score = normalized_scores[i]

            # Re-sort by reranker scores
            ranked.sort(key=lambda r: r.rrf_score, reverse=True)

            logger.info(
                "search.hybrid.reranking_complete",
                session_id=session_id,
                candidates_reranked=len(ranked),
            )

        # Limit to requested top_k after reranking
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

        logger.info(
            "search.hybrid.complete",
            session_id=session_id,
            results_count=len(results),
            total_chunks_searched=total_chunks,
            timing_ms=timing_ms,
            dense_results=len(dense_results),
            sparse_results=len(sparse_results),
        )

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
