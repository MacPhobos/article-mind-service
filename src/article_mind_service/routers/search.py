"""Search API endpoints.

Provides natural language search over session knowledge using hybrid
retrieval (dense vector search + sparse BM25 search).

Endpoints:
- POST /api/v1/sessions/{session_id}/search: Search session knowledge
- GET /api/v1/sessions/{session_id}/search/stats: Get index statistics
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from article_mind_service.database import get_db
from article_mind_service.embeddings import get_embedding_provider
from article_mind_service.logging_config import get_logger
from article_mind_service.schemas.search import (
    SearchMode,
    SearchRequest,
    SearchResponse,
)
from article_mind_service.search import BM25IndexCache, HybridSearch

router = APIRouter(prefix="/api/v1", tags=["search"])
logger = get_logger(__name__)


# Singleton search instance
_hybrid_search: HybridSearch | None = None


def get_hybrid_search() -> HybridSearch:
    """Get or create hybrid search instance.

    Design Decision: Singleton pattern
    - DenseSearch and SparseSearch are stateless
    - Safe to reuse across requests
    - Avoids recreating ChromaDB client connection
    """
    global _hybrid_search
    if _hybrid_search is None:
        _hybrid_search = HybridSearch()
    return _hybrid_search


@router.post(
    "/sessions/{session_id}/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search session knowledge",
    description="""
Search the session's indexed articles using natural language queries.

Supports three search modes:
- **hybrid** (default): Combines semantic and keyword search for best results
- **dense**: Semantic similarity search only (good for concept queries)
- **sparse**: BM25 keyword search only (good for exact matches)

Returns ranked results with source attribution for citations.

**Performance:** Typical latency <200ms for 1000 chunks.

**Requirements:**
- Session must have indexed articles (via R5 embedding pipeline)
- For hybrid/dense modes, query embedding is generated automatically
- For sparse mode only, works with BM25 index alone
    """,
)
async def search_session(
    session_id: int,
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
    search: HybridSearch = Depends(get_hybrid_search),
) -> SearchResponse:
    """Search session knowledge using hybrid retrieval.

    Args:
        session_id: Session ID to search within
        request: Search parameters
        db: Database session
        search: Hybrid search instance

    Returns:
        SearchResponse with ranked results and metadata

    Raises:
        HTTPException 404: If session not found
        HTTPException 501: If dense search requested but embedding service unavailable
        HTTPException 500: If search fails
    """
    # TODO: Verify session exists
    # from article_mind_service.models.session import Session
    # session = await db.get(Session, session_id)
    # if not session:
    #     raise HTTPException(
    #         status_code=status.HTTP_404_NOT_FOUND,
    #         detail=f"Session {session_id} not found",
    #     )

    # Check if BM25 index exists for session
    bm25_index = BM25IndexCache.get(session_id)
    if bm25_index is None or len(bm25_index) == 0:
        # No indexed content for this session
        logger.warning(
            "search.endpoint.no_index",
            session_id=session_id,
            query=request.query[:100],
        )
        return SearchResponse(
            query=request.query,
            results=[],
            total_chunks_searched=0,
            search_mode=request.search_mode,
            timing_ms=0,
        )

    try:
        # Get query embedding for dense search
        query_embedding: list[float] = []
        if request.search_mode in (SearchMode.DENSE, SearchMode.HYBRID):
            # Get embedding provider and generate query embedding
            # Pass db session to use database provider settings
            try:
                provider = await get_embedding_provider(db=db)
                embeddings = await provider.embed([request.query])
                query_embedding = embeddings[0]
            except Exception as e:
                # Fall back to sparse-only if embedding generation fails
                logger.warning(
                    "search.endpoint.embedding_failed",
                    session_id=session_id,
                    error=str(e),
                    fallback_mode="sparse" if request.search_mode == SearchMode.HYBRID else None,
                )
                if request.search_mode == SearchMode.DENSE:
                    raise HTTPException(
                        status_code=status.HTTP_501_NOT_IMPLEMENTED,
                        detail=f"Dense search unavailable: {str(e)}",
                    ) from e
                # For hybrid, fall back to sparse only
                request.search_mode = SearchMode.SPARSE

        # Execute search
        response = await search.search(
            session_id=session_id,
            request=request,
            query_embedding=query_embedding,
        )

        return response

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        ) from e


@router.get(
    "/sessions/{session_id}/search/stats",
    summary="Get search index statistics",
    description="Returns statistics about the session's search index.",
)
async def search_stats(
    session_id: int,
    db: AsyncSession = Depends(get_db),
    search: HybridSearch = Depends(get_hybrid_search),
) -> dict[str, str | int | list[str]]:
    """Get search index statistics for a session.

    Args:
        session_id: Session ID
        db: Database session
        search: Hybrid search instance

    Returns:
        Dictionary with index statistics

    Example Response:
        {
            "session_id": 1,
            "bm25_index_exists": true,
            "total_chunks": 547,
            "chromadb_chunks": 547,
            "search_modes_available": ["sparse", "dense", "hybrid"]
        }
    """
    bm25_index = BM25IndexCache.get(session_id)
    bm25_chunks = len(bm25_index) if bm25_index else 0

    # Check ChromaDB
    chromadb_chunks = search.dense.get_total_chunks(session_id)

    # Determine available search modes
    available_modes = []
    if bm25_chunks > 0:
        available_modes.append(SearchMode.SPARSE.value)
    if chromadb_chunks > 0:
        available_modes.append(SearchMode.DENSE.value)
    if bm25_chunks > 0 and chromadb_chunks > 0:
        available_modes.append(SearchMode.HYBRID.value)

    return {
        "session_id": session_id,
        "bm25_index_exists": bm25_index is not None,
        "total_chunks": bm25_chunks,
        "chromadb_chunks": chromadb_chunks,
        "search_modes_available": available_modes,
    }
