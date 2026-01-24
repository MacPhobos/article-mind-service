"""Search request and response schemas.

These schemas define the API contract for the knowledge query endpoint.
They support three search modes: hybrid (dense + sparse), dense only,
and sparse only.

Research Foundation:
- Top-K retrieval: 10-20 chunks optimal for synthesis
- Hybrid search: Best quality (combines semantic + keyword)
- Source attribution: Essential for citation support
- Metadata filtering: Enables filtering by article, content type, etc.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SearchMode(str, Enum):
    """Search mode selection.

    Values:
        DENSE: Vector search only (semantic similarity)
        SPARSE: BM25 search only (keyword matching)
        HYBRID: Combined search with RRF fusion (best quality)
    """

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class SearchRequest(BaseModel):
    """Search request parameters.

    Validates and constrains search parameters according to best practices
    from research:
    - Query length: 1-1000 characters (typical queries < 200 chars)
    - Top-K: 1-50 results (10-20 optimal for synthesis)
    - Search mode: Hybrid recommended for best quality

    Examples:
        >>> request = SearchRequest(
        ...     query="How does JWT authentication work?",
        ...     top_k=10,
        ...     search_mode=SearchMode.HYBRID
        ... )
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural language search query",
        examples=["How does authentication work?"],
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of results to return (1-50)",
    )
    include_content: bool = Field(
        default=True,
        description="Include chunk content in results",
    )
    search_mode: SearchMode = Field(
        default=SearchMode.HYBRID,
        description="Search mode: dense, sparse, or hybrid",
    )
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata filters (e.g., article_id, has_code)",
        examples=[{"article_id": 42}, {"has_code": True}],
    )


class SearchResult(BaseModel):
    """Individual search result with source attribution.

    Includes provenance information for citation support.
    Rank information from both dense and sparse search allows
    understanding which retrieval method found this result.

    Design Decision: Include ranks for transparency
    - dense_rank: Position in semantic search results
    - sparse_rank: Position in keyword search results
    - score: Combined RRF score (higher = more relevant)
    """

    chunk_id: str = Field(
        ...,
        description="Unique chunk identifier (doc_id:chunk_index)",
        examples=["doc_abc123:chunk_5"],
    )
    article_id: int = Field(
        ...,
        description="Source article ID",
    )
    content: str | None = Field(
        default=None,
        description="Chunk text content (if include_content=true)",
    )
    score: float = Field(
        ...,
        ge=0.0,
        description="Combined relevance score (RRF)",
    )
    source_url: str | None = Field(
        default=None,
        description="Original article URL",
    )
    source_title: str | None = Field(
        default=None,
        description="Article title",
    )
    dense_rank: int | None = Field(
        default=None,
        description="Rank from dense search (if applicable)",
    )
    sparse_rank: int | None = Field(
        default=None,
        description="Rank from sparse search (if applicable)",
    )


class SearchResponse(BaseModel):
    """Search response with results and metadata.

    Follows API contract specification for knowledge query endpoint.

    Performance Metrics:
    - timing_ms: Total search execution time
    - total_chunks_searched: Number of chunks in session index

    Design Decision: Include timing for monitoring
    - Helps track search performance over time
    - Identifies slow queries for optimization
    - Typical latency: <200ms for hybrid search
    """

    query: str = Field(
        ...,
        description="Original search query",
    )
    results: list[SearchResult] = Field(
        default_factory=list,
        description="Ranked search results",
    )
    total_chunks_searched: int = Field(
        ...,
        ge=0,
        description="Total chunks in session index",
    )
    search_mode: SearchMode = Field(
        ...,
        description="Search mode used",
    )
    timing_ms: int = Field(
        ...,
        ge=0,
        description="Search execution time in milliseconds",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "How does authentication work?",
                    "results": [
                        {
                            "chunk_id": "doc_abc123:chunk_5",
                            "article_id": 42,
                            "content": "Authentication uses JWT tokens...",
                            "score": 0.0156,
                            "source_url": "https://docs.example.com/auth",
                            "source_title": "Authentication Guide",
                            "dense_rank": 2,
                            "sparse_rank": 1,
                        }
                    ],
                    "total_chunks_searched": 1547,
                    "search_mode": "hybrid",
                    "timing_ms": 127,
                }
            ]
        }
    }
