"""Pydantic schemas for chat API endpoints."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ChatSource(BaseModel):
    """Source citation in chat response with full traceability.

    Represents a reference to an article chunk that was used
    to generate part of the response.

    Design Decision: Full content vs. excerpt
    - Previously returned 200-char excerpts (insufficient for verification)
    - Now returns full chunk content to enable answer grounding verification
    - Trade-off: Larger response size vs. user transparency
    """

    citation_index: int = Field(
        ...,
        description="Citation number [1], [2], etc.",
        examples=[1, 2],
    )
    article_id: int = Field(
        ...,
        description="ID of the source article",
    )
    chunk_id: str | None = Field(
        default=None,
        description="ID of the specific chunk",
    )
    title: str | None = Field(
        default=None,
        description="Article title",
        examples=["Introduction to RAG Systems"],
    )
    url: str | None = Field(
        default=None,
        description="Article URL",
        examples=["https://example.com/article"],
    )
    content: str | None = Field(
        default=None,
        description="Full chunk content used for this citation (not truncated)",
        examples=[
            "JWT authentication uses tokens for stateless authorization. "
            "The token contains encoded user claims and a signature that "
            "verifies authenticity without server-side session storage."
        ],
    )

    # Search metadata for transparency
    relevance_score: float | None = Field(
        default=None,
        description="Search relevance score (RRF combined score)",
        ge=0.0,
        examples=[0.0156],
    )
    search_method: str | None = Field(
        default=None,
        description="Search method used: semantic, keyword, or hybrid",
        examples=["hybrid"],
    )
    dense_rank: int | None = Field(
        default=None,
        description="Rank from semantic/dense vector search",
        ge=1,
        examples=[2],
    )
    sparse_rank: int | None = Field(
        default=None,
        description="Rank from keyword/sparse BM25 search",
        ge=1,
        examples=[1],
    )


class RetrievalMetadata(BaseModel):
    """Metadata about the retrieval process for transparency.

    Provides visibility into the search and retrieval process to help users
    understand how the answer was generated and why certain sources were selected.

    Design Decision: Transparency over simplicity
    - Exposes search internals to users for better trust and debugging
    - Allows frontend to show "N chunks retrieved, M cited" summary
    - Enables understanding of search timing and performance
    - Trade-off: Slightly larger response payload vs. user insight
    """

    chunks_retrieved: int = Field(
        description="Total number of chunks retrieved from search",
        examples=[5, 10],
    )
    chunks_cited: int = Field(
        description="Number of chunks actually cited in the response",
        examples=[2, 3],
    )
    search_mode: str = Field(
        description="Search mode used: hybrid, semantic, or keyword",
        examples=["hybrid"],
    )
    search_timing_ms: float | None = Field(
        default=None,
        description="Time taken for search in milliseconds",
        examples=[150.5],
    )
    total_chunks_in_session: int | None = Field(
        default=None,
        description="Total indexed chunks available in this session",
        examples=[50, 100],
    )


class ChatRequest(BaseModel):
    """Request body for sending a chat message."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="User's question or message",
        examples=["What are the key points about embeddings?"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "What are the main topics covered in my saved articles?"
                },
                {"message": "Summarize what I've read about vector databases."},
            ]
        }
    }


class ChatMessageResponse(BaseModel):
    """Single chat message in response."""

    id: int = Field(..., description="Message ID")
    role: Literal["user", "assistant"] = Field(
        ...,
        description="Message role",
    )
    content: str = Field(..., description="Message content")
    sources: list[ChatSource] | None = Field(
        default=None,
        description="Source citations (assistant messages only)",
    )
    created_at: datetime = Field(..., description="When the message was created")

    model_config = {"from_attributes": True}


class ChatResponse(BaseModel):
    """Response from sending a chat message.

    Contains the assistant's response with sources and usage metadata.
    """

    message_id: int = Field(..., description="ID of the assistant's response message")
    content: str = Field(..., description="Assistant's response text")
    sources: list[ChatSource] = Field(
        default_factory=list,
        description="Sources cited in the response",
    )
    llm_provider: str | None = Field(
        default=None,
        description="LLM provider used",
        examples=["openai", "anthropic"],
    )
    llm_model: str | None = Field(
        default=None,
        description="Specific model used",
        examples=["gpt-4o-mini", "claude-sonnet-4-5-20241022"],
    )
    tokens_used: int | None = Field(
        default=None,
        description="Total tokens consumed",
    )
    created_at: datetime = Field(..., description="When the response was generated")
    retrieval_metadata: RetrievalMetadata | None = Field(
        default=None,
        description="Search and retrieval context for this response (P2 enhancement)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message_id": 42,
                    "content": "Based on your saved articles, embeddings are... [1]",
                    "sources": [
                        {
                            "citation_index": 1,
                            "article_id": 123,
                            "title": "Understanding Embeddings",
                            "url": "https://example.com/embeddings",
                        }
                    ],
                    "llm_provider": "openai",
                    "llm_model": "gpt-4o-mini",
                    "tokens_used": 1250,
                    "created_at": "2026-01-19T12:00:00Z",
                    "retrieval_metadata": {
                        "chunks_retrieved": 5,
                        "chunks_cited": 2,
                        "search_mode": "hybrid",
                        "search_timing_ms": 150.5,
                    },
                }
            ]
        }
    }


class ChatHistoryResponse(BaseModel):
    """Response containing chat history for a session."""

    session_id: int = Field(..., description="Session ID")
    messages: list[ChatMessageResponse] = Field(
        default_factory=list,
        description="List of chat messages in chronological order",
    )
    total_messages: int = Field(..., description="Total number of messages")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": 1,
                    "messages": [
                        {
                            "id": 1,
                            "role": "user",
                            "content": "What are embeddings?",
                            "sources": None,
                            "created_at": "2026-01-19T12:00:00Z",
                        },
                        {
                            "id": 2,
                            "role": "assistant",
                            "content": "Embeddings are vector representations... [1]",
                            "sources": [
                                {
                                    "citation_index": 1,
                                    "article_id": 123,
                                    "title": "...",
                                }
                            ],
                            "created_at": "2026-01-19T12:00:01Z",
                        },
                    ],
                    "total_messages": 2,
                }
            ]
        }
    }
