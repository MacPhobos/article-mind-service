"""Pydantic schemas for chat API endpoints."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ChatSource(BaseModel):
    """Source citation in chat response.

    Represents a reference to an article chunk that was used
    to generate part of the response.
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
    excerpt: str | None = Field(
        default=None,
        description="Brief excerpt from the cited content",
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
