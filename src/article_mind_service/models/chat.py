"""Chat message models for Q&A history."""

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from article_mind_service.database import Base

if TYPE_CHECKING:
    from article_mind_service.models.session import ResearchSession


class ChatMessage(Base):
    """Chat message in a session Q&A conversation.

    Each message represents either a user question or an assistant response.
    Assistant responses include source citations for grounding.

    Attributes:
        id: Auto-incrementing primary key
        session_id: Foreign key to the session this chat belongs to
        role: Message role ("user" or "assistant")
        content: Message text content
        sources: JSON array of source citations (assistant messages only)
        llm_provider: Which LLM provider generated this response
        llm_model: Specific model used (e.g., "gpt-4o-mini")
        tokens_used: Total tokens consumed for this message
        created_at: When the message was created
    """

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    session_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("research_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="user or assistant",
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Message text content",
    )
    sources: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        default=None,
        comment="Source citations for assistant messages",
    )
    llm_provider: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        comment="LLM provider used (openai, anthropic)",
    )
    llm_model: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="Specific model identifier",
    )
    tokens_used: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Total tokens consumed",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    session: Mapped["ResearchSession"] = relationship(back_populates="chat_messages")

    def __repr__(self) -> str:
        return f"<ChatMessage(id={self.id}, role={self.role}, session_id={self.session_id})>"
