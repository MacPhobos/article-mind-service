"""Article database model."""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from sqlalchemy import JSON, DateTime, Enum, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from article_mind_service.database import Base

if TYPE_CHECKING:
    from .session import ResearchSession

# Article type as a Python type
ArticleType = Literal["url", "file"]

# Extraction status as a Python type
ExtractionStatus = Literal["pending", "processing", "completed", "failed"]

# Embedding status as a Python type
EmbeddingStatus = Literal["pending", "processing", "completed", "failed"]


class Article(Base):
    """Article model for storing research articles within sessions.

    Design Decisions:

    1. Article Type: Using PostgreSQL ENUM for type safety.
       - Ensures only 'url' or 'file' can be stored
       - Better performance than string comparison
       - Clear documentation of allowed values

    2. Storage Strategy: Filesystem + Database Hybrid
       - Large files stored on filesystem (storage_path)
       - Metadata stored in database
       - Benefits: Better performance, easier backup, scalable
       - Trade-offs: Must manage filesystem sync

    3. Content Storage: Optional text field
       - Smaller extracted content stored directly in DB
       - Larger content could be streamed from filesystem
       - NULL if extraction not yet completed

    4. Soft Delete: Using deleted_at timestamp
       - Allows recovery of accidentally deleted articles
       - Maintains referential integrity
       - Cascade deletes when parent session is deleted

    5. Extraction Status: Tracks async content extraction
       - 'pending': Article added, extraction not started
       - 'processing': Extraction in progress
       - 'completed': Extraction successful
       - 'failed': Extraction failed (with error details)

    Directory Structure for Files:
        data/uploads/{session_id}/{article_id}/
        ├── original.pdf       # Original uploaded file
        └── extracted.txt      # Extracted text (future)
    """

    __tablename__ = "articles"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("research_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Parent research session",
    )

    # Article type and source
    type: Mapped[str] = mapped_column(
        Enum("url", "file", name="article_type", create_constraint=True),
        nullable=False,
        comment="Article source type (url or file)",
    )
    original_url: Mapped[str | None] = mapped_column(
        String(2048),
        nullable=True,
        comment="Source URL for url-type articles",
    )
    original_filename: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Original filename for file-type articles",
    )

    # Storage
    storage_path: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="Filesystem path for uploaded files (relative to UPLOAD_BASE_PATH)",
    )

    # Metadata
    title: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="Article title (extracted or provided)",
    )
    extraction_status: Mapped[str] = mapped_column(
        Enum(
            "pending",
            "processing",
            "completed",
            "failed",
            name="extraction_status",
            create_constraint=True,
        ),
        nullable=False,
        default="pending",
        server_default="pending",
        index=True,
        comment="Content extraction status",
    )

    # Extracted content (stored directly for smaller text content)
    content_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Extracted text content",
    )
    content_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        comment="SHA-256 hash of content for deduplication",
    )

    # Embedding status tracking
    embedding_status: Mapped[str] = mapped_column(
        Enum(
            "pending",
            "processing",
            "completed",
            "failed",
            name="embedding_status",
            create_constraint=True,
        ),
        nullable=False,
        default="pending",
        server_default="pending",
        index=True,
        comment="Embedding generation status",
    )
    chunk_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of chunks created during embedding",
    )

    # Extraction metadata
    extraction_method: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        comment="Method used for extraction (trafilatura, pymupdf, etc.)",
    )
    extraction_error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Error message if extraction failed",
    )
    canonical_url: Mapped[str | None] = mapped_column(
        String(2048),
        nullable=True,
        comment="Final URL after redirects",
    )

    # Content metadata
    author: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="Article author",
    )
    published_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Publication date (if available)",
    )
    language: Mapped[str | None] = mapped_column(
        String(10),
        nullable=True,
        comment="Detected language code (e.g., 'en', 'fr')",
    )
    word_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of words in content",
    )
    reading_time_minutes: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Estimated reading time in minutes",
    )

    # Extended metadata (JSON) - using extraction_metadata to avoid conflict with SQLAlchemy metadata
    extraction_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="Additional metadata from extraction (keywords, tags, etc.)",
    )

    # Extraction timestamps
    extracted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When extraction completed successfully",
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="When article was added",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="When article was last updated",
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
        index=True,
        comment="Soft delete timestamp (null = not deleted)",
    )

    # Relationships
    session: Mapped["ResearchSession"] = relationship(
        "ResearchSession",
        back_populates="articles",
    )

    # Composite indexes for common queries
    __table_args__ = (
        Index("ix_articles_session_status", "session_id", "extraction_status"),
        Index("ix_articles_session_deleted", "session_id", "deleted_at"),
    )

    def __repr__(self) -> str:
        """String representation for debugging."""
        source = self.original_url or self.original_filename or "unknown"
        return f"<Article(id={self.id}, type='{self.type}', source='{source[:50]}')>"

    @property
    def is_deleted(self) -> bool:
        """Check if article is soft-deleted."""
        return self.deleted_at is not None

    @property
    def display_name(self) -> str:
        """Get display name for the article."""
        if self.title:
            return self.title
        if self.type == "url" and self.original_url:
            return self.original_url[:100]
        if self.type == "file" and self.original_filename:
            return self.original_filename
        return f"Article #{self.id}"
