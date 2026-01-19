"""Research session database model."""

from datetime import datetime
from typing import TYPE_CHECKING, Literal

from sqlalchemy import DateTime, Enum, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from article_mind_service.database import Base

if TYPE_CHECKING:
    from .article import Article

# Session status as a Python type
SessionStatus = Literal["draft", "active", "completed", "archived"]


class ResearchSession(Base):
    """Research session model for organizing article collections.

    Design Decisions:

    1. Status Enum: Using PostgreSQL ENUM type for data integrity.
       - Ensures only valid statuses can be stored
       - Better performance than string comparison
       - Clear documentation of allowed values

    2. Soft Delete: Using deleted_at timestamp instead of hard delete.
       - Allows recovery of accidentally deleted sessions
       - Maintains referential integrity with articles
       - Enables audit trail

    3. Timestamps: Using server_default=func.now() for database-level defaults.
       - Ensures consistent timestamps across application instances
       - Works correctly even if app server clock is wrong

    4. Article Count: Not stored, computed via relationship.
       - Avoids denormalization issues
       - Always accurate (no sync problems)
       - Performance acceptable for expected data volumes

    Status Lifecycle:
        draft -> active -> completed -> archived
                  |          |
                  v          v
               archived   archived
    """

    __tablename__ = "research_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Session display name",
    )
    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Optional session description",
    )
    status: Mapped[str] = mapped_column(
        Enum(
            "draft",
            "active",
            "completed",
            "archived",
            name="session_status",
            create_constraint=True,
        ),
        nullable=False,
        default="draft",
        server_default="draft",
        index=True,
        comment="Session lifecycle status",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="When session was created",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="When session was last updated",
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
        index=True,
        comment="Soft delete timestamp (null = not deleted)",
    )

    # Relationships
    articles: Mapped[list["Article"]] = relationship(
        "Article",
        back_populates="session",
        lazy="selectin",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"<ResearchSession(id={self.id}, name='{self.name}', status='{self.status}')>"

    @property
    def is_deleted(self) -> bool:
        """Check if session is soft-deleted."""
        return self.deleted_at is not None

    def can_transition_to(self, new_status: str) -> bool:
        """Check if status transition is valid.

        Valid transitions:
        - draft -> active, archived
        - active -> completed, archived
        - completed -> archived
        - archived -> (none)

        Args:
            new_status: The target status to transition to

        Returns:
            True if transition is valid, False otherwise
        """
        valid_transitions: dict[str, set[str]] = {
            "draft": {"active", "archived"},
            "active": {"completed", "archived"},
            "completed": {"archived"},
            "archived": set(),  # No transitions from archived
        }

        return new_status in valid_transitions.get(self.status, set())
