"""Provider settings database model."""

from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from article_mind_service.database import Base


class ProviderSettings(Base):
    """Provider settings model for storing runtime provider selection.

    Design Decisions:

    1. Singleton Pattern: Only one row allowed (id = 1)
       - Rationale: Provider settings apply globally to the service
       - Implementation: CheckConstraint ensures id = 1
       - Trade-offs: Simpler than configuration service, but less flexible
         for multi-tenant scenarios

    2. API Keys Remain in .env: Only provider selection stored in DB
       - Rationale: API keys are secrets and should not be in the database
       - Security: Keys stay in environment variables (.env file)
       - Database stores only: which provider to use (openai/anthropic/ollama)

    3. Automatic Timestamps: updated_at tracks last provider change
       - Useful for auditing when providers were switched
       - Server-side default ensures timestamp always set

    4. No Soft Delete: Settings row should never be deleted
       - Singleton constraint prevents deletion anyway
       - If needed, just update the values

    Usage:
        # Get or create settings (singleton)
        stmt = select(ProviderSettings).where(ProviderSettings.id == 1)
        result = await db.execute(stmt)
        settings = result.scalar_one_or_none()

        if not settings:
            settings = ProviderSettings(
                id=1,
                embedding_provider="openai",
                llm_provider="openai"
            )
            db.add(settings)
            await db.commit()
    """

    __tablename__ = "provider_settings"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        default=1,
        comment="Primary key (always 1 - singleton pattern)",
    )
    embedding_provider: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="openai",
        server_default="openai",
        comment="Current embedding provider (openai, ollama)",
    )
    llm_provider: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="openai",
        server_default="openai",
        comment="Current LLM provider (openai, anthropic)",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="Last updated timestamp",
    )

    # Singleton constraint - only one row allowed with id = 1
    __table_args__ = (CheckConstraint("id = 1", name="settings_singleton"),)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"<ProviderSettings(id={self.id}, "
            f"embedding='{self.embedding_provider}', "
            f"llm='{self.llm_provider}')>"
        )
