"""Settings service for provider configuration management.

Design Decisions:

1. Singleton Pattern: Only one settings row in database (id = 1)
   - Rationale: Provider settings are global to the service
   - get_or_create pattern ensures settings always exist
   - CheckConstraint in model enforces singleton

2. API Key Checking: Read from config.py Settings class
   - API keys stored in .env files (never in database)
   - Service checks if keys are configured before allowing provider switch
   - Returns "available" list to frontend for UI enablement

3. Dimension Change Detection: Compare old vs new embedding dimensions
   - OpenAI: 1536 dimensions (text-embedding-3-small)
   - Ollama: 1024 dimensions (nomic-embed-text)
   - Dimension change requires reindexing all documents
   - Returns boolean flag for dimension change

4. Transaction Safety: All operations use async/await with SQLAlchemy
   - Async session management for non-blocking I/O
   - Explicit commit not needed (handled by get_db dependency)
   - get_or_create pattern handles race conditions

Error Handling:
- Invalid providers caught by Pydantic schema validation (Literal types)
- Database errors propagate to router for HTTP error responses
- No try/except in service layer (let exceptions bubble up)
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from article_mind_service.config import settings as app_settings
from article_mind_service.models.provider_settings import ProviderSettings


async def get_settings(db: AsyncSession) -> ProviderSettings:
    """Get or create provider settings singleton.

    Implements get-or-create pattern for singleton settings row.

    Args:
        db: Async database session

    Returns:
        ProviderSettings instance (always id=1)

    Design Decision: Always create if missing
    - Rationale: Settings should always exist after first API call
    - Safe for concurrent access (id=1 constraint prevents duplicates)
    - Returns default values (openai, openai) if never configured
    """
    stmt = select(ProviderSettings).where(ProviderSettings.id == 1)
    result = await db.execute(stmt)
    settings = result.scalar_one_or_none()

    if not settings:
        settings = ProviderSettings(
            id=1,
            embedding_provider="openai",
            llm_provider="openai",
        )
        db.add(settings)
        await db.flush()  # Flush to get updated_at timestamp

    return settings


async def update_embedding_provider(
    db: AsyncSession,
    provider: str,
) -> tuple[ProviderSettings, bool]:
    """Update embedding provider and detect dimension changes.

    Args:
        db: Async database session
        provider: New embedding provider ("openai" or "ollama")

    Returns:
        Tuple of (updated_settings, dimension_changed)
        - updated_settings: ProviderSettings instance with new provider
        - dimension_changed: True if embedding dimensions changed

    Design Decision: Return dimension_changed flag
    - Rationale: Dimension changes require reindexing all documents
    - Caller (router) decides whether to trigger reindex based on confirm_reindex flag
    - Service layer doesn't know about reindex operations (separation of concerns)

    Dimension Mapping:
    - openai: 1536 dimensions (text-embedding-3-small)
    - ollama: 1024 dimensions (nomic-embed-text)

    Example:
        settings, changed = await update_embedding_provider(db, "ollama")
        if changed:
            # Trigger reindex operation
            await trigger_reindex(db, session_id)
    """
    settings = await get_settings(db)

    # Check if dimension changed
    old_dimensions = _get_embedding_dimensions(settings.embedding_provider)
    new_dimensions = _get_embedding_dimensions(provider)
    dimension_changed = old_dimensions != new_dimensions

    # Update provider
    settings.embedding_provider = provider
    await db.flush()  # Flush to update updated_at timestamp

    return settings, dimension_changed


async def update_llm_provider(
    db: AsyncSession,
    provider: str,
) -> ProviderSettings:
    """Update LLM provider.

    Args:
        db: Async database session
        provider: New LLM provider ("openai" or "anthropic")

    Returns:
        Updated ProviderSettings instance

    Design Decision: No dimension checking needed
    - Rationale: LLM provider only affects text generation, not embeddings
    - Switching LLM doesn't require reindexing (embeddings unchanged)
    - Simpler operation than embedding provider update

    Example:
        settings = await update_llm_provider(db, "anthropic")
    """
    settings = await get_settings(db)
    settings.llm_provider = provider
    await db.flush()  # Flush to update updated_at timestamp
    return settings


def get_available_providers() -> dict[str, list[str]]:
    """Get list of available providers based on configured API keys.

    Checks which providers have API keys configured in .env file.

    Returns:
        Dictionary with "embedding" and "llm" keys, each containing
        list of available provider names.

    Design Decision: Check app_settings for API keys
    - Rationale: API keys stored in .env (never in database)
    - Ollama always available (local, no API key needed)
    - OpenAI/Anthropic available only if API keys configured

    Example Response:
        {
            "embedding": ["openai", "ollama"],  # ollama always available
            "llm": ["openai"]  # anthropic missing (no API key)
        }

    Usage in Router:
        providers = get_available_providers()
        if request.provider not in providers["embedding"]:
            raise HTTPException(400, "Provider not configured")
    """
    embedding_providers = []
    llm_providers = []

    # Check embedding providers
    if app_settings.openai_api_key:
        embedding_providers.append("openai")
    embedding_providers.append("ollama")  # Ollama is always available (local)

    # Check LLM providers
    if app_settings.openai_api_key:
        llm_providers.append("openai")
    if app_settings.anthropic_api_key:
        llm_providers.append("anthropic")

    return {
        "embedding": embedding_providers,
        "llm": llm_providers,
    }


def _get_embedding_dimensions(provider: str) -> int:
    """Get embedding dimensions for a given provider.

    Args:
        provider: Embedding provider name ("openai" or "ollama")

    Returns:
        Number of embedding dimensions

    Design Decision: Match config.py embedding_dimensions property
    - Rationale: Centralize dimension mapping logic
    - OpenAI text-embedding-3-small: 1536 dimensions
    - Ollama nomic-embed-text: 1024 dimensions

    Note: This duplicates logic from config.py Settings.embedding_dimensions
    but avoids circular dependency (service -> config -> settings).
    Consider refactoring if dimension logic becomes more complex.
    """
    if provider == "openai":
        return 1536
    return 1024  # ollama nomic-embed-text
