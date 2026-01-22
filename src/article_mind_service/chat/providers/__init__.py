"""LLM provider factory and exports."""

from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession

from article_mind_service.chat.llm_providers import (
    LLMProvider,
    LLMProviderError,
    LLMResponse,
)
from article_mind_service.chat.providers.anthropic import AnthropicProvider
from article_mind_service.chat.providers.openai import OpenAIProvider
from article_mind_service.config import settings

ProviderName = Literal["openai", "anthropic"]


async def get_llm_provider(
    db: AsyncSession | None = None,
    provider_override: ProviderName | None = None,
) -> LLMProvider:
    """Get LLM provider instance based on configuration.

    Provider Resolution Priority:
    1. provider_override parameter (explicit override)
    2. Database settings (if db session provided)
    3. Environment variable settings (fallback)

    Design Decision: Three-level priority
    - Rationale: Explicit override > persistent settings > defaults
    - Allows testing without database
    - Allows admin panel to override via database
    - Falls back gracefully when database unavailable

    Args:
        db: Database session for reading settings (optional)
        provider_override: Explicit provider to use, bypasses database (optional)

    Returns:
        Configured LLM provider instance

    Raises:
        ValueError: If provider name is invalid

    Backward Compatibility:
        Works without database session (uses settings from .env)

    Example:
        # Use database settings
        async with get_db() as db:
            provider = await get_llm_provider(db=db)

        # Use explicit override (for testing)
        provider = await get_llm_provider(provider_override="anthropic")

        # Use .env settings (no database)
        provider = await get_llm_provider()
    """
    # Determine provider name using priority order
    if provider_override:
        provider_name = provider_override
    elif db:
        # Import here to avoid circular dependency
        from article_mind_service.services.settings_service import get_settings

        db_settings = await get_settings(db)
        provider_name = db_settings.llm_provider  # type: ignore[assignment]
    else:
        # Fallback to .env settings
        provider_name = settings.llm_provider  # type: ignore[assignment]

    # Instantiate provider
    if provider_name == "openai":
        return OpenAIProvider(
            api_key=settings.openai_api_key,
            model=(
                settings.llm_model
                if settings.llm_provider == "openai"
                else "gpt-4o-mini"
            ),
        )
    elif provider_name == "anthropic":
        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            model=(
                settings.llm_model
                if settings.llm_provider == "anthropic"
                else "claude-sonnet-4-5-20241022"
            ),
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}")


__all__ = [
    "LLMProvider",
    "LLMProviderError",
    "LLMResponse",
    "OpenAIProvider",
    "AnthropicProvider",
    "get_llm_provider",
    "ProviderName",
]
