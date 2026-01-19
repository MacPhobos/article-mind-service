"""LLM provider factory and exports."""

from typing import Literal

from article_mind_service.chat.llm_providers import (
    LLMProvider,
    LLMProviderError,
    LLMResponse,
)
from article_mind_service.chat.providers.anthropic import AnthropicProvider
from article_mind_service.chat.providers.openai import OpenAIProvider
from article_mind_service.config import settings

ProviderName = Literal["openai", "anthropic"]


def get_llm_provider(provider: ProviderName | None = None) -> LLMProvider:
    """Get LLM provider instance based on configuration.

    Args:
        provider: Explicit provider name, or None to use configured default

    Returns:
        Configured LLM provider instance

    Raises:
        ValueError: If provider name is invalid
    """
    provider_name = provider or settings.llm_provider

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
