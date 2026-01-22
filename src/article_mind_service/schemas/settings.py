"""Provider settings schemas for admin panel configuration."""

from typing import Literal

from pydantic import BaseModel, Field


class ProviderConfigResponse(BaseModel):
    """Current provider configuration with available options.

    Design Decision: Separate "current" and "available" providers

    Rationale: Admin panel needs to know:
    1. Which providers are currently selected (embedding_provider, llm_provider)
    2. Which providers are available (have API keys configured in .env)

    This allows the UI to:
    - Display current selection
    - Disable provider options that aren't configured
    - Show warnings if API keys are missing

    Trade-offs:
    - More data in response, but prevents UI errors
    - Exposes which providers have keys (acceptable for admin panel)
    """

    embedding_provider: Literal["openai", "ollama"] = Field(
        ...,
        description="Currently selected embedding provider",
        examples=["openai"],
    )
    embedding_provider_available: list[str] = Field(
        ...,
        description="List of available embedding providers (have API keys configured)",
        examples=[["openai", "ollama"]],
    )
    llm_provider: Literal["openai", "anthropic"] = Field(
        ...,
        description="Currently selected LLM provider",
        examples=["openai"],
    )
    llm_provider_available: list[str] = Field(
        ...,
        description="List of available LLM providers (have API keys configured)",
        examples=[["openai", "anthropic"]],
    )

    model_config = {"from_attributes": True}


class UpdateEmbeddingProviderRequest(BaseModel):
    """Request to update embedding provider.

    Design Decision: Explicit reindex confirmation

    Rationale: Changing embedding provider requires reindexing all documents
    because different providers use different embedding dimensions:
    - OpenAI text-embedding-3-small: 1536 dimensions
    - Ollama nomic-embed-text: 1024 dimensions

    The confirm_reindex flag forces admins to acknowledge this:
    - False: Provider updated, but warning returned
    - True: Provider updated and reindex triggered

    Trade-offs:
    - Requires extra step, but prevents accidental data loss
    - Explicit is better than implicit for destructive operations
    """

    provider: Literal["openai", "ollama"] = Field(
        ...,
        description="Embedding provider to switch to",
        examples=["openai"],
    )
    confirm_reindex: bool = Field(
        default=False,
        description="Confirm that you want to trigger reindexing (required if dimension changes)",
        examples=[False],
    )


class UpdateEmbeddingProviderResponse(BaseModel):
    """Response after updating embedding provider.

    Design Decision: Explicit feedback on reindex status

    Rationale: Admin needs to know:
    1. Was the provider changed?
    2. Was reindexing triggered?
    3. Is there a warning (dimension mismatch but no reindex)?

    This allows the UI to:
    - Show success/warning messages
    - Display reindex progress
    - Warn about dimension mismatches
    """

    provider: str = Field(
        ...,
        description="The new embedding provider",
        examples=["openai"],
    )
    reindex_triggered: bool = Field(
        ...,
        description="Whether reindexing was triggered",
        examples=[True],
    )
    warning: str | None = Field(
        default=None,
        description="Warning message if reindex is needed but not triggered",
        examples=[None, "Dimension changed from 1024 to 1536. Reindex required."],
    )


class UpdateLlmProviderRequest(BaseModel):
    """Request to update LLM provider.

    Design Decision: No reindex needed for LLM changes

    Rationale: LLM provider only affects text generation, not embeddings.
    Switching between OpenAI and Anthropic doesn't require reindexing
    because the vector database remains unchanged.

    Trade-offs:
    - Simpler than embedding provider update
    - No confirmation needed (non-destructive operation)
    """

    provider: Literal["openai", "anthropic"] = Field(
        ...,
        description="LLM provider to switch to",
        examples=["openai"],
    )


class UpdateLlmProviderResponse(BaseModel):
    """Response after updating LLM provider.

    Simple response confirming the provider change.
    """

    provider: str = Field(
        ...,
        description="The new LLM provider",
        examples=["openai"],
    )
