"""OpenAI LLM provider implementation."""

import openai
from openai import AsyncOpenAI

from article_mind_service.chat.llm_providers import (
    LLMProvider,
    LLMProviderError,
    LLMResponse,
)
from article_mind_service.config import settings


class OpenAIProvider(LLMProvider):
    """OpenAI GPT-4o-mini provider for cost-optimized generation.

    Pricing (as of 2026):
    - GPT-4o-mini: $0.15 input / $0.60 output per 1M tokens
    - 128K context window

    Best for:
    - High-volume, cost-sensitive RAG applications
    - Fast response times required
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
    ):
        self.api_key = api_key or settings.openai_api_key
        self.model = model
        self._client: AsyncOpenAI | None = None

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def client(self) -> AsyncOpenAI:
        """Lazy-initialize OpenAI client."""
        if self._client is None:
            if not self.api_key:
                raise LLMProviderError(
                    provider="openai", message="OPENAI_API_KEY not configured"
                )
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        context_chunks: list[str],
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Generate response using OpenAI GPT-4o-mini.

        Constructs messages in OpenAI chat format:
        - System message with instructions
        - User message with context + question
        """
        # Build context block
        context_block = self._format_context(context_chunks)

        # Construct full user message with context
        full_user_message = f"""Context:
{context_block}

Question: {user_message}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_user_message},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            choice = response.choices[0]
            usage = response.usage

            return LLMResponse(
                content=choice.message.content or "",
                tokens_input=usage.prompt_tokens if usage else 0,
                tokens_output=usage.completion_tokens if usage else 0,
                model=self.model,
                provider=self.provider_name,
            )

        except openai.APIError as e:
            raise LLMProviderError(
                provider="openai",
                message=f"API error: {e.message}",
                original_error=e,
            ) from e
        except openai.AuthenticationError as e:
            raise LLMProviderError(
                provider="openai",
                message="Invalid API key",
                original_error=e,
            ) from e
        except Exception as e:
            raise LLMProviderError(
                provider="openai",
                message=str(e),
                original_error=e,
            ) from e

    def _format_context(self, chunks: list[str]) -> str:
        """Format context chunks with citation numbers."""
        if not chunks:
            return "No relevant context found."

        formatted = []
        for i, chunk in enumerate(chunks, start=1):
            formatted.append(f"[{i}] {chunk}")

        return "\n\n".join(formatted)
