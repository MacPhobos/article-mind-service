"""Anthropic Claude provider implementation."""

import anthropic
from anthropic import AsyncAnthropic

from article_mind_service.chat.llm_providers import (
    LLMProvider,
    LLMProviderError,
    LLMResponse,
)
from article_mind_service.config import settings


class AnthropicProvider(LLMProvider):
    """Anthropic Claude Sonnet 4.5 provider for quality-optimized generation.

    Pricing (as of 2026):
    - Claude Sonnet 4.5: $3.00 input / $15.00 output per 1M tokens
    - 200K context window (1M at premium rates)
    - 90% savings with prompt caching

    Best for:
    - Complex reasoning tasks
    - High-quality synthesis
    - Coding and technical content
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5-20241022",
    ):
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model
        self._client: AsyncAnthropic | None = None

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def client(self) -> AsyncAnthropic:
        """Lazy-initialize Anthropic client."""
        if self._client is None:
            if not self.api_key:
                raise LLMProviderError(
                    provider="anthropic", message="ANTHROPIC_API_KEY not configured"
                )
            self._client = AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        context_chunks: list[str],
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Generate response using Claude Sonnet 4.5.

        Constructs messages in Anthropic format:
        - System parameter for instructions
        - Messages array with user content
        """
        # Build context block
        context_block = self._format_context(context_chunks)

        # Construct full user message with context
        full_user_message = f"""Context:
{context_block}

Question: {user_message}"""

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": full_user_message},
                ],
                temperature=temperature,
            )

            # Extract text from content blocks
            content = ""
            for block in response.content:
                if block.type == "text":
                    content += block.text

            return LLMResponse(
                content=content,
                tokens_input=response.usage.input_tokens,
                tokens_output=response.usage.output_tokens,
                model=self.model,
                provider=self.provider_name,
            )

        except anthropic.APIError as e:
            raise LLMProviderError(
                provider="anthropic",
                message=f"API error: {str(e)}",
                original_error=e,
            ) from e
        except anthropic.AuthenticationError as e:
            raise LLMProviderError(
                provider="anthropic",
                message="Invalid API key",
                original_error=e,
            ) from e
        except Exception as e:
            raise LLMProviderError(
                provider="anthropic",
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
