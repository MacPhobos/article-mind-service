"""LLM provider abstraction for chat generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Response from LLM provider.

    Attributes:
        content: Generated text response
        tokens_input: Input tokens consumed
        tokens_output: Output tokens generated
        model: Model identifier used
        provider: Provider name
    """

    content: str
    tokens_input: int
    tokens_output: int
    model: str
    provider: str

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed."""
        return self.tokens_input + self.tokens_output


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Implementations must provide async generate() method for chat completion.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider identifier (e.g., 'openai', 'anthropic')."""
        pass

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        context_chunks: list[str],
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Generate response using LLM.

        Args:
            system_prompt: System instructions for the model
            user_message: User's question
            context_chunks: List of context strings to include
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)

        Returns:
            LLMResponse with generated content and usage stats

        Raises:
            LLMProviderError: If generation fails
        """
        pass


class LLMProviderError(Exception):
    """Exception raised when LLM provider fails."""

    def __init__(
        self, provider: str, message: str, original_error: Exception | None = None
    ):
        self.provider = provider
        self.message = message
        self.original_error = original_error
        super().__init__(f"[{provider}] {message}")
