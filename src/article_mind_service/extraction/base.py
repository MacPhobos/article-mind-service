"""Abstract base class for content extractors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ExtractionResult:
    """Result of content extraction.

    Design Decision: Uses dataclass for simplicity and immutability.
    Alternative considered: Pydantic BaseModel (rejected to avoid mixing
    ORM and API concerns - this is internal to extraction logic).

    Attributes:
        content: Extracted text content
        title: Document/page title
        author: Author name (if available)
        published_date: Publication date (if available)
        language: Detected language code (e.g., 'en', 'fr')
        word_count: Number of words in content
        metadata: Additional metadata (sitename, tags, etc.)
        extraction_method: Which extractor was used (trafilatura, newspaper4k, etc.)
        extraction_time_ms: Time taken to extract (milliseconds)
        warnings: List of warnings during extraction
    """

    content: str
    title: str | None = None
    author: str | None = None
    published_date: datetime | None = None
    language: str | None = None
    word_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    extraction_method: str = ""
    extraction_time_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Calculate word count if not provided."""
        if not self.word_count and self.content:
            self.word_count = len(self.content.split())


class BaseExtractor(ABC):
    """Abstract base class for content extractors.

    All extractors (HTML, PDF, JS) inherit from this base to ensure
    consistent interface and behavior.
    """

    @abstractmethod
    async def extract(self, content: bytes | str, url: str) -> ExtractionResult:
        """Extract clean text from content.

        Args:
            content: Raw content (bytes for PDF, str for HTML)
            url: Source URL for context/metadata

        Returns:
            ExtractionResult with extracted text and metadata

        Raises:
            ExtractionError: If extraction fails
            EmptyContentError: If no content could be extracted
        """
        pass

    @abstractmethod
    def can_extract(self, content_type: str) -> bool:
        """Check if this extractor can handle the content type.

        Args:
            content_type: MIME type or content category

        Returns:
            True if this extractor can handle the content
        """
        pass
