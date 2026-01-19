"""Content extraction pipeline orchestration."""

import asyncio
from dataclasses import dataclass
from enum import Enum

import httpx

from .base import ExtractionResult
from .content_type import ContentType, detect_content_type
from .exceptions import (
    ContentTooLargeError,
    EmptyContentError,
    NetworkError,
    RateLimitError,
)
from .html_extractor import HTMLExtractor
from .js_extractor import JSExtractor
from .pdf_extractor import PDFExtractor


class ExtractionStatus(str, Enum):
    """Extraction status values."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineConfig:
    """Configuration for extraction pipeline.

    Design Decision: Configurable timeouts and limits
    - All settings have sensible defaults
    - Environment-driven configuration in production
    - Retry logic with exponential backoff

    Attributes:
        timeout_seconds: HTTP request timeout
        max_retries: Maximum retry attempts for transient failures
        max_content_size_mb: Maximum content size limit
        user_agent: User-Agent header for requests
        playwright_headless: Run Playwright in headless mode
        retry_with_js: Retry with JS rendering if static extraction fails
    """

    timeout_seconds: int = 30
    max_retries: int = 3
    max_content_size_mb: int = 50
    user_agent: str = "ArticleMind/1.0 (Content Extraction Bot)"
    playwright_headless: bool = True
    retry_with_js: bool = True

    @property
    def max_content_size_bytes(self) -> int:
        """Get max content size in bytes."""
        return self.max_content_size_mb * 1024 * 1024


class ExtractionPipeline:
    """Orchestrates content extraction from URLs.

    Design Decision: Tiered extraction strategy
    1. Detect content type (HEAD request + URL pattern)
    2. Fetch content with retry logic
    3. Route to appropriate extractor (HTML, PDF)
    4. Retry with JS rendering if HTML extraction fails

    Trade-offs:
    - ✅ High success rate through fallbacks
    - ✅ Automatic retry with exponential backoff
    - ✅ Lazy initialization of expensive resources (Playwright)
    - ❌ Multiple extraction attempts increase latency for failures
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Initialize pipeline with configuration.

        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or PipelineConfig()
        self._html_extractor = HTMLExtractor()
        self._pdf_extractor = PDFExtractor()
        self._js_extractor: JSExtractor | None = None

    def _get_js_extractor(self) -> JSExtractor:
        """Lazy-initialize JS extractor (expensive to create)."""
        if self._js_extractor is None:
            self._js_extractor = JSExtractor(
                headless=self.config.playwright_headless,
                timeout_ms=self.config.timeout_seconds * 1000,
            )
        return self._js_extractor

    async def extract(self, url: str) -> ExtractionResult:
        """Extract content from URL.

        Flow:
        1. Detect content type
        2. Fetch content
        3. Route to appropriate extractor
        4. If HTML extraction fails/weak, retry with JS rendering

        Args:
            url: URL to extract content from

        Returns:
            ExtractionResult with extracted content

        Raises:
            NetworkError: If network request fails
            ExtractionError: If extraction fails
            ContentTooLargeError: If content exceeds size limit
        """
        async with httpx.AsyncClient(
            timeout=self.config.timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": self.config.user_agent},
        ) as client:
            # Step 1: Detect content type
            content_type = await detect_content_type(url, client)

            # Step 2: Fetch content with retry
            content, final_url = await self._fetch_with_retry(client, url)

            # Step 3: Route to appropriate extractor
            try:
                if content_type == ContentType.PDF:
                    return await self._pdf_extractor.extract(content, final_url)
                else:
                    return await self._extract_html(content, final_url)
            except EmptyContentError:
                # Step 4: Retry with JS rendering for HTML
                if content_type == ContentType.HTML and self.config.retry_with_js:
                    return await self._extract_with_js(final_url)
                raise

    async def _fetch_with_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
    ) -> tuple[bytes, str]:
        """Fetch content with retry logic.

        Implements exponential backoff for transient failures.

        Args:
            client: HTTP client
            url: URL to fetch

        Returns:
            Tuple of (content bytes, final URL after redirects)

        Raises:
            RateLimitError: If rate limited by server
            ContentTooLargeError: If content exceeds size limit
            NetworkError: If all retries exhausted
        """
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries):
            try:
                response = await client.get(url)

                # Check for rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 5))
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(retry_after)
                        continue
                    raise RateLimitError(f"Rate limited by {url}")

                response.raise_for_status()

                # Check content size
                content_length = len(response.content)
                if content_length > self.config.max_content_size_bytes:
                    raise ContentTooLargeError(
                        f"Content size {content_length} exceeds limit "
                        f"{self.config.max_content_size_bytes}"
                    )

                return response.content, str(response.url)

            except httpx.TimeoutException as e:
                last_error = NetworkError(f"Timeout fetching {url}: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2**attempt)  # Exponential backoff
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500:
                    last_error = NetworkError(f"Server error {e.response.status_code}: {e}")
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(2**attempt)
                else:
                    raise NetworkError(f"HTTP error {e.response.status_code}: {e}") from e
            except httpx.RequestError as e:
                last_error = NetworkError(f"Request error: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2**attempt)

        raise last_error or NetworkError(f"Failed to fetch {url}")

    async def _extract_html(self, content: bytes, url: str) -> ExtractionResult:
        """Extract from HTML content."""
        return await self._html_extractor.extract(content, url)

    async def _extract_with_js(self, url: str) -> ExtractionResult:
        """Extract using JavaScript rendering."""
        js_extractor = self._get_js_extractor()
        result = await js_extractor.extract(b"", url)
        result.warnings.append("Used JavaScript rendering (static extraction failed)")
        return result
