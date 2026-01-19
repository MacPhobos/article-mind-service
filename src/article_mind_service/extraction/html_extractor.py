"""HTML content extraction using trafilatura with newspaper4k fallback."""

import time
from typing import Any

import trafilatura
from newspaper import Article

from .base import BaseExtractor, ExtractionResult
from .exceptions import EmptyContentError
from .utils import clean_text


class HTMLExtractor(BaseExtractor):
    """Extract content from HTML using trafilatura + newspaper4k fallback.

    Design Decision: Two-tier extraction strategy
    - Primary: trafilatura (highest accuracy F1: 0.958, fastest)
    - Fallback: newspaper4k (excellent metadata, NLP features)

    Trade-offs:
    - ✅ High accuracy with trafilatura
    - ✅ Robust fallback for edge cases
    - ❌ Double extraction attempt may slow down failures
    """

    MIN_CONTENT_LENGTH = 100  # Minimum characters for valid extraction

    def can_extract(self, content_type: str) -> bool:
        """Check if this extractor handles the content type."""
        return content_type.lower() in ("html", "text/html")

    async def extract(self, content: bytes | str, url: str) -> ExtractionResult:
        """Extract content from HTML.

        Strategy:
        1. Try trafilatura (highest accuracy)
        2. Fallback to newspaper4k if trafilatura fails or returns weak content

        Args:
            content: HTML content as string or bytes
            url: Source URL

        Returns:
            ExtractionResult with extracted text

        Raises:
            EmptyContentError: If no content could be extracted
        """
        start_time = time.perf_counter()

        # Convert bytes to string if needed
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")

        # Try trafilatura first (best accuracy)
        result = await self._extract_with_trafilatura(content, url)

        # Check quality and fallback if needed
        if not result.content or len(result.content) < self.MIN_CONTENT_LENGTH:
            newspaper_result = await self._extract_with_newspaper(content, url)

            if newspaper_result.content and len(newspaper_result.content) > len(
                result.content or ""
            ):
                result = newspaper_result
                result.warnings.append(
                    "Used newspaper4k fallback (trafilatura returned weak content)"
                )

        # Final validation
        if not result.content or len(result.content) < self.MIN_CONTENT_LENGTH:
            raise EmptyContentError(
                f"Extraction returned insufficient content ({len(result.content or '')} chars)"
            )

        result.extraction_time_ms = (time.perf_counter() - start_time) * 1000
        return result

    async def _extract_with_trafilatura(self, html: str, url: str) -> ExtractionResult:
        """Extract using trafilatura."""
        # Extract with metadata
        extracted = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            include_images=False,
            include_links=False,
            output_format="txt",
            favor_recall=True,
        )

        # Get metadata separately using dict format
        metadata_result = trafilatura.extract(
            html,
            url=url,
            output_format="json",
            favor_recall=True,
        )
        metadata_dict: dict[str, Any] = metadata_result if isinstance(metadata_result, dict) else {}

        content = clean_text(extracted) if extracted else ""

        return ExtractionResult(
            content=content,
            title=metadata_dict.get("title"),
            author=metadata_dict.get("author"),
            language=metadata_dict.get("language"),
            metadata={
                "sitename": metadata_dict.get("sitename"),
                "categories": metadata_dict.get("categories"),
                "tags": metadata_dict.get("tags"),
            },
            extraction_method="trafilatura",
        )

    async def _extract_with_newspaper(self, html: str, url: str) -> ExtractionResult:
        """Extract using newspaper4k."""
        article = Article(url)
        article.set_html(html)
        article.parse()

        # Optional NLP processing for keywords/summary
        try:
            article.nlp()
            keywords = article.keywords
            summary = article.summary
        except Exception:
            keywords = []
            summary = None

        content = clean_text(article.text) if article.text else ""

        return ExtractionResult(
            content=content,
            title=article.title,
            author=", ".join(article.authors) if article.authors else None,
            published_date=article.publish_date,
            language=article.meta_lang,
            metadata={
                "top_image": article.top_image,
                "keywords": keywords,
                "summary": summary,
                "movies": article.movies,
            },
            extraction_method="newspaper4k",
        )
