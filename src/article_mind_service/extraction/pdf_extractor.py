"""PDF content extraction using PyMuPDF + pymupdf4llm."""

import os
import tempfile
import time

import fitz  # PyMuPDF
import pymupdf4llm

from .base import BaseExtractor, ExtractionResult
from .exceptions import EmptyContentError, ExtractionError
from .utils import clean_text


class PDFExtractor(BaseExtractor):
    """Extract content from PDF using PyMuPDF + pymupdf4llm.

    Design Decision: Prioritize pymupdf4llm for markdown output
    - Primary: pymupdf4llm.to_markdown() - best for LLM ingestion
    - Fallback: PyMuPDF basic text extraction
    - Optimized for speed (0.12s benchmark) and quality (F1: 0.973)

    Trade-offs:
    - ✅ Fast extraction (~100-200ms)
    - ✅ Excellent markdown output for LLMs
    - ✅ Low memory usage
    - ❌ AGPL license (acceptable for this use case)
    - ❌ Multi-column layouts may be scrambled
    """

    MIN_CONTENT_LENGTH = 100

    def can_extract(self, content_type: str) -> bool:
        """Check if this extractor handles the content type."""
        return content_type.lower() in ("pdf", "application/pdf")

    async def extract(self, content: bytes | str, url: str) -> ExtractionResult:
        """Extract content from PDF.

        Strategy:
        1. Try pymupdf4llm for markdown output (best for LLMs)
        2. Fallback to basic PyMuPDF text extraction

        Args:
            content: PDF content as bytes
            url: Source URL

        Returns:
            ExtractionResult with extracted text

        Raises:
            EmptyContentError: If no content could be extracted
        """
        start_time = time.perf_counter()

        if isinstance(content, str):
            content = content.encode("utf-8")

        # Write to temp file (PyMuPDF works better with files)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            result = await self._extract_pdf(tmp_path, url)
        finally:
            os.unlink(tmp_path)

        if not result.content or len(result.content) < self.MIN_CONTENT_LENGTH:
            raise EmptyContentError(
                f"PDF extraction returned insufficient content ({len(result.content or '')} chars)"
            )

        result.extraction_time_ms = (time.perf_counter() - start_time) * 1000
        return result

    async def _extract_pdf(self, file_path: str, url: str) -> ExtractionResult:
        """Extract content from PDF file."""
        warnings: list[str] = []

        # Try pymupdf4llm first (best markdown output)
        try:
            md_text = pymupdf4llm.to_markdown(file_path)
            if md_text and len(md_text.strip()) > self.MIN_CONTENT_LENGTH:
                # Extract metadata
                doc = fitz.open(file_path)
                metadata = doc.metadata or {}
                page_count = doc.page_count
                doc.close()

                return ExtractionResult(
                    content=clean_text(md_text),
                    title=metadata.get("title"),
                    author=metadata.get("author"),
                    metadata={
                        "subject": metadata.get("subject"),
                        "keywords": metadata.get("keywords"),
                        "creator": metadata.get("creator"),
                        "producer": metadata.get("producer"),
                        "page_count": page_count,
                    },
                    extraction_method="pymupdf4llm",
                    warnings=warnings,
                )
        except Exception as e:
            warnings.append(f"pymupdf4llm failed: {e}")

        # Fallback to basic PyMuPDF extraction
        try:
            doc = fitz.open(file_path)
            text_parts: list[str] = []

            for page in doc:
                text_parts.append(page.get_text())

            text = "\n\n".join(text_parts)
            metadata = doc.metadata or {}
            page_count = doc.page_count
            doc.close()

            return ExtractionResult(
                content=clean_text(text),
                title=metadata.get("title"),
                author=metadata.get("author"),
                metadata={
                    "subject": metadata.get("subject"),
                    "keywords": metadata.get("keywords"),
                    "page_count": page_count,
                },
                extraction_method="pymupdf_basic",
                warnings=warnings,
            )
        except Exception as e:
            raise ExtractionError(f"PDF extraction failed: {e}") from e
