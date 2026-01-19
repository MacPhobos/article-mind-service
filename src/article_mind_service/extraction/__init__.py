"""Content extraction module.

Provides extraction of clean text content from URLs (HTML and PDF).

Usage:
    from article_mind_service.extraction import ExtractionPipeline, PipelineConfig

    config = PipelineConfig(timeout_seconds=60)
    pipeline = ExtractionPipeline(config)

    result = await pipeline.extract("https://example.com/article")
    print(result.content)
"""

from .base import BaseExtractor, ExtractionResult
from .content_type import ContentType, detect_content_type
from .exceptions import (
    ContentTooLargeError,
    ContentTypeError,
    EmptyContentError,
    ExtractionError,
    NetworkError,
    RateLimitError,
)
from .html_extractor import HTMLExtractor
from .js_extractor import JSExtractor
from .pdf_extractor import PDFExtractor
from .pipeline import ExtractionPipeline, ExtractionStatus, PipelineConfig
from .utils import clean_text, estimate_reading_time, is_boilerplate, normalize_whitespace

__all__ = [
    # Base classes
    "BaseExtractor",
    "ExtractionResult",
    # Content types
    "ContentType",
    "detect_content_type",
    # Extractors
    "HTMLExtractor",
    "PDFExtractor",
    "JSExtractor",
    # Pipeline
    "ExtractionPipeline",
    "PipelineConfig",
    "ExtractionStatus",
    # Exceptions
    "ExtractionError",
    "NetworkError",
    "EmptyContentError",
    "RateLimitError",
    "ContentTooLargeError",
    "ContentTypeError",
    # Utilities
    "clean_text",
    "normalize_whitespace",
    "is_boilerplate",
    "estimate_reading_time",
]
