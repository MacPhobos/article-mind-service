"""Content type detection utilities."""

from enum import Enum
from urllib.parse import urlparse

import httpx


class ContentType(str, Enum):
    """Supported content types."""

    HTML = "html"
    PDF = "pdf"
    UNKNOWN = "unknown"


async def detect_content_type(url: str, client: httpx.AsyncClient) -> ContentType:
    """Detect content type from URL and HEAD request.

    Detection strategy:
    1. URL pattern (*.pdf, *.html)
    2. HEAD request Content-Type header
    3. Default to HTML

    Args:
        url: URL to check
        client: HTTP client for HEAD request

    Returns:
        Detected ContentType
    """
    parsed = urlparse(url)
    path = parsed.path.lower()

    # Step 1: URL pattern
    if path.endswith(".pdf"):
        return ContentType.PDF
    if path.endswith((".html", ".htm")):
        return ContentType.HTML

    # Step 2: HEAD request
    try:
        response = await client.head(url, follow_redirects=True, timeout=10.0)
        content_type_header = response.headers.get("content-type", "").lower()

        if "application/pdf" in content_type_header:
            return ContentType.PDF
        if "text/html" in content_type_header:
            return ContentType.HTML
    except Exception:
        pass  # Fall through to default

    # Default to HTML
    return ContentType.HTML


def detect_content_type_from_bytes(content: bytes) -> ContentType:
    """Detect content type by inspecting content bytes.

    Args:
        content: Content bytes to inspect

    Returns:
        Detected ContentType
    """
    if content.startswith(b"%PDF"):
        return ContentType.PDF

    # Check for HTML markers in first 1KB
    preview = content[:1024].lower()
    if b"<!doctype html" in preview or b"<html" in preview:
        return ContentType.HTML

    return ContentType.UNKNOWN
