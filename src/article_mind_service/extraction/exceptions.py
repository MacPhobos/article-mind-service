"""Custom exceptions for content extraction."""


class ExtractionError(Exception):
    """Base exception for extraction errors."""

    pass


class NetworkError(ExtractionError):
    """Network-related errors (timeout, connection refused)."""

    pass


class ContentTypeError(ExtractionError):
    """Unable to determine or handle content type."""

    pass


class EmptyContentError(ExtractionError):
    """Extraction returned empty or minimal content."""

    pass


class RateLimitError(ExtractionError):
    """Rate limited by the target server."""

    pass


class ContentTooLargeError(ExtractionError):
    """Content exceeds maximum size limit."""

    pass
