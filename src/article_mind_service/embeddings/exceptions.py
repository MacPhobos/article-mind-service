"""Custom exceptions for embedding module."""


class EmbeddingError(Exception):
    """Base exception for embedding-related errors.

    Design Decision: Custom exception hierarchy for better error handling.
    - Allows catching all embedding errors with single except block
    - Specific subclasses for different failure modes
    - Preserves error chain with __cause__ for debugging
    """

    pass
