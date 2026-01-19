"""Content cleaning and processing utilities."""

import re
import unicodedata


def clean_text(text: str | None) -> str:
    """Clean and normalize extracted text.

    - Normalizes Unicode
    - Removes excessive whitespace
    - Removes control characters
    - Normalizes line endings

    Args:
        text: Raw extracted text

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Unicode normalization (NFC form)
    text = unicodedata.normalize("NFC", text)

    # Remove control characters except newlines and tabs
    text = "".join(char for char in text if unicodedata.category(char) != "Cc" or char in "\n\t")

    # Normalize whitespace
    text = normalize_whitespace(text)

    return text.strip()


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text.

    - Replaces multiple spaces with single space
    - Replaces multiple newlines with double newline (paragraph break)
    - Removes trailing whitespace from lines

    Args:
        text: Text to normalize

    Returns:
        Text with normalized whitespace
    """
    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Remove trailing whitespace from each line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Replace multiple spaces with single space
    text = re.sub(r" +", " ", text)

    # Replace 3+ newlines with double newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def is_boilerplate(text: str, threshold: float = 0.3) -> bool:
    """Check if text is likely boilerplate content.

    Uses lexical diversity as a heuristic - boilerplate tends to
    have low unique word ratio.

    Args:
        text: Text to check
        threshold: Minimum unique word ratio (default 0.3)

    Returns:
        True if text appears to be boilerplate
    """
    words = text.lower().split()
    if len(words) < 20:
        return False  # Too short to judge

    unique_ratio = len(set(words)) / len(words)
    return unique_ratio < threshold


def estimate_reading_time(text: str, wpm: int = 200) -> int:
    """Estimate reading time in minutes.

    Args:
        text: Content text
        wpm: Words per minute (default 200)

    Returns:
        Estimated reading time in minutes
    """
    word_count = len(text.split())
    return max(1, round(word_count / wpm))
