"""Database safety guards to prevent accidental production data destruction."""

import re
import sys
from urllib.parse import urlparse

# Production-like patterns that should NEVER be used in tests
FORBIDDEN_PATTERNS = [
    r"production",
    r"prod\b",
    r"\.rds\.amazonaws\.com",
    r"\.azure\.com",
    r"\.neon\.tech",
    r"\.supabase\.co",
]

# Required patterns for test database (at least one must match)
REQUIRED_TEST_PATTERNS = [
    r"_test\b",
    r"test_",
    r"\btest\b",
    r"localhost",
    r"127\.0\.0\.1",
]


def validate_test_database_url(url: str) -> None:
    """Validate that URL is safe for testing.

    Raises:
        SystemExit: If URL matches production patterns or lacks test patterns.
    """
    url_lower = url.lower()

    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, url_lower):
            print(f"FATAL: Database URL matches forbidden pattern '{pattern}'")
            print(f"URL: {url}")
            print("Tests MUST NOT run against production-like databases.")
            sys.exit(1)

    has_test_indicator = any(
        re.search(pattern, url_lower) for pattern in REQUIRED_TEST_PATTERNS
    )

    if not has_test_indicator:
        parsed = urlparse(url)
        print(f"FATAL: Database URL lacks test indicator.")
        print(f"URL: {url}")
        print(f"Database name '{parsed.path}' must contain '_test' or 'test_'")
        print(f"Or host must be localhost/127.0.0.1")
        sys.exit(1)
