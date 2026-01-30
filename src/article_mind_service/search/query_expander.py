"""Query expansion via domain-relevant synonym dictionary.

Applied to BM25 sparse queries only. Expands abbreviations and
technical shorthands to improve recall without adding latency.

Design Decision: Dictionary-based synonym expansion
- Zero-latency lookup (simple dict access)
- Preserves original terms for exact matching
- Expands common abbreviations and technical terms
- Deduplicates to avoid token repetition

Trade-offs:
- ✅ No external dependencies (no LLM calls, no network)
- ✅ Deterministic and testable
- ✅ Sub-millisecond performance (<0.1ms)
- ❌ Limited to predefined synonyms (no dynamic expansion)
- ❌ Requires manual curation of synonym dictionary

Alternatives Considered:
1. LLM-based expansion: Rejected due to latency (100-500ms)
2. WordNet expansion: Rejected due to poor technical term coverage
3. Query understanding model: Rejected due to complexity and latency

Usage:
    from article_mind_service.search.query_expander import expand_query

    expanded = expand_query("auth api")
    # Result: "auth authentication authorize login api application programming interface endpoint"
"""

import re

# Domain-relevant synonym dictionary
# Maps abbreviations and technical shorthands to their expansions
# Rationale: Common technical terms in software documentation and articles
SYNONYMS = {
    "auth": "authentication authorize login",
    "db": "database data storage",
    "api": "application programming interface endpoint",
    "ml": "machine learning model training",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "k8s": "kubernetes container orchestration",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "regex": "regular expression pattern",
    "env": "environment variable configuration",
    "config": "configuration settings",
    "async": "asynchronous concurrent",
    "perf": "performance optimization",
    "sec": "security vulnerability",
    "dep": "dependency package library",
    "css": "stylesheet style design",
    "html": "markup document page",
    "ui": "user interface frontend",
    "ux": "user experience design",
    "ci": "continuous integration pipeline",
    "cd": "continuous deployment delivery",
    "orm": "object relational mapping",
    "sdk": "software development kit",
    "cli": "command line interface terminal",
    "sso": "single sign on",
    "jwt": "json web token",
    "oauth": "open authorization",
    "cors": "cross origin resource sharing",
    "crud": "create read update delete",
}


def expand_query(query: str) -> str:
    """Expand abbreviations and technical terms in query.

    Preserves original terms and appends synonyms.
    Deduplicates while maintaining order.

    Args:
        query: Search query string to expand

    Returns:
        Expanded query string with synonyms appended

    Algorithm:
    1. Split query into words
    2. For each word, preserve original and append synonyms if found
    3. Deduplicate while maintaining order (seen set tracking)
    4. Join back into string

    Performance:
    - Time: O(n) where n is number of words
    - Space: O(m) where m is unique words in result
    - Typical: <0.1ms for queries with 1-10 words

    Examples:
        >>> expand_query("auth api")
        'auth authentication authorize login api application programming interface endpoint'

        >>> expand_query("JWT authentication")
        'jwt json web token authentication'

        >>> expand_query("unknown word")
        'unknown word'

        >>> expand_query("")
        ''

    Design Decision: Case-insensitive matching
    - Converts query to lowercase for synonym lookup
    - Preserves technical term standardization (JWT -> jwt)
    - Rationale: BM25 tokenization is case-insensitive anyway
    """
    if not query or not query.strip():
        return ""

    words = query.lower().split()
    expanded = []

    for word in words:
        # Preserve original word first
        expanded.append(word)

        # Strip punctuation for lookup but keep original in results
        # Example: "auth." -> lookup "auth" -> append synonyms
        clean = re.sub(r"[^\w]", "", word)

        # Lookup synonyms and append if found
        if clean in SYNONYMS:
            expanded.extend(SYNONYMS[clean].split())

    # Deduplicate while preserving order
    # Example: "auth auth" -> "auth" (single occurrence)
    seen = set()
    result = []
    for w in expanded:
        if w not in seen:
            seen.add(w)
            result.append(w)

    return " ".join(result)
