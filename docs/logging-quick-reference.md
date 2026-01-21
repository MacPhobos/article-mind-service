# Structured Logging Quick Reference

## Getting Started

### Import Logger

```python
from article_mind_service.logging_config import get_logger

logger = get_logger(__name__)
```

### Basic Usage

```python
# Info level - normal operations
logger.info("user.login", user_id=123, ip="192.168.1.1")

# Debug level - detailed debugging
logger.debug("cache.lookup", key="user:123", hit=True)

# Warning level - non-critical issues
logger.warning("rate.limit.approaching", user_id=456, requests=950, limit=1000)

# Error level - errors with stack trace
try:
    risky_operation()
except Exception as e:
    logger.error("operation.failed", error=str(e), exc_info=True)
```

## Event Naming Convention

Format: `<module>.<operation>.<stage>`

Examples:
- `rag.query.start` - RAG query begins
- `search.hybrid.complete` - Search completed
- `embedding.generate.failed` - Embedding failed
- `chat.message.created` - Chat message created

## Best Practices

### DO ✅

```python
# Use structured key-value pairs
logger.info("api.request",
            endpoint="/search",
            method="POST",
            status=200,
            duration_ms=45)

# Truncate long strings
logger.info("query.received", query=user_query[:100])

# Use descriptive event names
logger.info("user.password.reset.email.sent", user_id=123)

# Include context for debugging
logger.error("database.query.failed",
             table="users",
             query_type="SELECT",
             error=str(e),
             exc_info=True)
```

### DON'T ❌

```python
# Don't use string formatting
logger.info(f"User {user_id} logged in")  # BAD

# Don't log sensitive data at INFO level
logger.info("user.created", password=password)  # BAD - use DEBUG if needed

# Don't use generic event names
logger.info("error")  # BAD - too generic

# Don't forget exc_info for errors
logger.error("failed", error=str(e))  # BAD - missing stack trace
```

## Common Patterns

### Request/Response Logging

```python
logger.info("api.request.start",
            request_id=request_id,
            endpoint=request.url.path,
            method=request.method)

# ... process request ...

logger.info("api.request.complete",
            request_id=request_id,
            status_code=response.status_code,
            duration_ms=int((time.time() - start_time) * 1000))
```

### Database Operations

```python
logger.debug("db.query.start",
             table="articles",
             operation="SELECT",
             filters={"session_id": session_id})

result = await db.execute(query)

logger.info("db.query.complete",
            table="articles",
            rows_returned=len(result))
```

### External API Calls

```python
logger.info("external_api.call.start",
            provider="openai",
            endpoint="/embeddings")

try:
    response = await client.post(...)
    logger.info("external_api.call.success",
                provider="openai",
                status=response.status_code,
                latency_ms=response.elapsed.total_seconds() * 1000)
except Exception as e:
    logger.error("external_api.call.failed",
                 provider="openai",
                 error=str(e),
                 exc_info=True)
```

### Performance Timing

```python
import time

start_time = time.time()

# ... operation ...

logger.info("operation.complete",
            operation="embedding_generation",
            duration_ms=int((time.time() - start_time) * 1000),
            items_processed=len(documents))
```

## Log Levels

| Level   | When to Use | Example |
|---------|-------------|---------|
| DEBUG   | Detailed debugging info (content previews, intermediate values) | `logger.debug("context.preview", preview=context[:500])` |
| INFO    | Normal operations (requests, completions, counts) | `logger.info("chunks.retrieved", count=5)` |
| WARNING | Non-critical issues (fallbacks, degraded performance) | `logger.warning("cache.miss", key=cache_key)` |
| ERROR   | Errors requiring attention (always include exc_info) | `logger.error("db.connection.failed", exc_info=True)` |

## Environment Configuration

```env
# .env file
LOG_LEVEL=INFO  # Options: DEBUG, INFO, WARNING, ERROR
```

## Filtering Logs

### During Development

```bash
# Run server with DEBUG logs
LOG_LEVEL=DEBUG uv run uvicorn article_mind_service.main:app --reload

# Run with INFO logs (default)
uv run uvicorn article_mind_service.main:app --reload
```

### Searching Logs

```bash
# Find all RAG query logs
grep "rag.query" logs.txt

# Find errors only
grep "[error" logs.txt

# Find specific session
grep "session_id=123" logs.txt

# Count LLM API calls
grep "rag.query.llm_response" logs.txt | wc -l
```

## Example Output

### Development Mode (Console)

```
2026-01-20T22:18:17.031749Z [info     ] rag.query.start                [article_mind_service.chat.rag_pipeline]
    filename=rag_pipeline.py
    func_name=query
    lineno=102
    max_chunks=5
    question='What is the difference between OAuth2 and JWT?'
    session_id=123
```

### Production Mode (JSON) - Future

```json
{
  "timestamp": "2026-01-20T22:18:17.031749Z",
  "level": "info",
  "event": "rag.query.start",
  "logger": "article_mind_service.chat.rag_pipeline",
  "filename": "rag_pipeline.py",
  "func_name": "query",
  "lineno": 102,
  "max_chunks": 5,
  "question": "What is the difference between OAuth2 and JWT?",
  "session_id": 123
}
```

## Debugging Checklist

When investigating issues:

1. ✅ Check log level is appropriate (DEBUG for detailed info)
2. ✅ Search for event name (e.g., `grep "rag.query.start"`)
3. ✅ Filter by session_id or request_id
4. ✅ Look for ERROR logs with stack traces
5. ✅ Check timing_ms for performance issues
6. ✅ Verify expected data is present in logs

## Common Log Events

### RAG Pipeline

- `rag.query.start` - Query begins
- `rag.query.chunks_retrieved` - Chunks retrieved from search
- `rag.query.context_formatted` - Context formatted for LLM (DEBUG)
- `rag.query.llm_response` - LLM responded
- `rag.query.citations_extracted` - Citations extracted
- `rag.retrieve_chunks.embedding_failed` - Embedding generation failed

### Search

- `search.hybrid.start` - Search begins
- `search.hybrid.complete` - Search completed
- `search.endpoint.no_index` - No search index found
- `search.endpoint.embedding_failed` - Embedding generation failed

## Testing Logging

```bash
# Run test script
cd article-mind-service
uv run python scripts/test_logging.py

# Run with DEBUG level
LOG_LEVEL=DEBUG uv run python scripts/test_logging.py
```

## Migration Guide

### Old Logging (Standard library)

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Retrieved %d chunks for session %d", len(chunks), session_id)
```

### New Logging (Structured)

```python
from article_mind_service.logging_config import get_logger
logger = get_logger(__name__)

logger.info("chunks.retrieved",
            chunks_count=len(chunks),
            session_id=session_id)
```

## Additional Resources

- **Full Documentation**: `docs/logging-implementation.md`
- **structlog Docs**: https://www.structlog.org/
- **Implementation Plan**: `/export/workspace/article-mind/docs/plans/plan-R8-evidence-traceability.md`
