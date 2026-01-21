"""Structured logging configuration using structlog.

Design Decision: structlog over standard logging
- Structured logs enable better debugging and observability
- Machine-readable format for log aggregation tools
- Contextual logging with bind/unbind for request tracing
- Human-readable console output for development
- JSON output for production (future enhancement)

Usage:
    from article_mind_service.logging_config import configure_logging, get_logger

    # In main.py startup
    configure_logging()

    # In application code
    logger = get_logger(__name__)
    logger.info("processing_request", user_id=123, request_id="abc")
"""

import logging
import sys
from typing import Any

import structlog


def configure_logging(log_level: str = "INFO", json_logs: bool = False) -> None:
    """Configure structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_logs: If True, output JSON logs for production. If False, use
                   human-readable console output for development.

    Design Decision: Development vs Production Format
    - Development: ConsoleRenderer for human-readable output
    - Production: JSONRenderer for machine parsing (future enhancement)
    - Timestamper: ISO8601 format with UTC timezone
    - CallsiteParameterAdder: Adds filename, function, line number

    Processor Pipeline:
    1. Add log level
    2. Add logger name
    3. Add timestamp (ISO8601 UTC)
    4. Add callsite info (file, function, line)
    5. Format as console or JSON

    Error Handling:
    - Logs structured as dicts for type safety
    - Exception info automatically captured with exc_info=True
    - Stack traces included in error logs
    """
    # Configure standard logging to work with structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )

    # Choose processors based on environment
    processors: list[Any] = [
        # Add log level to event dict
        structlog.stdlib.add_log_level,
        # Add logger name to event dict
        structlog.stdlib.add_logger_name,
        # Add timestamp in ISO8601 format with UTC
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        # Add stack info (filename, function, line number)
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ),
    ]

    # Add final renderer
    if json_logs:
        # Production: JSON output for log aggregation
        # Format exceptions before JSON rendering
        processors.append(structlog.processors.format_exc_info)
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Development: Human-readable console output
        # ConsoleRenderer handles exceptions automatically, no need for format_exc_info
        processors.append(
            structlog.dev.ConsoleRenderer(
                colors=sys.stdout.isatty(),  # Only use colors if terminal
                exception_formatter=structlog.dev.plain_traceback,
            )
        )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    """Get a structured logger instance.

    Args:
        name: Logger name (typically __name__ from calling module)

    Returns:
        Configured structlog logger (BoundLogger)

    Usage:
        logger = get_logger(__name__)
        logger.info("user_login", user_id=123, ip="192.168.1.1")

    Design Decision: Explicit logger naming
    - Pass __name__ to get module-qualified logger names
    - Enables filtering by module in log aggregation
    - Better than singleton logger (allows per-module configuration)

    Note: Returns Any to avoid complex structlog type annotations.
    The actual type is structlog.stdlib.BoundLogger.
    """
    return structlog.get_logger(name)
