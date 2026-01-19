"""Health check endpoints."""

from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from article_mind_service.config import settings
from article_mind_service.database import get_db
from article_mind_service.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check endpoint",
    description=(
        "Returns service health status including database connectivity. "
        "No authentication required. This endpoint does NOT use the /api/v1 "
        "prefix as specified in the API contract."
    ),
    responses={
        200: {
            "description": "Service is healthy or degraded",
            "content": {
                "application/json": {
                    "examples": {
                        "healthy": {
                            "summary": "All systems operational",
                            "value": {
                                "status": "ok",
                                "version": "1.0.0",
                                "database": "connected",
                            },
                        },
                        "degraded": {
                            "summary": "Database unavailable",
                            "value": {
                                "status": "degraded",
                                "version": "1.0.0",
                                "database": "disconnected",
                            },
                        },
                    }
                }
            },
        }
    },
)
async def health_check(
    db: AsyncSession = Depends(get_db),
) -> HealthResponse:
    """Health check endpoint.

    Returns service status, version, and database connectivity.

    Design Decision: Graceful degradation instead of failing hard

    Rationale: Health checks should ALWAYS respond (even if DB is down)
    to distinguish between "service dead" vs "service alive but degraded".
    This helps monitoring systems and load balancers make better decisions.

    Trade-offs:
    - Availability: Returns 200 even with DB down (status field indicates state)
    - Monitoring: Enables distinguishing partial failures from total failures
    - Simplicity: Single endpoint vs separate /health and /health/db

    Error Handling:
    - Database connection errors: Caught and returned as "degraded" status
    - No exceptions propagated (health check must always succeed)
    - Database status reflects actual connectivity test (SELECT 1)

    Performance:
    - Complexity: O(1) - single SELECT 1 query
    - Expected time: <50ms for healthy database
    - Timeout: Inherits from database connection pool settings

    Args:
        db: Async database session (injected by FastAPI)

    Returns:
        HealthResponse with current service status
    """
    # Check database connectivity
    database_status: str = "disconnected"
    try:
        await db.execute(text("SELECT 1"))
        database_status = "connected"
    except Exception as e:
        # Log error but don't raise - return degraded status instead
        print(f"Database health check failed: {e}")
        # Don't propagate exception - health check must always respond

    # Determine overall status based on database connectivity
    overall_status: str = "ok" if database_status == "connected" else "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        database=database_status,
    )
