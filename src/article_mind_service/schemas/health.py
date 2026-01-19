"""Health check response schemas."""

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response following API contract specification.

    See: docs/auxiliary_info/initial_api_contract_instructions.md

    Design Decision: Using Literal types for status enums

    Rationale: Literal types provide compile-time type safety and better
    IDE autocomplete compared to string constants or enums. FastAPI's OpenAPI
    generation correctly translates Literal to enum in the spec.

    Trade-offs:
    - Type Safety: Literal catches typos at type-check time vs runtime Enum
    - OpenAPI: Auto-generates enum constraints in OpenAPI spec
    - Simplicity: No need to import/define separate Enum classes
    """

    status: Literal["ok", "degraded", "error"] = Field(
        ...,
        description="Service health status",
        examples=["ok"],
    )
    version: str = Field(
        ...,
        description="API version",
        examples=["1.0.0"],
    )
    database: Literal["connected", "disconnected"] = Field(
        ...,
        description="Database connection status",
        examples=["connected"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "ok",
                    "version": "1.0.0",
                    "database": "connected",
                }
            ]
        }
    }
