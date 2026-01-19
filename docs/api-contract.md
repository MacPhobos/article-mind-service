# Article Mind API Contract

> **Version**: 1.0.0
> **Last Updated**: 2026-01-18
> **Status**: FROZEN - Changes require version bump and UI sync

This document defines the API contract between `article-mind-service` (backend) and `article-mind-ui` (frontend).

---

## Table of Contents

1. [Base Configuration](#base-configuration)
2. [Type Generation Strategy](#type-generation-strategy)
3. [Common Types](#common-types)
4. [Endpoints](#endpoints)
5. [Error Handling](#error-handling)
6. [Status Codes](#status-codes)
7. [CORS Configuration](#cors-configuration)
8. [Changelog](#changelog)

---

## Base Configuration

| Environment | Base URL |
|-------------|----------|
| Development | `http://localhost:8000` |
| Production  | `https://api.article-mind.com` (TBD) |

All endpoints are prefixed with `/api/v1` **except** `/health` and `/openapi.json`.

---

## Type Generation Strategy

**This strategy is LOCKED. Do not deviate.**

### Backend (Python FastAPI)

- OpenAPI spec auto-generated at `/openapi.json`
- All response models are Pydantic schemas in `src/article_mind_service/schemas/`
- FastAPI automatically generates OpenAPI spec from Pydantic models
- Use `response_model` decorator on all endpoints for type-safe OpenAPI generation

### Frontend (SvelteKit TypeScript)

- Generated types at `src/lib/api/generated.ts`
- Generation script: `npm run gen:api`
- Command: `openapi-typescript http://localhost:8000/openapi.json -o src/lib/api/generated.ts`

**Workflow**: Backend changes models → Backend deploys → UI runs type generation → UI updates

---

## Common Types

### ErrorResponse

All errors follow this shape.

```typescript
interface ErrorResponse {
  error: {
    code: string;       // Machine-readable code
    message: string;    // Human-readable message
    details?: unknown;  // Optional additional context
  };
}
```

---

## Endpoints

### Health

#### `GET /health`

Health check endpoint. No authentication required. No `/api/v1` prefix.

**Response** `200 OK`

```json
{
  "status": "ok",
  "version": "1.0.0",
  "database": "connected"
}
```

**Schema:**

```typescript
interface HealthResponse {
  status: "ok" | "degraded" | "error";
  version: string;
  database: "connected" | "disconnected";
}
```

**Status Values:**
- `ok`: All systems operational (database connected)
- `degraded`: Service running but database unavailable
- `error`: Critical failure (not currently used, reserved for future)

**Examples:**

Healthy system:
```json
{
  "status": "ok",
  "version": "1.0.0",
  "database": "connected"
}
```

Degraded system (database down):
```json
{
  "status": "degraded",
  "version": "1.0.0",
  "database": "disconnected"
}
```

**Design Decision: Graceful Degradation**

The health endpoint ALWAYS returns HTTP 200, even when the database is down. This allows monitoring systems to distinguish between:
- **Service completely dead** (connection refused, timeout)
- **Service alive but degraded** (returns 200 with status: "degraded")

This pattern enables smarter load balancing and alerts.

---

## Error Handling

### Error Response Shape

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable description",
    "details": { ... }
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid request parameters |
| `NOT_FOUND` | 404 | Resource not found |
| `INTERNAL_ERROR` | 500 | Server error |

---

## Status Codes

| Code | Usage |
|------|-------|
| `200 OK` | Successful GET, PATCH, POST (actions) |
| `201 Created` | Successful POST (resource creation) |
| `204 No Content` | Successful DELETE |
| `400 Bad Request` | Validation error |
| `404 Not Found` | Resource not found |
| `500 Internal Server Error` | Unexpected error |

---

## CORS Configuration

### Development

```
Origins: http://localhost:5173
Credentials: Allowed
Methods: All
Headers: All
```

### Production

Configure via environment variable: `CORS_ORIGINS`

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-18 | Initial contract with /health endpoint |

---

_This contract is the source of truth. UI and service implementations must conform to these definitions._
