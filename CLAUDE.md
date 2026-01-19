# Article Mind Service - Backend Development Guide

## Project Overview

FastAPI-based backend service for the Article Mind knowledge management system. This service provides REST APIs for managing articles, embeddings, search, and analytics.

**Architecture:** Async-first, API contract-driven, type-safe Python backend with PostgreSQL persistence.

## Technology Stack

### Core Framework
- **Language:** Python 3.12.12 (via ASDF)
- **Framework:** FastAPI 0.115+ (automatic OpenAPI generation)
- **ASGI Server:** Uvicorn 0.34+ with hot reload
- **Package Manager:** uv 0.5+ (10-100x faster than pip)

### Database
- **Database:** PostgreSQL 16.x
- **ORM:** SQLAlchemy 2.0+ (async)
- **Migrations:** Alembic 1.13+
- **Driver:** asyncpg (for async PostgreSQL)
- **Sync Driver:** psycopg[binary] (for Alembic migrations)

### Code Quality
- **Linter/Formatter:** ruff 0.8+ (replaces black, flake8, isort)
- **Type Checker:** mypy 1.14+ (strict mode enabled)
- **Security:** bandit 1.7+ (security vulnerability scanner)

### Testing
- **Framework:** pytest 8.x
- **Async Support:** pytest-asyncio 0.24+
- **HTTP Client:** httpx 0.28+ (for testing API endpoints)
- **Coverage:** pytest-cov 6.0+

### Data Validation
- **Schemas:** Pydantic 2.10+
- **Settings:** pydantic-settings 2.7+

## Development Commands

All common commands are available via the Makefile. Run `make help` to see all available targets.

### Standard Workflow

```bash
# Install dependencies
make install

# Start development server (http://localhost:8000)
make dev

# Run tests
make test

# Run tests with coverage
make test-cov

# Run all quality checks (format + lint + test)
make check

# Format code
make format

# Run linters (ruff, mypy, bandit)
make lint

# Auto-fix linting issues
make lint-fix
```

### Database Migrations

```bash
# Create new migration (after changing models)
make migrate-create MSG="add users table"

# Apply migrations
make migrate

# Rollback one migration
make migrate-down
```

### Utility Commands

```bash
# Type checking only
make type-check

# Security scanning only
make security

# Python shell with app context
make shell

# Clean build artifacts
make clean
```

## Project Structure

```
article-mind-service/
├── src/article_mind_service/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # Settings (Pydantic Settings)
│   ├── database.py          # SQLAlchemy async engine & session
│   ├── dependencies.py      # FastAPI dependencies (auth, etc.)
│   ├── models/              # SQLAlchemy models (database schema)
│   │   └── __init__.py
│   ├── schemas/             # Pydantic schemas (API contracts)
│   │   └── __init__.py
│   └── routers/             # FastAPI routers (API endpoints)
│       └── __init__.py
├── tests/
│   ├── conftest.py          # Pytest fixtures
│   ├── test_main.py         # Main app tests
│   ├── unit/                # Unit tests
│   └── integration/         # Integration tests
├── alembic/                 # Database migrations
│   ├── env.py               # Alembic environment (async)
│   └── versions/            # Migration files
├── scripts/                 # Utility scripts
├── pyproject.toml           # Project metadata, dependencies, tool config
├── Makefile                 # Common commands
├── .tool-versions           # ASDF version pinning (python 3.12.12)
├── .env.example             # Environment variable template
├── .env                     # Environment variables (gitignored)
└── README.md                # Project README
```

## API Contract Workflow

**CRITICAL:** Pydantic schemas define the API contract, which automatically generates OpenAPI specs for frontend TypeScript types.

### 1. Define Pydantic Schema

All request and response models MUST be Pydantic schemas in `schemas/`:

```python
# src/article_mind_service/schemas/health.py
from pydantic import BaseModel

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    database: str
```

### 2. Use `response_model` in Route Decorator

```python
# src/article_mind_service/routers/health.py
from fastapi import APIRouter
from article_mind_service.schemas.health import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        database="connected"
    )
```

### 3. OpenAPI Auto-Generation

FastAPI automatically generates OpenAPI spec at:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **OpenAPI JSON:** http://localhost:8000/openapi.json

### 4. Frontend TypeScript Generation

Frontend runs `npm run gen:api` to generate TypeScript types from `/openapi.json`:

```typescript
// Auto-generated in article-mind-ui/src/api/types.ts
export interface HealthResponse {
  status: string;
  version: string;
  database: string;
}
```

### 5. Type-Safe Communication

```typescript
// Frontend usage (fully type-safe)
const health: HealthResponse = await api.get('/health');
console.log(health.status); // TypeScript knows this exists
```

**Key Principles:**
- ✅ **DO:** Define all API contracts as Pydantic schemas
- ✅ **DO:** Use `response_model` on all endpoints
- ✅ **DO:** Let FastAPI generate OpenAPI automatically
- ❌ **DON'T:** Return raw dicts from endpoints
- ❌ **DON'T:** Manually write OpenAPI specs
- ❌ **DON'T:** Manually write TypeScript types for API

## Database Management

### SQLAlchemy 2.0 Async Models

Define database tables in `models/` using SQLAlchemy 2.0 mapped classes:

```python
# src/article_mind_service/models/user.py
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from article_mind_service.database import Base

class User(Base):
    """User model."""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email})>"
```

**Important:** Import all models in `models/__init__.py` for Alembic autogenerate:

```python
# src/article_mind_service/models/__init__.py
from .user import User

__all__ = ["User"]
```

### Database Sessions with Dependency Injection

Use FastAPI's `Depends` to inject async database sessions:

```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends

from article_mind_service.database import get_db
from article_mind_service.models.user import User

@router.get("/users")
async def get_users(db: AsyncSession = Depends(get_db)) -> list[User]:
    """Get all users."""
    result = await db.execute(select(User))
    return result.scalars().all()
```

### Migrations with Alembic

After changing SQLAlchemy models, create and apply migrations:

```bash
# 1. Create migration (autogenerate from model changes)
make migrate-create MSG="add users table"

# 2. Review generated migration in alembic/versions/

# 3. Apply migration to database
make migrate

# Rollback if needed
make migrate-down
```

**Migration Best Practices:**
- Always review autogenerated migrations before applying
- Test migrations in development before production
- Write reversible migrations (implement `downgrade()`)
- Use descriptive migration messages
- Never edit applied migrations (create new ones instead)

## Python Async Patterns

### Async/Await Everywhere

**All I/O operations MUST use async/await:**

```python
# ✅ CORRECT
@app.get("/users/{user_id}")
async def get_user(user_id: int, db: AsyncSession = Depends(get_db)) -> User:
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# ❌ WRONG (missing async/await)
@app.get("/users/{user_id}")
def get_user(user_id: int, db: AsyncSession = Depends(get_db)) -> User:
    result = db.execute(select(User).where(User.id == user_id))  # Missing await!
    return result.scalar_one_or_none()
```

### AsyncIO Patterns

**Concurrent Task Execution:**

```python
import asyncio

async def fetch_multiple_users(user_ids: list[int], db: AsyncSession) -> list[User]:
    """Fetch multiple users concurrently."""
    tasks = [
        db.execute(select(User).where(User.id == user_id))
        for user_id in user_ids
    ]
    results = await asyncio.gather(*tasks)
    return [r.scalar_one_or_none() for r in results]
```

**Timeout Handling:**

```python
import asyncio

async def fetch_with_timeout(user_id: int, db: AsyncSession) -> User:
    """Fetch user with timeout."""
    try:
        async with asyncio.timeout(5.0):  # Python 3.11+
            result = await db.execute(select(User).where(User.id == user_id))
            return result.scalar_one()
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Database timeout")
```

## Code Quality Standards

### Ruff (Linter + Formatter)

Ruff replaces black, flake8, and isort with a single fast tool.

**Configuration in `pyproject.toml`:**
- Line length: 100 characters
- Python 3.12 target
- Rules: pycodestyle, pyflakes, isort, bugbear, comprehensions, pyupgrade

**Usage:**

```bash
# Check for issues
make lint

# Auto-fix issues
make lint-fix

# Format code
make format

# Check formatting without changes
make format-check
```

### Mypy (Type Checker)

Strict type checking is enabled. **All functions MUST have type annotations.**

```python
# ✅ CORRECT (all types annotated)
async def get_user(user_id: int, db: AsyncSession) -> User | None:
    """Fetch user by ID."""
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()

# ❌ WRONG (missing type annotations)
async def get_user(user_id, db):
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()
```

**Run type checker:**

```bash
make type-check  # or make lint (includes mypy)
```

### Bandit (Security Linter)

Scans for security vulnerabilities:

```bash
make security  # or make lint (includes bandit)
```

**Common issues detected:**
- SQL injection risks
- Hardcoded passwords/secrets
- Insecure random number generation
- Shell injection vulnerabilities
- Weak cryptographic algorithms

## Testing Strategy

### Pytest with Async Support

**Test Structure:**

```python
# tests/test_users.py
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_create_user(async_client: AsyncClient) -> None:
    """Test user creation endpoint."""
    response = await async_client.post(
        "/api/v1/users",
        json={"email": "test@example.com", "name": "Test User"}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "test@example.com"
    assert "id" in data
```

### Test Fixtures

Shared fixtures are defined in `tests/conftest.py`:

```python
@pytest.fixture
def client() -> TestClient:
    """Synchronous test client."""
    return TestClient(app)

@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
uv run pytest tests/test_users.py

# Run with verbose output
uv run pytest -v

# Run specific test
uv run pytest tests/test_users.py::test_create_user
```

**Coverage Requirements:**
- Minimum 80% coverage for new code
- Integration tests for all API endpoints
- Unit tests for business logic

## Environment Configuration

### Environment Variables

Configure via `.env` file (never commit `.env`, use `.env.example` as template):

```env
# Database
DATABASE_URL=postgresql://article_mind:article_mind@localhost:5432/article_mind

# API
API_V1_PREFIX=/api/v1
CORS_ORIGINS=http://localhost:5173

# App
DEBUG=true
LOG_LEVEL=INFO
```

### Settings Management

Access settings via Pydantic Settings:

```python
from article_mind_service.config import settings

# Use in code
print(settings.database_url)
print(settings.debug)
```

**Settings are type-safe and validated on startup.**

## Common Pitfalls and Guard Rails

### ❌ Pitfall 1: Forgetting Async/Await

**Problem:**
```python
@app.get("/users")
def get_users(db: AsyncSession = Depends(get_db)):
    result = db.execute(select(User))  # Missing await!
    return result.scalars().all()
```

**Solution:**
```python
@app.get("/users")
async def get_users(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User))
    return result.scalars().all()
```

**Guard Rail:** Mypy will catch this error with strict type checking enabled.

### ❌ Pitfall 2: Missing Response Model

**Problem:**
```python
@app.get("/health")
async def health_check():
    return {"status": "ok"}  # No type safety, no OpenAPI schema
```

**Solution:**
```python
@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok", version="0.1.0")
```

**Guard Rail:** Code review checklist requires `response_model` on all endpoints.

### ❌ Pitfall 3: Hardcoded Configuration

**Problem:**
```python
DATABASE_URL = "postgresql://localhost/mydb"
```

**Solution:**
```python
from article_mind_service.config import settings
database_url = settings.database_url
```

**Guard Rail:** Bandit security scanner detects hardcoded credentials.

### ❌ Pitfall 4: Not Running Migrations

**Problem:** Changing SQLAlchemy models without creating migrations causes database schema drift.

**Solution:**
```bash
# After changing models
make migrate-create MSG="add email field to users"
make migrate
```

**Guard Rail:** CI/CD pipeline checks for unapplied migrations.

### ❌ Pitfall 5: Models Not Imported in Alembic

**Problem:** Alembic autogenerate doesn't detect models if they're not imported.

**Solution:** Import all models in `models/__init__.py`:

```python
# src/article_mind_service/models/__init__.py
from .user import User
from .article import Article

__all__ = ["User", "Article"]
```

**Guard Rail:** Alembic env.py imports `article_mind_service.models` explicitly.

### ❌ Pitfall 6: Synchronous I/O in Async Context

**Problem:**
```python
import requests  # Synchronous HTTP library

@app.get("/external")
async def fetch_external():
    response = requests.get("https://api.example.com")  # Blocks event loop!
    return response.json()
```

**Solution:**
```python
import httpx  # Async HTTP library

@app.get("/external")
async def fetch_external():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com")
        return response.json()
```

**Guard Rail:** Code review guidelines mandate async libraries for I/O.

## FastAPI Best Practices

### Router Organization

Organize endpoints by domain:

```python
# src/article_mind_service/routers/users.py
from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/users", tags=["users"])

@router.get("/")
async def list_users():
    pass

@router.post("/")
async def create_user():
    pass
```

**Register in main.py:**

```python
from article_mind_service.routers import users

app.include_router(users.router)
```

### Dependency Injection

Use FastAPI's `Depends` for reusable dependencies:

```python
# src/article_mind_service/dependencies.py
from fastapi import Depends, HTTPException, Header

async def get_current_user(authorization: str = Header(...)) -> User:
    """Extract current user from Authorization header."""
    # Validate token, fetch user
    if not valid_token(authorization):
        raise HTTPException(status_code=401, detail="Invalid token")
    return fetch_user_from_token(authorization)

# Usage in router
@router.get("/me")
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user
```

### Exception Handling

Use FastAPI's `HTTPException` for API errors:

```python
from fastapi import HTTPException

@router.get("/users/{user_id}")
async def get_user(user_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

## Package Management with uv

uv is 10-100x faster than pip:

```bash
# Install package
uv pip install httpx

# Update pyproject.toml dependencies, then sync
make install

# Or use uv directly
uv pip install -e ".[dev]"
```

**Why uv?**
- 10-100x faster than pip
- Better dependency resolution
- Compatible with pip standards
- Single binary, no Python required

## ASDF Version Management

Python version is pinned in `.tool-versions`:

```
python 3.12.12
```

**Install Python version:**

```bash
asdf install
```

**Verify version:**

```bash
python --version  # Should show Python 3.12.12
```

## API Versioning

All API routes MUST use version prefix:

```python
from fastapi import APIRouter

router = APIRouter(prefix="/api/v1")

@router.get("/users")
async def get_users():
    pass

# Full path: /api/v1/users
```

**Why version?**
- Allows breaking changes in v2 without breaking v1
- Frontend can migrate gradually
- OpenAPI spec clearly shows version

## Health Check API

### Overview

The `/health` endpoint is the **first implemented API endpoint** and serves as the reference implementation for API contract patterns in this service.

**Endpoint:** `GET /health` (no `/api/v1` prefix as per API contract)

**Purpose:**
- Monitor service availability
- Check database connectivity
- Provide version information
- Enable load balancer health checks

### API Contract

The health check follows the frozen API contract in `docs/api-contract.md`.

**Response Schema:**

```typescript
interface HealthResponse {
  status: "ok" | "degraded" | "error";
  version: string;
  database: "connected" | "disconnected";
}
```

**Response Examples:**

Healthy system:
```json
{
  "status": "ok",
  "version": "0.1.0",
  "database": "connected"
}
```

Degraded system (database down):
```json
{
  "status": "degraded",
  "version": "0.1.0",
  "database": "disconnected"
}
```

### Implementation Pattern

The health check demonstrates the **standard API endpoint pattern** used throughout this service:

1. **Pydantic Schema** (`schemas/health.py`):
   - Define response structure with Literal types for enums
   - Include docstrings explaining design decisions
   - Add OpenAPI examples in `model_config`

2. **FastAPI Router** (`routers/health.py`):
   - Use `response_model` decorator for type-safe OpenAPI generation
   - Include detailed endpoint documentation
   - Implement graceful error handling
   - Document trade-offs and design decisions

3. **Router Registration** (`main.py`):
   - Include router in FastAPI app
   - No prefix for health endpoint (exception to versioning rule)

4. **Comprehensive Tests** (`tests/test_main.py`, `tests/unit/test_health.py`):
   - Test response structure and schema validation
   - Test database connectivity scenarios (connected/disconnected)
   - Test OpenAPI spec generation
   - Mock database failures to test degraded state
   - Performance tests (<1 second response time)

### Design Decisions

#### Decision 1: Graceful Degradation

**Rationale:** Health checks should ALWAYS respond (even if DB is down) to distinguish between "service dead" vs "service alive but degraded".

**Implementation:**
- Catches database exceptions instead of propagating
- Returns HTTP 200 with `status: "degraded"` when DB is down
- Allows monitoring systems to make smarter decisions

**Trade-offs:**
- ✅ Better observability (can distinguish partial vs total failure)
- ✅ Enables smarter load balancing
- ❌ HTTP 200 even when degraded (status field indicates state)

#### Decision 2: Literal Types for Enums

**Rationale:** Literal types provide compile-time type safety and better IDE autocomplete compared to string constants or Enum classes.

**Implementation:**
```python
status: Literal["ok", "degraded", "error"]
database: Literal["connected", "disconnected"]
```

**Trade-offs:**
- ✅ Type safety: Catches typos at type-check time
- ✅ OpenAPI: Auto-generates enum constraints in spec
- ✅ Simplicity: No need for separate Enum classes
- ❌ Less explicit than Enum (no centralized definition)

#### Decision 3: No /api/v1 Prefix

**Rationale:** Health checks are infrastructure endpoints, not part of the versioned API.

**Implementation:**
- Registered directly on app without prefix
- Documented as exception in API contract

**Trade-offs:**
- ✅ Standard practice for health checks
- ✅ Stable endpoint for monitoring tools
- ❌ Inconsistent with other API endpoints

### Testing Strategy

**Test Coverage:** 16 tests across 2 test files

**Test Categories:**
1. **Smoke Tests:** Endpoint exists, returns 200
2. **Schema Validation:** Response matches Pydantic schema
3. **Database Scenarios:** Connected and disconnected states
4. **Error Handling:** Various database error types
5. **OpenAPI Generation:** Schema properly documented
6. **Performance:** Response time < 1 second

**Mock Strategy:**
- Mock `get_db` dependency at router level
- Use `AsyncMock` for session.execute
- Bind loop variables properly to avoid closure issues

### Manual Testing

```bash
# Start development server
make dev

# Test health endpoint
curl http://localhost:8000/health

# Expected response (if DB is up):
{
  "status": "ok",
  "version": "0.1.0",
  "database": "connected"
}

# View OpenAPI docs
open http://localhost:8000/docs

# View OpenAPI spec
curl http://localhost:8000/openapi.json | jq '.paths["/health"]'
```

### Frontend Type Generation

After implementing health check, frontend can generate TypeScript types:

```bash
cd article-mind-ui
npm run gen:api
```

This generates:

```typescript
// src/lib/api/generated.ts
export interface HealthResponse {
  status: "ok" | "degraded" | "error";
  version: string;
  database: "connected" | "disconnected";
}
```

### Key Takeaways

**For Future API Endpoints:**
1. Always define Pydantic schema in `schemas/` first
2. Use `response_model` decorator on all endpoints
3. Document design decisions in docstrings
4. Write comprehensive tests (8+ test cases)
5. Add OpenAPI examples in schema and router
6. Test with database mocking for failure scenarios
7. Verify OpenAPI spec generation
8. Follow the pattern established by `/health`

## Database Setup

### PostgreSQL Installation

The project uses PostgreSQL 16.x for local development. You have several installation options:

#### Option A: System PostgreSQL (Recommended if already installed)

Check if PostgreSQL is already running:

```bash
systemctl status postgresql
# or
pg_ctl status
```

If already installed, create the database and user:

```bash
psql postgres << 'EOF'
CREATE USER article_mind WITH PASSWORD 'article_mind';
CREATE DATABASE article_mind OWNER article_mind;
GRANT ALL PRIVILEGES ON DATABASE article_mind TO article_mind;
\q
EOF
```

Verify connection:

```bash
PGPASSWORD=article_mind psql -U article_mind -d article_mind -c '\conninfo'
```

#### Option B: Docker (Recommended for isolation)

Start PostgreSQL container:

```bash
docker run -d \
  --name article-mind-postgres \
  -e POSTGRES_USER=article_mind \
  -e POSTGRES_PASSWORD=article_mind \
  -e POSTGRES_DB=article_mind \
  -p 5432:5432 \
  -v article-mind-pgdata:/var/lib/postgresql/data \
  postgres:16-alpine
```

Stop and remove container:

```bash
docker stop article-mind-postgres
docker rm article-mind-postgres
```

Remove volume (deletes all data):

```bash
docker volume rm article-mind-pgdata
```

#### Option C: ASDF Version Manager

```bash
# Install PostgreSQL plugin
asdf plugin add postgres

# Install PostgreSQL 16.6
asdf install postgres 16.6
asdf local postgres 16.6

# Initialize PostgreSQL data directory
export PGDATA="$HOME/.asdf/installs/postgres/16.6/data"
pg_ctl init -D "$PGDATA"

# Start PostgreSQL
pg_ctl start -D "$PGDATA" -l "$PGDATA/postgres.log"
```

### Database Connection

Connection string format in `.env`:

```env
DATABASE_URL=postgresql://article_mind:article_mind@localhost:5432/article_mind
```

The `database.py` module automatically converts this to async format:

```python
# Automatically converted to:
postgresql+asyncpg://article_mind:article_mind@localhost:5432/article_mind
```

### Testing Database Connection

Run the test script:

```bash
uv run python scripts/test_db.py
```

Expected output:

```
✅ Database connection successful!
PostgreSQL version: PostgreSQL 16.x on ...
Test query result: 1 + 1 = 2
```

### Common Database Operations

#### Connect to Database

Using Docker:

```bash
docker exec -it article-mind-postgres psql -U article_mind -d article_mind
```

Using system PostgreSQL:

```bash
PGPASSWORD=article_mind psql -U article_mind -d article_mind -h localhost
# or without password if using peer authentication
psql -U article_mind -d article_mind
```

#### Useful psql Commands

```bash
\l          # List all databases
\dt         # List all tables
\d table    # Describe table schema
\du         # List users/roles
\conninfo   # Show connection info
\q          # Quit psql
```

#### View Current Migration Status

```bash
uv run alembic current
```

#### View Migration History

```bash
uv run alembic history
```

### Troubleshooting

#### Connection Refused

**Symptom:** `connection to server on socket failed: Connection refused`

**Solutions:**

1. Check PostgreSQL is running:
   ```bash
   # Docker
   docker ps | grep postgres

   # System service
   systemctl status postgresql

   # ASDF
   pg_ctl status -D "$PGDATA"
   ```

2. Verify port 5432 is not blocked:
   ```bash
   ss -tlnp | grep :5432
   ```

3. Check firewall settings

#### Authentication Failed

**Symptom:** `password authentication failed for user "article_mind"`

**Solutions:**

1. Verify credentials in `.env` match database user
2. Check `pg_hba.conf` authentication method (should allow password or md5)
3. For Docker, recreate container with correct environment variables:
   ```bash
   docker rm -f article-mind-postgres
   docker run -d \
     --name article-mind-postgres \
     -e POSTGRES_USER=article_mind \
     -e POSTGRES_PASSWORD=article_mind \
     -e POSTGRES_DB=article_mind \
     -p 5432:5432 \
     postgres:16-alpine
   ```

#### Port Already in Use

**Symptom:** `port 5432 is already allocated`

**Solutions:**

1. Find process using port 5432:
   ```bash
   lsof -i :5432
   ```

2. Stop conflicting service or use different port:
   ```bash
   # Docker with custom port
   docker run -p 5433:5432 ... postgres:16-alpine

   # Update .env
   DATABASE_URL=postgresql://article_mind:article_mind@localhost:5433/article_mind
   ```

#### Migration Conflicts

**Symptom:** Alembic migration errors or conflicts

**Solutions:**

1. Check current migration status:
   ```bash
   uv run alembic current
   ```

2. Rollback if needed:
   ```bash
   make migrate-down
   ```

3. Recreate migration:
   ```bash
   make migrate-create MSG="description"
   make migrate
   ```

4. If Alembic is out of sync, you may need to stamp the database:
   ```bash
   # DANGEROUS: Only if you know what you're doing
   uv run alembic stamp head
   ```

#### Alembic Can't Detect Models

**Symptom:** `alembic revision --autogenerate` creates empty migration

**Solutions:**

1. Ensure models are imported in `models/__init__.py`:
   ```python
   from .user import User
   from .article import Article

   __all__ = ["User", "Article"]
   ```

2. Verify `alembic/env.py` imports models correctly
3. Check `target_metadata = Base.metadata` is set in `alembic/env.py`

#### Docker Volume Permissions

**Symptom:** PostgreSQL container fails to start with permission errors

**Solutions:**

```bash
# Remove volume and recreate
docker volume rm article-mind-pgdata
docker run -d ... postgres:16-alpine
```

### Production Considerations

While not required for local development, keep these in mind for production:

#### Connection Pooling

Already configured in `database.py`. Adjust settings if needed:

```python
engine = create_async_engine(
    database_url,
    echo=settings.debug,
    pool_size=20,          # Max connections in pool
    max_overflow=10,       # Connections beyond pool_size
    pool_timeout=30,       # Seconds to wait for connection
    pool_recycle=3600,     # Recycle connections after 1 hour
)
```

#### Database Backups

Automated backups for production:

```bash
# Backup
docker exec article-mind-postgres pg_dump -U article_mind article_mind > backup.sql

# Restore
docker exec -i article-mind-postgres psql -U article_mind article_mind < backup.sql
```

#### Read Replicas

For production scaling:
- Configure separate read-only connection for queries
- Use write connection only for INSERT/UPDATE/DELETE
- Implement connection routing based on operation type

## Development Workflow

### Daily Development Loop

1. **Start dev server:**
   ```bash
   make dev
   ```

2. **Make changes** to routes/models/schemas
   - Hot reload happens automatically
   - Check http://localhost:8000/docs for OpenAPI updates

3. **Run tests:**
   ```bash
   make test
   ```

4. **Check code quality:**
   ```bash
   make check  # Runs format-check, lint, and test
   ```

5. **Create migration** (if models changed):
   ```bash
   make migrate-create MSG="add articles table"
   make migrate
   ```

6. **Commit changes** (quality checks pass)

### Pre-Commit Checklist

- [ ] All tests pass: `make test`
- [ ] Code formatted: `make format`
- [ ] Linters pass: `make lint`
- [ ] Type checking passes: `make type-check`
- [ ] Security scan passes: `make security`
- [ ] Migrations created (if models changed)
- [ ] `.env` not committed (in `.gitignore`)

## Guard Rails Summary

### MUST DO ✅

- ✅ Use Pydantic schemas for all API requests/responses
- ✅ Use async/await for all database operations
- ✅ Add type hints to all functions (mypy strict)
- ✅ Create migrations after model changes
- ✅ Use dependency injection (FastAPI Depends)
- ✅ Run tests before committing
- ✅ Use `response_model` on all endpoints
- ✅ Import models in `models/__init__.py` for Alembic
- ✅ Use environment variables for configuration
- ✅ Prefix all routes with `/api/v1`

### NEVER DO ❌

- ❌ Hardcode configuration values
- ❌ Use synchronous database operations
- ❌ Skip type annotations
- ❌ Modify database schema without migrations
- ❌ Return raw dicts from endpoints (use Pydantic)
- ❌ Commit `.env` file
- ❌ Use synchronous I/O libraries in async context
- ❌ Skip `response_model` decorator
- ❌ Forget to import models in Alembic
- ❌ Edit applied migrations (create new ones)

## Resources

- **FastAPI:** https://fastapi.tiangolo.com/
- **Pydantic:** https://docs.pydantic.dev/
- **SQLAlchemy 2.0:** https://docs.sqlalchemy.org/en/20/
- **Alembic:** https://alembic.sqlalchemy.org/
- **uv:** https://github.com/astral-sh/uv
- **Ruff:** https://docs.astral.sh/ruff/
- **pytest:** https://docs.pytest.org/

## Troubleshooting

### Port 8000 Already in Use

```bash
# Find and kill process using port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
uv run uvicorn article_mind_service.main:app --reload --port 8001
```

### Alembic Can't Find Models

**Symptom:** `alembic revision --autogenerate` doesn't detect changes.

**Solution:**
1. Ensure models are imported in `models/__init__.py`
2. Check `alembic/env.py` imports `article_mind_service.models`
3. Verify `target_metadata = Base.metadata` is set

### Import Errors in Tests

**Symptom:** `ModuleNotFoundError: No module named 'article_mind_service'`

**Solution:**
- Check `pyproject.toml` has `pythonpath = ["src"]` in `[tool.pytest.ini_options]`
- Run `make install` to install package in editable mode

### Async Database Errors

**Symptom:** `RuntimeError: Event loop is closed`

**Solution:**
- Use `async def` for all route handlers
- Use `await` for all database operations
- Ensure `AsyncSession` from `get_db` dependency

### OpenAPI Spec Not Updating

**Symptom:** Changes to Pydantic schemas don't appear in `/openapi.json`

**Solution:**
- Restart dev server (hot reload may not catch schema changes)
- Ensure `response_model` is set on route decorator
- Check for syntax errors in schema definitions

---

**Last Updated:** 2026-01-18
**Python Version:** 3.12.12
**FastAPI Version:** 0.115+
