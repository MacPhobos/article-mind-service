# Article Mind Service

Backend service for the Article Mind knowledge management system.

## Quick Start

```bash
# Install dependencies
make install

# Start development server
make dev

# Run tests
make test

# Run linters
make lint
```

## Documentation

See [CLAUDE.md](./CLAUDE.md) for comprehensive development guidelines.

## Technology Stack

- **Python 3.12.12**
- **FastAPI 0.115+**
- **PostgreSQL 16.x**
- **SQLAlchemy 2.0 (async)**
- **Alembic** for migrations
- **uv** for package management

## API Documentation

When running the dev server, access:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

## Development

All common commands are available via the Makefile:

```bash
make help  # Show all available commands
```
