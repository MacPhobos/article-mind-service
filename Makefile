.PHONY: help install dev test test-cov test-watch lint lint-fix format format-check migrate migrate-down migrate-create clean shell check

PYTHON := python
UV := uv

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies with uv
	$(UV) pip install -e ".[dev]"

dev: ## Start development server with hot reload
	$(UV) run uvicorn article_mind_service.main:app --reload --host 0.0.0.0 --port 8000

test: ## Run tests
	$(UV) run pytest

test-cov: ## Run tests with coverage
	$(UV) run pytest --cov=article_mind_service --cov-report=html --cov-report=term

test-watch: ## Run tests in watch mode
	$(UV) run pytest-watch

lint: ## Run all linters (ruff, mypy, bandit)
	$(UV) run ruff check src tests
	$(UV) run mypy src
	$(UV) run bandit -r src

lint-fix: ## Auto-fix linting issues
	$(UV) run ruff check --fix src tests

format: ## Format code with ruff
	$(UV) run ruff format src tests

format-check: ## Check code formatting
	$(UV) run ruff format --check src tests

type-check: ## Run mypy type checker
	$(UV) run mypy src

security: ## Run security checks with bandit
	$(UV) run bandit -r src

check: format-check lint test ## Run all quality checks (format, lint, test)

migrate: ## Run database migrations
	$(UV) run alembic upgrade head

migrate-down: ## Rollback one migration
	$(UV) run alembic downgrade -1

migrate-create: ## Create new migration (use: make migrate-create MSG="description")
	@if [ -z "$(MSG)" ]; then \
		echo "Error: MSG is required. Usage: make migrate-create MSG=\"description\""; \
		exit 1; \
	fi
	$(UV) run alembic revision --autogenerate -m "$(MSG)"

clean: ## Remove build artifacts and cache
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

shell: ## Start Python shell with app context
	$(UV) run python -i -c "from article_mind_service.main import app; from article_mind_service.config import settings; print('App loaded. Available: app, settings')"
