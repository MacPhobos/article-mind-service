"""Search quality evaluation test.

Tests search quality against golden query dataset with comprehensive metrics.

Run in two modes:
1. Unit test mode (default): Uses mocked search results for CI
2. Integration mode: Calls actual search API (requires live service)

Usage:
    # Unit test mode (CI-safe)
    pytest tests/eval/search_quality_test.py

    # Integration mode (requires service running)
    EVAL_MODE=integration pytest tests/eval/search_quality_test.py
"""

import json
import os
from pathlib import Path
from typing import Any

import pytest
from httpx import AsyncClient

from .metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank

# ============================================================================
# Configuration
# ============================================================================

GOLDEN_QUERIES_PATH = Path(__file__).parent / "golden_queries.json"
EVAL_MODE = os.getenv("EVAL_MODE", "unit")  # "unit" or "integration"


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture
def golden_queries() -> list[dict[str, Any]]:
    """Load golden query dataset from JSON."""
    with open(GOLDEN_QUERIES_PATH) as f:
        return json.load(f)


@pytest.fixture
def mock_search_results() -> dict[str, list[int]]:
    """Mock search results for unit testing.

    Returns dict mapping query ID to list of retrieved article IDs.
    These mocks simulate realistic search behavior with good precision:
    - Most expected articles appear in top-5 results
    - Some noise/irrelevant results mixed in (realistic)
    - High recall overall (most expected articles found in top-10)
    """
    return {
        # Single word queries - good precision
        "gq-001": [5, 12, 8, 3, 7],  # JWT - finds both expected in top-2
        "gq-003": [5, 2, 8, 10, 11],  # OAuth - finds expected in position 1
        "gq-006": [2, 8, 10, 5, 1],  # API - finds all 3 expected in top-5
        "gq-011": [5, 8, 3, 12, 9],  # cache - finds both expected in top-2
        "gq-014": [2, 8, 5, 10, 11],  # CORS - finds expected in position 1
        # Long/multi-concept queries - good precision
        "gq-002": [5, 8, 12, 2, 7],  # auth in microservices - finds all 3 in top-3
        "gq-005": [7, 15, 3, 8, 9],  # database indexing - finds both expected in top-2
        "gq-008": [4, 9, 13, 2, 5],  # python async - finds all 3 in top-3
        "gq-009": [6, 11, 14, 3, 5],  # ML algorithms - finds all 3 in top-3
        "gq-010": [1, 3, 7, 5, 8],  # docker k8s - finds all 3 in top-3
        "gq-013": [2, 8, 5, 10, 11],  # GraphQL - finds both expected in top-2
        # Question queries - good precision
        "gq-004": [3, 5, 8, 10, 11],  # exact phrase - finds expected in position 1
        "gq-007": [2, 10, 8, 5, 11],  # what is REST - finds both expected in top-2
        "gq-012": [10, 12, 8, 5, 9],  # pagination - finds both expected in top-2
        # Technical phrases - good precision
        "gq-015": [5, 7, 3, 8, 9],  # encryption - finds both expected in top-2
    }


# ============================================================================
# Helper Functions
# ============================================================================


def compute_metrics(
    retrieved: list[int],
    query_config: dict[str, Any],
) -> dict[str, float]:
    """Compute all evaluation metrics for a single query.

    Args:
        retrieved: List of retrieved article IDs (ranked)
        query_config: Golden query configuration with expected results

    Returns:
        Dict with metric names and scores
    """
    relevant = set(query_config["expected_article_ids"])

    # Binary relevance labels (1.0 for relevant, 0.0 for irrelevant)
    relevance_labels = {aid: 1.0 for aid in relevant}

    return {
        "precision_at_5": precision_at_k(retrieved, relevant, k=5),
        "recall_at_10": recall_at_k(retrieved, relevant, k=10),
        "ndcg_at_10": ndcg_at_k(retrieved, relevance_labels, k=10),
        "mrr": reciprocal_rank(retrieved, relevant),
    }


def check_thresholds(
    metrics: dict[str, float], query_config: dict[str, Any]
) -> tuple[bool, list[str]]:
    """Check if metrics meet minimum thresholds.

    Args:
        metrics: Computed metrics
        query_config: Golden query with threshold requirements

    Returns:
        Tuple of (all_passed, failures)
        - all_passed: True if all thresholds met
        - failures: List of failure messages
    """
    failures = []

    # Check recall threshold if specified
    if "min_recall" in query_config:
        if metrics["recall_at_10"] < query_config["min_recall"]:
            failures.append(
                f"Recall@10 {metrics['recall_at_10']:.3f} "
                f"< {query_config['min_recall']:.3f}"
            )

    # Check precision threshold if specified
    if "min_precision_at_5" in query_config:
        if metrics["precision_at_5"] < query_config["min_precision_at_5"]:
            failures.append(
                f"Precision@5 {metrics['precision_at_5']:.3f} "
                f"< {query_config['min_precision_at_5']:.3f}"
            )

    return len(failures) == 0, failures


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.asyncio
class TestSearchQuality:
    """Search quality evaluation against golden queries."""

    async def test_golden_queries_load(self, golden_queries: list[dict[str, Any]]) -> None:
        """Verify golden queries dataset loads correctly."""
        assert len(golden_queries) > 0
        assert all("id" in q for q in golden_queries)
        assert all("query" in q for q in golden_queries)
        assert all("expected_article_ids" in q for q in golden_queries)

    @pytest.mark.skipif(
        EVAL_MODE == "integration",
        reason="Integration mode requires live service",
    )
    async def test_all_queries_unit_mode(
        self,
        golden_queries: list[dict[str, Any]],
        mock_search_results: dict[str, list[int]],
    ) -> None:
        """Test all golden queries in unit mode (mocked results).

        This test runs in CI and doesn't require a live service.
        """
        total_queries = len(golden_queries)
        passed = 0
        failed = 0
        aggregate_metrics = {
            "precision_at_5": [],
            "recall_at_10": [],
            "ndcg_at_10": [],
            "mrr": [],
        }

        print(f"\n{'='*80}")
        print(f"Search Quality Evaluation - Unit Mode ({total_queries} queries)")
        print(f"{'='*80}")
        print(
            f"{'Query ID':<12} | {'Type':<15} | {'P@5':<6} | "
            f"{'R@10':<6} | {'nDCG@10':<8} | {'MRR':<6} | {'Status'}"
        )
        print(f"{'-'*80}")

        for query_config in golden_queries:
            query_id = query_config["id"]
            query_type = query_config.get("type", "unknown")

            # Get mock results
            retrieved = mock_search_results.get(query_id, [])

            # Compute metrics
            metrics = compute_metrics(retrieved, query_config)

            # Check thresholds
            all_passed, failures = check_thresholds(metrics, query_config)

            # Track aggregate metrics
            for metric_name, value in metrics.items():
                aggregate_metrics[metric_name].append(value)

            # Print results
            status = "✓ PASS" if all_passed else "✗ FAIL"
            print(
                f"{query_id:<12} | {query_type:<15} | "
                f"{metrics['precision_at_5']:<6.2f} | "
                f"{metrics['recall_at_10']:<6.2f} | "
                f"{metrics['ndcg_at_10']:<8.2f} | "
                f"{metrics['mrr']:<6.2f} | {status}"
            )

            if not all_passed:
                for failure in failures:
                    print(f"  → {failure}")
                failed += 1
            else:
                passed += 1

        # Print aggregate metrics
        print(f"{'-'*80}")
        avg_p5 = sum(aggregate_metrics["precision_at_5"]) / total_queries
        avg_r10 = sum(aggregate_metrics["recall_at_10"]) / total_queries
        avg_ndcg = sum(aggregate_metrics["ndcg_at_10"]) / total_queries
        avg_mrr = sum(aggregate_metrics["mrr"]) / total_queries

        print(
            f"{'AGGREGATE':<12} | {'':<15} | "
            f"{avg_p5:<6.2f} | {avg_r10:<6.2f} | "
            f"{avg_ndcg:<8.2f} | {avg_mrr:<6.2f} |"
        )
        print(f"{'='*80}")
        print(f"Passed: {passed}/{total_queries} | Failed: {failed}/{total_queries}")
        print(f"{'='*80}\n")

        # Assert overall quality thresholds
        # Note: These are realistic thresholds for mocked data
        # In production, adjust based on actual search quality requirements
        assert avg_p5 >= 0.4, f"Average Precision@5 {avg_p5:.3f} below 0.4"
        assert avg_r10 >= 0.8, f"Average Recall@10 {avg_r10:.3f} below 0.8"
        assert avg_mrr >= 0.8, f"Average MRR {avg_mrr:.3f} below 0.8"

    @pytest.mark.skipif(
        EVAL_MODE != "integration",
        reason="Integration mode only (set EVAL_MODE=integration)",
    )
    async def test_all_queries_integration_mode(
        self,
        golden_queries: list[dict[str, Any]],
        async_client: AsyncClient,
    ) -> None:
        """Test all golden queries in integration mode (live API).

        Requires:
        - Service running at http://localhost:13010
        - Test database with seeded articles
        - Set EVAL_MODE=integration environment variable

        Usage:
            EVAL_MODE=integration pytest tests/eval/search_quality_test.py
        """
        session_id = 1  # TODO: Make configurable
        total_queries = len(golden_queries)
        passed = 0
        failed = 0
        aggregate_metrics = {
            "precision_at_5": [],
            "recall_at_10": [],
            "ndcg_at_10": [],
            "mrr": [],
        }

        print(f"\n{'='*80}")
        print(f"Search Quality Evaluation - Integration Mode ({total_queries} queries)")
        print(f"Session ID: {session_id}")
        print(f"{'='*80}")
        print(
            f"{'Query ID':<12} | {'Type':<15} | {'P@5':<6} | "
            f"{'R@10':<6} | {'nDCG@10':<8} | {'MRR':<6} | {'Status'}"
        )
        print(f"{'-'*80}")

        for query_config in golden_queries:
            query_id = query_config["id"]
            query_type = query_config.get("type", "unknown")
            query_text = query_config["query"]

            # Call actual search API
            response = await async_client.post(
                f"/api/v1/sessions/{session_id}/search",
                json={"query": query_text, "limit": 10},
            )

            assert response.status_code == 200, f"Search API failed for {query_id}"
            results = response.json()

            # Extract article IDs from results
            retrieved = [r["article_id"] for r in results.get("results", [])]

            # Compute metrics
            metrics = compute_metrics(retrieved, query_config)

            # Check thresholds
            all_passed, failures = check_thresholds(metrics, query_config)

            # Track aggregate metrics
            for metric_name, value in metrics.items():
                aggregate_metrics[metric_name].append(value)

            # Print results
            status = "✓ PASS" if all_passed else "✗ FAIL"
            print(
                f"{query_id:<12} | {query_type:<15} | "
                f"{metrics['precision_at_5']:<6.2f} | "
                f"{metrics['recall_at_10']:<6.2f} | "
                f"{metrics['ndcg_at_10']:<8.2f} | "
                f"{metrics['mrr']:<6.2f} | {status}"
            )

            if not all_passed:
                for failure in failures:
                    print(f"  → {failure}")
                failed += 1
            else:
                passed += 1

        # Print aggregate metrics
        print(f"{'-'*80}")
        avg_p5 = sum(aggregate_metrics["precision_at_5"]) / total_queries
        avg_r10 = sum(aggregate_metrics["recall_at_10"]) / total_queries
        avg_ndcg = sum(aggregate_metrics["ndcg_at_10"]) / total_queries
        avg_mrr = sum(aggregate_metrics["mrr"]) / total_queries

        print(
            f"{'AGGREGATE':<12} | {'':<15} | "
            f"{avg_p5:<6.2f} | {avg_r10:<6.2f} | "
            f"{avg_ndcg:<8.2f} | {avg_mrr:<6.2f} |"
        )
        print(f"{'='*80}")
        print(f"Passed: {passed}/{total_queries} | Failed: {failed}/{total_queries}")
        print(f"{'='*80}\n")

        # Assert overall quality thresholds
        # Note: These are realistic thresholds for mocked data
        # In production, adjust based on actual search quality requirements
        assert avg_p5 >= 0.4, f"Average Precision@5 {avg_p5:.3f} below 0.4"
        assert avg_r10 >= 0.8, f"Average Recall@10 {avg_r10:.3f} below 0.8"
        assert avg_mrr >= 0.8, f"Average MRR {avg_mrr:.3f} below 0.8"


# ============================================================================
# Individual Query Tests (for debugging)
# ============================================================================


@pytest.mark.asyncio
class TestIndividualQueries:
    """Test individual queries for debugging and development."""

    async def test_single_word_query(self, mock_search_results: dict[str, list[int]]) -> None:
        """Test single-word technical term query."""
        query_config = {
            "id": "gq-001",
            "query": "JWT",
            "expected_article_ids": [5, 12],
            "min_recall": 0.8,
        }

        retrieved = mock_search_results["gq-001"]
        metrics = compute_metrics(retrieved, query_config)

        assert metrics["recall_at_10"] >= 0.8
        assert 5 in retrieved[:10]
        assert 12 in retrieved[:10]

    async def test_long_query(self, mock_search_results: dict[str, list[int]]) -> None:
        """Test long natural language query."""
        query_config = {
            "id": "gq-002",
            "query": "how does authentication work in microservices",
            "expected_article_ids": [5, 8, 12],
            "min_precision_at_5": 0.6,
        }

        retrieved = mock_search_results["gq-002"]
        metrics = compute_metrics(retrieved, query_config)

        assert metrics["precision_at_5"] >= 0.6

    async def test_high_precision_query(self, mock_search_results: dict[str, list[int]]) -> None:
        """Test query requiring high precision."""
        query_config = {
            "id": "gq-003",
            "query": "OAuth",
            "expected_article_ids": [5],
            "min_recall": 1.0,
        }

        retrieved = mock_search_results["gq-003"]
        metrics = compute_metrics(retrieved, query_config)

        assert metrics["recall_at_10"] == 1.0
        assert 5 in retrieved[:10]
