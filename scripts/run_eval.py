#!/usr/bin/env python
"""Standalone search quality evaluation runner.

Runs golden query evaluation against live search API and displays results.

Usage:
    # Run against session ID 1 (default)
    uv run python scripts/run_eval.py

    # Run against specific session
    uv run python scripts/run_eval.py --session-id 2

    # Use custom golden queries file
    uv run python scripts/run_eval.py --golden tests/eval/custom_queries.json

    # Specify API base URL
    uv run python scripts/run_eval.py --api-url http://localhost:13010

Note: Always use 'uv run python' to ensure dependencies are available.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx

# Add tests to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from tests.eval.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank


def compute_metrics(
    retrieved: list[int],
    query_config: dict[str, Any],
) -> dict[str, float]:
    """Compute all evaluation metrics for a single query."""
    relevant = set(query_config["expected_article_ids"])
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
    """Check if metrics meet minimum thresholds."""
    failures = []

    if "min_recall" in query_config:
        if metrics["recall_at_10"] < query_config["min_recall"]:
            failures.append(
                f"Recall@10 {metrics['recall_at_10']:.3f} < {query_config['min_recall']:.3f}"
            )

    if "min_precision_at_5" in query_config:
        if metrics["precision_at_5"] < query_config["min_precision_at_5"]:
            failures.append(
                f"Precision@5 {metrics['precision_at_5']:.3f} "
                f"< {query_config['min_precision_at_5']:.3f}"
            )

    return len(failures) == 0, failures


async def run_evaluation(
    session_id: int,
    golden_queries_path: Path,
    api_url: str,
) -> tuple[int, int]:
    """Run evaluation and return (passed, failed) counts.

    Args:
        session_id: Research session ID to evaluate
        golden_queries_path: Path to golden queries JSON
        api_url: Base URL of API (e.g., http://localhost:13010)

    Returns:
        Tuple of (passed_count, failed_count)
    """
    # Load golden queries
    with open(golden_queries_path) as f:
        golden_queries = json.load(f)

    total_queries = len(golden_queries)
    passed = 0
    failed = 0
    aggregate_metrics = {
        "precision_at_5": [],
        "recall_at_10": [],
        "ndcg_at_10": [],
        "mrr": [],
    }

    # Print header
    print(f"\n{'='*90}")
    print(f"Search Quality Evaluation")
    print(f"Session ID: {session_id}")
    print(f"API URL: {api_url}")
    print(f"Golden Queries: {golden_queries_path}")
    print(f"Total Queries: {total_queries}")
    print(f"{'='*90}")
    print(
        f"{'Query ID':<12} | {'Type':<15} | {'P@5':<6} | "
        f"{'R@10':<6} | {'nDCG@10':<8} | {'MRR':<6} | {'Status'}"
    )
    print(f"{'-'*90}")

    async with httpx.AsyncClient(base_url=api_url, timeout=30.0) as client:
        for query_config in golden_queries:
            query_id = query_config["id"]
            query_type = query_config.get("type", "unknown")
            query_text = query_config["query"]

            try:
                # Call search API
                response = await client.post(
                    f"/api/v1/sessions/{session_id}/search",
                    json={"query": query_text, "limit": 10},
                )

                if response.status_code != 200:
                    print(
                        f"{query_id:<12} | {query_type:<15} | "
                        f"{'ERROR':<6} | {'---':<6} | {'---':<8} | "
                        f"{'---':<6} | ✗ API Error {response.status_code}"
                    )
                    failed += 1
                    continue

                results = response.json()
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

            except httpx.RequestError as e:
                print(
                    f"{query_id:<12} | {query_type:<15} | "
                    f"{'ERROR':<6} | {'---':<6} | {'---':<8} | "
                    f"{'---':<6} | ✗ Connection Error"
                )
                print(f"  → {e}")
                failed += 1

    # Print aggregate metrics
    print(f"{'-'*90}")
    if aggregate_metrics["precision_at_5"]:
        avg_p5 = sum(aggregate_metrics["precision_at_5"]) / len(
            aggregate_metrics["precision_at_5"]
        )
        avg_r10 = sum(aggregate_metrics["recall_at_10"]) / len(
            aggregate_metrics["recall_at_10"]
        )
        avg_ndcg = sum(aggregate_metrics["ndcg_at_10"]) / len(aggregate_metrics["ndcg_at_10"])
        avg_mrr = sum(aggregate_metrics["mrr"]) / len(aggregate_metrics["mrr"])

        print(
            f"{'AGGREGATE':<12} | {'':<15} | "
            f"{avg_p5:<6.2f} | {avg_r10:<6.2f} | "
            f"{avg_ndcg:<8.2f} | {avg_mrr:<6.2f} |"
        )
    else:
        print(f"{'AGGREGATE':<12} | {'':<15} | {'---':<6} | {'---':<6} | {'---':<8} | {'---':<6} |")

    print(f"{'='*90}")
    print(f"Passed: {passed}/{total_queries} | Failed: {failed}/{total_queries}")
    print(f"{'='*90}\n")

    return passed, failed


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run search quality evaluation against golden queries"
    )
    parser.add_argument(
        "--session-id",
        type=int,
        default=1,
        help="Research session ID to evaluate (default: 1)",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path(__file__).parent.parent / "tests" / "eval" / "golden_queries.json",
        help="Path to golden queries JSON file",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:13010",
        help="Base URL of API (default: http://localhost:13010)",
    )

    args = parser.parse_args()

    # Validate golden queries file exists
    if not args.golden.exists():
        print(f"Error: Golden queries file not found: {args.golden}")
        sys.exit(1)

    # Run evaluation
    try:
        passed, failed = asyncio.run(
            run_evaluation(
                session_id=args.session_id,
                golden_queries_path=args.golden,
                api_url=args.api_url,
            )
        )

        # Exit with non-zero if any failures
        sys.exit(0 if failed == 0 else 1)

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nError running evaluation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
