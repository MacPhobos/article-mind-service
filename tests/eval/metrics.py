"""Search quality evaluation metrics.

Provides standard information retrieval metrics for evaluating search quality:
- Precision@K: Fraction of top-k results that are relevant
- Recall@K: Fraction of relevant items found in top-k
- Reciprocal Rank (RR): 1 / rank of first relevant result
- Normalized Discounted Cumulative Gain (nDCG): Ranking quality metric

These metrics are used to evaluate search quality against golden query datasets.
"""

import math


def precision_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Calculate precision at k.

    Precision@K measures the fraction of the top-k retrieved results that are relevant.

    Design Decisions:
    - Returns 0.0 for empty results (no false positives is still precision 0)
    - Uses set intersection for O(min(k, |relevant_ids|)) complexity
    - K is applied before intersection to avoid counting beyond top-k

    Args:
        retrieved_ids: Ordered list of retrieved document IDs (ranked by relevance)
        relevant_ids: Set of known relevant document IDs
        k: Number of top results to consider

    Returns:
        Precision score in range [0.0, 1.0]

    Examples:
        >>> precision_at_k([1, 2, 3, 4, 5], {1, 3, 5}, k=5)
        0.6  # 3 out of 5 are relevant

        >>> precision_at_k([1, 2, 3], {4, 5}, k=3)
        0.0  # None are relevant

        >>> precision_at_k([], {1, 2}, k=5)
        0.0  # No results returned
    """
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & relevant_ids) / len(top_k)


def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Calculate recall at k.

    Recall@K measures the fraction of all relevant documents found in the top-k results.

    Design Decisions:
    - Returns 1.0 when no relevant items exist (vacuous truth: found all 0 items)
    - Uses set intersection for O(min(k, |relevant_ids|)) complexity
    - Complements Precision@K: precision measures accuracy, recall measures coverage

    Args:
        retrieved_ids: Ordered list of retrieved document IDs (ranked by relevance)
        relevant_ids: Set of known relevant document IDs
        k: Number of top results to consider

    Returns:
        Recall score in range [0.0, 1.0]

    Examples:
        >>> recall_at_k([1, 2, 3], {1, 2, 3, 4, 5}, k=3)
        0.6  # Found 3 out of 5 relevant items

        >>> recall_at_k([1, 2, 3, 4, 5], {1, 3, 5}, k=5)
        1.0  # Found all 3 relevant items

        >>> recall_at_k([1, 2, 3], {4, 5}, k=3)
        0.0  # Found 0 out of 2 relevant items
    """
    if not relevant_ids:
        return 1.0  # Vacuous truth: no relevant items means perfect recall
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    """Calculate reciprocal rank.

    Reciprocal Rank (RR) is 1 / rank of the first relevant result.
    Used to measure how quickly a search engine finds a relevant result.

    Design Decisions:
    - Returns 0.0 when no relevant items found (worst case)
    - 1-indexed ranking (first position is rank 1, not 0)
    - Only considers first relevant result (typical for "find first answer" tasks)

    Args:
        retrieved_ids: Ordered list of retrieved document IDs (ranked by relevance)
        relevant_ids: Set of known relevant document IDs

    Returns:
        Reciprocal rank score in range [0.0, 1.0]
        - 1.0: First result is relevant (perfect)
        - 0.5: Second result is relevant
        - 0.33: Third result is relevant
        - 0.0: No relevant results found

    Examples:
        >>> reciprocal_rank([1, 2, 3], {1})
        1.0  # First result is relevant

        >>> reciprocal_rank([1, 2, 3], {2})
        0.5  # Second result is relevant (1/2)

        >>> reciprocal_rank([1, 2, 3], {4})
        0.0  # No relevant results
    """
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)  # 1-indexed ranking
    return 0.0


def ndcg_at_k(
    retrieved_ids: list[int], relevance_labels: dict[int, float], k: int
) -> float:
    """Calculate normalized discounted cumulative gain at k.

    nDCG@K measures ranking quality by considering both relevance and position.
    Higher relevance scores and higher positions contribute more to the score.

    Design Decisions:
    - Uses log2(i + 2) discounting (standard in IR, not log2(i + 1))
    - Returns 0.0 when ideal DCG is 0 (no relevant items)
    - Normalizes by ideal DCG to make scores comparable across queries
    - Supports graded relevance (0.0 to 1.0 scores, not just binary)

    Complexity Analysis:
    - Time: O(k log k) for sorting ideal scores
    - Space: O(k) for actual and ideal score lists

    Args:
        retrieved_ids: Ordered list of retrieved document IDs (ranked by relevance)
        relevance_labels: Dict mapping document IDs to relevance scores (0.0 to 1.0)
        k: Number of top results to consider

    Returns:
        nDCG score in range [0.0, 1.0]
        - 1.0: Perfect ranking (ideal order)
        - 0.0: No relevant items or worst ranking

    Examples:
        >>> ndcg_at_k([1, 2, 3], {1: 1.0, 2: 0.5, 3: 0.0}, k=3)
        1.0  # Perfect ranking (relevance decreases)

        >>> ndcg_at_k([3, 2, 1], {1: 1.0, 2: 0.5, 3: 0.0}, k=3)
        0.58  # Worst ranking (reversed order)

    References:
        - Järvelin & Kekäläinen (2002): Cumulated gain-based evaluation of IR techniques
        - Standard IR metric used in TREC and web search evaluation
    """

    def dcg(scores: list[float]) -> float:
        """Calculate Discounted Cumulative Gain.

        Uses log2(i + 2) discounting to reduce value of lower-ranked results.
        """
        return sum(score / math.log2(i + 2) for i, score in enumerate(scores))

    # Get actual relevance scores for retrieved results
    actual = [relevance_labels.get(rid, 0.0) for rid in retrieved_ids[:k]]

    # Get ideal relevance scores (sorted descending)
    ideal = sorted(relevance_labels.values(), reverse=True)[:k]

    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return 0.0  # No relevant items exist

    return dcg(actual) / ideal_dcg
