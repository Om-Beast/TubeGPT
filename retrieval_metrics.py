"""Small deterministic helpers for evaluating retrieval runs."""
from __future__ import annotations

from typing import Iterable


def retrieval_metrics(
    retrieved_chunk_ids: Iterable[str | int],
    gold_chunk_ids: Iterable[str | int],
    k: int = 4,
) -> dict:
    """Calculate hit, Recall@k and reciprocal rank for one query.

    Hit@k is 1 when any gold evidence chunk appears in the top k, otherwise 0.
    Gold recall@k is the fraction of manually labelled gold chunks recovered.
    """
    if k < 1:
        raise ValueError("k must be positive")
    retrieved = [str(item) for item in retrieved_chunk_ids][:k]
    gold = {str(item) for item in gold_chunk_ids}
    if not gold:
        raise ValueError("Each evaluation case needs at least one gold chunk ID")

    hits = [rank for rank, chunk_id in enumerate(retrieved, start=1) if chunk_id in gold]
    found_gold = len(set(retrieved) & gold)
    reciprocal_rank = 1.0 / hits[0] if hits else 0.0
    return {
        "hit_at_k": int(bool(hits)),
        "gold_recall_at_k": found_gold / len(gold),
        "reciprocal_rank": reciprocal_rank,
        "first_relevant_rank": hits[0] if hits else None,
    }


def aggregate_retrieval_metrics(cases: Iterable[dict], k: int = 4) -> dict:
    """Aggregate per-question retrieval metrics; errors on empty gold sets."""
    rows = [
        retrieval_metrics(case.get("retrieved_chunk_ids", []), case.get("gold_chunk_ids", []), k)
        for case in cases
    ]
    if not rows:
        raise ValueError("At least one evaluation case is required")
    count = len(rows)
    return {
        "queries_evaluated": count,
        "k": k,
        "hit_rate_at_k": sum(row["hit_at_k"] for row in rows) / count,
        "gold_recall_at_k": sum(row["gold_recall_at_k"] for row in rows) / count,
        "mean_reciprocal_rank": sum(row["reciprocal_rank"] for row in rows) / count,
        "per_query": rows,
    }
