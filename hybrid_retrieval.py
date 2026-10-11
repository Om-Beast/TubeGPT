"""Hybrid lexical + dense retrieval utilities.

Dense ranking comes from the vector index; lexical ranking uses BM25 so exact
product names, acronyms, and technical terms can surface when semantic ranking
is weak. Reciprocal-rank fusion combines rank positions rather than raw scores.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable

_TOKEN_PATTERN = re.compile(r"[\w']+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase Unicode word tokens; supports both Latin and Indic scripts."""
    return [token.casefold() for token in _TOKEN_PATTERN.findall(str(text or ""))]


def _doc_text(document: Any) -> str:
    if isinstance(document, dict):
        return str(document.get("page_content", document.get("text", "")) or "")
    return str(getattr(document, "page_content", getattr(document, "text", "")) or "")


def _doc_id(document: Any, fallback_index: int) -> str:
    metadata = document.get("metadata", {}) if isinstance(document, dict) else getattr(document, "metadata", {})
    if isinstance(metadata, dict):
        chunk_index = metadata.get("chunk_index")
        if chunk_index is not None:
            return "chunk:" + str(chunk_index)
    return "text:" + str(fallback_index) + ":" + _doc_text(document)


def bm25_scores(query: str, documents: Iterable[Any], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Return BM25 scores in the same order as input documents."""
    docs = list(documents)
    if k1 <= 0 or not 0 <= b <= 1:
        raise ValueError("BM25 requires k1 > 0 and b in [0, 1]")
    query_terms = tokenize(query)
    if not docs or not query_terms:
        return [0.0] * len(docs)

    tokenized = [tokenize(_doc_text(doc)) for doc in docs]
    term_frequencies = [Counter(tokens) for tokens in tokenized]
    lengths = [len(tokens) for tokens in tokenized]
    average_length = sum(lengths) / len(lengths) if lengths else 0.0
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))

    query_frequency = Counter(query_terms)
    scores = []
    count = len(docs)
    for frequencies, length in zip(term_frequencies, lengths):
        score = 0.0
        for term, qf in query_frequency.items():
            tf = frequencies.get(term, 0)
            if not tf:
                continue
            df = document_frequency.get(term, 0)
            idf = math.log(1.0 + (count - df + 0.5) / (df + 0.5))
            norm = tf + k1 * (1.0 - b + b * (length / average_length if average_length else 0.0))
            score += idf * (tf * (k1 + 1.0) / norm) * qf
        scores.append(score)
    return scores


def reciprocal_rank_fusion(
    rankings: Iterable[Iterable[Any]],
    top_k: int = 4,
    rank_constant: int = 60,
) -> list[Any]:
    """Fuse ranked lists with RRF while deduplicating by chunk identity."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if rank_constant < 1:
        raise ValueError("rank_constant must be positive")

    fused: dict[str, float] = {}
    documents: dict[str, Any] = {}
    insertion_order: dict[str, int] = {}
    seen_order = 0

    for ranking in rankings:
        local_seen = set()
        for rank, document in enumerate(ranking, start=1):
            doc_id = _doc_id(document, rank)
            if doc_id in local_seen:
                continue
            local_seen.add(doc_id)
            if doc_id not in documents:
                documents[doc_id] = document
                insertion_order[doc_id] = seen_order
                seen_order += 1
                fused[doc_id] = 0.0
            fused[doc_id] += 1.0 / (rank_constant + rank)

    ordered_ids = sorted(
        fused,
        key=lambda doc_id: (-fused[doc_id], insertion_order[doc_id]),
    )
    return [documents[doc_id] for doc_id in ordered_ids[:top_k]]


def hybrid_rank_documents(
    query: str,
    dense_ranked_documents: Iterable[Any],
    corpus_documents: Iterable[Any],
    top_k: int = 4,
) -> list[Any]:
    """Fuse dense candidates with BM25's corpus-wide ranking."""
    dense = list(dense_ranked_documents)
    corpus = list(corpus_documents)
    scores = bm25_scores(query, corpus)
    lexical = [
        corpus[index]
        for index in sorted(range(len(corpus)), key=lambda i: (-scores[i], i))
        if scores[index] > 0
    ]
    if not lexical:
        return dense[:top_k]
    return reciprocal_rank_fusion([dense, lexical], top_k=top_k)
