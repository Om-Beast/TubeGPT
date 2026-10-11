#!/usr/bin/env python3
"""Compare dense FAISS retrieval with hybrid dense + BM25 retrieval.

Requires GOOGLE_API_KEY and installed project requirements. The bundled dataset
is a small smoke set: replace or extend it with manually labelled passages and
questions from representative videos before making resume-level claims.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hybrid_retrieval import hybrid_rank_documents
from retrieval_metrics import aggregate_retrieval_metrics


def load_dataset(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not data.get("documents") or not data.get("cases"):
        raise ValueError("Dataset must include non-empty documents and cases.")
    valid_ids = {str(document["id"]) for document in data["documents"]}
    for case in data["cases"]:
        if not case.get("question") or not case.get("gold_chunk_ids"):
            raise ValueError("Every case needs a question and at least one gold_chunk_id.")
        unknown = {str(item) for item in case["gold_chunk_ids"]} - valid_ids
        if unknown:
            raise ValueError("Unknown gold chunk IDs in " + str(case.get("id")) + ": " + ", ".join(sorted(unknown)))
    return data


def chunk_ids(documents) -> list[str]:
    return [str(document.metadata["chunk_index"]) for document in documents]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        default=str(ROOT / "data" / "evaluation" / "retrieval_gold.json"),
        help="JSON dataset with documents and manually labelled gold chunk IDs",
    )
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--candidate-k", type=int, default=12)
    parser.add_argument("--embedding-model", default=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2"))
    parser.add_argument("--output", help="Optional path to save the full JSON report")
    args = parser.parse_args()

    if args.top_k < 1 or args.candidate_k < args.top_k:
        parser.error("--top-k must be positive and --candidate-k must be >= --top-k")
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: Set GOOGLE_API_KEY before running the embedding-backed evaluation.", file=sys.stderr)
        return 2

    dataset = load_dataset(Path(args.dataset))
    docs = [
        Document(
            page_content=str(item["text"]),
            metadata={"chunk_index": str(item["id"]), "timestamp_available": False},
        )
        for item in dataset["documents"]
    ]
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=args.embedding_model)
        vector_store = FAISS.from_documents(docs, embeddings)
        dense_cases = []
        hybrid_cases = []
        per_case = []

        for case in dataset["cases"]:
            dense_results = vector_store.similarity_search_with_score(
                case["question"],
                k=min(args.candidate_k, len(docs)),
            )
            dense_ranked = [document for document, _score in dense_results]
            hybrid_ranked = hybrid_rank_documents(
                case["question"],
                dense_ranked,
                docs,
                top_k=args.top_k,
            )
            gold = [str(item) for item in case["gold_chunk_ids"]]
            dense_ids = chunk_ids(dense_ranked[:args.top_k])
            hybrid_ids = chunk_ids(hybrid_ranked[:args.top_k])
            dense_cases.append({"retrieved_chunk_ids": dense_ids, "gold_chunk_ids": gold})
            hybrid_cases.append({"retrieved_chunk_ids": hybrid_ids, "gold_chunk_ids": gold})
            per_case.append({
                "case_id": case.get("id"),
                "question": case["question"],
                "gold_chunk_ids": gold,
                "dense_top_k": dense_ids,
                "hybrid_top_k": hybrid_ids,
            })

        report = {
            "dataset": dataset.get("title", Path(args.dataset).name),
            "dataset_version": dataset.get("version"),
            "dataset_note": dataset.get("description", ""),
            "embedding_model": args.embedding_model,
            "candidate_k": args.candidate_k,
            "top_k": args.top_k,
            "document_count": len(docs),
            "question_count": len(dataset["cases"]),
            "dense": aggregate_retrieval_metrics(dense_cases, k=args.top_k),
            "hybrid": aggregate_retrieval_metrics(hybrid_cases, k=args.top_k),
            "per_case": per_case,
            "interpretation": (
                "This small hand-labelled smoke set only compares retrieval ranking on the bundled corpus. "
                "It does not measure answer factuality, citation entailment, production traffic, or general YouTube performance."
            ),
        }
    except Exception as exc:
        print("ERROR: Evaluation failed: " + str(exc), file=sys.stderr)
        return 1

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
