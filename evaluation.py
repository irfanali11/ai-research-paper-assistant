"""Research-grade retrieval evaluation for Scholar.

Run basic retrieval evaluation:
    python evaluation.py path/to/paper.pdf

Run with end-to-end faithfulness testing:
    python evaluation.py path/to/paper.pdf --faithfulness

Save benchmark to JSON:
    python evaluation.py path/to/paper.pdf --faithfulness --output benchmark.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pdf_processor import (
    PDFProcessingError,
    chunk_text,
    extract_text_from_pdf,
)
from rag_pipeline import (
    RAGPipeline,
    RAGPipelineError,
)


# ------------------------------------------------------------------
# Test suite
# ------------------------------------------------------------------

DEFAULT_TEST_CASES: list[dict[str, object]] = [
    {
        "question": "What is the main research question or problem?",
        "expected_keywords": [
            "research",
            "problem",
            "question",
            "objective",
        ],
        "expected_sections": [
            "abstract",
            "introduction",
            "conclusion",
        ],
    },
    {
        "question": "What methodology or methods were used?",
        "expected_keywords": [
            "method",
            "data",
            "analysis",
            "approach",
        ],
        "expected_sections": [
            "methodology",
            "methods",
            "materials and methods",
            "experiments",
        ],
    },
    {
        "question": "What are the key findings or results?",
        "expected_keywords": [
            "result",
            "finding",
            "performance",
            "accuracy",
        ],
        "expected_sections": [
            "results",
            "findings",
            "discussion",
            "conclusion",
        ],
    },
    {
        "question": "What limitations does the paper mention?",
        "expected_keywords": [
            "limitation",
            "constraint",
            "challenge",
            "future",
        ],
        "expected_sections": [
            "limitations",
            "discussion",
            "future work",
            "conclusion",
        ],
    },
    {
        "question": "What is the conclusion of the paper?",
        "expected_keywords": [
            "conclusion",
            "contribution",
            "propose",
            "demonstrate",
        ],
        "expected_sections": [
            "conclusion",
            "discussion",
        ],
    },
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _extract_section_from_chunk(chunk: str) -> str:
    """Extract the section label from a chunk."""
    import re
    match = re.match(r"\[Section:\s*(.*?)\]\s*", chunk, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    return "unknown"


def _run_retrieval_test(
    pipeline: RAGPipeline,
    test_cases: list[dict[str, object]],
    top_k: int = 5,
) -> list[dict[str, object]]:
    """Run retrieval-only evaluation."""
    results = pipeline.evaluate_retrieval(test_cases, top_k=top_k)
    enriched = []

    for case, result in zip(test_cases, results):
        # Extract section names from retrieved chunks
        retrieved_sections = []
        for idx in result["retrieved_sections"]:
            if 1 <= idx <= len(pipeline._chunks):
                section = _extract_section_from_chunk(pipeline._chunks[idx - 1])
                retrieved_sections.append(section)

        # Check if any expected section was retrieved
        expected = [s.lower() for s in case.get("expected_sections", [])]
        section_hit = any(s in retrieved_sections for s in expected)

        # Get cross-encoder scores if available
        # We re-run retrieve to get scores (lightweight, no LLM)
        try:
            detailed = pipeline.retrieve(case["question"], top_k=top_k)
            rerank_scores = detailed.scores if hasattr(detailed, "scores") else None
            avg_score = sum(rerank_scores) / len(rerank_scores) if rerank_scores else 0.0
        except Exception:
            avg_score = 0.0
            rerank_scores = None

        enriched.append(
            {
                "question": result["question"],
                "retrieved_indices": result["retrieved_sections"],
                "retrieved_sections": retrieved_sections,
                "keyword_hit_rate": result["hit_rate"],
                "keywords_found": result["keywords_found"],
                "expected_sections": expected,
                "section_match": section_hit,
                "avg_rerank_score": round(avg_score, 4),
                "rerank_scores": rerank_scores,
            }
        )

    return enriched


def _run_faithfulness_test(
    pipeline: RAGPipeline,
    test_cases: list[dict[str, object]],
    api_key: str,
    top_k: int = 5,
) -> list[dict[str, object]]:
    """Run end-to-end faithfulness evaluation."""
    results = []

    for case in test_cases:
        question = case["question"]

        try:
            result = pipeline.answer_question(
                question,
                api_key,
                top_k=top_k,
                evaluate=True,
            )

            faith = result.faithfulness or {}
            results.append(
                {
                    "question": question,
                    "answer": result.answer[:500],  # Truncate for display
                    "faithfulness_score": faith.get("faithfulness_score", 0.0),
                    "total_claims": faith.get("total_claims", 0),
                    "supported_claims": faith.get("supported_claims", 0),
                    "unsupported_claims": faith.get("unsupported_claims", []),
                    "explanation": faith.get("explanation", ""),
                }
            )

        except RAGPipelineError as exc:
            results.append(
                {
                    "question": question,
                    "answer": f"ERROR: {exc}",
                    "faithfulness_score": 0.0,
                    "total_claims": 0,
                    "supported_claims": 0,
                    "unsupported_claims": [],
                    "explanation": str(exc),
                }
            )

    return results


def _print_retrieval_report(retrieval_results: list[dict[str, object]]) -> None:
    """Print a formatted retrieval evaluation report."""
    print("\n" + "=" * 70)
    print("RETRIEVAL EVALUATION REPORT")
    print("=" * 70)

    total_keyword_rate = 0.0
    section_successes = 0
    total_rerank_score = 0.0

    for number, result in enumerate(retrieval_results, start=1):
        hit_rate = float(result["keyword_hit_rate"])
        total_keyword_rate += hit_rate

        if result["section_match"]:
            section_successes += 1

        total_rerank_score += result.get("avg_rerank_score", 0.0)

        print(f"\nQ{number}: {result['question']}")
        print(f"  Retrieved chunks: {result['retrieved_indices']}")
        print(f"  Retrieved sections: {result['retrieved_sections']}")
        print(f"  Keyword hit rate: {hit_rate:.0%}")
        print(f"  Keywords found: {result['keywords_found']}")
        print(f"  Section signal: {'PASS' if result['section_match'] else 'WEAK'}")
        print(f"  Avg rerank score: {result.get('avg_rerank_score', 0.0):.4f}")

    avg_keyword_rate = total_keyword_rate / len(retrieval_results) if retrieval_results else 0.0
    section_rate = section_successes / len(retrieval_results) if retrieval_results else 0.0
    avg_rerank = total_rerank_score / len(retrieval_results) if retrieval_results else 0.0

    print("\n" + "=" * 70)
    print(f"Average keyword hit rate:  {avg_keyword_rate:.0%}")
    print(f"Section retrieval success: {section_rate:.0%}")
    print(f"Average rerank score:      {avg_rerank:.4f}")
    print("=" * 70)


def _print_faithfulness_report(faithfulness_results: list[dict[str, object]]) -> None:
    """Print a formatted faithfulness evaluation report."""
    print("\n" + "=" * 70)
    print("END-TO-END FAITHFULNESS REPORT")
    print("=" * 70)

    total_score = 0.0
    total_claims = 0
    total_supported = 0

    for number, result in enumerate(faithfulness_results, start=1):
        score = result.get("faithfulness_score", 0.0)
        claims = result.get("total_claims", 0)
        supported = result.get("supported_claims", 0)

        total_score += score
        total_claims += claims
        total_supported += supported

        print(f"\nQ{number}: {result['question']}")
        print(f"  Faithfulness: {score:.0%} ({supported}/{claims} claims)")
        print(f"  Answer preview: {result['answer'][:120]}...")

        unsupported = result.get("unsupported_claims", [])
        if unsupported:
            for claim in unsupported:
                print(f"  \u26a0\ufe0f  Unsupported: {claim}")

    avg_score = total_score / len(faithfulness_results) if faithfulness_results else 0.0

    print("\n" + "=" * 70)
    print(f"Average faithfulness score: {avg_score:.0%}")
    print(f"Total claims evaluated:     {total_claims}")
    print(f"Total supported claims:     {total_supported}")
    print("=" * 70)


def _save_benchmark(
    retrieval_results: list[dict[str, object]],
    faithfulness_results: list[dict[str, object]] | None,
    output_path: str,
) -> None:
    """Save evaluation results to JSON for benchmarking."""
    benchmark = {
        "system": "Scholar Advanced",
        "pipeline_features": [
            "dense_semantic_retrieval",
            "sparse_bm25_retrieval",
            "reciprocal_rank_fusion",
            "cross_encoder_reranking",
            "diversity_filter",
            "section_aware_chunking",
            "table_extraction",
            "multi_document_support",
        ],
        "retrieval_evaluation": retrieval_results,
    }

    if faithfulness_results is not None:
        benchmark["faithfulness_evaluation"] = faithfulness_results
        # Calculate aggregate metrics
        scores = [r["faithfulness_score"] for r in faithfulness_results]
        benchmark["aggregate_faithfulness"] = {
            "average_score": sum(scores) / len(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
        }

    # Clean non-serializable items
    for r in retrieval_results:
        r.pop("rerank_scores", None)  # Remove list of floats for cleaner JSON

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)

    print(f"\nBenchmark saved to: {output_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run_evaluation(
    pdf_path: str,
    faithfulness: bool = False,
    output: str | None = None,
) -> None:
    """Index a PDF and evaluate retrieval quality."""
    print("Reading PDF...")

    try:
        text = extract_text_from_pdf(pdf_path)
    except PDFProcessingError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    chunks = chunk_text(text)

    print(f"Indexed {len(chunks)} chunks from: {pdf_path}\n")

    pipeline = RAGPipeline()
    print("Generating local embeddings and building indices...")

    # CHANGED: index_chunks() now requires a document_id and
    # document_name since the pipeline supports multiple documents
    # in one collection. A single-shot CLI eval run just needs a
    # stable id, so the file path/name is reused for both.
    doc_name = Path(pdf_path).name
    pipeline.index_chunks(chunks, document_id=doc_name, document_name=doc_name)

    # --- Retrieval evaluation (always runs, no API cost) ---
    retrieval_results = _run_retrieval_test(
        pipeline,
        DEFAULT_TEST_CASES,
        top_k=5,
    )
    _print_retrieval_report(retrieval_results)

    # --- Faithfulness evaluation (optional, uses Gemini) ---
    faithfulness_results = None

    if faithfulness:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()

        if not api_key:
            print(
                "\n\u26a0\ufe0f  GEMINI_API_KEY not set. "
                "Skipping faithfulness evaluation."
            )
            print("Set it with: export GEMINI_API_KEY='your-key'")
        else:
            print("\nRunning end-to-end faithfulness evaluation...")
            print("(This uses 1 Gemini call per test case)\n")

            faithfulness_results = _run_faithfulness_test(
                pipeline,
                DEFAULT_TEST_CASES,
                api_key,
                top_k=5,
            )
            _print_faithfulness_report(faithfulness_results)

    # --- Save benchmark ---
    if output:
        _save_benchmark(
            retrieval_results,
            faithfulness_results,
            output,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Scholar RAG pipeline on a research paper."
    )
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument(
        "--faithfulness",
        action="store_true",
        help="Run end-to-end faithfulness evaluation (requires GEMINI_API_KEY)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Save benchmark results to JSON file",
    )

    args = parser.parse_args()

    run_evaluation(
        pdf_path=args.pdf,
        faithfulness=args.faithfulness,
        output=args.output,
    )