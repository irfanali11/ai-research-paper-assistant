"""Research-grade evaluation for Scholar.

Run basic retrieval evaluation on one paper:
    python evaluation.py papers/paper1.pdf

Run on multiple papers at once, indexed into the same shared pipeline
(exercises multi-document scoping — each paper is evaluated against
ONLY its own chunks, the way Scholar's chat scope picker works):
    python evaluation.py papers/paper1.pdf papers/paper2.pdf

Run with end-to-end faithfulness testing (uses Gemini):
    python evaluation.py papers/paper1.pdf --faithfulness

Run an ablation comparing query decomposition ON vs OFF on compound
questions (uses Gemini for the decomposition step itself):
    python evaluation.py papers/paper1.pdf --decompose-ablation

Save everything to JSON for a report/README:
    python evaluation.py papers/paper1.pdf --faithfulness --decompose-ablation --output benchmark.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
# Test suites
# ------------------------------------------------------------------

# Simple, single-intent questions — used for the baseline retrieval
# and faithfulness checks. Deliberately generic so they apply to any
# academic paper, regardless of field.
DEFAULT_TEST_CASES: list[dict[str, object]] = [
    {
        "question": "What is the main research question or problem?",
        "expected_keywords": ["research", "problem", "question", "objective"],
        "expected_sections": ["abstract", "introduction", "conclusion"],
    },
    {
        "question": "What methodology or methods were used?",
        "expected_keywords": ["method", "data", "analysis", "approach"],
        "expected_sections": [
            "methodology", "methods", "materials and methods", "experiments",
        ],
    },
    {
        "question": "What are the key findings or results?",
        "expected_keywords": ["result", "finding", "performance", "accuracy"],
        "expected_sections": ["results", "findings", "discussion", "conclusion"],
    },
    {
        "question": "What limitations does the paper mention?",
        "expected_keywords": ["limitation", "constraint", "challenge", "future"],
        "expected_sections": [
            "limitations", "discussion", "future work", "conclusion",
        ],
    },
    {
        "question": "What is the conclusion of the paper?",
        "expected_keywords": ["conclusion", "contribution", "propose", "demonstrate"],
        "expected_sections": ["conclusion", "discussion"],
    },
]

# NEW: deliberately COMPOUND questions — bundling two distinct intents
# into one question. These are the cases query decomposition exists
# for. Used only by --decompose-ablation, comparing retrieval with
# decomposition on vs. off on the exact same questions.
COMPOUND_TEST_CASES: list[dict[str, object]] = [
    {
        "question": (
            "What methodology did the authors use, and what limitations "
            "did they mention?"
        ),
        "expected_keywords": [
            "method", "data", "analysis", "limitation", "constraint", "future",
        ],
    },
    {
        "question": (
            "What are the key findings, and how do they relate to the "
            "paper's stated research question?"
        ),
        "expected_keywords": [
            "result", "finding", "research", "question", "objective",
        ],
    },
    {
        "question": (
            "What data or methods were used, and what conclusions did "
            "the authors draw from them?"
        ),
        "expected_keywords": [
            "method", "data", "conclusion", "propose", "demonstrate",
        ],
    },
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _extract_section_from_chunk(chunk: str) -> str:
    """Extract the section label from a chunk."""
    match = re.match(r"\[Section:\s*(.*?)\]\s*", chunk, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    return "unknown"


def _load_and_index_documents(
    pipeline: RAGPipeline,
    pdf_paths: list[str],
) -> dict[str, str]:
    """Extract, chunk, and index each PDF into the shared pipeline.

    Returns a dict of document_id -> display name, in the same shape
    Scholar's own app.py session state uses, so this script exercises
    the real multi-document code path rather than a simplified stand-in.
    """
    doc_map: dict[str, str] = {}

    for pdf_path in pdf_paths:
        path = Path(pdf_path)
        print(f"Reading PDF: {path.name} ...")

        try:
            text = extract_text_from_pdf(str(path))
        except PDFProcessingError as exc:
            print(f"  Error reading {path.name}: {exc}")
            continue

        chunks = chunk_text(text)

        if not chunks:
            print(f"  Warning: no usable chunks extracted from {path.name}, skipping.")
            continue

        document_id = f"{path.name}:{path.stat().st_size}"
        pipeline.index_chunks(chunks, document_id=document_id, document_name=path.name)
        doc_map[document_id] = path.name

        print(f"  Indexed {len(chunks)} chunks from {path.name}")

    return doc_map


# ------------------------------------------------------------------
# Retrieval evaluation (per document, scoped via doc_ids — exercises
# the same multi-document filtering Scholar's chat scope picker uses)
# ------------------------------------------------------------------

def _run_retrieval_test(
    pipeline: RAGPipeline,
    test_cases: list[dict[str, object]],
    doc_id: str | None,
    top_k: int = 5,
) -> list[dict[str, object]]:
    """Run retrieval-only evaluation, optionally scoped to one document."""
    doc_ids = [doc_id] if doc_id else None

    results = pipeline.evaluate_retrieval(test_cases, top_k=top_k, doc_ids=doc_ids)
    enriched = []

    for case, result in zip(test_cases, results):
        retrieved_sections = []
        for idx in result["retrieved_sections"]:
            if 1 <= idx <= len(pipeline._chunks):
                section = _extract_section_from_chunk(pipeline._chunks[idx - 1])
                retrieved_sections.append(section)

        expected = [s.lower() for s in case.get("expected_sections", [])]
        section_hit = any(s in retrieved_sections for s in expected)

        try:
            detailed = pipeline.retrieve(case["question"], top_k=top_k, doc_ids=doc_ids)
            rerank_scores = detailed.scores
            # FIX: the cross-encoder returns NumPy float32 values, which
            # Python's json module cannot serialize. Cast to native
            # Python float here so _save_benchmark's json.dump doesn't
            # crash on TypeError: Object of type float32 is not JSON
            # serializable.
            rerank_scores = (
                [float(s) for s in rerank_scores] if rerank_scores else None
            )
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


def _print_retrieval_report(
    retrieval_results: list[dict[str, object]],
    label: str = "",
) -> dict[str, float]:
    """Print a formatted retrieval evaluation report. Returns summary metrics."""
    heading = f"RETRIEVAL EVALUATION REPORT{f' — {label}' if label else ''}"
    print("\n" + "=" * 70)
    print(heading)
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

    n = len(retrieval_results) or 1
    avg_keyword_rate = total_keyword_rate / n
    section_rate = section_successes / n
    avg_rerank = total_rerank_score / n

    print("\n" + "-" * 70)
    print(f"Average keyword hit rate:  {avg_keyword_rate:.0%}")
    print(f"Section retrieval success: {section_rate:.0%}")
    print(f"Average rerank score:      {avg_rerank:.4f}")
    print("=" * 70)

    return {
        "avg_keyword_hit_rate": avg_keyword_rate,
        "section_retrieval_success": section_rate,
        "avg_rerank_score": avg_rerank,
    }


# ------------------------------------------------------------------
# NEW: Query decomposition ablation — compares retrieval quality on
# compound questions WITH decomposition vs WITHOUT it, on the exact
# same questions. This is the "before/after" evidence for the feature.
# ------------------------------------------------------------------

def _run_decomposition_ablation(
    pipeline: RAGPipeline,
    doc_id: str | None,
    api_key: str,
    top_k: int = 5,
) -> dict[str, object]:
    """Run compound questions with decomposition on vs off, compare hit rates."""
    doc_ids = [doc_id] if doc_id else None

    without_results = pipeline.evaluate_retrieval(
        COMPOUND_TEST_CASES, top_k=top_k, doc_ids=doc_ids,
        decompose=False,
    )
    with_results = pipeline.evaluate_retrieval(
        COMPOUND_TEST_CASES, top_k=top_k, doc_ids=doc_ids,
        decompose=True, api_key=api_key,
    )

    print("\n" + "=" * 70)
    print("QUERY DECOMPOSITION ABLATION (compound questions only)")
    print("=" * 70)

    rows = []
    for case, without_r, with_r in zip(COMPOUND_TEST_CASES, without_results, with_results):
        print(f"\nQ: {case['question']}")
        print(f"  Without decomposition — hit rate: {without_r['hit_rate']:.0%}, "
              f"keywords found: {without_r['keywords_found']}")
        print(f"  With decomposition    — hit rate: {with_r['hit_rate']:.0%}, "
              f"sub-questions: {with_r.get('sub_question_count', 1)}, "
              f"keywords found: {with_r['keywords_found']}")

        rows.append({
            "question": case["question"],
            "hit_rate_without_decomposition": without_r["hit_rate"],
            "hit_rate_with_decomposition": with_r["hit_rate"],
            "sub_question_count": with_r.get("sub_question_count", 1),
        })

    avg_without = sum(r["hit_rate_without_decomposition"] for r in rows) / len(rows)
    avg_with = sum(r["hit_rate_with_decomposition"] for r in rows) / len(rows)

    print("\n" + "-" * 70)
    print(f"Average hit rate WITHOUT decomposition: {avg_without:.0%}")
    print(f"Average hit rate WITH decomposition:    {avg_with:.0%}")
    delta = avg_with - avg_without
    print(f"Delta: {delta:+.0%}")
    print("=" * 70)

    return {
        "per_question": rows,
        "avg_hit_rate_without_decomposition": avg_without,
        "avg_hit_rate_with_decomposition": avg_with,
        "delta": delta,
    }


# ------------------------------------------------------------------
# Faithfulness evaluation (end-to-end, uses Gemini)
# ------------------------------------------------------------------

def _run_faithfulness_test(
    pipeline: RAGPipeline,
    test_cases: list[dict[str, object]],
    api_key: str,
    doc_id: str | None,
    top_k: int = 5,
) -> list[dict[str, object]]:
    """Run end-to-end faithfulness evaluation, optionally scoped to one doc."""
    doc_ids = [doc_id] if doc_id else None
    results = []

    for case in test_cases:
        question = case["question"]

        try:
            result = pipeline.answer_question(
                question, api_key, top_k=top_k, evaluate=True, doc_ids=doc_ids,
            )

            faith = result.faithfulness or {}
            results.append(
                {
                    "question": question,
                    "answer": result.answer[:500],
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


def _print_faithfulness_report(
    faithfulness_results: list[dict[str, object]],
    label: str = "",
) -> dict[str, float]:
    """Print a formatted faithfulness evaluation report. Returns summary metrics."""
    heading = f"END-TO-END FAITHFULNESS REPORT{f' — {label}' if label else ''}"
    print("\n" + "=" * 70)
    print(heading)
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

        for claim in result.get("unsupported_claims", []):
            print(f"  \u26a0\ufe0f  Unsupported: {claim}")

    n = len(faithfulness_results) or 1
    avg_score = total_score / n

    print("\n" + "-" * 70)
    print(f"Average faithfulness score: {avg_score:.0%}")
    print(f"Total claims evaluated:     {total_claims}")
    print(f"Total supported claims:     {total_supported}")
    print("=" * 70)

    return {
        "avg_faithfulness_score": avg_score,
        "total_claims_evaluated": total_claims,
        "total_supported_claims": total_supported,
    }


# ------------------------------------------------------------------
# Benchmark export
# ------------------------------------------------------------------

def _save_benchmark(
    per_document_results: dict[str, dict[str, object]],
    decomposition_ablation: dict[str, object] | None,
    output_path: str,
) -> None:
    """Save all evaluation results to JSON for a README/report."""
    benchmark = {
        "system": "Scholar Advanced RAG",
        "pipeline_features": [
            "dense_semantic_retrieval",
            "sparse_bm25_retrieval",
            "reciprocal_rank_fusion",
            "cross_encoder_reranking",
            "diversity_filter",
            "section_aware_chunking",
            "table_extraction",
            "multi_document_support",
            "query_decomposition",
            "faithfulness_evaluation",
        ],
        "per_document": per_document_results,
    }

    if decomposition_ablation is not None:
        benchmark["decomposition_ablation"] = decomposition_ablation

    def _json_safe(obj):
        """Fallback for json.dump: converts NumPy scalar types (which
        aren't natively JSON-serializable) to plain Python numbers,
        so a stray float32/int64 anywhere in the results doesn't crash
        the whole save. Raises for anything genuinely unexpected."""
        if hasattr(obj, "item"):  # NumPy scalar types implement .item()
            return obj.item()
        raise TypeError(
            f"Object of type {obj.__class__.__name__} is not JSON serializable"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False, default=_json_safe)

    print(f"\nBenchmark saved to: {output_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run_evaluation(
    pdf_paths: list[str],
    faithfulness: bool = False,
    decompose_ablation: bool = False,
    output: str | None = None,
) -> None:
    """Index one or more PDFs into a shared pipeline and evaluate quality.

    When multiple PDFs are given, each is evaluated independently via
    doc_ids scoping — this exercises Scholar's real multi-document
    filtering, not just a simplified single-doc stand-in.
    """
    pipeline = RAGPipeline()
    print("Loading local embedding + reranking models (first run downloads them)...")

    doc_map = _load_and_index_documents(pipeline, pdf_paths)

    if not doc_map:
        print("No documents were successfully indexed. Exiting.")
        sys.exit(1)

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if (faithfulness or decompose_ablation) and not api_key:
        print(
            "\n\u26a0\ufe0f  GEMINI_API_KEY not set — faithfulness and/or "
            "decomposition ablation will be skipped.\n"
            "Set it with: export GEMINI_API_KEY='your-key'"
        )

    per_document_results: dict[str, dict[str, object]] = {}
    decomposition_ablation_result = None

    for document_id, name in doc_map.items():
        print(f"\n{'#' * 70}\n# Evaluating: {name}\n{'#' * 70}")

        # Scope to this one document only when multiple are loaded, so
        # results reflect THAT paper's retrieval quality specifically —
        # same behavior as selecting one paper in Scholar's chat scope.
        doc_id_filter = document_id if len(doc_map) > 1 else None

        retrieval_results = _run_retrieval_test(
            pipeline, DEFAULT_TEST_CASES, doc_id=doc_id_filter, top_k=5,
        )
        retrieval_summary = _print_retrieval_report(retrieval_results, label=name)

        doc_report: dict[str, object] = {
            "document_name": name,
            "retrieval_evaluation": retrieval_results,
            "retrieval_summary": retrieval_summary,
        }

        if faithfulness and api_key:
            print(f"\nRunning faithfulness evaluation for {name} "
                  f"(1 Gemini call per test case)...")
            faithfulness_results = _run_faithfulness_test(
                pipeline, DEFAULT_TEST_CASES, api_key, doc_id=doc_id_filter, top_k=5,
            )
            faithfulness_summary = _print_faithfulness_report(
                faithfulness_results, label=name,
            )
            doc_report["faithfulness_evaluation"] = faithfulness_results
            doc_report["faithfulness_summary"] = faithfulness_summary

        per_document_results[document_id] = doc_report

    # Decomposition ablation runs once, against the first loaded
    # document (compound questions are generic enough not to need
    # per-document repetition for a demonstrative comparison).
    if decompose_ablation and api_key:
        first_doc_id = next(iter(doc_map)) if len(doc_map) > 1 else None
        decomposition_ablation_result = _run_decomposition_ablation(
            pipeline, doc_id=first_doc_id, api_key=api_key, top_k=5,
        )

    if output:
        _save_benchmark(per_document_results, decomposition_ablation_result, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Scholar's RAG pipeline on one or more research papers."
    )
    parser.add_argument(
        "pdfs",
        nargs="+",
        help="Path(s) to one or more PDF files. Multiple PDFs are indexed "
             "together and evaluated with per-document scoping, exercising "
             "Scholar's multi-document support.",
    )
    parser.add_argument(
        "--faithfulness",
        action="store_true",
        help="Run end-to-end faithfulness evaluation (requires GEMINI_API_KEY).",
    )
    parser.add_argument(
        "--decompose-ablation",
        action="store_true",
        help="Compare retrieval quality on compound questions WITH vs WITHOUT "
             "query decomposition (requires GEMINI_API_KEY).",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Save all benchmark results to a JSON file.",
    )

    args = parser.parse_args()

    run_evaluation(
        pdf_paths=args.pdfs,
        faithfulness=args.faithfulness,
        decompose_ablation=args.decompose_ablation,
        output=args.output,
    )