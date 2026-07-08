"""Basic retrieval evaluation for the Scholar RAG pipeline.

Run from the scholar directory after indexing a document:
    python evaluation.py path/to/sample.pdf
"""

from __future__ import annotations

import sys

from pdf_processor import chunk_text, extract_text_from_pdf
from rag_pipeline import RAGPipeline

DEFAULT_TEST_CASES: list[dict[str, object]] = [
    {
        "question": "What is the main research question or problem?",
        "expected_keywords": ["research", "problem", "question", "study", "aim"],
    },
    {
        "question": "What methodology or methods were used?",
        "expected_keywords": ["method", "experiment", "data", "analysis", "approach"],
    },
    {
        "question": "What are the key findings or results?",
        "expected_keywords": ["result", "finding", "performance", "accuracy", "show"],
    },
    {
        "question": "What limitations does the paper mention?",
        "expected_keywords": ["limitation", "future", "constraint", "challenge"],
    },
    {
        "question": "What is the conclusion of the paper?",
        "expected_keywords": ["conclusion", "contribution", "demonstrate", "propose"],
    },
]


def run_evaluation(pdf_path: str) -> None:
    """Index a PDF and print retrieval evaluation results.

    Args:
        pdf_path: Path to a sample academic PDF.
    """
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    pipeline = RAGPipeline()
    pipeline.index_chunks(chunks)

    print(f"Indexed {len(chunks)} chunks from: {pdf_path}\n")
    print("=" * 60)

    results = pipeline.evaluate_retrieval(DEFAULT_TEST_CASES)

    total_hit_rate = 0.0
    for i, result in enumerate(results, 1):
        print(f"\nQ{i}: {result['question']}")
        print(f"  Retrieved sections: {result['retrieved_sections']}")
        print(f"  Keywords found: {result['keywords_found']}")
        print(f"  Hit rate: {result['hit_rate']:.0%}")
        total_hit_rate += float(result["hit_rate"])

    avg_hit_rate = total_hit_rate / len(results) if results else 0.0
    print("\n" + "=" * 60)
    print(f"Average keyword hit rate: {avg_hit_rate:.0%}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluation.py <path-to-pdf>")
        sys.exit(1)
    run_evaluation(sys.argv[1])
