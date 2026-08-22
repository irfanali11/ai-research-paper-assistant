"""Lightweight retrieval evaluation for Scholar.

Run:

    python evaluation.py path/to/paper.pdf
"""

from __future__ import annotations

import sys

from pdf_processor import (
    chunk_text,
    extract_text_from_pdf,
)
from rag_pipeline import RAGPipeline


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


def run_evaluation(
    pdf_path: str,
) -> None:
    """Index a PDF and evaluate retrieval quality."""
    print("Reading PDF...")

    text = extract_text_from_pdf(
        pdf_path
    )

    chunks = chunk_text(
        text
    )

    print(
        f"Indexed {len(chunks)} chunks from: "
        f"{pdf_path}\n"
    )

    pipeline = RAGPipeline()

    print("Generating local embeddings...")
    pipeline.index_chunks(
        chunks
    )

    print("\n" + "=" * 70)

    results = pipeline.evaluate_retrieval(
        DEFAULT_TEST_CASES,
        top_k=5,
    )

    total_keyword_rate = 0.0
    section_successes = 0

    for number, (
        case,
        result,
    ) in enumerate(
        zip(
            DEFAULT_TEST_CASES,
            results,
        ),
        start=1,
    ):
        retrieved = [
            str(section)
            for section in result[
                "retrieved_sections"
            ]
        ]

        # Extract section names from retrieved chunks.
        retrieved_section_names = []

        for chunk in pipeline._chunks:
            if chunk not in result["retrieved_sections"]:
                continue

        expected_sections = [
            str(section).lower()
            for section in case.get(
                "expected_sections",
                [],
            )
        ]

        retrieved_chunks = [
            pipeline._chunks[index - 1]
            for index in result[
                "retrieved_sections"
            ]
            if 0 < index <= len(
                pipeline._chunks
            )
        ]

        section_text = " ".join(
            retrieved_chunks
        ).lower()

        section_hit = any(
            expected in section_text
            for expected in expected_sections
        )

        if section_hit:
            section_successes += 1

        hit_rate = float(
            result["hit_rate"]
        )

        total_keyword_rate += hit_rate

        print(
            f"\nQ{number}: "
            f"{result['question']}"
        )

        print(
            "  Retrieved chunks: "
            f"{result['retrieved_sections']}"
        )

        print(
            "  Keyword hit rate: "
            f"{hit_rate:.0%}"
        )

        print(
            "  Keywords found: "
            f"{result['keywords_found']}"
        )

        print(
            "  Expected section signal: "
            f"{'PASS' if section_hit else 'WEAK'}"
        )

    average_keyword_rate = (
        total_keyword_rate / len(results)
        if results
        else 0.0
    )

    section_rate = (
        section_successes / len(results)
        if results
        else 0.0
    )

    print("\n" + "=" * 70)

    print(
        f"Average keyword hit rate: "
        f"{average_keyword_rate:.0%}"
    )

    print(
        f"Section retrieval success: "
        f"{section_rate:.0%}"
    )

    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python evaluation.py "
            "<path-to-pdf>"
        )
        sys.exit(1)

    run_evaluation(
        sys.argv[1]
    )
