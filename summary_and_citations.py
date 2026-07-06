"""Structured summary generation and citation extraction for academic papers."""

from __future__ import annotations

import re

from rag_pipeline import RAGPipeline, RAGPipelineError


class SummaryError(Exception):
    """Raised when summary or citation extraction fails."""


def generate_structured_summary(
    full_text: str,
    pipeline: RAGPipeline,
    api_key: str,
) -> str:
    """Generate a four-section structured summary of an academic paper.

    Sections: Research Question, Methodology, Key Findings, Limitations.

    Args:
        full_text: The complete extracted text of the paper.
        pipeline: An initialized RAGPipeline instance for LLM calls.
        api_key: Google Gemini API key.

    Returns:
        A formatted summary string with four labeled sections.

    Raises:
        SummaryError: If the text is empty or generation fails.
    """
    if not full_text or not full_text.strip():
        raise SummaryError("No text available to summarize.")

    truncated = full_text[:12000]

    prompt = (
        "You are a scholarly research assistant. Read the following academic paper "
        "text and produce a structured summary with exactly these four sections. "
        "Use the exact section headings shown below.\n\n"
        "## Research Question\n"
        "(What problem or question does this paper address?)\n\n"
        "## Methodology\n"
        "(What methods, data, or experimental design did the authors use?)\n\n"
        "## Key Findings\n"
        "(What are the main results and contributions?)\n\n"
        "## Limitations\n"
        "(What limitations or caveats do the authors acknowledge, or that you can infer?)\n\n"
        "Be concise but informative. Base your summary only on the provided text.\n\n"
        f"Paper text:\n{truncated}"
    )

    try:
        return pipeline.generate_with_prompt(prompt, api_key, max_tokens=2048)
    except RAGPipelineError as exc:
        raise SummaryError(str(exc)) from exc


def extract_references_section(full_text: str) -> str:
    """Extract the References or Bibliography section from paper text via regex.

    Args:
        full_text: The complete extracted text of the paper.

    Returns:
        Raw reference section text, or an empty string if not found.
    """
    patterns = [
        r"(?:^|\n)\s*(?:REFERENCES|References|BIBLIOGRAPHY|Bibliography|Works Cited|WORKS CITED)\s*\n",
        r"(?:^|\n)\s*(?:REFERENCES|References|BIBLIOGRAPHY|Bibliography)\s*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, full_text, re.MULTILINE)
        if match:
            start = match.end()
            end_patterns = [
                r"\n\s*(?:APPENDIX|Appendix|ACKNOWLEDGMENTS|Acknowledgments|SUPPLEMENTARY|Supplementary)\b",
            ]
            end = len(full_text)
            for end_pat in end_patterns:
                end_match = re.search(end_pat, full_text[start:], re.MULTILINE)
                if end_match:
                    end = start + end_match.start()
                    break
            return full_text[start:end].strip()

    return ""


def format_citations(
    full_text: str,
    pipeline: RAGPipeline,
    api_key: str,
) -> str:
    """Extract and format the paper's reference list.

    Uses regex to locate the References section, then LLM-assisted cleanup
    if the raw extraction is messy.

    Args:
        full_text: The complete extracted text of the paper.
        pipeline: An initialized RAGPipeline instance for LLM calls.
        api_key: Google Gemini API key.

    Returns:
        A clean, formatted list of citations.

    Raises:
        SummaryError: If no references can be found or formatting fails.
    """
    if not full_text or not full_text.strip():
        raise SummaryError("No text available for citation extraction.")

    raw_refs = extract_references_section(full_text)

    if not raw_refs:
        raise SummaryError(
            "Could not locate a References or Bibliography section in this paper."
        )

    if len(raw_refs) > 8000:
        raw_refs = raw_refs[:8000]

    prompt = (
        "You are a scholarly research assistant. The following text was extracted "
        "from the References section of an academic paper. It may contain formatting "
        "artifacts from PDF extraction (broken lines, missing spaces, etc.).\n\n"
        "Clean up and format this into a numbered list of citations. "
        "Each citation should be on its own line, numbered sequentially (1., 2., 3., ...). "
        "Preserve the original citation content as faithfully as possible. "
        "Do not invent citations that are not in the source text.\n\n"
        f"Raw references text:\n{raw_refs}"
    )

    try:
        return pipeline.generate_with_prompt(prompt, api_key, max_tokens=4096)
    except RAGPipelineError as exc:
        raise SummaryError(str(exc)) from exc
