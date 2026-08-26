"""Structured summaries and citation extraction for Scholar."""

from __future__ import annotations

import re
from typing import Any

from rag_pipeline import (
    RAGPipeline,
    RAGPipelineError,
    RetrievalResult,
)


class SummaryError(Exception):
    """Raised when summary or citation extraction fails."""


# ------------------------------------------------------------------
# Meta-text detection (NEW)
# ------------------------------------------------------------------
# NEW: Detects chunks that contain writing instructions, style guides,
# author guidelines, or generic academic advice rather than actual
# paper content. These are common PDF artifacts that poison RAG.

_META_TEXT_PATTERNS = [
    r"the introduction (section|paragraph) (should|must|needs to)",
    r"a (scientific|research|academic) paper (should|must|needs to)",
    r"how to (write|structure|organize) (a|an) (paper|essay|thesis)",
    r"author guidelines",
    r"submission instructions",
    r"manuscript preparation",
    r"this section should (contain|include|state)",
    r"your (abstract|introduction|conclusion) should",
    r"tips for writing",
    r"writing (advice|guidelines|standards)",
    r"formatting requirements",
    r"journal (policy|requirements|guidelines)",
    r"peer review (process|guidelines)",
    r"figure \d+ should (show|illustrate|demonstrate)",
    r"table \d+ should (contain|show|list)",
    r"keywords?:\s*(select|choose|include)",
    r"corresponding author",
    r"conflict of interest",
    r"ethical approval",
    r"informed consent",
    r"data availability statement",
    r"supplementary (material|data|information)",
    r"acknowledgments? \(optional\)",
    r"references? (should|must) be (formatted|styled)",
    r"apa|mla|chicago|ieee|harvard style",
]

_META_TEXT_REGEX = re.compile(
    "|".join(f"(?:{p})" for p in _META_TEXT_PATTERNS),
    re.IGNORECASE,
)


def _is_meta_text(chunk: str) -> bool:
    """Check if a chunk contains writing instructions or meta-text.

    These chunks appear in PDFs as author guidelines, submission
    instructions, or embedded style guides. They poison RAG because
    they contain academic vocabulary (research, introduction, aim)
    but describe generic paper structure, not the actual paper.
    """
    if not chunk or len(chunk) < 50:
        return True  # Too short to be substantive content

    # Check against meta-text patterns
    if _META_TEXT_REGEX.search(chunk):
        return True

    # Heuristic: chunks that are purely instructional often contain
    # many modal verbs (should, must, needs to) in the first 200 chars
    opening = chunk[:300].lower()
    modal_count = sum(opening.count(w) for w in ["should ", "must ", "needs to ", " ought to "])
    if modal_count >= 3:
        return True

    return False


# ------------------------------------------------------------------
# RAG-based summary context
# ------------------------------------------------------------------

_SUMMARY_SECTION_QUERIES: dict[str, str] = {
    "research_question": (
        "What is the main research question, problem, objective, "
        "or aim of this paper?"
    ),
    "methodology": (
        "What methodology, methods, experimental setup, data collection, "
        "approach, or algorithm was used?"
    ),
    "findings": (
        "What are the key findings, results, performance metrics, "
        "accuracy, outcomes, or discoveries?"
    ),
    "limitations": (
        "What limitations, constraints, weaknesses, challenges, "
        "or future work are mentioned?"
    ),
}


def _build_rag_summary_context(
    pipeline: RAGPipeline,
    max_chunks_per_section: int = 5,
) -> dict[str, str]:
    """Retrieve targeted chunks for each summary section via RAG.

    Uses the same hybrid retrieval pipeline (dense + sparse + RRF +
    cross-encoder re-ranking) that powers the chat interface.
    """
    contexts: dict[str, str] = {}

    for key, query in _SUMMARY_SECTION_QUERIES.items():
        try:
            result: RetrievalResult = pipeline.retrieve(
                query,
                top_k=max_chunks_per_section,
            )

            # NEW: Filter out meta-text / noise chunks
            clean_chunks: list[str] = []
            for i, chunk in enumerate(result.chunks, 1):
                if _is_meta_text(chunk):
                    continue  # Skip writing instructions and style guides

                section = _extract_section_from_chunk(chunk)
                clean_chunks.append(f"[Source {i} | Section: {section}]\n{chunk}")

            if clean_chunks:
                contexts[key] = "\n\n---\n\n".join(clean_chunks)
            else:
                contexts[key] = ""

        except Exception:
            contexts[key] = ""

    return contexts


def _extract_section_from_chunk(chunk: str) -> str:
    """Extract section label from a chunk for display."""
    match = re.match(r"\[Section:\s*(.*?)\]\s*", chunk, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "Document"


def _has_tables_in_context(context: str) -> bool:
    """Check if the retrieved context contains table data."""
    return "[Table" in context and "|" in context


# ------------------------------------------------------------------
# Legacy raw-text section extraction (kept as fallback)
# ------------------------------------------------------------------

def _extract_sections_for_summary(
    full_text: str,
    max_chars: int = 24000,
) -> str:
    """Select high-value academic sections locally."""
    if not full_text.strip():
        return ""

    section_pattern = re.compile(
        r"\[Page \d+\]"
        r"|(?<=\n)"
        r"(?:Abstract|Introduction|Related Work|"
        r"Literature Review|Background|Methodology|Methods|"
        r"Materials And Methods|Experimental Setup|Experiments|"
        r"Results|Findings|Discussion|Results And Discussion|"
        r"Conclusion|Conclusions|Limitations|Future Work)"
        r"\s*:?\s*\n",
        re.IGNORECASE,
    )

    matches = list(section_pattern.finditer(full_text))

    if len(matches) < 2:
        if len(full_text) <= max_chars:
            return full_text

        part = max_chars // 3
        return (
            full_text[:part]
            + "\n\n[...middle of document omitted...]\n\n"
            + full_text[
                len(full_text) // 2 - part // 2 :
                len(full_text) // 2 + part // 2
            ]
            + "\n\n[...]\n\n"
            + full_text[-part:]
        )

    sections: list[str] = []
    sections.append(full_text[: matches[0].start()].strip())

    for index, match in enumerate(matches):
        start = match.start()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(full_text)
        )
        block = full_text[start:end].strip()
        if block:
            sections.append(block)

    priority_keywords = [
        "abstract",
        "introduction",
        "methodology",
        "methods",
        "materials",
        "experiments",
        "results",
        "findings",
        "discussion",
        "conclusion",
        "limitations",
        "future work",
    ]

    prioritized: list[str] = []
    for keyword in priority_keywords:
        for section in sections:
            if keyword in section[:250].lower():
                if section not in prioritized:
                    prioritized.append(section)

    for section in sections:
        if section not in prioritized:
            prioritized.append(section)

    output_parts: list[str] = []
    current_length = 0

    for section in prioritized:
        if not section:
            continue

        remaining = max_chars - current_length
        if remaining <= 500:
            break

        if len(section) > remaining:
            section = section[:remaining]

        output_parts.append(section)
        current_length += len(section)

    return "\n\n".join(output_parts)


# ------------------------------------------------------------------
# Prompt builder (UPDATED with anti-meta-text hardening)
# ------------------------------------------------------------------

def _format_summary_prompt(
    rag_contexts: dict[str, str],
    fallback_text: str,
) -> str:
    """Build the summary prompt from RAG contexts or fallback text."""
    rag_success = sum(1 for v in rag_contexts.values() if v.strip())

    if rag_success >= 3:
        table_note = ""
        if any(_has_tables_in_context(v) for v in rag_contexts.values()):
            table_note = (
                "\nNote: The supplied material includes extracted tables. "
                "When summarizing findings, reference specific numerical "
                "results from tables where available."
            )

        # NEW: Strong anti-meta-text instruction
        prompt = f"""You are Scholar, an academic research assistant.

Create a structured summary of the supplied paper material.

Use EXACTLY these four headings:

## Research Question
## Methodology
## Key Findings
## Limitations

For each heading, use ONLY the source material provided under that heading.
Do not mix material across headings unless explicitly relevant.

CRITICAL ANTI-NOISE INSTRUCTIONS:

1. Some retrieved chunks may contain generic writing advice, author
   guidelines, style guides, or meta-instructions about how papers
   SHOULD be structured (e.g., "the introduction should state the aim...",
   "a scientific paper must include..."). These are NOT part of the
   actual paper content. IGNORE any chunk that reads like writing
   instructions, submission guidelines, or generic academic advice.

2. Only use chunks that describe the ACTUAL content, arguments,
   findings, or claims of THIS specific paper.

3. If the actual paper does not explicitly state a research question,
   say so clearly: "The paper does not explicitly state a research
   question. Instead, it addresses..." and then describe what the
   paper actually does.

4. Do not use outside knowledge.
5. Do not invent information.
6. If the paper does not explicitly state something, say that it
   is not clearly stated in the supplied material.
7. Do not confuse the paper's background with its research question.
8. Do not confuse proposed methods with actual findings.
9. For limitations, report limitations explicitly acknowledged by
   the authors. Do not present your own speculation as fact.
10. Be concise but academically useful.{table_note}

--- RESEARCH QUESTION CONTEXT ---
{rag_contexts.get("research_question", "No relevant sections retrieved.")}

--- METHODOLOGY CONTEXT ---
{rag_contexts.get("methodology", "No relevant sections retrieved.")}

--- KEY FINDINGS CONTEXT ---
{rag_contexts.get("findings", "No relevant sections retrieved.")}

--- LIMITATIONS CONTEXT ---
{rag_contexts.get("limitations", "No relevant sections retrieved.")}

SUMMARY:
""".strip()

    else:
        # Fallback: raw text extraction with same anti-noise instruction
        prompt = f"""You are Scholar, an academic research assistant.

Create a structured summary of the supplied paper material.

Use EXACTLY these four headings:

## Research Question
## Methodology
## Key Findings
## Limitations

CRITICAL ANTI-NOISE INSTRUCTIONS:

1. The supplied material may contain embedded writing instructions,
   author guidelines, or style guide text. IGNORE any text that reads
   like generic advice about how papers should be structured.
2. Only describe the ACTUAL content of THIS specific paper.
3. If the paper does not explicitly state a research question, say so
   and describe what the paper actually addresses instead.
4. Do not use outside knowledge.
5. Do not invent information.
6. Be concise but academically useful.

PAPER MATERIAL:

{fallback_text}

SUMMARY:
""".strip()

    return prompt


# ------------------------------------------------------------------
# Public: Structured Summary
# ------------------------------------------------------------------

def generate_structured_summary(
    full_text: str,
    pipeline: RAGPipeline,
    api_key: str,
) -> str:
    """Generate a grounded four-part academic summary."""
    if not full_text or not full_text.strip():
        raise SummaryError("No text available to summarize.")

    rag_contexts = _build_rag_summary_context(pipeline)
    fallback_text = _extract_sections_for_summary(
        full_text,
        max_chars=24000,
    )

    if not fallback_text and not any(rag_contexts.values()):
        raise SummaryError(
            "Could not identify enough document content "
            "to generate a summary."
        )

    prompt = _format_summary_prompt(rag_contexts, fallback_text)

    try:
        return pipeline.generate_with_prompt(
            prompt,
            api_key,
            # CHANGED: raised from 1800 -> 6000. Gemini 2.5's hidden
            # "thinking" tokens count against this budget, so 1800 was
            # frequently leaving too little room to finish all four
            # summary sections, causing mid-sentence cutoffs.
            max_tokens=6000,
        )

    except RAGPipelineError as exc:
        raise SummaryError(str(exc)) from exc


# ------------------------------------------------------------------
# Summary faithfulness evaluation
# ------------------------------------------------------------------

def evaluate_summary_faithfulness(
    summary: str,
    pipeline: RAGPipeline,
    api_key: str,
) -> dict[str, Any]:
    """Evaluate whether a generated summary is faithful to the paper."""
    if not summary or not summary.strip():
        return {
            "faithfulness_score": 0.0,
            "total_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": [],
            "explanation": "No summary provided.",
        }

    try:
        result = pipeline.retrieve(
            "What is the main content of this paper?",
            top_k=10,
        )
        source_chunks = result.chunks
    except Exception:
        return {
            "faithfulness_score": 0.0,
            "total_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": [],
            "explanation": "Could not retrieve source chunks for evaluation.",
        }

    try:
        faith = pipeline.evaluate_faithfulness(
            summary,
            source_chunks,
            api_key,
        )
        return faith
    except Exception as exc:
        return {
            "faithfulness_score": 0.0,
            "total_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": [],
            "explanation": f"Evaluation failed: {str(exc)}",
        }


# ------------------------------------------------------------------
# Reference / Citation Extraction
# ------------------------------------------------------------------

def extract_references_section(full_text: str) -> str:
    """Extract a References/Bibliography section robustly."""
    if not full_text:
        return ""

    heading_pattern = re.compile(
        r"(?im)^\s*"
        r"(?:\d+(?:\.\d+)*\s*)?"
        r"(references|bibliography|works cited)"
        r"\s*:?\s*$"
    )

    match = heading_pattern.search(full_text)
    if not match:
        return ""

    start = match.end()

    end_pattern = re.compile(
        r"(?im)^\s*"
        r"(?:\d+(?:\.\d+)*\s*)?"
        r"(appendix|acknowledgments|acknowledgements|"
        r"supplementary material)"
        r"\s*:?\s*$"
    )

    end_match = end_pattern.search(full_text, pos=start)
    end = end_match.start() if end_match else len(full_text)

    return full_text[start:end].strip()


def _clean_reference_text(raw_refs: str) -> str:
    """Perform safe deterministic cleanup before LLM formatting."""
    text = raw_refs.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# NEW: Rough entry-count estimators used to detect when the LLM has
# silently dropped references — a real failure mode observed where a
# 30-reference bibliography came back with only 16 entries and no
# error, because the model just... stopped listing some of them.

def _estimate_reference_count(raw_refs: str) -> int:
    """Estimate how many individual references are in the raw text.

    Academic reference lists almost always end each entry with a
    4-digit year followed by a period (e.g. "...1998."), so counting
    those is a reasonable proxy for entry count regardless of the
    source PDF's exact formatting.
    """
    return len(re.findall(r"(?:19|20)\d{2}\.", raw_refs))


def _count_formatted_entries(formatted: str) -> int:
    """Count numbered list items in the LLM's formatted output."""
    return len(re.findall(r"(?m)^\s*\d+\.\s", formatted))


def format_citations(
    full_text: str,
    pipeline: RAGPipeline,
    api_key: str,
) -> str:
    """Extract and clean a paper's bibliography."""
    if not full_text or not full_text.strip():
        raise SummaryError("No text available for citation extraction.")

    raw_refs = extract_references_section(full_text)
    if not raw_refs:
        raise SummaryError(
            "Could not locate a References or Bibliography "
            "section in this paper."
        )

    raw_refs = _clean_reference_text(raw_refs)

    # NOTE: this truncation is on the INPUT (raw reference text sent
    # to the model), separate from the output token cutoff issue.
    # Left as-is since 12000 chars of references is already generous;
    # the fix below is about the model having enough OUTPUT budget to
    # actually format all of what it's given.
    if len(raw_refs) > 12000:
        raw_refs = raw_refs[:12000]

    expected_count = _estimate_reference_count(raw_refs)

    prompt = f"""
You are formatting references extracted from an academic paper.

Convert the supplied references into a clean numbered list.

Rules:

- Preserve citation information exactly as much as possible.
- Do not invent authors, titles, years, journals, URLs, or DOIs.
- Do not add citations that are not present.
- Do not remove distinct references merely because formatting is messy.
- Repair obvious line-break and whitespace artifacts.
- Each reference should be on its own numbered item.
- If a reference is incomplete in the source, leave it incomplete.
- The source contains approximately {expected_count if expected_count else "an unknown number of"} references.
  You MUST include every single one. Do not stop early. Do not summarize
  or skip any entry, even if it looks similar to another one.
- Preserve the original ordering of the source references.

SOURCE REFERENCES:

{raw_refs}

FORMATTED REFERENCES:
""".strip()

    def _generate() -> str:
        return pipeline.generate_with_prompt(
            prompt,
            api_key,
            # Raised from 3000 -> 6000. A full reference list needs
            # real room to be fully re-numbered and formatted without
            # cutting off partway through.
            max_tokens=6000,
        )

    try:
        formatted = _generate()

        # NEW: Completeness check. If the model dropped a large chunk
        # of references, retry once with an even more explicit prompt
        # before giving up and warning the user, rather than silently
        # returning a partial list as if it were complete.
        if expected_count > 0:
            actual_count = _count_formatted_entries(formatted)

            if actual_count < expected_count * 0.85:
                formatted = _generate()
                actual_count = _count_formatted_entries(formatted)

            if actual_count < expected_count * 0.85:
                formatted = (
                    f"⚠️ **Note:** This paper appears to have around "
                    f"{expected_count} references, but only {actual_count} "
                    f"were formatted below. Some entries may be missing — "
                    f"try re-extracting, or check the PDF's References "
                    f"section directly for the full list.\n\n"
                    + formatted
                )

        return formatted

    except RAGPipelineError as exc:
        raise SummaryError(str(exc)) from exc