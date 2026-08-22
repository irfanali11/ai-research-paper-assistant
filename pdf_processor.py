"""PDF text extraction and academic-aware chunking for Scholar."""

from __future__ import annotations

import re
from typing import BinaryIO

import pdfplumber


class PDFProcessingError(Exception):
    """Raised when PDF text cannot be extracted or processed."""


# Academic section names commonly found in research papers.
SECTION_NAMES = (
    "abstract",
    "introduction",
    "related work",
    "literature review",
    "background",
    "theoretical framework",
    "methodology",
    "methods",
    "materials and methods",
    "research methods",
    "experimental setup",
    "experiments",
    "results",
    "findings",
    "discussion",
    "results and discussion",
    "conclusion",
    "conclusions",
    "limitations",
    "future work",
    "references",
    "bibliography",
    "works cited",
    "acknowledgments",
    "acknowledgements",
    "appendix",
    "supplementary material",
)


_SECTION_PATTERN = re.compile(
    r"^\s*(?:"
    + "|".join(re.escape(name) for name in SECTION_NAMES)
    + r")\s*(?:[:.]|\d+(?:\.\d+)*\s*)?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _clean_page_text(text: str) -> str:
    """Normalize common PDF extraction artifacts."""
    text = text.replace("\x00", " ")
    text = text.replace("\u00ad", "")

    # Join words split by a PDF line break:
    # "re- \n search" -> "research"
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Normalize remaining whitespace.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_text_from_pdf(pdf_source: BinaryIO | str) -> str:
    """Extract readable text from a PDF.

    Args:
        pdf_source: File-like object or path to PDF.

    Returns:
        Full extracted document text.

    Raises:
        PDFProcessingError: If the PDF cannot be read or contains no text.
    """
    try:
        with pdfplumber.open(pdf_source) as pdf:
            if not pdf.pages:
                raise PDFProcessingError(
                    "The uploaded PDF appears to be empty or has no readable pages."
                )

            page_texts: list[str] = []

            for page_number, page in enumerate(pdf.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""

                text = _clean_page_text(text)

                if text:
                    page_texts.append(
                        f"[Page {page_number}]\n{text}"
                    )

            if not page_texts:
                raise PDFProcessingError(
                    "No text could be extracted from this PDF. "
                    "It may be a scanned image-only document."
                )

            full_text = "\n\n".join(page_texts)

            # Basic extraction sanity check.
            alpha_chars = sum(char.isalpha() for char in full_text)

            if alpha_chars < 100:
                raise PDFProcessingError(
                    "Very little readable text was extracted from this PDF. "
                    "It may be scanned, image-based, or poorly encoded."
                )

            return full_text

    except PDFProcessingError:
        raise

    except Exception as exc:
        raise PDFProcessingError(
            "Unable to read the uploaded PDF. "
            "The file may be corrupted or in an unsupported format."
        ) from exc


def _normalize_heading_name(heading: str) -> str:
    """Convert a detected heading to a clean display name."""
    heading = heading.strip()

    # Remove leading numbering such as:
    # 1 Introduction
    # 2.1 Methodology
    # 3 Results
    heading = re.sub(r"^\d+(?:\.\d+)*\s*", "", heading)

    heading = heading.rstrip(":.")

    return heading.strip().title()


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split extracted paper text into academic sections.

    The function is deliberately conservative. It only treats a line as a
    section heading when the whole line resembles a known academic heading.

    Args:
        text: Full extracted document text.

    Returns:
        List of (section_name, section_text).
    """
    if not text or not text.strip():
        return []

    matches = list(_SECTION_PATTERN.finditer(text))

    if len(matches) < 1:
        return [("Document", text.strip())]

    sections: list[tuple[str, str]] = []

    # Content before the first recognized section.
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("Preamble", preamble))

    for index, match in enumerate(matches):
        section_name = _normalize_heading_name(match.group(0))

        start = match.end()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )

        section_text = text[start:end].strip()

        if section_text:
            sections.append((section_name, section_text))

    return sections or [("Document", text.strip())]


def _split_long_sentence(
    sentence: str,
    max_length: int,
    overlap: int,
) -> list[str]:
    """Split an unusually long sentence without losing all context."""
    if len(sentence) <= max_length:
        return [sentence]

    step = max(max_length - overlap, 100)

    pieces: list[str] = []

    for start in range(0, len(sentence), step):
        piece = sentence[start : start + max_length].strip()

        if piece:
            pieces.append(piece)

        if start + max_length >= len(sentence):
            break

    return pieces


def _chunk_section_text(
    section_name: str,
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Create sentence-aware chunks for one academic section."""
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    prefix = f"[Section: {section_name}] "
    effective_size = max(chunk_size - len(prefix), 300)

    if len(text) <= effective_size:
        return [f"{prefix}{text}"]

    # Better sentence splitting for academic text.
    sentence_pattern = re.compile(
        r"(?<=[.!?])\s+(?=[A-Z0-9\[])"
    )

    sentences = sentence_pattern.split(text)

    # If extraction doesn't give useful sentences, fall back to words.
    if len(sentences) <= 1:
        return [
            f"{prefix}{piece}"
            for piece in _split_long_sentence(
                text,
                effective_size,
                overlap,
            )
        ]

    raw_chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        if len(sentence) > effective_size:
            if current:
                raw_chunks.append(current)
                current = ""

            raw_chunks.extend(
                _split_long_sentence(
                    sentence,
                    effective_size,
                    overlap,
                )
            )
            continue

        proposed = (
            f"{current} {sentence}".strip()
            if current
            else sentence
        )

        if len(proposed) <= effective_size:
            current = proposed
            continue

        if current:
            raw_chunks.append(current)

        # Character overlap is intentionally small and local.
        overlap_text = (
            current[-overlap:].strip()
            if current and overlap > 0
            else ""
        )

        current = (
            f"{overlap_text} {sentence}".strip()
            if overlap_text
            else sentence
        )

    if current:
        raw_chunks.append(current)

    return [
        f"{prefix}{chunk}"
        for chunk in raw_chunks
        if chunk.strip()
    ]


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 120,
) -> list[str]:
    """Chunk an academic paper while preserving section information.

    Args:
        text: Full extracted paper text.
        chunk_size: Target maximum character size.
        overlap: Character overlap between chunks.
    
    Returns:
        List of section-aware chunks.
    """
    if not text or not text.strip():
        return []

    if chunk_size < 400:
        raise ValueError("chunk_size must be at least 400.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and smaller than chunk_size."
        )

    sections = split_into_sections(text)

    chunks: list[str] = []

    for section_name, section_text in sections:
        chunks.extend(
            _chunk_section_text(
                section_name=section_name,
                text=section_text,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )

    if not chunks:
        return _chunk_section_text(
            "Document",
            text,
            chunk_size,
            overlap,
        )

    return chunks