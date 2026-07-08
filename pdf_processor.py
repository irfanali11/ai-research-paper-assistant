"""PDF text extraction and chunking for the Scholar RAG pipeline."""

from __future__ import annotations

import re
from typing import BinaryIO

import pdfplumber

SECTION_HEADING_PATTERN = re.compile(
    r"(?:^|\n)\s*("
    r"Abstract|ABSTRACT|"
    r"Introduction|INTRODUCTION|"
    r"Related Work|RELATED WORK|"
    r"Background|BACKGROUND|"
    r"Methodology|METHODOLOGY|Methods|METHODS|"
    r"Materials and Methods|MATERIALS AND METHODS|"
    r"Experiments|EXPERIMENTS|"
    r"Results|RESULTS|"
    r"Discussion|DISCUSSION|"
    r"Conclusion|CONCLUSIONS?|"
    r"References|REFERENCES|Bibliography|BIBLIOGRAPHY"
    r")\s*(?:\n|:)",
    re.MULTILINE,
)


class PDFProcessingError(Exception):
    """Raised when PDF text cannot be extracted or processed."""


def extract_text_from_pdf(pdf_source: BinaryIO | str) -> str:
    """Extract all text from a PDF file.

    Args:
        pdf_source: A file-like object (bytes) or path to a PDF file.

    Returns:
        Concatenated text from all pages, with pages separated by newlines.

    Raises:
        PDFProcessingError: If the PDF is unreadable or yields no text.
    """
    try:
        with pdfplumber.open(pdf_source) as pdf:
            if not pdf.pages:
                raise PDFProcessingError(
                    "The uploaded PDF appears to be empty or has no readable pages."
                )

            page_texts: list[str] = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    page_texts.append(text.strip())

            if not page_texts:
                raise PDFProcessingError(
                    "No text could be extracted from this PDF. "
                    "It may be a scanned image-only document."
                )

            return "\n\n".join(page_texts)

    except PDFProcessingError:
        raise
    except Exception as exc:
        raise PDFProcessingError(
            "Unable to read the uploaded PDF. "
            "The file may be corrupted or in an unsupported format."
        ) from exc


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split document text into named sections using academic heading patterns.

    Args:
        text: The full document text.

    Returns:
        A list of (section_name, section_text) tuples.
    """
    matches = list(SECTION_HEADING_PATTERN.finditer(text))
    if len(matches) < 2:
        return [("Document", text)]

    sections: list[tuple[str, str]] = []

    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("Preamble", preamble))

    for i, match in enumerate(matches):
        section_name = match.group(1).strip().title()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append((section_name, section_text))

    return sections if sections else [("Document", text)]


def _chunk_section_text(
    section_name: str,
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Chunk a single section's text with sentence-aware splitting.

    Args:
        section_name: The academic section label.
        text: Text content for this section.
        chunk_size: Target maximum characters per chunk.
        overlap: Overlap between consecutive chunks.

    Returns:
        A list of prefixed chunk strings.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    prefix = f"[{section_name}] "
    effective_size = max(chunk_size - len(prefix), 200)

    if len(text) <= effective_size:
        return [f"{prefix}{text}"]

    sentence_pattern = re.compile(r"(?<=[.!?])\s+")
    sentences = sentence_pattern.split(text)
    raw_chunks: list[str] = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= effective_size:
            current_chunk = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
        else:
            if current_chunk:
                raw_chunks.append(current_chunk)
                overlap_text = (
                    current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                )
                current_chunk = f"{overlap_text} {sentence}".strip()
            else:
                for i in range(0, len(sentence), effective_size - overlap):
                    piece = sentence[i : i + effective_size]
                    if piece:
                        raw_chunks.append(piece)
                current_chunk = ""

    if current_chunk:
        raw_chunks.append(current_chunk)

    return [f"{prefix}{chunk}" for chunk in raw_chunks]


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[str]:
    """Split text into overlapping chunks, preferring academic section boundaries.

    When section headings are detected, chunks are created per section with
    section labels prefixed. Falls back to sentence-based splitting otherwise.

    Args:
        text: The full document text to chunk.
        chunk_size: Target maximum characters per chunk (default 800).
        overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        A list of text chunk strings.
    """
    if not text or not text.strip():
        return []

    sections = split_into_sections(text)
    if len(sections) > 1:
        chunks: list[str] = []
        for section_name, section_text in sections:
            chunks.extend(_chunk_section_text(section_name, section_text, chunk_size, overlap))
        if chunks:
            return chunks

    return _chunk_section_text("Document", text, chunk_size, overlap)
