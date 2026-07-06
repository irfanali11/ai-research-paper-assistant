"""PDF text extraction and chunking for the Scholar RAG pipeline."""

from __future__ import annotations

import re
from typing import BinaryIO

import pdfplumber


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


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[str]:
    """Split text into overlapping chunks of approximately ``chunk_size`` characters.

    Chunks are split on sentence boundaries when possible to preserve context.

    Args:
        text: The full document text to chunk.
        chunk_size: Target maximum characters per chunk (default 800).
        overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        A list of text chunk strings.
    """
    if not text or not text.strip():
        return []

    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= chunk_size:
        return [text]

    sentence_pattern = re.compile(r"(?<=[.!?])\s+")
    sentences = sentence_pattern.split(text)

    chunks: list[str] = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = f"{overlap_text} {sentence}".strip()
            else:
                # Single sentence exceeds chunk_size — hard-split it
                for i in range(0, len(sentence), chunk_size - overlap):
                    piece = sentence[i : i + chunk_size]
                    if piece:
                        chunks.append(piece)
                current_chunk = ""

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
