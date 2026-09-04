"""Streamlit UI entry point for the Scholar research paper assistant.

Scholar is a session-scoped academic PDF assistant using:

    PDF extraction      -> local (with tables + figures)
    Section-aware chunks -> local
    Embeddings          -> local SentenceTransformer
    Cross-encoder rerank-> local (neural precision scoring)
    Vector retrieval    -> local Chroma (persistent)
    Keyword retrieval   -> local BM25
    Faithfulness eval   -> Gemini (opt-in)
    Answer generation   -> Google Gemini only when requested
    Related papers      -> Semantic Scholar (free, keyless)

Scholar supports multiple loaded documents at once: chat can be scoped
to one, several, or all loaded papers; Summary/Citations/Related Papers
operate on one "focused" document at a time, chosen from the sidebar
document library.

No document contents are intentionally sent to any external service
other than the Gemini API (on explicit generation request) and the
Semantic Scholar search API (on explicit "Find Related Papers" click,
which sends only the loaded paper's title/abstract, not its full text).
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any

import streamlit as st
from sentence_transformers import SentenceTransformer

from pdf_processor import (
    PDFProcessingError,
    chunk_text,
    extract_text_from_pdf,
)
from rag_pipeline import (
    EMBEDDING_MODEL_NAME,
    RAGPipeline,
    RAGPipelineError,
)
from semantic_scholar import (
    RelatedPaper,
    SemanticScholarError,
    find_related_papers,
)
from summary_and_citations import (
    SummaryError,
    format_citations,
    generate_structured_summary,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_PDF_SIZE_MB = 25
MAX_PDF_SIZE_BYTES = MAX_PDF_SIZE_MB * 1024 * 1024
MAX_DOCUMENTS = 8

EXAMPLE_QUESTIONS = [
    "What is the main research question?",
    "What methodology did the authors use?",
    "What are the key findings?",
    "What limitations does the paper mention?",
]

PROCESSING_STEPS = [
    "Extracting text, tables, and figures from PDF",
    "Chunking by academic sections",
    "Loading local embedding model",
    "Building hybrid search index (dense + sparse)",
]


@dataclass
class _SourceRefs:
    """Lightweight source container used by the UI."""

    chunks: list[str]
    indices: list[int]
    scores: list[float] | None = None
    doc_names: list[str] | None = None


# ---------------------------------------------------------------------------
# Styling — restored original hero-banner look, extended for new features
# ---------------------------------------------------------------------------

def _inject_styles(dark_mode: bool) -> None:
    """Inject custom CSS for the Scholar interface."""

    if dark_mode:
        bg = "#0f172a"
        surface = "#1e293b"
        text = "#e2e8f0"
        muted = "#94a3b8"
        border = "#334155"
        hero_start = "#1e293b"
        hero_end = "#334155"
        header_bg = "rgba(15, 23, 42, 0.95)"
        empty_bg = "#1e293b"
        metric_bg = "#1e293b"
        about_bg = "#1e293b"
        accent = "#60a5fa"
        status_ready_bg = "#064e3b"
        status_ready_text = "#6ee7b7"
        status_ready_border = "#047857"
        chat_user_bg = "#1e3a5f"
        chat_assist_bg = "#111827"

        widget_rules = f"""
            p, li, label, .stCaption,
            [data-testid="stMarkdown"] {{
                color: {text} !important;
            }}

            div[data-testid="stVerticalBlockBorderWrapper"] {{
                background: {surface} !important;
                border-color: {border} !important;
            }}

            [data-testid="stFileUploader"] section {{
                background: {surface} !important;
                border-color: {border} !important;
            }}
        """

    else:
        bg = "#f8f9fb"
        surface = "#ffffff"
        text = "#1a2332"
        muted = "#64748b"
        border = "#e2e8f0"
        hero_start = "#1e3a5f"
        hero_end = "#2c5282"
        header_bg = "rgba(248, 249, 251, 0.92)"
        empty_bg = "#ffffff"
        metric_bg = "#ffffff"
        about_bg = "#f8fafc"
        accent = "#1e3a5f"
        status_ready_bg = "#ecfdf5"
        status_ready_text = "#047857"
        status_ready_border = "#a7f3d0"
        chat_user_bg = "#eef2ff"
        chat_assist_bg = "#f8fafc"

        widget_rules = ""

    st.markdown(
        f"""
        <style>

            .stApp {{
                background-color: {bg} !important;
                color: {text};
            }}

            [data-testid="stAppViewContainer"] {{
                background-color: {bg};
            }}

            [data-testid="stHeader"] {{
                background: {header_bg} !important;
                border-bottom: 1px solid {border};
            }}

            section[data-testid="stSidebar"] {{
                background-color: {surface} !important;
                border-right: 1px solid {border};
            }}

            section[data-testid="stSidebar"] > div {{
                background-color: {surface} !important;
            }}

            .main .block-container {{
                padding-top: 1.5rem;
                padding-bottom: 3rem;
                max-width: 1100px;
            }}

            [data-testid="stSidebar"] .block-container {{
                padding-top: 1.25rem;
                padding-bottom: 2rem;
            }}

            .scholar-hero {{
                background: linear-gradient(
                    135deg,
                    {hero_start} 0%,
                    {hero_end} 100%
                );
                border-radius: 12px;
                padding: 2rem 2.25rem;
                margin: 0 0 1.75rem 0;
                color: #ffffff;
            }}

            .scholar-hero h1 {{
                font-family: Georgia, "Times New Roman", serif;
                font-size: 2rem;
                font-weight: 600;
                margin: 0 0 0.4rem 0;
                color: #ffffff !important;
            }}

            .scholar-hero p {{
                margin: 0;
                font-size: 1.05rem;
                opacity: 0.92;
                line-height: 1.5;
            }}

            .section-title {{
                font-family: Georgia, "Times New Roman", serif;
                font-size: 1.35rem;
                font-weight: 600;
                color: {accent};
                margin: 0.75rem 0 0.25rem 0;
            }}

            .section-subtitle {{
                color: {muted};
                font-size: 0.95rem;
                margin: 0 0 1.25rem 0;
                line-height: 1.5;
            }}

            .status-pill {{
                display: inline-block;
                padding: 0.35rem 0.75rem;
                border-radius: 999px;
                font-size: 0.8rem;
                font-weight: 600;
                margin: 0.5rem 0 1rem 0;
            }}

            .status-ready {{
                background: {status_ready_bg};
                color: {status_ready_text};
                border: 1px solid {status_ready_border};
            }}

            .status-waiting {{
                background: {surface};
                color: {muted};
                border: 1px solid {border};
            }}

            .doc-chip {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 0.5rem;
                padding: 0.5rem 0.65rem;
                border-radius: 8px;
                border: 1px solid {border};
                margin-bottom: 0.4rem;
                font-size: 0.83rem;
                background: {surface};
            }}

            .doc-chip-focused {{
                border-color: {accent};
                border-width: 2px;
            }}

            .empty-state {{
                background: {empty_bg};
                border: 1px dashed {border};
                border-radius: 10px;
                padding: 2.5rem 2rem;
                text-align: center;
                color: {muted};
                margin-top: 0.5rem;
            }}

            .empty-state strong {{
                display: block;
                color: {accent};
                font-size: 1.05rem;
                margin-bottom: 0.5rem;
            }}

            .about-box {{
                background: {about_bg};
                border-left: 3px solid {accent};
                padding: 0.85rem 1rem;
                border-radius: 0 8px 8px 0;
                font-size: 0.88rem;
                line-height: 1.55;
                color: {muted};
                margin-top: 0.5rem;
            }}

            .about-box strong {{
                color: {accent};
            }}

            .source-badge {{
                display: inline-block;
                background: {surface};
                border: 1px solid {border};
                color: {accent};
                padding: 0.15rem 0.55rem;
                border-radius: 6px;
                font-size: 0.78rem;
                font-weight: 600;
                margin: 0.25rem 0.35rem 0.25rem 0;
            }}

            .faith-badge {{
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                padding: 0.25rem 0.6rem;
                border-radius: 6px;
                font-size: 0.8rem;
                font-weight: 600;
                margin: 0.5rem 0;
            }}

            .faith-high {{
                background: {status_ready_bg};
                color: {status_ready_text};
                border: 1px solid {status_ready_border};
            }}

            .faith-medium {{
                background: #fef3c7;
                color: #92400e;
                border: 1px solid #f59e0b;
            }}

            .faith-low {{
                background: #fee2e2;
                color: #991b1b;
                border: 1px solid #ef4444;
            }}

            div[data-testid="stMetric"] {{
                background: {metric_bg};
                border: 1px solid {border};
                border-radius: 10px;
                padding: 0.75rem 1rem;
            }}

            div[data-testid="stMetricLabel"] {{
                color: {muted} !important;
            }}

            div[data-testid="stMetricValue"] {{
                color: {text} !important;
            }}

            [data-testid="stTabs"] {{
                margin-top: 0.5rem;
            }}

            [data-testid="stTabs"] [data-baseweb="tab-panel"] {{
                padding-top: 1rem;
            }}

            [data-testid="stFileUploader"] {{
                margin-bottom: 0.75rem;
            }}

            /* Style raw markdown headings inside generated content
               (summary ## headings etc.) so they match the design
               system instead of oversized default browser headings. */
            div[data-testid="stMarkdownContainer"] h1 {{
                font-family: Georgia, "Times New Roman", serif;
                font-size: 1.3rem;
                font-weight: 600;
                color: {accent};
                margin: 1.1rem 0 0.5rem 0;
                border-bottom: 1px solid {border};
                padding-bottom: 0.4rem;
            }}

            div[data-testid="stMarkdownContainer"] h2 {{
                font-family: Georgia, "Times New Roman", serif;
                font-size: 1.15rem;
                font-weight: 600;
                color: {accent};
                margin: 1.1rem 0 0.4rem 0;
            }}

            div[data-testid="stMarkdownContainer"] h3 {{
                font-family: Georgia, "Times New Roman", serif;
                font-size: 1.02rem;
                font-weight: 600;
                color: {text};
                margin: 0.9rem 0 0.35rem 0;
            }}

            /* Chat */

            .chat-welcome {{
                text-align: center;
                padding: 2.5rem 1.5rem 1.5rem;
                color: {muted};
            }}

            .chat-welcome h3 {{
                color: {text};
                font-size: 1.2rem;
                font-weight: 600;
                margin: 0 0 0.5rem 0;
            }}

            .chat-welcome p {{
                margin: 0;
                font-size: 0.92rem;
                line-height: 1.5;
            }}

            div[data-testid="stChatMessage"] {{
                padding: 0.85rem 0.5rem !important;
                margin-bottom: 0.25rem;
                background: transparent !important;
                border: none !important;
            }}

            div[data-testid="stChatMessage"]
            [data-testid="stMarkdownContainer"] {{
                font-size: 0.95rem;
                line-height: 1.65;
            }}

            div[data-testid="stChatMessage"]:has(
                [data-testid="chatAvatarIcon-user"]
            ) {{
                background: {chat_user_bg} !important;
                border-radius: 12px;
                padding: 0.85rem 1rem !important;
                margin: 0.35rem 0 0.35rem 2rem !important;
            }}

            div[data-testid="stChatMessage"]:has(
                [data-testid="chatAvatarIcon-assistant"]
            ) {{
                background: {chat_assist_bg} !important;
                border-radius: 12px;
                padding: 0.85rem 1rem !important;
                margin: 0.35rem 2rem 0.35rem 0 !important;
                border: 1px solid {border} !important;
            }}

            [data-testid="stChatInput"] {{
                border-top: 1px solid {border};
                padding-top: 0.75rem;
            }}

            .typing-indicator {{
                color: {muted};
                font-size: 0.88rem;
                font-style: italic;
            }}

            .related-card {{
                border: 1px solid {border};
                border-radius: 10px;
                padding: 0.9rem 1rem;
                margin-bottom: 0.6rem;
                background: {surface};
            }}

            .related-card-title {{
                font-weight: 600;
                font-size: 0.95rem;
                margin-bottom: 0.2rem;
                color: {text};
            }}

            .related-card-meta {{
                color: {muted};
                font-size: 0.8rem;
                margin-bottom: 0.4rem;
            }}

            .related-card-abstract {{
                color: {muted};
                font-size: 0.85rem;
                line-height: 1.5;
            }}

            .scholar-footer {{
                text-align: center;
                color: {muted};
                font-size: 0.8rem;
                margin-top: 2.5rem;
                padding-top: 1rem;
                border-top: 1px solid {border};
            }}

            {widget_rules}

            #MainMenu {{
                visibility: hidden;
            }}

            footer {{
                visibility: hidden;
            }}

            @media (max-width: 768px) {{
                .main .block-container {{
                    padding: 1rem 0.75rem 2rem;
                }}

                .scholar-hero {{
                    padding: 1.25rem 1.5rem;
                }}

                .scholar-hero h1 {{
                    font-size: 1.5rem;
                }}

                .scholar-hero p {{
                    font-size: 0.95rem;
                }}
            }}

        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Cached local model / shared pipeline
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_embedder() -> SentenceTransformer:
    """Load the local embedding model once per Streamlit process."""

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def _load_pipeline(_embedder: SentenceTransformer) -> RAGPipeline:
    """Create a single shared RAGPipeline for the app process.

    One pipeline instance holds every loaded document, so it is created
    once and reused rather than rebuilt per upload.
    """

    return RAGPipeline(embedder=_embedder)


def _get_pipeline() -> RAGPipeline:
    """Return the shared pipeline, creating it on first use."""

    embedder = _load_embedder()
    return _load_pipeline(embedder)


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    """Read Gemini API key from Streamlit secrets."""

    try:
        value = st.secrets.get("GEMINI_API_KEY", "")
        return str(value).strip()
    except (KeyError, FileNotFoundError):
        return ""


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    """Initialize all session-scoped application state."""

    defaults: dict[str, Any] = {
        "documents": {},
        "focused_doc_id": None,
        "active_doc_ids": [],
        "chat_history": [],
        "failed_upload_key": "",
        "dark_mode": False,
        "processing": False,
        "evaluate_faithfulness": False,
        "show_retrieval_details": False,
        "enable_decomposition": False,
        "related_papers_cache": {},
        "uploader_version": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _remove_document(document_id: str) -> None:
    """Remove one document from the library and the vector index."""

    pipeline = _get_pipeline()
    try:
        pipeline.remove_document(document_id)
    except RAGPipelineError:
        pass

    st.session_state.documents.pop(document_id, None)
    st.session_state.active_doc_ids = [
        d for d in st.session_state.active_doc_ids if d != document_id
    ]
    st.session_state.related_papers_cache.pop(document_id, None)

    if st.session_state.focused_doc_id == document_id:
        remaining = list(st.session_state.documents.keys())
        st.session_state.focused_doc_id = remaining[0] if remaining else None

    st.session_state.chat_history = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_source_label(indices: list[int]) -> str:
    """Create a human-readable source label."""

    if not indices:
        return ""

    joined = ", ".join(str(i) for i in indices)
    noun = "section" if len(indices) == 1 else "sections"

    return f"Based on {noun} {joined}"


def _format_chunk_for_display(chunk: str) -> str:
    """Format a chunk for safe display, handling tables specially."""
    if "[Table" in chunk and "|" in chunk:
        lines = chunk.split("\n")
        table_lines = []
        other_lines = []
        in_table = False

        for line in lines:
            if "[Table" in line:
                in_table = True
                table_lines.append(line)
            elif in_table and ("|" in line or line.strip() == ""):
                table_lines.append(line)
            elif in_table:
                in_table = False
                other_lines.append(line)
            else:
                other_lines.append(line)

        formatted = "\n".join(other_lines)
        if table_lines:
            table_text = "\n".join(table_lines)
            formatted += f"\n\n```\n{table_text}\n```"
        return formatted

    return chunk


def _extract_display_section(chunk: str) -> str:
    """Extract the section name from a chunk prefix."""
    match = re.match(r"\[Section:\s*(.*?)\]\s*", chunk, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "Document"


def _extract_title_and_abstract(full_text: str) -> tuple[str, str]:
    """Best-effort extraction of a paper's title and abstract."""
    title = ""
    title_match = re.search(r"\[Title:\s*(.*?)\]", full_text)
    if title_match and title_match.group(1).strip():
        title = title_match.group(1).strip()
    else:
        for line in full_text.splitlines():
            cleaned = line.strip().lstrip("[").rstrip("]")
            if cleaned and not cleaned.lower().startswith(("page ", "title", "authors")):
                title = cleaned[:200]
                break

    abstract = ""
    abstract_match = re.search(
        r"abstract\s*[:.]?\s*\n(.+?)(?:\n\s*\n|\n(?:1\.?\s*)?introduction)",
        full_text,
        re.IGNORECASE | re.DOTALL,
    )
    if abstract_match:
        abstract = re.sub(r"\s+", " ", abstract_match.group(1)).strip()[:1000]

    return title, abstract


def _render_source_citations(
    indices: list[int],
    chunks: list[str],
    scores: list[float] | None = None,
    doc_names: list[str] | None = None,
) -> None:
    """Render retrieved source chunks safely with section names and scores."""

    if not indices:
        return

    section_list = ", ".join(f"§{i}" for i in indices)
    multi_doc = doc_names is not None and len(set(doc_names)) > 1

    with st.expander(
        f"Sources · {section_list}",
        expanded=False,
    ):
        st.caption(_format_source_label(indices))

        for position, (idx, chunk) in enumerate(
            zip(indices, chunks)
        ):
            section_name = _extract_display_section(chunk)
            score_text = ""
            if scores and position < len(scores):
                score_text = f" · Score: {scores[position]:.3f}"

            doc_label = ""
            if multi_doc and doc_names and position < len(doc_names):
                doc_label = f"{html.escape(doc_names[position])} · "

            st.markdown(
                f"**{doc_label}Section {idx}** · *{section_name}*{score_text}"
            )

            display_chunk = _format_chunk_for_display(chunk)
            preview = display_chunk[:800]

            if len(display_chunk) > 800:
                preview += "…"

            safe_preview = html.escape(preview)

            st.markdown(f"> {safe_preview}")

            if position < len(indices) - 1:
                st.divider()


def _render_faithfulness(faithfulness: dict[str, Any] | None) -> None:
    """Display faithfulness score and claim details."""
    if not faithfulness:
        return

    score = faithfulness.get("faithfulness_score", 0.0)
    total = faithfulness.get("total_claims", 0)
    supported = faithfulness.get("supported_claims", 0)
    unsupported = faithfulness.get("unsupported_claims", [])
    explanation = faithfulness.get("explanation", "")

    if score >= 0.8:
        badge_class = "faith-high"
        label = "● High Faithfulness"
    elif score >= 0.5:
        badge_class = "faith-medium"
        label = "● Medium Faithfulness"
    else:
        badge_class = "faith-low"
        label = "● Low Faithfulness"

    st.markdown(
        f"""
        <div class="faith-badge {badge_class}">
            {label} · {score:.0%} ({supported}/{total} claims)
        </div>
        """,
        unsafe_allow_html=True,
    )

    if unsupported:
        with st.expander(f"⚠️ {len(unsupported)} unsupported claim(s)"):
            for claim in unsupported:
                st.markdown(f"- {html.escape(claim)}")
            if explanation:
                st.caption(explanation)


def _render_empty_state(
    message: str,
    hint: str,
) -> None:
    """Render an empty state."""

    st.markdown(
        f"""
        <div class="empty-state">
            <strong>{html.escape(message)}</strong>
            {html.escape(hint)}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Retrieval / answer handling
# ---------------------------------------------------------------------------

def _current_chat_doc_ids() -> list[str] | None:
    """Doc ids to scope chat retrieval to, or None for search-everything."""

    active = st.session_state.active_doc_ids
    return active if active else None


def _generate_assistant_response(question: str) -> None:
    """Generate one grounded response with optional evaluation."""

    if not st.session_state.documents:
        st.error("No document is currently loaded.")
        return

    pipeline = _get_pipeline()
    api_key = _get_api_key()

    if not api_key:
        st.error(
            "Gemini API key is not configured. "
            "Add GEMINI_API_KEY to Streamlit secrets."
        )
        return

    doc_ids = _current_chat_doc_ids()

    with st.chat_message("assistant", avatar="📚"):

        status = st.empty()

        evaluate_mode = st.session_state.get("evaluate_faithfulness", False)

        if evaluate_mode:
            status.markdown(
                '<p class="typing-indicator">'
                "Retrieving and evaluating answer quality…"
                "</p>",
                unsafe_allow_html=True,
            )

            try:
                result = pipeline.answer_question(
                    question,
                    api_key,
                    top_k=4,
                    evaluate=True,
                    doc_ids=doc_ids,
                    decompose=st.session_state.get("enable_decomposition", False),
                )

                answer = result.answer
                sources = result.sources
                faithfulness = result.faithfulness

                status.empty()

                if not answer:
                    st.warning(
                        "The AI service returned an empty response. "
                        "Please try again."
                    )
                    return

                st.markdown(answer)

                _render_faithfulness(faithfulness)

                if st.session_state.get("show_retrieval_details", False):
                    with st.expander("🔍 Retrieval Details"):
                        st.markdown("**Pipeline stages used:**")
                        if st.session_state.get("enable_decomposition", False):
                            st.markdown(
                                "- Query decomposition (splits compound "
                                "questions into sub-questions when needed)"
                            )
                        st.markdown("- Dense semantic retrieval (ChromaDB)")
                        st.markdown("- Sparse keyword retrieval (BM25)")
                        st.markdown("- Reciprocal Rank Fusion (RRF)")
                        st.markdown("- Cross-encoder neural re-ranking")
                        st.markdown("- Diversity filter (Jaccard similarity)")

                _render_source_citations(
                    sources.indices,
                    sources.chunks,
                    getattr(sources, "scores", None),
                    getattr(sources, "doc_names", None),
                )

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": str(answer),
                        "source_indices": list(sources.indices),
                        "source_chunks": list(sources.chunks),
                        "source_doc_names": list(sources.doc_names or []),
                        "faithfulness": faithfulness,
                    }
                )

            except RAGPipelineError as exc:
                status.empty()
                st.error(str(exc))

            except Exception:
                status.empty()
                st.error(
                    "Could not generate an answer. "
                    "Please try again."
                )

        else:
            status.markdown(
                '<p class="typing-indicator">'
                "Retrieving relevant sections…"
                "</p>",
                unsafe_allow_html=True,
            )

            try:
                stream_iter, sources = pipeline.stream_answer_question(
                    question,
                    api_key,
                    top_k=4,
                    doc_ids=doc_ids,
                    decompose=st.session_state.get("enable_decomposition", False),
                )

                status.markdown(
                    '<p class="typing-indicator">'
                    "Scholar is thinking…"
                    "</p>",
                    unsafe_allow_html=True,
                )

                answer = st.write_stream(stream_iter)

                status.empty()

                if not answer:
                    st.warning(
                        "The AI service returned an empty response. "
                        "Please try again."
                    )
                    return

                _render_source_citations(
                    sources.indices,
                    sources.chunks,
                    getattr(sources, "scores", None),
                    getattr(sources, "doc_names", None),
                )

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": str(answer),
                        "source_indices": list(sources.indices),
                        "source_chunks": list(sources.chunks),
                        "source_doc_names": list(sources.doc_names or []),
                        "faithfulness": None,
                    }
                )

            except RAGPipelineError as exc:
                status.empty()
                st.error(str(exc))

            except Exception:
                status.empty()
                st.error(
                    "Could not generate an answer. "
                    "Please try again."
                )


def _chat_needs_response() -> bool:
    """Return whether the latest message is waiting for an answer."""

    history = st.session_state.chat_history

    return (
        bool(history)
        and history[-1].get("role") == "user"
    )


def _submit_user_message(question: str) -> None:
    """Add a user question and rerun the app."""

    question = question.strip()

    if not question:
        return

    question = question[:4000]

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question,
        }
    )

    st.rerun()


# ---------------------------------------------------------------------------
# Chat rendering
# ---------------------------------------------------------------------------

def _render_chat_message(message: dict[str, Any]) -> None:
    """Render one chat message with optional faithfulness display."""

    role = message.get("role", "assistant")

    if role not in {"user", "assistant"}:
        return

    avatar = "🧑‍🎓" if role == "user" else "📚"

    with st.chat_message(role, avatar=avatar):

        content = str(message.get("content", ""))

        st.markdown(content)

        if (
            role == "assistant"
            and message.get("source_indices")
        ):
            _render_source_citations(
                message["source_indices"],
                message.get("source_chunks", []),
                None,
                message.get("source_doc_names"),
            )

        if role == "assistant" and message.get("faithfulness"):
            _render_faithfulness(message["faithfulness"])


def _render_chat_welcome() -> None:
    """Render the initial chat welcome."""

    doc_count = len(st.session_state.documents)
    if st.session_state.active_doc_ids:
        scope_note = "Scoped to your selected paper(s)."
    elif doc_count > 1:
        scope_note = f"Searching across all {doc_count} loaded paper(s)."
    else:
        scope_note = (
            "Scholar retrieves relevant sections from your document "
            "and generates grounded answers with source references."
        )

    st.markdown(
        f"""
        <div class="chat-welcome">
            <h3>Ask anything about your paper</h3>
            <p>{html.escape(scope_note)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_example_questions() -> None:
    """Render inexpensive suggested questions."""

    st.markdown(
        '<p style="text-align:center;'
        'color:#64748b;font-size:0.85rem;'
        'margin:0 0 0.5rem 0;">'
        "Suggested questions"
        "</p>",
        unsafe_allow_html=True,
    )

    columns = st.columns(len(EXAMPLE_QUESTIONS))

    for column, example in zip(
        columns,
        EXAMPLE_QUESTIONS,
    ):
        with column:
            if st.button(
                example,
                key=f"example_{example}",
                use_container_width=True,
            ):
                _submit_user_message(example)


def _render_chat_scope_picker() -> None:
    """Let the user scope chat to specific loaded documents."""

    doc_items = list(st.session_state.documents.items())
    labels = [meta["name"] for _, meta in doc_items]
    ids = [doc_id for doc_id, _ in doc_items]

    current_active = st.session_state.active_doc_ids
    default_selection = (
        [labels[ids.index(d)] for d in current_active if d in ids]
        if current_active
        else labels
    )

    selected_labels = st.multiselect(
        "Chat scope — which papers should Scholar search?",
        options=labels,
        default=default_selection,
        help="Leave all selected to search across every loaded paper.",
    )

    selected_ids = [ids[labels.index(lbl)] for lbl in selected_labels]

    st.session_state.active_doc_ids = (
        [] if len(selected_ids) == len(labels) else selected_ids
    )


def _render_chat_tab() -> None:
    """Render the chat interface."""

    if not st.session_state.documents:
        st.markdown(
            '<p class="section-title">Chat</p>',
            unsafe_allow_html=True,
        )

        _render_empty_state(
            "No document loaded",
            "Upload a PDF in the sidebar to start a conversation.",
        )

        return

    header_col, clear_col = st.columns([5, 1])

    with header_col:
        st.markdown(
            '<p class="section-title" style="margin-top:0">'
            "Chat"
            "</p>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<p class="section-subtitle" '
            'style="margin-bottom:0.75rem">'
            "Grounded answers with cited document sections"
            "</p>",
            unsafe_allow_html=True,
        )

    with clear_col:
        if st.session_state.chat_history:

            if st.button(
                "Clear",
                type="secondary",
                use_container_width=True,
            ):
                st.session_state.chat_history = []
                st.rerun()

    if len(st.session_state.documents) > 1:
        _render_chat_scope_picker()

    with st.container(border=True):

        if not st.session_state.chat_history:
            _render_chat_welcome()
            _render_example_questions()

        else:
            for message in st.session_state.chat_history:
                _render_chat_message(message)

        if _chat_needs_response():
            question = st.session_state.chat_history[-1]["content"]

            _generate_assistant_response(question)

    prompt = st.chat_input(
        "Message Scholar about your paper…"
    )

    if prompt:
        _submit_user_message(prompt)


# ---------------------------------------------------------------------------
# PDF processing
# ---------------------------------------------------------------------------

def _validate_uploaded_file(uploaded_file) -> None:
    """Validate basic PDF upload constraints."""

    if uploaded_file is None:
        raise PDFProcessingError(
            "No PDF file was selected."
        )

    filename = str(uploaded_file.name)

    if not filename.lower().endswith(".pdf"):
        raise PDFProcessingError(
            "Only PDF files are supported."
        )

    if uploaded_file.size <= 0:
        raise PDFProcessingError(
            "The uploaded PDF is empty."
        )

    if uploaded_file.size > MAX_PDF_SIZE_BYTES:
        raise PDFProcessingError(
            f"The PDF is too large. "
            f"Maximum supported size is {MAX_PDF_SIZE_MB} MB."
        )


def _process_pdf(
    uploaded_file,
    file_key: str,
) -> None:
    """Extract, chunk, embed, and index one PDF as a library document."""

    if st.session_state.processing:
        return

    if len(st.session_state.documents) >= MAX_DOCUMENTS:
        st.warning(
            f"You've reached the {MAX_DOCUMENTS}-document limit for this "
            "session. Remove a paper from the sidebar before adding another."
        )
        return

    st.session_state.processing = True

    try:
        _validate_uploaded_file(uploaded_file)

        uploaded_file.seek(0)

        with st.status(
            "Processing document…",
            expanded=True,
        ) as status:

            st.write(PROCESSING_STEPS[0])

            full_text = extract_text_from_pdf(
                uploaded_file
            )

            if not full_text.strip():
                raise PDFProcessingError(
                    "No readable text was found in the PDF."
                )

            st.write(PROCESSING_STEPS[1])

            chunks = chunk_text(full_text)

            if not chunks:
                raise PDFProcessingError(
                    "No usable text chunks could be created."
                )

            st.write(PROCESSING_STEPS[2])

            pipeline = _get_pipeline()

            st.write(PROCESSING_STEPS[3])

            pipeline.index_chunks(
                chunks,
                document_id=file_key,
                document_name=uploaded_file.name,
            )

            status.update(
                label="Document ready",
                state="complete",
            )

        table_count = sum(1 for c in chunks if "[Table" in c)
        figure_count = sum(1 for c in chunks if "[Figure" in c)

        st.session_state.documents[file_key] = {
            "name": uploaded_file.name,
            "full_text": full_text,
            "chunk_count": len(chunks),
            "table_count": table_count,
            "figure_count": figure_count,
        }
        st.session_state.focused_doc_id = file_key
        st.session_state.failed_upload_key = ""

        st.success(
            f"Document indexed successfully — "
            f"{len(chunks):,} chunks ready for analysis."
        )

    except PDFProcessingError as exc:

        st.session_state.failed_upload_key = file_key

        st.error(str(exc))

    except RAGPipelineError as exc:

        st.session_state.failed_upload_key = file_key

        st.error(str(exc))

    except MemoryError:

        st.session_state.failed_upload_key = file_key

        st.error(
            "The document is too large for the available memory. "
            "Try a smaller PDF."
        )

    except Exception:

        st.session_state.failed_upload_key = file_key

        st.error(
            "The document could not be processed. "
            "Try a smaller or different PDF."
        )

    finally:
        st.session_state.processing = False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_document_library() -> None:
    """List loaded documents with focus/remove controls."""

    if not st.session_state.documents:
        return

    st.markdown("**Your papers**")

    for doc_id, meta in list(st.session_state.documents.items()):
        is_focused = doc_id == st.session_state.focused_doc_id
        chip_class = "doc-chip doc-chip-focused" if is_focused else "doc-chip"

        col_info, col_focus, col_remove = st.columns([5, 1, 1])

        with col_info:
            display_name = meta["name"][:32] + ("…" if len(meta["name"]) > 32 else "")
            st.markdown(
                f"<div class='{chip_class}'><span style='overflow-wrap:anywhere;'>"
                f"{html.escape(display_name)}</span></div>",
                unsafe_allow_html=True,
            )

        with col_focus:
            if st.button(
                "●" if is_focused else "◎",
                key=f"focus_{doc_id}",
                help="Set as focused document for Summary/Citations/Related",
            ):
                st.session_state.focused_doc_id = doc_id
                st.rerun()

        with col_remove:
            if st.button("✕", key=f"remove_{doc_id}", help="Remove this paper"):
                _remove_document(doc_id)
                st.rerun()


def _render_sidebar() -> None:
    """Render upload controls and application information."""

    with st.sidebar:

        st.markdown("### Scholar")
        st.caption("Research Paper Assistant")

        st.markdown(
            "<div style='height:0.5rem'></div>",
            unsafe_allow_html=True,
        )

        st.toggle(
            "Dark mode",
            key="dark_mode",
        )

        st.toggle(
            "Evaluate answer faithfulness",
            key="evaluate_faithfulness",
            help="Uses one extra Gemini call per answer to verify claims against sources. Enable when testing quality.",
        )

        st.toggle(
            "Show retrieval details",
            key="show_retrieval_details",
            help="Display technical information about which retrieval stages were used.",
        )

        st.toggle(
            "Enable query decomposition",
            key="enable_decomposition",
            help=(
                "For compound/comparative questions (e.g. 'compare X and Y'), "
                "splits the question into sub-questions and retrieves for each "
                "separately before answering. Simple questions skip this "
                "automatically. Uses one extra Gemini call only when the "
                "question is actually complex."
            ),
        )

        if st.session_state.documents:

            st.markdown(
                '<span class="status-pill status-ready">'
                f"● {len(st.session_state.documents)} paper(s) ready"
                "</span>",
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                '<span class="status-pill status-waiting">'
                "○ Awaiting upload"
                "</span>",
                unsafe_allow_html=True,
            )

        uploaded_files = st.file_uploader(
            "Upload research papers (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            help=(
                f"Upload academic PDFs. "
                f"Maximum size per file: {MAX_PDF_SIZE_MB} MB."
            ),
            # Versioned key resets the widget after a successful upload
            # so its own "selected file" chip doesn't sit duplicated
            # above the "Your papers" library list below it.
            key=f"uploader_{st.session_state.get('uploader_version', 0)}",
        )

        if uploaded_files:
            any_new_processed = False

            for uploaded_file in uploaded_files:

                file_key = (
                    f"{uploaded_file.name}:"
                    f"{uploaded_file.size}"
                )

                if file_key in st.session_state.documents:
                    continue

                if st.session_state.failed_upload_key == file_key:

                    st.warning(
                        f"Processing failed for {uploaded_file.name}."
                    )

                    if st.button(
                        "Retry processing",
                        key=f"retry_{file_key}",
                        use_container_width=True,
                    ):
                        st.session_state.failed_upload_key = ""
                        _process_pdf(uploaded_file, file_key)
                        st.rerun()

                else:
                    _process_pdf(uploaded_file, file_key)
                    if file_key in st.session_state.documents:
                        any_new_processed = True

            if any_new_processed:
                st.session_state.uploader_version = (
                    st.session_state.get("uploader_version", 0) + 1
                )
                st.rerun()

        _render_document_library()

        st.divider()

        st.markdown(
            """
            <div class="about-box">
                <strong>About Scholar</strong><br><br>
                Local embeddings + Chroma vector search +
                BM25 keyword retrieval + <strong>Cross-encoder re-ranking</strong>
                + Gemini generation.<br><br>
                <strong>Advanced features:</strong><br>
                • Multi-document chat scoping<br>
                • Query decomposition for compound questions<br>
                • Section-aware chunking<br>
                • Table & figure extraction<br>
                • Faithfulness evaluation<br>
                • Related-paper discovery (Semantic Scholar)<br><br>
                PDFs and embeddings remain local.
                Gemini and Semantic Scholar are only called
                when you explicitly request it.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Document metrics
# ---------------------------------------------------------------------------

def _render_document_metrics() -> None:
    """Display basic statistics for the focused document."""

    if not st.session_state.documents:
        return

    doc_id = st.session_state.focused_doc_id
    if doc_id not in st.session_state.documents:
        doc_id = next(iter(st.session_state.documents))
        st.session_state.focused_doc_id = doc_id

    meta = st.session_state.documents[doc_id]

    col1, col2, col3, col4 = st.columns(4)

    filename = meta.get("name", "Document")

    with col1:
        st.metric(
            "Document",
            filename[:20]
            + ("…" if len(filename) > 20 else ""),
        )

    with col2:
        st.metric(
            "Chunks",
            f"{meta.get('chunk_count', 0):,}",
        )

    with col3:
        st.metric(
            "Tables",
            f"{meta.get('table_count', 0)}",
        )

    with col4:
        st.metric(
            "Figures",
            f"{meta.get('figure_count', 0)}",
        )


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

def _render_hero() -> None:
    """Render application header."""

    doc_count = len(st.session_state.documents)
    subtitle = (
        "AI-powered research paper assistant — "
        "ask questions, generate summaries, "
        "and explore citations grounded in your document."
    )
    if doc_count > 1:
        subtitle = (
            f"{doc_count} papers loaded — ask questions across your "
            "library, generate summaries, explore citations, and "
            "discover related work."
        )

    st.markdown(
        f"""
        <div class="scholar-hero">
            <h1>Scholar</h1>
            <p>{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_focused_doc_picker(purpose: str) -> str | None:
    """Let the user pick which loaded document a per-document tab targets."""

    if not st.session_state.documents:
        return None

    doc_items = list(st.session_state.documents.items())
    labels = [meta["name"] for _, meta in doc_items]
    ids = [doc_id for doc_id, _ in doc_items]

    current = st.session_state.focused_doc_id
    default_index = ids.index(current) if current in ids else 0

    if len(doc_items) > 1:
        selected_label = st.selectbox(
            f"Document for {purpose}",
            options=labels,
            index=default_index,
        )
        selected_id = ids[labels.index(selected_label)]
        st.session_state.focused_doc_id = selected_id
        return selected_id

    return ids[0]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _render_summary_tab() -> None:
    """Render structured paper summary for the focused document."""

    st.markdown(
        '<p class="section-title">Paper Summary</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="section-subtitle">'
        "Structured overview: Research Question, "
        "Methodology, Key Findings, and Limitations."
        "</p>",
        unsafe_allow_html=True,
    )

    if not st.session_state.documents:

        _render_empty_state(
            "No document loaded",
            "Upload a PDF in the sidebar to generate a summary.",
        )

        return

    doc_id = _render_focused_doc_picker("summary")
    meta = st.session_state.documents[doc_id]
    summary = meta.get("summary", "")

    if summary:

        with st.container(border=True):
            st.markdown(summary)

        col1, col2, col3 = st.columns(
            [1, 1, 1]
        )

        filename = meta["name"].rsplit(".", 1)[0]

        with col1:
            st.download_button(
                "Download .md",
                summary,
                file_name=f"{filename}_summary.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with col2:
            st.download_button(
                "Download .txt",
                summary,
                file_name=f"{filename}_summary.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with col3:
            if st.button(
                "Regenerate",
                type="secondary",
                use_container_width=True,
            ):
                st.session_state.documents[doc_id]["summary"] = ""
                st.rerun()

        return

    col1, col2, col3 = st.columns(
        [1, 2, 1]
    )

    with col2:

        if st.button(
            "Generate Summary",
            type="primary",
            use_container_width=True,
        ):

            api_key = _get_api_key()

            if not api_key:
                st.error(
                    "Gemini API key is not configured."
                )
                return

            progress = st.empty()

            try:

                progress.caption(
                    "Generating structured summary…"
                )

                pipeline = _get_pipeline()

                summary = generate_structured_summary(
                    meta["full_text"],
                    pipeline,
                    api_key,
                )

                progress.empty()

                st.session_state.documents[doc_id]["summary"] = summary

                st.rerun()

            except SummaryError as exc:

                progress.empty()

                st.error(str(exc))

            except Exception:

                progress.empty()

                st.error(
                    "The summary could not be generated. "
                    "Please try again."
                )


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------

def _render_citations_tab() -> None:
    """Render extracted bibliography for the focused document."""

    st.markdown(
        '<p class="section-title">Key Citations</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="section-subtitle">'
        "Extracted and formatted reference list "
        "from the paper's bibliography."
        "</p>",
        unsafe_allow_html=True,
    )

    if not st.session_state.documents:

        _render_empty_state(
            "No document loaded",
            "Upload a PDF to extract its references.",
        )

        return

    doc_id = _render_focused_doc_picker("citations")
    meta = st.session_state.documents[doc_id]
    citations = meta.get("citations", "")

    if citations:

        with st.container(border=True):
            st.markdown(citations)

        col1, col2, col3 = st.columns(
            [1, 1, 1]
        )

        filename = meta["name"].rsplit(".", 1)[0]

        with col1:
            st.download_button(
                "Download .md",
                citations,
                file_name=f"{filename}_citations.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with col2:
            st.download_button(
                "Download .txt",
                citations,
                file_name=f"{filename}_citations.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with col3:

            if st.button(
                "Re-extract",
                type="secondary",
                use_container_width=True,
            ):
                st.session_state.documents[doc_id]["citations"] = ""
                st.rerun()

        return

    col1, col2, col3 = st.columns(
        [1, 2, 1]
    )

    with col2:

        if st.button(
            "Extract Citations",
            type="primary",
            use_container_width=True,
        ):

            api_key = _get_api_key()

            if not api_key:
                st.error(
                    "Gemini API key is not configured."
                )
                return

            progress = st.empty()

            try:

                progress.caption(
                    "Locating References section…"
                )

                pipeline = _get_pipeline()

                citations = format_citations(
                    meta["full_text"],
                    pipeline,
                    api_key,
                )

                progress.empty()

                st.session_state.documents[doc_id]["citations"] = citations

                st.rerun()

            except SummaryError as exc:

                progress.empty()

                st.error(str(exc))

            except Exception:

                progress.empty()

                st.error(
                    "The references could not be extracted. "
                    "Please try again."
                )


# ---------------------------------------------------------------------------
# Related papers
# ---------------------------------------------------------------------------

def _render_related_card(paper: RelatedPaper) -> None:
    """Render one related-paper result card."""

    authors_text = ", ".join(paper.authors[:4])
    if len(paper.authors) > 4:
        authors_text += " et al."

    meta_parts = [p for p in [authors_text, str(paper.year) if paper.year else ""] if p]
    if paper.citation_count is not None:
        meta_parts.append(f"{paper.citation_count:,} citations")
    meta_line = " · ".join(meta_parts)

    abstract_preview = paper.abstract[:280] + ("…" if len(paper.abstract) > 280 else "")

    link_html = (
        f'<a href="{html.escape(paper.url)}" target="_blank" '
        f'style="font-size:0.8rem;">View on Semantic Scholar →</a>'
        if paper.url else ""
    )

    st.markdown(
        f"""
        <div class="related-card">
            <div class="related-card-title">{html.escape(paper.title)}</div>
            <div class="related-card-meta">{html.escape(meta_line)}</div>
            <div class="related-card-abstract">{html.escape(abstract_preview)}</div>
            <div style="margin-top:0.5rem;">{link_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_related_tab() -> None:
    """Render related-papers discovery for the focused document."""

    st.markdown(
        '<p class="section-title">Related Papers</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="section-subtitle">'
        "Discover related work via Semantic Scholar, based on this "
        "paper's title and abstract."
        "</p>",
        unsafe_allow_html=True,
    )

    if not st.session_state.documents:

        _render_empty_state(
            "No document loaded",
            "Upload a PDF to find related work.",
        )

        return

    doc_id = _render_focused_doc_picker("related papers")
    meta = st.session_state.documents[doc_id]

    cached = st.session_state.related_papers_cache.get(doc_id)

    if cached:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"Related to: {meta['name']}")
        with col2:
            if st.button("Refresh", use_container_width=True):
                st.session_state.related_papers_cache.pop(doc_id, None)
                st.rerun()

        for paper in cached:
            _render_related_card(paper)
        return

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Find Related Papers", type="primary", use_container_width=True):
            title, abstract = _extract_title_and_abstract(meta["full_text"])

            if not title:
                st.error("Could not identify this paper's title to search from.")
                return

            progress = st.empty()
            try:
                progress.caption("Searching Semantic Scholar…")
                results = find_related_papers(
                    title=title,
                    abstract=abstract,
                    limit=6,
                    exclude_title=title,
                )
                progress.empty()

                if not results:
                    st.info("No related papers were found for this document.")
                else:
                    st.session_state.related_papers_cache[doc_id] = results
                    st.rerun()

            except SemanticScholarError as exc:
                progress.empty()
                st.error(str(exc))
            except Exception:
                progress.empty()
                st.error("Could not fetch related papers. Please try again.")


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the Scholar Streamlit application."""

    st.set_page_config(
        page_title="Scholar — AI Research Paper Assistant",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _init_session_state()

    # Sidebar must run before styling because the toggle
    # determines the active visual theme.
    _render_sidebar()

    _inject_styles(
        st.session_state.get(
            "dark_mode",
            False,
        )
    )

    _render_hero()

    _render_document_metrics()

    tab_chat, tab_summary, tab_citations, tab_related = st.tabs(
        [
            "Chat",
            "Summary",
            "Citations",
            "Related",
        ]
    )

    with tab_chat:
        _render_chat_tab()

    with tab_summary:
        _render_summary_tab()

    with tab_citations:
        _render_citations_tab()

    with tab_related:
        _render_related_tab()

    st.markdown(
        '<p class="scholar-footer">'
        "Scholar · Advanced Hybrid RAG · "
        "Dense + Sparse + Cross-Encoder + Faithfulness Eval + Related Papers"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()