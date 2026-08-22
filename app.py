"""Streamlit UI entry point for the Scholar research paper assistant.

Scholar is a session-scoped academic PDF assistant using:

    PDF extraction      -> local
    Section-aware chunks -> local
    Embeddings          -> local SentenceTransformer
    Vector retrieval    -> local Chroma
    Keyword retrieval   -> local BM25
    Answer generation   -> Google Gemini only when requested

No document contents are intentionally sent to any external service other
than the Gemini API during an explicit generation request.
"""

from __future__ import annotations

import html
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

EXAMPLE_QUESTIONS = [
    "What is the main research question?",
    "What methodology did the authors use?",
    "What are the key findings?",
    "What limitations does the paper mention?",
]

PROCESSING_STEPS = [
    "Extracting text from PDF",
    "Chunking by academic sections",
    "Loading local embedding model",
    "Building hybrid search index",
]


# ---------------------------------------------------------------------------
# Lightweight source compatibility object
# ---------------------------------------------------------------------------

@dataclass
class _SourceRefs:
    """Lightweight source container used by the UI."""

    chunks: list[str]
    indices: list[int]


# ---------------------------------------------------------------------------
# Styling
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
# Cached local model
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_embedder() -> SentenceTransformer:
    """Load the local embedding model once per Streamlit process."""

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    """Read Gemini API key from Streamlit secrets.

    The key is never displayed in the UI.
    """

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
        "pipeline": None,
        "full_text": "",
        "chunks": [],
        "chat_history": [],
        "summary": "",
        "citations": "",
        "pdf_processed": False,
        "failed_file_key": "",
        "last_upload_key": "",
        "last_filename": "",
        "dark_mode": False,
        "processing": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_document_state() -> None:
    """Clear all state associated with the currently loaded document."""

    st.session_state.pipeline = None
    st.session_state.full_text = ""
    st.session_state.chunks = []
    st.session_state.chat_history = []
    st.session_state.summary = ""
    st.session_state.citations = ""
    st.session_state.pdf_processed = False
    st.session_state.failed_file_key = ""
    st.session_state.last_filename = ""


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


def _render_source_citations(
    indices: list[int],
    chunks: list[str],
) -> None:
    """Render retrieved source chunks safely."""

    if not indices:
        return

    section_list = ", ".join(f"§{i}" for i in indices)

    with st.expander(
        f"Sources · {section_list}",
        expanded=False,
    ):
        st.caption(_format_source_label(indices))

        for position, (idx, chunk) in enumerate(
            zip(indices, chunks)
        ):
            st.markdown(f"**Section {idx}**")

            preview = chunk[:700]

            if len(chunk) > 700:
                preview += "…"

            # Escape HTML while preserving normal markdown blockquote display.
            safe_preview = html.escape(preview)

            st.markdown(f"> {safe_preview}")

            if position < len(indices) - 1:
                st.divider()


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

def _resolve_answer_result(
    pipeline: RAGPipeline,
    question: str,
    api_key: str,
):
    """Return a normalized answer/source pair."""

    result = pipeline.answer_question(
        question,
        api_key,
    )

    if hasattr(result, "answer") and hasattr(result, "sources"):
        return result.answer, result.sources

    # Defensive fallback for older pipeline objects.
    answer = str(result)
    sources = pipeline.retrieve(question)

    if hasattr(sources, "indices"):
        return answer, sources

    chunks = list(sources)

    return (
        answer,
        _SourceRefs(
            chunks=chunks,
            indices=list(range(1, len(chunks) + 1)),
        ),
    )


def _generate_assistant_response(question: str) -> None:
    """Generate one grounded response for the latest user question."""

    pipeline = st.session_state.pipeline

    if pipeline is None:
        st.error("No document is currently loaded.")
        return

    api_key = _get_api_key()

    if not api_key:
        st.error(
            "Gemini API key is not configured. "
            "Add GEMINI_API_KEY to Streamlit secrets."
        )
        return

    with st.chat_message("assistant", avatar="📚"):

        status = st.empty()

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
            )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": str(answer),
                    "source_indices": list(sources.indices),
                    "source_chunks": list(sources.chunks),
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

    # Prevent accidentally storing enormous prompts.
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
    """Render one chat message."""

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
            )


def _render_chat_welcome() -> None:
    """Render the initial chat welcome."""

    st.markdown(
        """
        <div class="chat-welcome">
            <h3>Ask anything about your paper</h3>
            <p>
                Scholar retrieves relevant sections from your document
                and generates grounded answers with source references.
            </p>
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


def _render_chat_tab() -> None:
    """Render the chat interface."""

    if not st.session_state.pdf_processed:
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

    with st.container(border=True):

        if not st.session_state.chat_history:
            _render_chat_welcome()
            _render_example_questions()

        else:
            for message in st.session_state.chat_history:
                _render_chat_message(message)

        # Only generate when the newest message is unanswered.
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
    """Extract, chunk, embed, and index one PDF."""

    if st.session_state.processing:
        return

    st.session_state.processing = True

    try:
        _validate_uploaded_file(uploaded_file)

        # Make sure a previous document cannot remain active if
        # processing the new document fails.
        _reset_document_state()

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

            embedder = _load_embedder()

            pipeline = RAGPipeline(
                embedder=embedder
            )

            st.write(PROCESSING_STEPS[3])

            pipeline.index_chunks(chunks)

            status.update(
                label="Document ready",
                state="complete",
            )

        # Commit state only after the complete pipeline succeeds.
        st.session_state.pipeline = pipeline
        st.session_state.full_text = full_text
        st.session_state.chunks = chunks
        st.session_state.chat_history = []
        st.session_state.summary = ""
        st.session_state.citations = ""
        st.session_state.pdf_processed = True
        st.session_state.last_upload_key = file_key
        st.session_state.last_filename = uploaded_file.name
        st.session_state.failed_file_key = ""

        st.success(
            f"Document indexed successfully — "
            f"{len(chunks):,} chunks ready for analysis."
        )

    except PDFProcessingError as exc:

        st.session_state.failed_file_key = file_key

        st.error(str(exc))

    except RAGPipelineError as exc:

        st.session_state.failed_file_key = file_key

        st.error(str(exc))

    except MemoryError:

        st.session_state.failed_file_key = file_key

        st.error(
            "The document is too large for the available memory. "
            "Try a smaller PDF."
        )

    except Exception:

        st.session_state.failed_file_key = file_key

        # Deliberately do not expose raw exception details.
        st.error(
            "The document could not be processed. "
            "Try a smaller or different PDF."
        )

    finally:
        st.session_state.processing = False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

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

        if st.session_state.pdf_processed:

            st.markdown(
                '<span class="status-pill status-ready">'
                "● Document ready"
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

        uploaded_file = st.file_uploader(
            "Upload a research paper (PDF)",
            type=["pdf"],
            help=(
                f"Upload one academic PDF. "
                f"Maximum size: {MAX_PDF_SIZE_MB} MB."
            ),
        )

        if uploaded_file is not None:

            file_key = (
                f"{uploaded_file.name}:"
                f"{uploaded_file.size}"
            )

            is_new_file = (
                st.session_state.get("last_upload_key")
                != file_key
            )

            if is_new_file:

                st.session_state.failed_file_key = ""

                # If a different document is uploaded,
                # immediately clear the old document state.
                if st.session_state.pdf_processed:
                    _reset_document_state()

                st.session_state.last_upload_key = file_key

            if (
                st.session_state.failed_file_key == file_key
            ):

                st.warning(
                    "The last processing attempt failed."
                )

                if st.button(
                    "Retry processing",
                    use_container_width=True,
                ):
                    st.session_state.failed_file_key = ""
                    st.rerun()

            elif not st.session_state.pdf_processed:

                _process_pdf(
                    uploaded_file,
                    file_key,
                )

        else:

            # Don't destroy the active document merely because
            # Streamlit temporarily returns no uploader value.
            pass

        if st.session_state.pdf_processed:

            st.caption(
                f"**Active file:** "
                f"{st.session_state.get('last_filename', '—')}"
            )

            st.caption(
                f"{len(st.session_state.chunks):,} "
                "retrieval chunks"
            )

        st.divider()

        st.markdown(
            """
            <div class="about-box">
                <strong>About Scholar</strong><br><br>
                Local embeddings + Chroma vector search +
                BM25 keyword retrieval + Gemini generation.
                <br><br>
                PDFs and embeddings remain session-scoped.
                Gemini is only called when you explicitly
                request an AI-generated response.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Document metrics
# ---------------------------------------------------------------------------

def _render_document_metrics() -> None:
    """Display basic document statistics."""

    if not st.session_state.pdf_processed:
        return

    col1, col2, col3 = st.columns(3)

    filename = st.session_state.get(
        "last_filename",
        "Document",
    )

    with col1:
        st.metric(
            "Document",
            filename[:28]
            + ("…" if len(filename) > 28 else ""),
        )

    with col2:
        st.metric(
            "Text Sections",
            f"{len(st.session_state.chunks):,}",
        )

    with col3:
        character_count = len(
            st.session_state.full_text
        )

        st.metric(
            "Characters Extracted",
            f"{character_count:,}",
        )


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

def _render_hero() -> None:
    """Render application header."""

    st.markdown(
        """
        <div class="scholar-hero">
            <h1>Scholar</h1>
            <p>
                AI-powered research paper assistant —
                ask questions, generate summaries,
                and explore citations grounded in your document.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _render_summary_tab() -> None:
    """Render structured paper summary."""

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

    if not st.session_state.pdf_processed:

        _render_empty_state(
            "No document loaded",
            "Upload a PDF in the sidebar to generate a summary.",
        )

        return

    if st.session_state.summary:

        with st.container(border=True):
            st.markdown(
                st.session_state.summary
            )

        col1, col2, col3 = st.columns(
            [1, 1, 1]
        )

        filename = (
            st.session_state
            .get("last_filename", "paper")
            .rsplit(".", 1)[0]
        )

        with col1:
            st.download_button(
                "Download .md",
                st.session_state.summary,
                file_name=f"{filename}_summary.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with col2:
            st.download_button(
                "Download .txt",
                st.session_state.summary,
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
                st.session_state.summary = ""
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

                summary = generate_structured_summary(
                    st.session_state.full_text,
                    st.session_state.pipeline,
                    api_key,
                )

                progress.empty()

                st.session_state.summary = summary

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
    """Render extracted bibliography."""

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

    if not st.session_state.pdf_processed:

        _render_empty_state(
            "No document loaded",
            "Upload a PDF to extract its references.",
        )

        return

    if st.session_state.citations:

        with st.container(border=True):
            st.markdown(
                st.session_state.citations
            )

        col1, col2, col3 = st.columns(
            [1, 1, 1]
        )

        filename = (
            st.session_state
            .get("last_filename", "paper")
            .rsplit(".", 1)[0]
        )

        with col1:
            st.download_button(
                "Download .md",
                st.session_state.citations,
                file_name=f"{filename}_citations.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with col2:
            st.download_button(
                "Download .txt",
                st.session_state.citations,
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
                st.session_state.citations = ""
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

                citations = format_citations(
                    st.session_state.full_text,
                    st.session_state.pipeline,
                    api_key,
                )

                progress.empty()

                st.session_state.citations = citations

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

    tab_chat, tab_summary, tab_citations = st.tabs(
        [
            "Chat",
            "Summary",
            "Citations",
        ]
    )

    with tab_chat:
        _render_chat_tab()

    with tab_summary:
        _render_summary_tab()

    with tab_citations:
        _render_citations_tab()

    st.markdown(
        '<p class="scholar-footer">'
        "Scholar · Hybrid RAG · "
        "Local Vector + BM25 Retrieval"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()