"""Streamlit UI entry point for the Scholar research paper assistant."""

from __future__ import annotations

import html
from dataclasses import dataclass

import streamlit as st
from sentence_transformers import SentenceTransformer

from pdf_processor import PDFProcessingError, chunk_text, extract_text_from_pdf
from rag_pipeline import EMBEDDING_MODEL_NAME, RAGPipeline, RAGPipelineError
from summary_and_citations import SummaryError, format_citations, generate_structured_summary


@dataclass
class _SourceRefs:
    """Lightweight container for retrieved chunk references in the UI layer."""

    chunks: list[str]
    indices: list[int]

EXAMPLE_QUESTIONS = [
    "What is the main research question?",
    "What methods were used in this study?",
    "What are the key findings?",
]

PROCESSING_STEPS = [
    "Extracting text from PDF",
    "Chunking by academic sections",
    "Generating embeddings",
    "Building hybrid search index",
]


def _inject_styles(dark_mode: bool) -> None:
    """Inject custom CSS for a clean, academic visual style.

    Args:
        dark_mode: Whether to apply the dark color palette.
    """
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
        chat_panel_bg = "#111827"
        chat_user_bg = "#1e3a5f"
        chat_assist_bg = "#1e293b"
        widget_rules = f"""
            p, li, label, .stCaption, [data-testid="stMarkdown"] {{
                color: {text} !important;
            }}
            div[data-testid="stVerticalBlockBorderWrapper"] {{
                background: {surface} !important;
                border-color: {border} !important;
            }}
            [data-testid="stChatMessage"] {{
                background: {surface} !important;
                border: 1px solid {border};
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
        chat_panel_bg = "#ffffff"
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
                background: linear-gradient(135deg, {hero_start} 0%, {hero_end} 100%);
                border-radius: 12px;
                padding: 2rem 2.25rem;
                margin: 0 0 1.75rem 0;
                color: #ffffff;
                clear: both;
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
                line-height: 1.3;
                display: block;
                clear: both;
            }}
            .section-subtitle {{
                color: {muted};
                font-size: 0.95rem;
                margin: 0 0 1.25rem 0;
                line-height: 1.5;
                display: block;
                clear: both;
            }}
            .status-pill {{
                display: inline-block;
                padding: 0.35rem 0.75rem;
                border-radius: 999px;
                font-size: 0.8rem;
                font-weight: 600;
                margin: 0.5rem 0 1rem 0;
                clear: both;
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
                clear: both;
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
            .about-box strong {{ color: {accent}; }}
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

            /* —— Chat panel (ChatGPT-style) —— */
            .chat-panel-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.75rem 1rem;
                border-bottom: 1px solid {border};
                margin: -1rem -1rem 0.75rem -1rem;
                background: {chat_assist_bg};
                border-radius: 12px 12px 0 0;
            }}
            .chat-panel-header span {{
                font-weight: 600;
                color: {accent};
                font-size: 0.95rem;
            }}
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
            .chat-sources-label {{
                font-size: 0.8rem;
                color: {muted};
                margin-top: 0.75rem;
                padding-top: 0.75rem;
                border-top: 1px solid {border};
            }}
            div[data-testid="stChatMessage"] {{
                padding: 0.85rem 0.5rem !important;
                margin-bottom: 0.25rem;
                background: transparent !important;
                border: none !important;
            }}
            div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {{
                font-size: 0.95rem;
                line-height: 1.65;
            }}
            div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
                background: {chat_user_bg} !important;
                border-radius: 12px;
                padding: 0.85rem 1rem !important;
                margin: 0.35rem 0 0.35rem 2rem !important;
            }}
            div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {{
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
            [data-testid="stChatInput"] textarea {{
                border-radius: 12px !important;
            }}
            .typing-indicator {{
                color: {muted};
                font-size: 0.88rem;
                font-style: italic;
                padding: 0.25rem 0;
            }}

            .scholar-footer {{
                text-align: center;
                color: {muted};
                font-size: 0.8rem;
                margin-top: 2.5rem;
                padding-top: 1rem;
                border-top: 1px solid {border};
                clear: both;
            }}
            {widget_rules}
            #MainMenu {{visibility: hidden;}}
            footer {{visibility: hidden;}}
            @media (max-width: 768px) {{
                .main .block-container {{ padding: 1rem 0.75rem 2rem; }}
                .scholar-hero {{ padding: 1.25rem 1.5rem; }}
                .scholar-hero h1 {{ font-size: 1.5rem; }}
                .scholar-hero p {{ font-size: 0.95rem; }}
                [data-testid="stTabs"] [data-baseweb="tab"] {{
                    padding: 0.5rem 0.75rem;
                    font-size: 0.9rem;
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def _load_embedder() -> SentenceTransformer:
    """Load and cache the sentence-transformers embedding model.

    Returns:
        A cached SentenceTransformer instance.
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def _create_pipeline(embedder: SentenceTransformer) -> RAGPipeline:
    """Create a RAGPipeline wired to a cached embedding model.

    Args:
        embedder: Pre-loaded SentenceTransformer instance.

    Returns:
        A configured RAGPipeline.
    """
    import inspect

    params = inspect.signature(RAGPipeline.__init__).parameters
    if "embedder" in params:
        return RAGPipeline(embedder=embedder)

    pipeline = RAGPipeline()
    pipeline._embedder = embedder
    return pipeline


def _get_api_key() -> str:
    """Read the Google Gemini API key from Streamlit secrets."""
    try:
        return st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        return ""


def _init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    defaults = {
        "pipeline": None,
        "full_text": "",
        "chunks": [],
        "chat_history": [],
        "summary": "",
        "citations": "",
        "pdf_processed": False,
        "failed_file_key": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _format_source_label(indices: list[int]) -> str:
    """Format retrieved chunk indices as a human-readable label.

    Args:
        indices: 1-based chunk indices.

    Returns:
        A label string such as 'Based on sections 2, 4'.
    """
    if not indices:
        return ""
    joined = ", ".join(str(i) for i in indices)
    noun = "section" if len(indices) == 1 else "sections"
    return f"Based on {noun} {joined}"


def _render_source_citations(indices: list[int], chunks: list[str]) -> None:
    """Render source citations in a compact expander below an assistant message.

    Args:
        indices: 1-based chunk indices.
        chunks: Retrieved chunk texts.
    """
    if not indices:
        return

    label = _format_source_label(indices)
    section_list = ", ".join(f"§{i}" for i in indices)
    with st.expander(f"Sources · {section_list}", expanded=False):
        st.caption(label)
        for idx, chunk in zip(indices, chunks):
            preview = chunk[:500] + ("…" if len(chunk) > 500 else "")
            st.markdown(f"**Section {idx}**")
            st.markdown(f"> {html.escape(preview)}")
            if idx != indices[-1]:
                st.divider()


def _chat_needs_response() -> bool:
    """Return True if the last chat message is from the user awaiting a reply."""
    history = st.session_state.chat_history
    return bool(history) and history[-1]["role"] == "user"


def _render_chat_welcome() -> None:
    """Render the empty-state welcome message inside the chat panel."""
    st.markdown(
        """
        <div class="chat-welcome">
            <h3>Ask anything about your paper</h3>
            <p>Scholar retrieves relevant sections from your document and generates
            a grounded, cited answer — like a research assistant.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_chat_message(message: dict) -> None:
    """Render a single message from chat history.

    Args:
        message: A dict with role, content, and optional source fields.
    """
    avatar = "🧑‍🎓" if message["role"] == "user" else "📚"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("source_indices"):
            _render_source_citations(
                message["source_indices"],
                message.get("source_chunks", []),
            )


def _resolve_answer_result(pipeline: RAGPipeline, question: str, api_key: str):
    """Call answer_question and normalize legacy or modern return types."""
    result = pipeline.answer_question(question, api_key)
    if hasattr(result, "answer"):
        return result.answer, result.sources

    answer = str(result)
    sources = pipeline.retrieve(question)
    if hasattr(sources, "indices"):
        return answer, sources
    chunks = list(sources)
    indices = list(range(1, len(chunks) + 1))
    return answer, _SourceRefs(chunks=chunks, indices=indices)


def _generate_assistant_response(question: str) -> None:
    """Stream an assistant reply and append it to chat history.

    Args:
        question: The user's question to answer.
    """
    api_key = _get_api_key()
    pipeline = st.session_state.pipeline

    with st.chat_message("assistant", avatar="📚"):
        status = st.empty()
        status.markdown(
            '<p class="typing-indicator">Retrieving relevant sections…</p>',
            unsafe_allow_html=True,
        )

        try:
            if hasattr(pipeline, "stream_answer_question"):
                stream_iter, sources = pipeline.stream_answer_question(question, api_key)
                status.markdown(
                    '<p class="typing-indicator">Scholar is thinking…</p>',
                    unsafe_allow_html=True,
                )

                def _stream_tokens():
                    for token in stream_iter:
                        yield token

                answer = st.write_stream(_stream_tokens)
            elif hasattr(pipeline, "answer_question"):
                status.markdown(
                    '<p class="typing-indicator">Scholar is thinking…</p>',
                    unsafe_allow_html=True,
                )
                answer, sources = _resolve_answer_result(pipeline, question, api_key)
                st.markdown(answer)
            else:
                raise RAGPipelineError(
                    "Chat is unavailable — please re-upload your PDF to refresh the session."
                )

            status.empty()

            if answer:
                _render_source_citations(sources.indices, sources.chunks)
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "source_indices": sources.indices,
                        "source_chunks": sources.chunks,
                    }
                )
            else:
                st.warning("No response received. Please try again.")

        except RAGPipelineError as exc:
            status.empty()
            st.error(str(exc))
        except Exception as exc:
            status.empty()
            try:
                answer, sources = _resolve_answer_result(pipeline, question, api_key)
                st.markdown(answer)
                _render_source_citations(sources.indices, sources.chunks)
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "source_indices": sources.indices,
                        "source_chunks": sources.chunks,
                    }
                )
            except Exception:
                st.error(
                    f"Could not generate an answer. Detail: {exc}. "
                    "Try re-uploading your PDF or waiting a minute if rate-limited."
                )


def _submit_user_message(question: str) -> None:
    """Append a user message and trigger a rerun to generate the reply.

    Args:
        question: The user's message text.
    """
    if not question.strip():
        return
    st.session_state.chat_history.append({"role": "user", "content": question.strip()})
    st.rerun()


def _render_example_questions() -> None:
    """Render suggested prompt chips below the welcome message."""
    st.markdown(
        '<p style="text-align:center;color:#64748b;font-size:0.85rem;margin:0 0 0.5rem 0;">'
        "Suggested questions"
        "</p>",
        unsafe_allow_html=True,
    )
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, example in zip(cols, EXAMPLE_QUESTIONS):
        with col:
            if st.button(example, key=f"example_{example}", use_container_width=True):
                _submit_user_message(example)


def _process_pdf(uploaded_file, file_key: str) -> None:
    """Extract text, chunk, and index an uploaded PDF with step-by-step progress."""
    try:
        uploaded_file.seek(0)
        embedder = _load_embedder()

        with st.status("Processing document…", expanded=True) as status:
            st.write(PROCESSING_STEPS[0])
            full_text = extract_text_from_pdf(uploaded_file)

            st.write(PROCESSING_STEPS[1])
            chunks = chunk_text(full_text)

            if not chunks:
                status.update(label="Processing failed", state="error")
                st.session_state.failed_file_key = file_key
                st.error("No text could be chunked from this PDF.")
                return

            st.write(PROCESSING_STEPS[2])
            pipeline = _create_pipeline(embedder)

            st.write(PROCESSING_STEPS[3])
            pipeline.index_chunks(chunks)

            status.update(label="Document ready", state="complete")

        st.session_state.pipeline = pipeline
        st.session_state.full_text = full_text
        st.session_state.chunks = chunks
        st.session_state.chat_history = []
        st.session_state.summary = ""
        st.session_state.citations = ""
        st.session_state.pdf_processed = True
        st.session_state.failed_file_key = ""
        st.success(f"Document indexed — {len(chunks)} sections ready for analysis.")

    except PDFProcessingError as exc:
        st.session_state.failed_file_key = file_key
        st.error(str(exc))
    except RAGPipelineError as exc:
        st.session_state.failed_file_key = file_key
        st.error(str(exc))
    except Exception as exc:
        st.session_state.failed_file_key = file_key
        st.error(
            f"Processing failed during embedding or indexing. "
            f"Try a smaller PDF or restart the app. Detail: {exc}"
        )


def _render_hero() -> None:
    """Render the main page hero header."""
    st.markdown(
        """
        <div class="scholar-hero">
            <h1>Scholar</h1>
            <p>AI-powered research paper assistant — ask questions, generate summaries,
            and explore citations grounded in your document.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_document_metrics() -> None:
    """Show document statistics when a PDF has been processed."""
    if not st.session_state.pdf_processed:
        return

    col1, col2, col3 = st.columns(3)
    filename = st.session_state.get("last_filename", "Document")
    with col1:
        st.metric("Document", filename[:28] + ("…" if len(filename) > 28 else ""))
    with col2:
        st.metric("Text Sections", len(st.session_state.chunks))
    with col3:
        char_count = len(st.session_state.full_text)
        st.metric("Characters Extracted", f"{char_count:,}")


def _render_empty_state(message: str, hint: str) -> None:
    """Render a styled empty-state placeholder."""
    st.markdown(
        f"""
        <div class="empty-state">
            <strong>{message}</strong>
            {hint}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar() -> None:
    """Render the sidebar with upload and about sections."""
    with st.sidebar:
        st.markdown("### Scholar")
        st.caption("Research Paper Assistant")
        st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

        st.toggle("Dark mode", key="dark_mode")

        if st.session_state.pdf_processed:
            st.markdown(
                '<span class="status-pill status-ready">● Document ready</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-pill status-waiting">○ Awaiting upload</span>',
                unsafe_allow_html=True,
            )

        uploaded_file = st.file_uploader(
            "Upload a research paper (PDF)",
            type=["pdf"],
            help="Upload a single academic paper to analyze.",
        )

        if uploaded_file is not None:
            file_key = f"{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.get("last_upload_key") != file_key:
                st.session_state.last_upload_key = file_key
                st.session_state.failed_file_key = ""

            if not st.session_state.pdf_processed:
                if st.session_state.get("failed_file_key") == file_key:
                    st.warning("Last upload failed. Click Retry or re-upload the PDF.")
                    if st.button("Retry processing", use_container_width=True):
                        st.session_state.failed_file_key = ""
                        st.rerun()
                else:
                    st.session_state.last_filename = uploaded_file.name
                    _process_pdf(uploaded_file, file_key)
        else:
            st.session_state.last_upload_key = ""

        if st.session_state.pdf_processed:
            st.caption(f"**Active file:** {st.session_state.get('last_filename', '—')}")

        st.divider()

        st.markdown(
            """
            <div class="about-box">
                <strong>About Scholar</strong><br><br>
                Hybrid RAG pipeline: vector search + BM25 keyword retrieval,
                section-aware chunking, and Gemini-powered generation.
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_chat_tab() -> None:
    """Render the ChatGPT-style chat interface tab."""
    if not st.session_state.pdf_processed:
        st.markdown('<p class="section-title">Chat</p>', unsafe_allow_html=True)
        _render_empty_state(
            "No document loaded",
            "Upload a PDF in the sidebar to start a conversation about your paper.",
        )
        return

    header_col, clear_col = st.columns([5, 1])
    with header_col:
        st.markdown('<p class="section-title" style="margin-top:0">Chat</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-subtitle" style="margin-bottom:0.75rem">'
            "Grounded answers with cited document sections"
            "</p>",
            unsafe_allow_html=True,
        )
    with clear_col:
        if st.session_state.chat_history:
            if st.button("Clear", type="secondary", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

    with st.container(border=True):
        if not st.session_state.chat_history:
            _render_chat_welcome()
            _render_example_questions()
        else:
            for message in st.session_state.chat_history:
                _render_chat_message(message)

        if _chat_needs_response():
            _generate_assistant_response(st.session_state.chat_history[-1]["content"])

    if prompt := st.chat_input("Message Scholar about your paper…"):
        _submit_user_message(prompt)


def _render_summary_tab() -> None:
    """Render the structured summary tab."""
    st.markdown('<p class="section-title">Paper Summary</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-subtitle">'
        "Structured overview: Research Question, Methodology, Key Findings, and Limitations."
        "</p>",
        unsafe_allow_html=True,
    )

    if not st.session_state.pdf_processed:
        _render_empty_state(
            "No document loaded",
            "Upload a PDF in the sidebar to generate a structured summary.",
        )
        return

    if st.session_state.summary:
        with st.container(border=True):
            st.markdown(st.session_state.summary)

        dl_col1, dl_col2, btn_col = st.columns([1, 1, 1])
        filename = st.session_state.get("last_filename", "paper").replace(".pdf", "")
        with dl_col1:
            st.download_button(
                "Download .md",
                st.session_state.summary,
                file_name=f"{filename}_summary.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with dl_col2:
            st.download_button(
                "Download .txt",
                st.session_state.summary,
                file_name=f"{filename}_summary.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with btn_col:
            if st.button("Regenerate", type="secondary", use_container_width=True):
                st.session_state.summary = ""
                st.rerun()
        return

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Generate Summary", type="primary", use_container_width=True):
            api_key = _get_api_key()
            progress = st.empty()
            progress.caption("Reading document structure…")
            try:
                progress.caption("Generating structured summary via Gemini…")
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
                st.error("An unexpected error occurred while generating the summary.")


def _render_citations_tab() -> None:
    """Render the key citations tab."""
    st.markdown('<p class="section-title">Key Citations</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-subtitle">'
        "Extracted and formatted reference list from the paper's bibliography."
        "</p>",
        unsafe_allow_html=True,
    )

    if not st.session_state.pdf_processed:
        _render_empty_state(
            "No document loaded",
            "Upload a PDF in the sidebar to extract the reference list.",
        )
        return

    if st.session_state.citations:
        with st.container(border=True):
            st.markdown(st.session_state.citations)

        dl_col1, dl_col2, btn_col = st.columns([1, 1, 1])
        filename = st.session_state.get("last_filename", "paper").replace(".pdf", "")
        with dl_col1:
            st.download_button(
                "Download .md",
                st.session_state.citations,
                file_name=f"{filename}_citations.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with dl_col2:
            st.download_button(
                "Download .txt",
                st.session_state.citations,
                file_name=f"{filename}_citations.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with btn_col:
            if st.button("Re-extract", type="secondary", use_container_width=True):
                st.session_state.citations = ""
                st.rerun()
        return

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Extract Citations", type="primary", use_container_width=True):
            api_key = _get_api_key()
            progress = st.empty()
            progress.caption("Locating References section…")
            try:
                progress.caption("Formatting citations via Gemini…")
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
                st.error("An unexpected error occurred while extracting citations.")


def main() -> None:
    """Run the Scholar Streamlit application."""
    st.set_page_config(
        page_title="Scholar — AI Research Paper Assistant",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _init_session_state()
    _render_sidebar()
    _inject_styles(st.session_state.get("dark_mode", False))
    _load_embedder()

    _render_hero()
    _render_document_metrics()

    tab_chat, tab_summary, tab_citations = st.tabs(["Chat", "Summary", "Citations"])

    with tab_chat:
        _render_chat_tab()

    with tab_summary:
        _render_summary_tab()

    with tab_citations:
        _render_citations_tab()

    st.markdown(
        '<p class="scholar-footer">Scholar · Hybrid RAG · Vector + BM25 retrieval</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
