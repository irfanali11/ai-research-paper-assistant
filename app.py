"""Streamlit UI entry point for the Scholar research paper assistant."""

from __future__ import annotations

import streamlit as st

from pdf_processor import PDFProcessingError, chunk_text, extract_text_from_pdf
from rag_pipeline import RAGPipeline, RAGPipelineError
from summary_and_citations import SummaryError, format_citations, generate_structured_summary


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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _process_pdf(uploaded_file) -> None:
    """Extract text, chunk, and index an uploaded PDF."""
    try:
        full_text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(full_text)

        if not chunks:
            st.error("No text could be chunked from this PDF.")
            return

        pipeline = RAGPipeline()
        with st.spinner("Generating embeddings and building search index…"):
            pipeline.index_chunks(chunks)

        st.session_state.pipeline = pipeline
        st.session_state.full_text = full_text
        st.session_state.chunks = chunks
        st.session_state.chat_history = []
        st.session_state.summary = ""
        st.session_state.citations = ""
        st.session_state.pdf_processed = True
        st.success(f"Processed successfully — {len(chunks)} chunks indexed.")

    except PDFProcessingError as exc:
        st.error(str(exc))
    except RAGPipelineError as exc:
        st.error(str(exc))
    except Exception:
        st.error("An unexpected error occurred while processing the PDF.")


def _render_sidebar() -> None:
    """Render the sidebar with upload and about sections."""
    with st.sidebar:
        st.title("Scholar")
        st.caption("AI Research Paper Assistant")

        uploaded_file = st.file_uploader(
            "Upload a research paper (PDF)",
            type=["pdf"],
            help="Upload a single academic paper to analyze.",
        )

        if uploaded_file is not None:
            if (
                not st.session_state.pdf_processed
                or st.session_state.get("last_filename") != uploaded_file.name
            ):
                st.session_state.last_filename = uploaded_file.name
                with st.spinner("Extracting text from PDF…"):
                    _process_pdf(uploaded_file)

        st.divider()
        st.markdown(
            "**About this project**\n\n"
            "Scholar is a retrieval-augmented generation (RAG) tool that lets you "
            "upload an academic paper and interact with it using natural language. "
            "It extracts text from your PDF, indexes it with local embeddings, "
            "and uses a large language model to answer questions, generate structured "
            "summaries, and extract citations — all grounded in the paper's content.\n\n"
            "Built as a portfolio project demonstrating applied AI engineering."
        )


def _render_chat_tab() -> None:
    """Render the chat interface tab."""
    st.subheader("Ask a Question")

    if not st.session_state.pdf_processed:
        st.info("Upload a PDF in the sidebar to start asking questions.")
        return

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about the paper…")

    if question:
        if not question.strip():
            st.warning("Please enter a valid question.")
            return

        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        api_key = _get_api_key()
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    answer = st.session_state.pipeline.answer_question(question, api_key)
                    st.markdown(answer)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )
                except RAGPipelineError as exc:
                    st.error(str(exc))
                except Exception:
                    st.error("An unexpected error occurred while generating an answer.")


def _render_summary_tab() -> None:
    """Render the structured summary tab."""
    st.subheader("Paper Summary")

    if not st.session_state.pdf_processed:
        st.info("Upload a PDF in the sidebar to generate a summary.")
        return

    if st.session_state.summary:
        st.markdown(st.session_state.summary)
        if st.button("Regenerate Summary"):
            st.session_state.summary = ""
            st.rerun()
        return

    if st.button("Generate Summary", type="primary"):
        api_key = _get_api_key()
        with st.spinner("Generating structured summary…"):
            try:
                summary = generate_structured_summary(
                    st.session_state.full_text,
                    st.session_state.pipeline,
                    api_key,
                )
                st.session_state.summary = summary
                st.rerun()
            except SummaryError as exc:
                st.error(str(exc))
            except Exception:
                st.error("An unexpected error occurred while generating the summary.")


def _render_citations_tab() -> None:
    """Render the key citations tab."""
    st.subheader("Key Citations")

    if not st.session_state.pdf_processed:
        st.info("Upload a PDF in the sidebar to extract citations.")
        return

    if st.session_state.citations:
        st.markdown(st.session_state.citations)
        if st.button("Re-extract Citations"):
            st.session_state.citations = ""
            st.rerun()
        return

    if st.button("Extract Citations", type="primary"):
        api_key = _get_api_key()
        with st.spinner("Extracting and formatting citations…"):
            try:
                citations = format_citations(
                    st.session_state.full_text,
                    st.session_state.pipeline,
                    api_key,
                )
                st.session_state.citations = citations
                st.rerun()
            except SummaryError as exc:
                st.error(str(exc))
            except Exception:
                st.error("An unexpected error occurred while extracting citations.")


def main() -> None:
    """Run the Scholar Streamlit application."""
    st.set_page_config(
        page_title="Scholar — AI Research Paper Assistant",
        page_icon="📄",
        layout="wide",
    )

    _init_session_state()
    _render_sidebar()

    st.title("Scholar")
    st.caption("Ask questions, get summaries, and explore citations from your research paper.")

    tab_chat, tab_summary, tab_citations = st.tabs(["Chat", "Summary", "Citations"])

    with tab_chat:
        _render_chat_tab()

    with tab_summary:
        _render_summary_tab()

    with tab_citations:
        _render_citations_tab()


if __name__ == "__main__":
    main()
