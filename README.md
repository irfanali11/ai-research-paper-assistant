# Scholar — AI Research Paper Assistant

A retrieval-augmented generation (RAG) web app for academic PDFs. Upload a research paper, ask questions in natural language, generate a structured summary, and extract citations — all grounded in the document's content.

Built as a portfolio project demonstrating applied AI engineering for Master's admissions (CS, IT, AI, Software Engineering, Data Science) and technical recruiting.

**Repository:** [github.com/irfanali11/ai-research-paper-assistant](https://github.com/irfanali11/ai-research-paper-assistant)

---

## Features

### Document processing
- **PDF upload** — Session-scoped, single-document analysis via the sidebar
- **Section-aware chunking** — Splits text on academic headings (Abstract, Methods, Results, etc.)
- **Processing pipeline** — Step-by-step progress: Extract → Chunk → Embed → Index
- **Document metrics** — Filename, section count, and character count after upload

### Chat & Q&A
- **ChatGPT-style interface** — Welcome screen, message history, and suggested questions
- **Hybrid retrieval** — Dense vector search + BM25 fused with reciprocal rank fusion (RRF)
- **Streaming answers** — Token-by-token response generation via Gemini
- **Source citations** — Expandable references showing retrieved chunk indices and excerpts
- **Grounded prompting** — Answers constrained to retrieved context; explicit fallback when context is insufficient

### Summary & citations
- **Structured summary** — Four sections: Research Question, Methodology, Key Findings, Limitations
- **Citation extraction** — Regex parsing of the References section with LLM-assisted formatting
- **Export** — Download summaries and citations as `.md` or `.txt`

### UI & performance
- **Light / dark theme** — Toggle in the sidebar
- **Cached embedding model** — `@st.cache_resource` avoids reloading the model on every rerun
- **Batch embedding** — Chunks encoded in batches of 32 for faster indexing

### Evaluation
- **`evaluation.py`** — CLI script to measure retrieval quality with keyword hit-rate on sample questions

---

## How It Works

```
PDF Upload
    ↓
Text Extraction (pdfplumber)
    ↓
Section-Aware Chunking
    ↓
Local Embeddings (all-MiniLM-L6-v2) ──→ Chroma Vector Store
    ↓                                        +
BM25 Index (rank-bm25) ──────────────────→ Hybrid Retrieval (RRF)
                                                ↓
User Question ──────────────────────→ Top-K Chunks → Gemini API → Grounded Answer
```

No models are trained or fine-tuned. The pipeline uses off-the-shelf pretrained components only.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ |
| Frontend | Streamlit |
| PDF parsing | pdfplumber |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) — local, free |
| Sparse retrieval | rank-bm25 (BM25Okapi) |
| Vector database | Chroma (in-memory) |
| LLM | Google Gemini API (`gemini-2.5-flash`) — free tier |
| Fusion | Reciprocal Rank Fusion (RRF, k=60) |

---

## APIs Used

| Service | Purpose | Cost |
|---------|---------|------|
| **Google Gemini API** | Q&A, structured summaries, citation formatting | Free tier ([get a key](https://aistudio.google.com/apikey)) |
| **sentence-transformers** | Local embedding generation | Free — no API key required |

The only external network call is to the Gemini API. Embeddings run entirely on your machine.

---

## Getting Started

### Prerequisites

- Python 3.11 or later
- A free [Google Gemini API key](https://aistudio.google.com/apikey)

### Installation

```bash
git clone https://github.com/irfanali11/ai-research-paper-assistant.git
cd ai-research-paper-assistant
pip install -r requirements.txt
```

### Configure secrets

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-gemini-api-key-here"
```

> **Never commit `secrets.toml`.** It is listed in `.gitignore`.

### Run

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Usage

1. **Upload** a PDF research paper via the sidebar.
2. Wait for processing (Extract → Chunk → Embed → Index).
3. **Chat** — ask questions; answers stream in with source citations.
4. **Summary** — click *Generate Summary* for a structured four-section overview.
5. **Citations** — click *Extract Citations* for a cleaned reference list.
6. **Export** — download summary or citations as `.md` / `.txt`.
7. **Theme** — toggle light/dark mode from the sidebar.

### Suggested questions

After upload, try:
- *What is the main research question?*
- *What methods were used in this study?*
- *What are the key findings?*

---

## Retrieval evaluation

Run the evaluation script from the project root to measure retrieval keyword hit-rate on a sample PDF:

```bash
python evaluation.py path/to/sample.pdf
```

The script indexes the PDF, runs five standard academic questions, and reports per-question and average hit rates.

---

## Project structure

```
scholar/
├── app.py                    # Streamlit UI (chat, summary, citations, themes)
├── pdf_processor.py          # PDF extraction + section-aware chunking
├── rag_pipeline.py           # Embeddings, Chroma, hybrid retrieval, LLM calls
├── summary_and_citations.py  # Structured summary + citation extraction
├── evaluation.py             # Retrieval evaluation CLI
├── requirements.txt
├── .streamlit/
│   ├── config.toml           # Theme and server settings
│   └── secrets.toml.example  # API key template
├── README.md
└── DATA_HANDLING.md          # Privacy and data-flow notes
```

---

## Data handling

Uploaded PDFs are **session-scoped only** — nothing is persisted to disk beyond the active session. No user data, questions, or document content is logged to external services (except the text sent to Gemini for generation).

See [DATA_HANDLING.md](DATA_HANDLING.md) for full details.

---

## Error handling

The app surfaces clear, user-friendly messages for:

- Corrupted or unreadable PDFs
- Scanned/image-only PDFs with no extractable text
- Missing or invalid API keys
- Gemini API rate limits (with automatic retry/backoff)
- Empty or invalid questions

Raw stack traces are never shown in the UI.

---

## Live demo

_Demo link coming soon — deploy via [Streamlit Community Cloud](https://streamlit.io/cloud) and add `GEMINI_API_KEY` in the app secrets._

---

## Roadmap

- [ ] Deploy live demo on Streamlit Cloud
- [ ] Expanded evaluation (Recall@k, vector vs BM25 vs hybrid ablation)
- [ ] Technical report with experiment results
- [ ] Unit tests and GitHub Actions CI

---

## Author

**Irfan Ali** — [GitHub](https://github.com/irfanali11)

---

## License

This project is open source and available for portfolio and educational use.
