# Scholar

**An academic research paper assistant built on a hybrid retrieval-augmented generation (RAG) pipeline, with explicit faithfulness evaluation and multi-document support.**

Scholar lets you upload one or more academic PDFs and ask questions, generate structured summaries, extract formatted citations, and discover related work — with every generated answer traceable back to the source text and automatically checked for factual grounding before you have to trust it.

This project was built to explore a specific question: *how do you make a RAG system's answers verifiable rather than just plausible?* Most of the engineering decisions below exist to answer that question, not to add features for their own sake.

---

## Why this project is structured the way it is

Naive RAG — embed a query, grab the top-k nearest chunks, stuff them in a prompt — is easy to build and easy to get subtly wrong. It has three well-known failure modes that Scholar is deliberately built to address:

1. **Embedding similarity is not the same as relevance.** A single dense vector search can miss chunks that share exact keywords with the query but drift slightly in phrasing, or vice versa.
2. **LLMs hallucinate confidently.** A grounded-sounding answer with citations is not the same as an accurate one.
3. **Compound questions get answered badly.** A single embedding of a two-part question represents neither part well.

Each of the three sections below maps directly to one of these problems.

---

## Architecture

```
pdf_processor.py       → PDF extraction, section-aware chunking
rag_pipeline.py         → hybrid retrieval, reranking, generation, faithfulness eval
summary_and_citations.py → structured summaries, citation extraction
semantic_scholar.py     → related-paper discovery (free, keyless API)
evaluation.py           → CLI benchmarking (retrieval + faithfulness + ablations)
app.py                  → Streamlit UI
```

Each file owns exactly one responsibility. This isn't incidental — it's what makes the evaluation harness (`evaluation.py`) possible at all: it can exercise `rag_pipeline.py`'s retrieval logic directly, with no UI dependency, which is what lets the numbers below be reproducible from the command line.

### 1. Retrieval — solving "embedding similarity ≠ relevance"

- **Dense retrieval**: `BAAI/bge-base-en-v1.5` embeddings (upgraded from `all-MiniLM-L6-v2` after evaluation showed the smaller model was missing content a hybrid+reranked pipeline could find — see [Findings](#findings-from-testing) below), stored in a persistent Chroma vector index.
- **Sparse retrieval**: BM25 keyword search, run in parallel with dense retrieval.
- **Fusion**: the two ranked lists are merged with **Reciprocal Rank Fusion (RRF)**, which combines rankings without requiring the two methods' scores to be on the same scale.
- **Reranking**: a cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores the fused candidates by processing the query and each chunk *together*, which is more precise than comparing independently-computed embedding vectors.
- **Diversity filtering**: near-duplicate chunks (measured via Jaccard similarity on tokens) are removed before the final selection, so the LLM's context window isn't wasted on redundant text.
- **Query decomposition**: compound or comparative questions (*"compare X's methodology and limitations"*) are split into standalone sub-questions by the LLM, each retrieved independently, then fused back together with the same RRF mechanism — one more layer of the same algorithm already used for dense+sparse fusion. A cheap heuristic detects genuinely simple questions and skips the extra LLM call entirely for those.

### 2. Faithfulness evaluation — solving "confident ≠ accurate"

After an answer is generated, an independent evaluation step extracts every individual factual claim from it and checks each one against the retrieved source chunks, returning:
- a 0.0–1.0 faithfulness score
- which specific claims are unsupported
- an explanation of *why*

This is opt-in (it costs one extra LLM call) and has caught real issues during testing — including a case where an answer cited the wrong source number for a correctly-stated fact, which the evaluator flagged as a claim-attribution mismatch rather than treating "the fact was true" as sufficient.

### 3. Multi-document support and scoping

All loaded documents live in a single Chroma collection, each chunk tagged with a `document_id` and a stable `global_index`. Chat can be scoped to one, several, or all loaded papers via a metadata filter (`where={"document_id": {"$in": doc_ids}}`) applied before both the dense and sparse retrieval steps — not just filtered afterward, which would waste retrieval budget on documents that were never going to be searched.

---

## Findings from testing

These numbers come from `evaluation.py`, run against real papers, not curated to look good. Reproducing them is one command (`python evaluation.py <pdf> --faithfulness --decompose-ablation --output benchmark.json`).

### Embedder comparison (MiniLM → BGE)

Before the embedder swap, retrieval on a real pharmacology paper (36 chunks) returned **"insufficient sources"** for straightforward, answerable questions — including *"What is [the paper's core subject] defined as?"*, despite that definition being explicitly present in the paper. After switching to BGE-base and adding the retrieval-appropriate query instruction prefix BGE expects, the same questions returned complete, correctly-cited answers, including a case where the retrieved content correctly identified a nuanced discussion the paper had (the tension between two related governance frameworks) that MiniLM had missed entirely.

### Retrieval and faithfulness benchmark (IMRAD writing-guide paper, 24 chunks)

| Metric | Result |
|---|---|
| Average keyword hit-rate (5 standard questions) | 45% |
| Average faithfulness score | 93% (30/31 claims supported) |
| Section-retrieval match | 0% |

The 0% section-match number looks alarming in isolation but has a specific, diagnosable cause, not a retrieval failure: this particular PDF's section headings weren't detected by the heading-recognition regex in `pdf_processor.py`, so every chunk was labeled `"Preamble"` rather than `"Methods"`/`"Results"`/etc. Since the evaluation checks whether a retrieved chunk's *labeled* section matches an expected section name, and no chunk had the expected label, the metric could not succeed regardless of retrieval quality. Manual testing confirmed the actual retrieved content was correct — the keyword hit-rate (which doesn't depend on section labeling) is the more trustworthy number for this document. This is left in the README rather than fixed and hidden, because it's a real, specific limitation of regex-based heading detection worth documenting honestly.

### Query decomposition ablation

On the same 24-chunk document, decomposition showed **no measurable improvement** (0% delta in hit-rate) on compound test questions. The likely explanation: with only 24 chunks in one undifferentiated section, a single query's retrieval already covers most of the searchable content — there's little room for splitting the question to surface anything a single query would have missed. This produced a testable hypothesis (decomposition's value should scale with document length and structural diversity), currently being validated against a longer, more structurally varied paper.

---

## Known limitations

Stated plainly, because a system's boundaries are as informative as its capabilities:

- **Heading detection is regex-based** and fails silently on PDFs with non-standard section formatting, degrading the section-match evaluation metric (though not necessarily actual retrieval quality — see above).
- **The query-decomposition fast-path is a prefix heuristic**, not a full semantic classifier. A genuinely compound question phrased as a sequence of "What is X? What is Y?" can be misclassified as simple and skip decomposition, since the heuristic only inspects the question's opening words.
- **Citation extraction on long reference lists** can still under-deliver on a single generation attempt; a completeness check with automatic retry mitigates this but does not eliminate it for very long bibliographies.
- **The faithfulness evaluator uses the same model family (Gemini) that generated the original answer**, which introduces a theoretical risk of correlated blind spots between generation and evaluation — an independent second model as judge would be a stronger design, not yet implemented.

---

## Tech stack

| Component | Choice | Why |
|---|---|---|
| Embeddings | `BAAI/bge-base-en-v1.5` (local, free) | Retrieval-tuned training objective, outperformed general-purpose MiniLM in testing |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, free) | Query-document joint scoring, more precise than bi-encoder similarity alone |
| Vector store | ChromaDB (local, persistent) | Purpose-built approximate nearest-neighbor search; avoids brute-force linear scan a relational store would require |
| Keyword search | BM25 (`rank_bm25`) | Catches exact/rare-term matches embeddings can miss |
| LLM | Gemini 2.5 Flash | Generation, faithfulness evaluation, query decomposition |
| PDF processing | `pdfplumber` | Text, table, and figure-caption extraction |
| Related-paper search | Semantic Scholar Academic Graph API | Free, keyless |
| UI | Streamlit | Rapid iteration for a research-tool interface |

No paid APIs are required beyond Gemini's free tier. Embeddings and reranking run entirely locally.

---

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Add your Gemini API key to `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-key-here"
```

### Running the evaluation harness

```bash
# Single paper, retrieval-only (no API cost)
python evaluation.py papers/example.pdf

# Full evaluation: faithfulness + decomposition ablation
export GEMINI_API_KEY="your-key-here"
python evaluation.py papers/example.pdf --faithfulness --decompose-ablation --output benchmark.json

# Multiple papers, evaluated with per-document scoping
python evaluation.py papers/paper1.pdf papers/paper2.pdf --output benchmark.json
```

---

## What I'd build next

- An independent second model as faithfulness judge, to remove the same-model evaluation risk noted above.
- Replacing regex-based heading detection with a lighter statistical/NLP approach for more robust section labeling across inconsistently-formatted PDFs.
- Completing the decomposition ablation across documents of varying length to properly test the "value scales with complexity" hypothesis rather than leaving it as a single data point.
