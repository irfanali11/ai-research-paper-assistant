# Data Handling Requirements

## Session-Scoped Data

- Uploaded PDFs are session-scoped only — files are not persisted to disk beyond the active session unless explicitly cached for performance within that session.
- No user data, questions, or documents are logged to any external service.

## Permitted External Network Calls

The only external network calls permitted are:

1. **Google Gemini API** — for LLM generation (question answering, structured summaries, citation formatting).
2. **No external call for embeddings** — the embedding model (`sentence-transformers/all-MiniLM-L6-v2`) runs locally with no API key required.

## Error Handling

The application handles and displays clear, user-friendly error messages for:

- Corrupted or unreadable PDFs
- Empty text extraction (e.g., scanned image-only documents)
- API failures or rate limits
- Empty or invalid questions

Raw stack traces are never surfaced to the user.
