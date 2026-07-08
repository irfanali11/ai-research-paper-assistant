"""Embedding generation, Chroma storage, hybrid retrieval, and LLM generation."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

import chromadb
import google.generativeai as genai
from chromadb.api.models.Collection import Collection
from google.api_core import exceptions as google_exceptions
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"
COLLECTION_NAME = "scholar_chunks"
MAX_LLM_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 5
RRF_K = 60


class RAGPipelineError(Exception):
    """Raised when the RAG pipeline encounters a recoverable error."""


@dataclass
class RetrievalResult:
    """Result of a hybrid retrieval query."""

    chunks: list[str]
    indices: list[int]


@dataclass
class AnswerResult:
    """LLM answer with retrieved source chunk references."""

    answer: str
    sources: RetrievalResult


class RAGPipeline:
    """Manages chunk embedding, vector storage, hybrid retrieval, and LLM generation."""

    def __init__(self, embedder: SentenceTransformer | None = None) -> None:
        """Initialize the embedding model and an in-memory Chroma collection.

        Args:
            embedder: Optional pre-loaded SentenceTransformer (for caching).
        """
        self._embedder = embedder or SentenceTransformer(EMBEDDING_MODEL_NAME)
        self._client = chromadb.Client()
        self._collection: Collection | None = None
        self._chunks: list[str] = []
        self._bm25: BM25Okapi | None = None

    def index_chunks(self, chunks: list[str]) -> None:
        """Embed chunks and store them in a fresh Chroma collection.

        Args:
            chunks: List of text chunks to embed and index.

        Raises:
            RAGPipelineError: If no chunks are provided.
        """
        if not chunks:
            raise RAGPipelineError("No text chunks available to index.")

        chunks = [c.strip() for c in chunks if c and c.strip()]
        if not chunks:
            raise RAGPipelineError("No text chunks available to index.")

        self._chunks = chunks
        tokenized = [chunk.lower().split() for chunk in chunks]
        self._bm25 = BM25Okapi(tokenized)

        try:
            self._client.delete_collection(COLLECTION_NAME)
        except (ValueError, Exception):
            pass

        self._collection = self._client.create_collection(name=COLLECTION_NAME)

        embeddings: list[list[float]] = []
        batch_size = 32
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            batch_embeddings = self._embedder.encode(batch, show_progress_bar=False).tolist()
            embeddings.extend(batch_embeddings)

        self._collection.add(
            ids=[str(i) for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings,
        )

    def _validate_query(self, question: str) -> str:
        """Validate and normalize a user question.

        Args:
            question: The user's natural-language question.

        Returns:
            Stripped question string.

        Raises:
            RAGPipelineError: If the question is blank or index is empty.
        """
        if not question or not question.strip():
            raise RAGPipelineError("Please enter a question before submitting.")

        if self._collection is None or self._collection.count() == 0:
            raise RAGPipelineError(
                "No document has been indexed yet. Please upload a PDF first."
            )

        return question.strip()

    def _reciprocal_rank_fusion(
        self,
        ranked_lists: list[list[int]],
        top_k: int,
    ) -> list[int]:
        """Merge ranked chunk indices using reciprocal rank fusion.

        Args:
            ranked_lists: Multiple ranked lists of chunk indices.
            top_k: Number of final results to return.

        Returns:
            Fused list of chunk indices ordered by combined score.
        """
        scores: dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, idx in enumerate(ranked):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

        sorted_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
        return sorted_indices[:top_k]

    def retrieve(self, question: str, top_k: int = 4) -> RetrievalResult:
        """Return the most relevant chunks using hybrid vector + BM25 retrieval.

        Args:
            question: The user's natural-language question.
            top_k: Number of top chunks to return.

        Returns:
            RetrievalResult with chunk texts and 1-based section indices.

        Raises:
            RAGPipelineError: If the index is empty or the question is blank.
        """
        question = self._validate_query(question)
        candidate_count = min(top_k * 3, len(self._chunks))

        query_embedding = self._embedder.encode([question], show_progress_bar=False).tolist()
        vector_results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=candidate_count,
        )
        vector_ids = [int(idx) for idx in vector_results.get("ids", [[]])[0]]

        bm25_ids: list[int] = []
        if self._bm25 is not None:
            bm25_scores = self._bm25.get_scores(question.lower().split())
            bm25_ids = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True,
            )[:candidate_count]

        fused_indices = self._reciprocal_rank_fusion([vector_ids, bm25_ids], top_k)
        chunks = [self._chunks[i] for i in fused_indices]
        display_indices = [i + 1 for i in fused_indices]

        return RetrievalResult(chunks=chunks, indices=display_indices)

    def _build_qa_prompt(self, question: str, chunks: list[str]) -> str:
        """Build a grounded Q&A prompt from retrieved chunks.

        Args:
            question: The user's question.
            chunks: Retrieved context chunks.

        Returns:
            The full prompt string.
        """
        context = "\n\n---\n\n".join(chunks)
        return (
            "You are a scholarly research assistant. Answer the user's question "
            "using ONLY the provided context from an academic paper. "
            "If the context does not contain enough information to answer, "
            "say so clearly rather than guessing.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question.strip()}\n\n"
            "Answer:"
        )

    def answer_question(
        self,
        question: str,
        api_key: str,
        top_k: int = 4,
    ) -> AnswerResult:
        """Retrieve relevant chunks and generate a grounded answer via the Gemini API.

        Args:
            question: The user's natural-language question.
            api_key: Google Gemini API key.
            top_k: Number of chunks to retrieve for context.

        Returns:
            AnswerResult with the answer and source chunk indices.

        Raises:
            RAGPipelineError: On invalid input or API failures.
        """
        sources = self.retrieve(question, top_k=top_k)
        prompt = self._build_qa_prompt(question, sources.chunks)
        answer = self._call_llm(prompt, api_key)
        return AnswerResult(answer=answer, sources=sources)

    def stream_answer_question(
        self,
        question: str,
        api_key: str,
        top_k: int = 4,
    ) -> tuple[Iterator[str], RetrievalResult]:
        """Stream a grounded answer and return the source chunks used.

        Args:
            question: The user's natural-language question.
            api_key: Google Gemini API key.
            top_k: Number of chunks to retrieve for context.

        Returns:
            A tuple of (text iterator, RetrievalResult).

        Raises:
            RAGPipelineError: On invalid input or API failures.
        """
        sources = self.retrieve(question, top_k=top_k)
        prompt = self._build_qa_prompt(question, sources.chunks)
        return self._stream_llm(prompt, api_key), sources

    def _quota_error_message(self, error: Exception) -> str:
        """Return a user-friendly message for Gemini quota or rate-limit errors."""
        message = str(error).lower()
        if "perday" in message or "per day" in message or "daily" in message:
            return (
                "Gemini free-tier daily quota reached. Wait until tomorrow or check "
                "usage at https://aistudio.google.com/apikey"
            )
        if "limit: 0" in message:
            return (
                "This Gemini API key has no free-tier quota (limit: 0). "
                "Create a new key at https://aistudio.google.com/apikey or enable "
                "billing on your Google Cloud project."
            )
        return (
            "Gemini rate limit reached. Wait 30–60 seconds and try again. "
            "Avoid clicking Generate Summary multiple times in a row."
        )

    def _get_model(self, api_key: str) -> genai.GenerativeModel:
        """Configure Gemini and return a GenerativeModel instance.

        Args:
            api_key: Google Gemini API key.

        Returns:
            Configured GenerativeModel.

        Raises:
            RAGPipelineError: If the API key is missing.
        """
        if not api_key:
            raise RAGPipelineError(
                "Gemini API key is not configured. "
                "Add GEMINI_API_KEY to your Streamlit secrets."
            )
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(LLM_MODEL)

    def _call_llm(self, prompt: str, api_key: str, max_tokens: int = 1024) -> str:
        """Send a prompt to the Gemini API and return the response text.

        Args:
            prompt: The full prompt string.
            api_key: Google Gemini API key.
            max_tokens: Maximum tokens in the response.

        Returns:
            The assistant's response text.

        Raises:
            RAGPipelineError: On API failures or rate limits.
        """
        model = self._get_model(api_key)
        last_error: Exception | None = None

        for attempt in range(MAX_LLM_RETRIES):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                    ),
                )

                if not response.text:
                    raise RAGPipelineError(
                        "The AI service returned an empty response. Please try again."
                    )
                return response.text

            except RAGPipelineError:
                raise
            except google_exceptions.ResourceExhausted as exc:
                last_error = exc
                if attempt < MAX_LLM_RETRIES - 1:
                    time.sleep(RETRY_BASE_DELAY_SECONDS * (attempt + 1))
                    continue
                raise RAGPipelineError(self._quota_error_message(exc)) from None
            except (google_exceptions.Unauthenticated, google_exceptions.PermissionDenied):
                raise RAGPipelineError(
                    "Invalid Gemini API key. Please check your configuration."
                ) from None
            except google_exceptions.GoogleAPIError as exc:
                if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                    last_error = exc
                    if attempt < MAX_LLM_RETRIES - 1:
                        time.sleep(RETRY_BASE_DELAY_SECONDS * (attempt + 1))
                        continue
                    raise RAGPipelineError(self._quota_error_message(exc)) from None
                raise RAGPipelineError(
                    "The AI service encountered an error. Please try again later."
                ) from None
            except Exception as exc:
                if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                    last_error = exc
                    if attempt < MAX_LLM_RETRIES - 1:
                        time.sleep(RETRY_BASE_DELAY_SECONDS * (attempt + 1))
                        continue
                    raise RAGPipelineError(self._quota_error_message(exc)) from None
                raise RAGPipelineError(
                    "An unexpected error occurred while generating a response."
                ) from None

        if last_error is not None:
            raise RAGPipelineError(self._quota_error_message(last_error)) from None
        raise RAGPipelineError(
            "An unexpected error occurred while generating a response."
        )

    def _stream_llm(self, prompt: str, api_key: str, max_tokens: int = 1024) -> Iterator[str]:
        """Stream tokens from the Gemini API for a given prompt.

        Args:
            prompt: The full prompt string.
            api_key: Google Gemini API key.
            max_tokens: Maximum tokens in the response.

        Yields:
            Successive text fragments from the model.

        Raises:
            RAGPipelineError: On API failures.
        """
        model = self._get_model(api_key)

        try:
            response = model.generate_content(
                prompt,
                stream=True,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                ),
            )
            yielded = False
            for chunk in response:
                try:
                    text = chunk.text
                except (ValueError, AttributeError):
                    continue
                if text:
                    yielded = True
                    yield text

            if not yielded:
                raise RAGPipelineError(
                    "The AI service returned an empty response. Please try again."
                )

        except RAGPipelineError:
            raise
        except (google_exceptions.Unauthenticated, google_exceptions.PermissionDenied):
            raise RAGPipelineError(
                "Invalid Gemini API key. Please check your configuration."
            ) from None
        except google_exceptions.ResourceExhausted as exc:
            raise RAGPipelineError(self._quota_error_message(exc)) from None
        except google_exceptions.GoogleAPIError as exc:
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                raise RAGPipelineError(self._quota_error_message(exc)) from None
            raise RAGPipelineError(
                "The AI service encountered an error. Please try again later."
            ) from None
        except Exception as exc:
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                raise RAGPipelineError(self._quota_error_message(exc)) from None
            raise RAGPipelineError(
                "An unexpected error occurred while generating a response."
            ) from None

    def generate_with_prompt(self, prompt: str, api_key: str, max_tokens: int = 2048) -> str:
        """Generate text from an arbitrary prompt via the Gemini API.

        Args:
            prompt: The full prompt string.
            api_key: Google Gemini API key.
            max_tokens: Maximum tokens in the response.

        Returns:
            The assistant's response text.

        Raises:
            RAGPipelineError: On API failures.
        """
        return self._call_llm(prompt, api_key, max_tokens=max_tokens)

    def evaluate_retrieval(
        self,
        test_cases: list[dict[str, object]],
        top_k: int = 4,
    ) -> list[dict[str, object]]:
        """Run a basic retrieval evaluation against keyword expectations.

        Args:
            test_cases: List of dicts with 'question' and 'expected_keywords' keys.
            top_k: Number of chunks to retrieve per question.

        Returns:
            A list of per-question evaluation result dicts.
        """
        results: list[dict[str, object]] = []

        for case in test_cases:
            question = str(case["question"])
            keywords = [str(k).lower() for k in case["expected_keywords"]]
            retrieval = self.retrieve(question, top_k=top_k)
            combined = " ".join(retrieval.chunks).lower()
            hits = [kw for kw in keywords if kw in combined]

            results.append(
                {
                    "question": question,
                    "expected_keywords": keywords,
                    "retrieved_sections": retrieval.indices,
                    "keywords_found": hits,
                    "hit_rate": len(hits) / len(keywords) if keywords else 0.0,
                }
            )

        return results
