"""Embedding generation, Chroma storage, retrieval, and LLM question answering."""

from __future__ import annotations

import time

import chromadb
import google.generativeai as genai
from chromadb.api.models.Collection import Collection
from google.api_core import exceptions as google_exceptions
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"
COLLECTION_NAME = "scholar_chunks"
MAX_LLM_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 5


class RAGPipelineError(Exception):
    """Raised when the RAG pipeline encounters a recoverable error."""


class RAGPipeline:
    """Manages chunk embedding, vector storage, retrieval, and LLM generation."""

    def __init__(self) -> None:
        """Initialize the embedding model and an in-memory Chroma collection."""
        self._embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self._client = chromadb.Client()
        self._collection: Collection | None = None

    def index_chunks(self, chunks: list[str]) -> None:
        """Embed chunks and store them in a fresh Chroma collection.

        Args:
            chunks: List of text chunks to embed and index.

        Raises:
            RAGPipelineError: If no chunks are provided.
        """
        if not chunks:
            raise RAGPipelineError("No text chunks available to index.")

        try:
            self._client.delete_collection(COLLECTION_NAME)
        except (ValueError, Exception):
            pass

        self._collection = self._client.create_collection(name=COLLECTION_NAME)
        embeddings = self._embedder.encode(chunks, show_progress_bar=False).tolist()

        self._collection.add(
            ids=[str(i) for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings,
        )

    def retrieve(self, question: str, top_k: int = 4) -> list[str]:
        """Return the most relevant chunks for a given question.

        Args:
            question: The user's natural-language question.
            top_k: Number of top chunks to return.

        Returns:
            A list of the most relevant chunk texts.

        Raises:
            RAGPipelineError: If the index is empty or the question is blank.
        """
        if not question or not question.strip():
            raise RAGPipelineError("Please enter a question before submitting.")

        if self._collection is None or self._collection.count() == 0:
            raise RAGPipelineError(
                "No document has been indexed yet. Please upload a PDF first."
            )

        query_embedding = self._embedder.encode([question.strip()], show_progress_bar=False).tolist()

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self._collection.count()),
        )

        documents: list[str] = results.get("documents", [[]])[0]
        return documents

    def answer_question(
        self,
        question: str,
        api_key: str,
        top_k: int = 4,
    ) -> str:
        """Retrieve relevant chunks and generate a grounded answer via the Gemini API.

        Args:
            question: The user's natural-language question.
            api_key: Google Gemini API key.
            top_k: Number of chunks to retrieve for context.

        Returns:
            The LLM-generated answer grounded in retrieved chunks.

        Raises:
            RAGPipelineError: On invalid input or API failures.
        """
        chunks = self.retrieve(question, top_k=top_k)
        context = "\n\n---\n\n".join(chunks)

        prompt = (
            "You are a scholarly research assistant. Answer the user's question "
            "using ONLY the provided context from an academic paper. "
            "If the context does not contain enough information to answer, "
            "say so clearly rather than guessing.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question.strip()}\n\n"
            "Answer:"
        )

        return self._call_llm(prompt, api_key)

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
        if not api_key:
            raise RAGPipelineError(
                "Gemini API key is not configured. "
                "Add GEMINI_API_KEY to your Streamlit secrets."
            )

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(LLM_MODEL)
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
