"""Simple-RAG + Cohere reranker.

Identical to Simple-RAG but adds a Cohere reranking stage over the retrieved
chunks. It reuses Simple-RAG's persistent Chroma index (same chunks, vectors and
embedding model) instead of building its own, and performs no indexing of its
own. Retrieval over-fetches `retrieve_k` candidates from the shared store, Cohere
reranks them, and the top `top_k` are kept for the prompt.

Cohere rerank is a Cohere-native API (not OpenAI-compatible): it uses
COHERE_API_KEY directly, independent of the OpenRouter endpoint used elsewhere.
"""
from __future__ import annotations

import logging
import os

from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

from ragbench.models import Document, IndexStats
from ragbench.systems.simple_rag import SimpleRAGSystem

logger = logging.getLogger(__name__)


class SimpleRagRerankerSystem(SimpleRAGSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Reuse the source system's Chroma index (same chunks/vectors). Keep the
        # inherited collection_name ("simple_rag"); only redirect the persist dir.
        self.source_system = self.system_cfg.get("source_system", "simple-rag")
        self.persist_dir = (
            self.config.global_.paths.outputs_dir
            / self.config.model_subdir(self.source_system)
            / self.source_system
            / "chroma"
        )
        self.retrieve_k = int(self.system_cfg.get("retrieve_k", 20))
        self.rerank_model = self.system_cfg.get("rerank_model", "rerank-v3.5")
        self._compression: ContextualCompressionRetriever | None = None

    def index(self, documents: list[Document]) -> IndexStats:
        # No-op: the vector index is owned and built by the source system.
        logger.info(
            "[%s] reuses %s's index; nothing to index.", self.name, self.source_system
        )
        return IndexStats(passages=0, tokens=0, cost=0.0)

    def load(self) -> None:
        if not self.persist_dir.exists():
            raise FileNotFoundError(
                f"[{self.name}] shared index not found at {self.persist_dir}; "
                f"run `index -s {self.source_system}` first."
            )
        self._get_store()

    def _cohere_key(self) -> str:
        key = os.environ.get("COHERE_API_KEY")
        if not key:
            raise RuntimeError(
                f"[{self.name}] COHERE_API_KEY is not set; required for Cohere rerank."
            )
        return key

    def _get_compression_retriever(self) -> ContextualCompressionRetriever:
        if self._compression is None:
            base = self._get_store().as_retriever(
                search_kwargs={"k": self.retrieve_k}
            )
            compressor = CohereRerank(
                model=self.rerank_model,
                top_n=self.top_k,
                cohere_api_key=self._cohere_key(),
            )
            self._compression = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=base
            )
        return self._compression

    def _retrieve(self, question: str):
        # Over-fetch retrieve_k candidates, Cohere-rerank, keep top_k.
        return self._get_compression_retriever().invoke(question)
