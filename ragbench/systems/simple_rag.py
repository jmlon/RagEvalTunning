"""Simple-RAG: naive vector RAG over a persistent Chroma index."""
from __future__ import annotations

import logging
import time

from langchain_chroma import Chroma
from langchain_core.documents import Document as LCDocument
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragbench.llm import count_tokens, usage_from_message
from ragbench.models import Document, IndexStats, QueryOutput
from ragbench.systems.base import RAGSystem
from ragbench.vectorstore import open_chroma

logger = logging.getLogger(__name__)

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the question using ONLY the "
            "provided context. If the answer is not in the context, say you "
            "don't know. Be concise.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


class SimpleRAGSystem(RAGSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.persist_dir = self.output_dir / "chroma"
        self.collection_name = "simple_rag"
        self.chunk_size = int(self.system_cfg.get("chunk_size", 1000))
        self.chunk_overlap = int(self.system_cfg.get("chunk_overlap", 200))
        self.top_k = int(self.system_cfg.get("top_k", 4))
        self.temperature = float(self.system_cfg.get("temperature", 0.0))
        self._store: Chroma | None = None

    def _embeddings(self):
        return self.llm.embeddings(self.embedding_model)

    def _get_store(self) -> Chroma:
        """Open the persistent collection, creating it if absent."""
        if self._store is None:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._store = open_chroma(
                self.collection_name, self._embeddings(), self.persist_dir
            )
        return self._store

    def index(self, documents: list[Document]) -> IndexStats:
        # Additive: add/replace only the given documents in the existing index.
        store = self._get_store()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        total_chunks = 0
        total_tokens = 0
        for doc in documents:
            # Drop any prior chunks for this document (handles re-index of a change).
            existing = store.get(where={"doc_id": doc.id})
            if existing["ids"]:
                store.delete(ids=existing["ids"])
            chunks = [
                LCDocument(
                    page_content=piece,
                    metadata={"source": doc.source, "doc_id": doc.id, "chunk": i},
                )
                for i, piece in enumerate(splitter.split_text(doc.text))
            ]
            if chunks:
                store.add_documents(chunks)
            total_chunks += len(chunks)
            # Embedding tokens are exactly the tokens of the chunks being embedded.
            total_tokens += count_tokens([c.page_content for c in chunks])
        cost = self.llm.embedding_cost(self.embedding_model, total_tokens)
        logger.info(
            "[%s] indexed %d chunk(s) from %d doc(s) (%d embedding tokens)",
            self.name, total_chunks, len(documents), total_tokens,
        )
        return IndexStats(passages=total_chunks, tokens=total_tokens, cost=cost)

    def load(self) -> None:
        if not self.persist_dir.exists():
            raise FileNotFoundError(
                f"[{self.name}] no index at {self.persist_dir}; run `index` first."
            )
        self._get_store()

    def _retrieve(self, question: str):
        """Return the retrieved documents for a question. Override to add reranking."""
        retriever = self._get_store().as_retriever(search_kwargs={"k": self.top_k})
        return retriever.invoke(question)

    def query(self, question: str) -> QueryOutput:
        start = time.perf_counter()
        try:
            docs = self._retrieve(question)
            context = "\n\n".join(d.page_content for d in docs)

            chat = self.llm.chat(self.llm_model, temperature=self.temperature)
            message = (_PROMPT | chat).invoke({"context": context, "question": question})

            in_tok, out_tok = usage_from_message(message)
            return QueryOutput(
                answer=message.content,
                contexts=[d.page_content for d in docs],
                latency_s=time.perf_counter() - start,
                tokens=in_tok + out_tok,
                cost=self.llm.cost(self.llm_model, in_tok, out_tok),
            )
        except Exception:  # noqa: BLE001 — must not abort the run (FR-1.4)
            logger.exception("[%s] query failed for: %s", self.name, question)
            return QueryOutput(answer=None, latency_s=time.perf_counter() - start)
