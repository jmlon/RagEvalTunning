"""Graph-RAG: knowledge-graph retrieval over Neo4j.

Indexing: each document is chunked, and `LLMGraphTransformer` extracts entities
and relationships from every chunk into graph documents, which are written to a
Neo4j database (`add_graph_documents(..., include_source=True)` so each chunk is
stored as a Document node linked to its entities via MENTIONS).

Retrieval (subgraph-as-context): the question is matched against entity nodes via
a Neo4j vector index (seed entities), their 1-hop neighbourhood is expanded into
relationship triples, and the source passages that mention the seed entities are
collected. Triples + passages are serialized to text and answered by the LLM.

Neo4j runs as a local Docker container (see scripts/neo4j.sh); connection comes
from per-system config (uri/username) and an env var for the password. This is a
local DB, distinct from any cloud NEO4J_URI in the environment.
"""
from __future__ import annotations

import logging
import os
import time

from langchain_core.documents import Document as LCDocument
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragbench.index_registry import load_registry
from ragbench.llm import usage_from_message
from ragbench.models import Document, IndexStats, QueryOutput
from ragbench.systems.base import RAGSystem

logger = logging.getLogger(__name__)

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the question using ONLY the "
            "provided knowledge-graph facts and passages. If the answer is not in "
            "the context, say you don't know. Be concise.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

# Directed, de-duplicated 1-hop relationships among/around the seed entities.
_SUBGRAPH_CYPHER = """
UNWIND $ids AS sid
MATCH (e:__Entity__ {id: sid})-[r]-(m:__Entity__)
RETURN DISTINCT startNode(r).id AS source, type(r) AS rel, endNode(r).id AS target
LIMIT $limit
"""

# Source passages that mention the seed entities.
_PASSAGES_CYPHER = """
UNWIND $ids AS sid
MATCH (e:__Entity__ {id: sid})<-[:MENTIONS]-(d:Document)
RETURN DISTINCT d.text AS text
LIMIT $limit
"""


class GraphRAGSystem(RAGSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # URI: env override (set when ragbench runs in a container on a shared
        # docker network, e.g. bolt://ragbench-neo4j:7687) wins over the config
        # default (bolt://localhost:7687, used for host runs).
        uri_env = self.system_cfg.get("neo4j_uri_env", "NEO4J_LOCAL_URI")
        self.neo4j_uri = os.environ.get(uri_env) or self.system_cfg.get(
            "neo4j_uri", "bolt://localhost:7687"
        )
        self.neo4j_username = self.system_cfg.get("neo4j_username", "neo4j")
        pw_env = self.system_cfg.get("neo4j_password_env", "NEO4J_LOCAL_PASSWORD")
        self.neo4j_password = os.environ.get(pw_env, "hola1234")
        self.chunk_size = int(self.system_cfg.get("chunk_size", 1000))
        self.chunk_overlap = int(self.system_cfg.get("chunk_overlap", 200))
        self.seed_k = int(self.system_cfg.get("seed_k", 8))
        self.max_triples = int(self.system_cfg.get("max_triples", 60))
        self.max_passages = int(self.system_cfg.get("max_passages", 6))
        self.temperature = float(self.system_cfg.get("temperature", 0.0))
        self._graph: Neo4jGraph | None = None
        self._vindex: Neo4jVector | None = None

    def _connect(self) -> Neo4jGraph:
        if self._graph is None:
            try:
                self._graph = Neo4jGraph(
                    url=self.neo4j_uri,
                    username=self.neo4j_username,
                    password=self.neo4j_password,
                )
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    f"[{self.name}] cannot connect to Neo4j at {self.neo4j_uri}; "
                    f"start it with `scripts/neo4j.sh up`. ({e})"
                ) from e
        return self._graph

    def _get_vector_index(self) -> Neo4jVector:
        """Vector index over entity nodes; embeds any entities lacking one."""
        if self._vindex is None:
            self._vindex = Neo4jVector.from_existing_graph(
                embedding=self.llm.embeddings(self.embedding_model),
                url=self.neo4j_uri,
                username=self.neo4j_username,
                password=self.neo4j_password,
                node_label="__Entity__",
                text_node_properties=["id"],
                embedding_node_property="embedding",
            )
        return self._vindex

    def index(self, documents: list[Document]) -> IndexStats:
        graph = self._connect()
        # Fresh/forced build (empty registry) → wipe the whole graph first so no
        # stale nodes/relationships linger. Incremental adds keep existing data.
        if not load_registry(self.output_dir):
            logger.info("[%s] empty registry; wiping Neo4j graph.", self.name)
            graph.query("MATCH (n) DETACH DELETE n")

        transformer = LLMGraphTransformer(
            llm=self.llm.chat(self.llm_model, temperature=self.temperature)
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        total_chunks = 0
        for doc in documents:
            # Replace this document's prior source chunks (handles re-index).
            graph.query("MATCH (d:Document {doc_id: $id}) DETACH DELETE d", {"id": doc.id})
            chunks = [
                LCDocument(
                    page_content=piece,
                    metadata={"doc_id": doc.id, "source": doc.source},
                )
                for piece in splitter.split_text(doc.text)
            ]
            if not chunks:
                continue
            graph_docs = transformer.convert_to_graph_documents(chunks)
            graph.add_graph_documents(
                graph_docs, baseEntityLabel=True, include_source=True
            )
            total_chunks += len(chunks)

        # (Re)build entity embeddings for vector seeding; invalidate cache.
        self._vindex = None
        if total_chunks:
            self._get_vector_index()
        logger.info(
            "[%s] indexed %d chunk(s) from %d doc(s) into Neo4j.",
            self.name, total_chunks, len(documents),
        )
        # LLMGraphTransformer's extraction LLM calls do not surface token usage to
        # the harness; indexing tokens/cost reported as 0 (see spec OI-9).
        return IndexStats(passages=total_chunks, tokens=0, cost=0.0)

    def load(self) -> None:
        graph = self._connect()
        count = graph.query("MATCH (e:__Entity__) RETURN count(e) AS c")[0]["c"]
        if not count:
            raise FileNotFoundError(
                f"[{self.name}] Neo4j graph at {self.neo4j_uri} has no entities; "
                f"run `index -s {self.name}` first (and ensure Neo4j is running)."
            )

    def _seed_entity_ids(self, question: str) -> list[str]:
        docs = self._get_vector_index().similarity_search(question, k=self.seed_k)
        ids = []
        for d in docs:
            # from_existing_graph puts the embedded property into page_content as
            # "\nid: <value>"; metadata is empty.
            text = d.page_content
            ids.append(text.split("id:", 1)[-1].strip() if "id:" in text else text.strip())
        return [i for i in ids if i]

    def query(self, question: str) -> QueryOutput:
        start = time.perf_counter()
        try:
            graph = self._connect()
            seed_ids = self._seed_entity_ids(question)

            triples = graph.query(
                _SUBGRAPH_CYPHER, {"ids": seed_ids, "limit": self.max_triples}
            )
            passages = graph.query(
                _PASSAGES_CYPHER, {"ids": seed_ids, "limit": self.max_passages}
            )

            facts = "\n".join(
                f"- {r['source']} -[{r['rel']}]-> {r['target']}"
                for r in triples
                if r["source"] != r["target"]
            )
            passage_text = "\n---\n".join(p["text"] for p in passages if p.get("text"))
            context = (
                f"Knowledge graph facts:\n{facts or '(none)'}\n\n"
                f"Relevant passages:\n{passage_text or '(none)'}"
            )

            chat = self.llm.chat(self.llm_model, temperature=self.temperature)
            message = (_PROMPT | chat).invoke({"context": context, "question": question})

            in_tok, out_tok = usage_from_message(message)
            return QueryOutput(
                answer=message.content,
                contexts=[f"{r['source']} -[{r['rel']}]-> {r['target']}" for r in triples]
                + [p["text"] for p in passages if p.get("text")],
                latency_s=time.perf_counter() - start,
                tokens=in_tok + out_tok,
                cost=self.llm.cost(self.llm_model, in_tok, out_tok),
            )
        except Exception:  # noqa: BLE001 — must not abort the run (FR-1.4)
            logger.exception("[%s] query failed for: %s", self.name, question)
            return QueryOutput(answer=None, latency_s=time.perf_counter() - start)
