"""Maps system names to their RAGSystem implementations."""
from __future__ import annotations

from ragbench.systems.base import RAGSystem
from ragbench.systems.graph_rag import GraphRAGSystem
from ragbench.systems.hybrid_rag import HybridRAGSystem
from ragbench.systems.hybrid_rag_reranker import HybridRagRerankerSystem
from ragbench.systems.llm_wiki import LLMWikiSystem
from ragbench.systems.simple_rag import SimpleRAGSystem
from ragbench.systems.simple_rag_reranker import SimpleRagRerankerSystem

REGISTRY: dict[str, type[RAGSystem]] = {
    "simple-rag": SimpleRAGSystem,
    "hybrid-rag": HybridRAGSystem,
    "simple-rag-reranker": SimpleRagRerankerSystem,
    "hybrid-rag-reranker": HybridRagRerankerSystem,
    "graph-rag": GraphRAGSystem,
    "llm-wiki": LLMWikiSystem,
}


def get_system_class(name: str) -> type[RAGSystem]:
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown system '{name}'. Registered: {sorted(REGISTRY)}"
        )
    return REGISTRY[name]
