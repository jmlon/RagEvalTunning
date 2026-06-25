"""Abstract base class all RAG systems implement."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ragbench.config import BenchmarkConfig
from ragbench.llm import LLMFactory
from ragbench.models import Document, IndexStats, QueryOutput


class RAGSystem(ABC):
    #: filesystem-safe identifier, e.g. "simple-rag"; used as outputs/<name>/.
    name: str

    def __init__(
        self,
        name: str,
        output_dir: Path,
        system_cfg,
        config: BenchmarkConfig,
        llm_factory: LLMFactory,
    ):
        self.name = name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.system_cfg = system_cfg
        self.config = config
        self.llm = llm_factory
        self.llm_model, self.embedding_model = config.resolve_models(name)

    @classmethod
    def index_variant_subdir(cls, system_cfg) -> str | None:
        """Optional sub-path under outputs/<name>/ that isolates one index
        build keyed by its index-affecting params.

        Returning None (the default) means the system writes directly to
        outputs/<name>/, as before. A system whose index depends on tunable
        params can return e.g. "size=400-overlap=150" so distinct configs are
        preserved side by side instead of overwriting each other. Computed from
        `system_cfg` alone so index/run/report all resolve the same directory.
        """
        return None

    @abstractmethod
    def index(self, documents: list[Document]) -> IndexStats:
        """Build and persist the system's representation under output_dir.

        v1 always does a full rebuild. Returns indexing efficiency metrics
        (passage count, tokens, cost). Systems that cannot expose token usage
        report tokens/cost as 0.
        """

    @abstractmethod
    def query(self, question: str) -> QueryOutput:
        """Answer one question.

        Must NOT raise on a normal failure to answer: return
        QueryOutput(answer=None, ...) instead.
        """

    def load(self) -> None:
        """Hydrate an already-built index from output_dir before querying.

        Default is a no-op for systems that load lazily in index/query.
        """
        return None
