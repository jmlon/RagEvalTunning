"""Core data models for the RAG benchmark harness."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Question(BaseModel):
    """One entry from test_questions.json (user-provided, ids assigned)."""

    id: str
    question: str
    answer: str  # single, sole ground truth


class Document(BaseModel):
    """A normalized corpus item shared by all systems."""

    id: str
    text: str
    source: str
    metadata: dict = Field(default_factory=dict)


class GradeLabel(str, Enum):
    perfect = "perfect"
    good = "good"
    partial = "partial"
    poor = "poor"
    wrong = "wrong"
    no_answer = "no answer"


#: Numeric score for each grade label (5-level ordinal; no answer scores 0).
SCORES: dict[GradeLabel, int] = {
    GradeLabel.perfect: 5,
    GradeLabel.good: 4,
    GradeLabel.partial: 2,
    GradeLabel.poor: 1,
    GradeLabel.wrong: 0,
    GradeLabel.no_answer: 0,
}


class QueryOutput(BaseModel):
    """What a RAGSystem.query returns for a single question."""

    answer: str | None
    contexts: list[str] = Field(default_factory=list)
    latency_s: float = 0.0
    tokens: int = 0
    cost: float = 0.0


class IndexStats(BaseModel):
    """Efficiency metrics returned by RAGSystem.index."""

    passages: int = 0
    tokens: int = 0
    cost: float = 0.0


class QueryResult(BaseModel):
    """One graded row in results.json."""

    id: str
    question: str
    ground_truth: str
    answer: str | None
    grade: GradeLabel
    score: int
    judge_rationale: str
    latency_s: float
    tokens: int
    estimated_cost: float


class Aggregate(BaseModel):
    """Query-time metrics aggregated across all questions."""

    mean_score: float
    grade_counts: dict[str, int]
    total_latency_s: float
    total_tokens: int
    estimated_cost: float


class IndexingReport(BaseModel):
    """Efficiency metrics of the LAST indexing operation for this system.

    Not accumulated across incremental runs: it reflects only the documents
    processed by the most recent `index` invocation. For a full-corpus
    measurement, run `index --force`.
    """

    documents: int = 0          # documents (re)indexed in the last operation
    passages: int = 0
    tokens: int = 0
    estimated_cost: float = 0.0
    time_s: float = 0.0
    last_indexed_at: str | None = None
    llm_model: str | None = None
    embedding_model: str | None = None


class SystemReport(BaseModel):
    """Root object written to outputs/<system>/results.json."""

    system: str
    questions: int
    indexing: IndexingReport
    aggregate: Aggregate
    results: list[QueryResult]
