"""LLM judge: grades a system answer against the ground truth."""
from __future__ import annotations

import logging
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ragbench.config import GlobalConfig
from ragbench.llm import LLMFactory
from ragbench.models import GradeLabel

logger = logging.getLogger(__name__)


class JudgeVerdict(BaseModel):
    """Structured output schema for the judge."""

    grade: Literal["perfect", "good", "partial", "poor", "wrong"] = Field(
        description="perfect: fully answers all asked parts, no contradictions; "
        "good: correct and answers the question but a minor omission/imprecision "
        "(no fabrication); partial: some asked parts right, others missing or "
        "wrong; poor: mostly wrong, only a small correct fragment; wrong: nothing "
        "correct or contradicts the core of the question."
    )
    rationale: str = Field(description="Brief justification for the grade.")


_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful, fair grader for a question-answering benchmark. "
            "Decide how well the SYSTEM ANSWER answers the QUESTION, using the "
            "GROUND TRUTH as the reference for what is correct.\n\n"
            "Method:\n"
            "1. Break the QUESTION into the parts it actually asks for.\n"
            "2. For each part, check whether the SYSTEM ANSWER is correct and "
            "consistent with the GROUND TRUTH. Judge semantic meaning, not wording "
            "(paraphrases and reworded answers are fine).\n"
            "3. The GROUND TRUTH may contain extra elaboration beyond what the "
            "question asks. Omitting that extra detail is NOT an error. Only "
            "penalize content that is missing for a part the question explicitly "
            "asks for, or that is incorrect / contradicts the GROUND TRUTH.\n\n"
            "Grades (5-level ordinal, best to worst):\n"
            "- perfect: correctly answers everything the question asks and "
            "contradicts nothing in the GROUND TRUTH. Extra omitted GROUND TRUTH "
            "detail that the question did not ask for does NOT lower this.\n"
            "- good: correctly answers the question, but with a minor omission of "
            "an asked detail or slight imprecision — and NO incorrect or "
            "fabricated content.\n"
            "- partial: some asked parts are correct while others are missing or "
            "wrong (e.g. a multi-part question where part is right and part is "
            "fabricated).\n"
            "- poor: mostly wrong, but at least one element the question asks for "
            "is correct (e.g. 1 of 4 requested items right, the rest wrong).\n"
            "- wrong: NOTHING in the answer is correct, or it contradicts the "
            "GROUND TRUTH on the core of what was asked. If even one asked element "
            "is correct, grade 'poor' (or higher), not 'wrong'.\n\n"
            "Distinguish OMISSION (leaving out non-essential detail — not "
            "penalized) from CONTRADICTION/FABRICATION (stating something false — "
            "penalized). The presence of fabricated/incorrect content caps the "
            "grade at 'partial' or below, never 'good' or 'perfect'. Provide a "
            "brief rationale that names which parts were correct, missing, or wrong.",
        ),
        (
            "human",
            "QUESTION:\n{question}\n\nGROUND TRUTH:\n{ground_truth}\n\n"
            "SYSTEM ANSWER:\n{answer}",
        ),
    ]
)


class Judge:
    def __init__(self, global_cfg: GlobalConfig, llm_factory: LLMFactory):
        self.cfg = global_cfg.judge
        self.llm = llm_factory

    def grade(self, question: str, ground_truth: str, answer: str) -> tuple[GradeLabel, str]:
        """Grade an answer. Caller handles the no-answer case before calling this."""
        chat = self.llm.chat(self.cfg.model, temperature=self.cfg.temperature)
        structured = chat.with_structured_output(JudgeVerdict)
        # Retry transient structured-output failures (occasional malformed/rate-limited
        # responses, more likely under parallel `run`).
        last_err: Exception | None = None
        for _ in range(3):
            try:
                verdict: JudgeVerdict = (_PROMPT | structured).invoke(
                    {"question": question, "ground_truth": ground_truth, "answer": answer}
                )
                return GradeLabel(verdict.grade), verdict.rationale
            except Exception as e:  # noqa: BLE001
                last_err = e
        logger.warning("Judge failed for question after retries: %s (%s)", question, last_err)
        return GradeLabel.wrong, "Judge error; defaulted to wrong."
