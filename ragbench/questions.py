"""Aggregate per-document question files into a single test_questions.json.

Each document in inputs/ (e.g. `2005.11401v4.pdf`) has a sibling questions file
with the same stem and a `.json` extension (`2005.11401v4.json`) containing a
list of {id, question, answer} objects. At benchmark time these are joined
across all documents, their ids namespaced by document stem (`<stem>:<id>`),
and their order randomized (seeded for reproducibility when configured).
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from ragbench.config import BenchmarkConfig
from ragbench.ingestion import LOADERS
from ragbench.models import Question

logger = logging.getLogger(__name__)


def _load_doc_questions(json_path: Path, stem: str) -> list[Question]:
    text = json_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{json_path}: file is empty.")
    raw = json.loads(text)
    if not isinstance(raw, list):
        raise ValueError(f"{json_path}: expected a JSON list of questions.")
    questions: list[Question] = []
    for item in raw:
        # Namespace the id with the document stem for uniqueness + traceability.
        original_id = str(item["id"])
        questions.append(
            Question(
                id=f"{stem}:{original_id}",
                question=item["question"],
                answer=item["answer"],
            )
        )
    return questions


def build_test_questions(config: BenchmarkConfig) -> list[Question]:
    """Join per-document question files, randomize, write test_questions.json."""
    inputs_dir = Path(config.global_.paths.inputs_dir)
    if not inputs_dir.exists():
        raise FileNotFoundError(f"Inputs directory not found: {inputs_dir}")

    doc_exts = set(LOADERS)
    questions: list[Question] = []
    doc_count = 0
    for path in sorted(inputs_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in doc_exts:
            continue
        doc_count += 1
        sidecar = path.with_suffix(".json")
        if not sidecar.exists():
            logger.warning("No questions file for %s (expected %s); skipping.",
                           path.name, sidecar.name)
            continue
        try:
            doc_qs = _load_doc_questions(sidecar, path.stem)
        except (ValueError, json.JSONDecodeError, KeyError) as e:
            # A bad/empty/incomplete questions file must not abort the benchmark.
            logger.warning("Skipping %s: %s", sidecar.name, e)
            continue
        logger.info("Loaded %d question(s) from %s", len(doc_qs), sidecar.name)
        questions.extend(doc_qs)

    if not questions:
        raise RuntimeError(
            f"No questions found. Expected <document-stem>.json files in {inputs_dir}."
        )

    rng = random.Random(config.global_.seed)
    rng.shuffle(questions)

    out_path = Path(config.global_.paths.questions_file)
    out_path.write_text(
        json.dumps([q.model_dump() for q in questions], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "Wrote %d question(s) from %d document(s) to %s (seed=%s)",
        len(questions), doc_count, out_path, config.global_.seed,
    )
    return questions
