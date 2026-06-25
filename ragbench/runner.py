"""Orchestration for index / run / report."""
from __future__ import annotations

import json
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from ragbench.config import BenchmarkConfig
from ragbench.index_registry import load_registry, make_entry, needs_index, save_registry
from ragbench.ingestion import load_corpus
from ragbench.judge import Judge
from ragbench.llm import LLMFactory
from ragbench.models import GradeLabel, Question, QueryResult, SCORES
from ragbench.questions import build_test_questions
from ragbench.registry import get_system_class
from ragbench.results import build_report, read_report, write_markdown_report, write_report

logger = logging.getLogger(__name__)


def load_questions(path: str | Path) -> list[Question]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Question.model_validate(item) for item in raw]


def _output_dir(config: BenchmarkConfig, name: str) -> Path:
    # outputs/<llm>_<emb>/<system>/ — the model-combination level keeps distinct
    # LLM/embedding choices' indexes & results side by side (resolved from config,
    # so index/run/report agree).
    base = Path(config.global_.paths.outputs_dir) / config.model_subdir(name) / name
    # A system may isolate each index build in a param-keyed sub-path so distinct
    # configs (whose indexing is costly) are preserved side by side. Resolved
    # from config, so index/run/report agree.
    subdir = get_system_class(name).index_variant_subdir(config.systems[name])
    return base / subdir if subdir else base


def _instantiate(config: BenchmarkConfig, name: str, llm_factory: LLMFactory):
    cls = get_system_class(name)
    return cls(
        name=name,
        output_dir=_output_dir(config, name),
        system_cfg=config.systems[name],
        config=config,
        llm_factory=llm_factory,
    )


def _resolve_systems(config: BenchmarkConfig, selected: list[str] | None) -> list[str]:
    enabled = config.enabled_systems()
    if not selected:
        return enabled
    chosen = []
    for name in selected:
        if name not in config.systems:
            raise KeyError(f"System '{name}' not in config. Known: {sorted(config.systems)}")
        if not config.systems[name].enabled:
            logger.warning("System '%s' is disabled in config; skipping.", name)
            continue
        chosen.append(name)
    return chosen


def _index_meta_path(output_dir: Path) -> Path:
    return output_dir / "index_meta.json"


def _write_index_meta(
    output_dir: Path, config: BenchmarkConfig, name: str,
    stats, elapsed: float, doc_count: int,
) -> None:
    """Record the LAST indexing operation's efficiency metrics (not accumulated)."""
    llm_model, emb_model = config.resolve_models(name)
    meta = {
        "documents": doc_count,
        "passages": stats.passages,
        "tokens": stats.tokens,
        "estimated_cost": round(stats.cost, 6),
        "time_s": round(elapsed, 4),
        "last_indexed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "llm_model": llm_model,
        "embedding_model": emb_model,
    }
    _index_meta_path(output_dir).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def run_index(
    config: BenchmarkConfig,
    selected: list[str] | None = None,
    force: bool = False,
) -> None:
    llm_factory = LLMFactory(config.global_)
    documents = load_corpus(config.global_.paths.inputs_dir)
    inputs_dir = Path(config.global_.paths.inputs_dir)
    systems = _resolve_systems(config, selected)

    for name in systems:
        try:
            output_dir = _output_dir(config, name)
            # --force removes the whole index directory so the rebuild starts
            # from scratch. Costlier than an incremental update, but robust
            # against half-built state left by a mid-process failure. With a
            # param-keyed variant dir this drops only the current config's
            # build; sibling variants (and any shared llm_cache) are untouched.
            if force and output_dir.exists():
                logger.info("[%s] --force: removing %s", name, output_dir)
                shutil.rmtree(output_dir)

            registry = load_registry(output_dir)
            to_index = [
                d for d in documents
                if needs_index(inputs_dir / d.source, registry.get(d.source))
            ]
            skipped = len(documents) - len(to_index)
            if not to_index:
                logger.info(
                    "[%s] all %d document(s) already indexed and unchanged; nothing to do.",
                    name, len(documents),
                )
                continue

            system = _instantiate(config, name, llm_factory)
            start = time.perf_counter()
            stats = system.index(to_index)
            elapsed = time.perf_counter() - start

            for d in to_index:
                registry[d.source] = make_entry(inputs_dir / d.source)
            save_registry(output_dir, registry)
            _write_index_meta(output_dir, config, name, stats, elapsed, len(to_index))

            logger.info(
                "[%s] indexed %d doc(s) (%d skipped, %d passages, %d tokens, $%s) in %.2fs",
                name, len(to_index), skipped, stats.passages, stats.tokens,
                round(stats.cost, 6), elapsed,
            )
        except Exception:  # noqa: BLE001 — isolate failures per system (FR-10.5)
            logger.exception("[%s] indexing failed; skipping.", name)


def run_eval(config: BenchmarkConfig, selected: list[str] | None = None) -> None:
    llm_factory = LLMFactory(config.global_)
    # Aggregate per-document question files into test_questions.json (randomized).
    questions = build_test_questions(config)
    judge = Judge(config.global_, llm_factory)
    systems = _resolve_systems(config, selected)

    for name in systems:
        try:
            system = _instantiate(config, name, llm_factory)
            system.load()
        except Exception:  # noqa: BLE001
            logger.exception("[%s] failed to load index; skipping. Run `index` first.", name)
            continue

        output_dir = _output_dir(config, name)
        meta_path = _index_meta_path(output_dir)
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())

        def evaluate(q: Question) -> QueryResult:
            out = system.query(q.question)
            if out.answer is None or not str(out.answer).strip():
                grade, rationale = GradeLabel.no_answer, "System returned no answer."
            else:
                grade, rationale = judge.grade(q.question, q.answer, out.answer)
            logger.info("[%s] %s -> %s", name, q.id, grade.value)
            return QueryResult(
                id=q.id,
                question=q.question,
                ground_truth=q.answer,
                answer=out.answer,
                grade=grade,
                score=SCORES[grade],
                judge_rationale=rationale,
                latency_s=round(out.latency_s, 4),
                tokens=out.tokens,
                estimated_cost=round(out.cost, 6),
            )

        # Query + judge are I/O-bound (LLM calls); run them across worker threads.
        # The first question is evaluated serially to warm the system's lazily-built
        # caches (Chroma store / BM25 ensemble / vector index) before concurrency.
        workers = max(1, int(config.global_.workers))
        if workers <= 1 or len(questions) <= 1:
            results = [evaluate(q) for q in questions]
        else:
            first = evaluate(questions[0])
            with ThreadPoolExecutor(max_workers=workers) as pool:
                rest = list(pool.map(evaluate, questions[1:]))  # map preserves order
            results = [first, *rest]

        report = build_report(name, meta, results)
        path = write_report(output_dir, report)
        logger.info(
            "[%s] mean_score=%.2f written to %s",
            name,
            report.aggregate.mean_score,
            path,
        )


def run_report(config: BenchmarkConfig, selected: list[str] | None = None) -> list[str]:
    """Print a per-system summary and write a comparison table to report.md.

    Returns formatted console lines for each selected system with results.
    """
    systems = _resolve_systems(config, selected)
    lines: list[str] = []
    reports: list = []
    for name in systems:
        try:
            report = read_report(_output_dir(config, name))
        except FileNotFoundError as e:
            lines.append(f"[{name}] {e}")
            continue
        reports.append(report)
        agg = report.aggregate
        idx = report.indexing
        counts = ", ".join(f"{k}={v}" for k, v in agg.grade_counts.items())
        lines.append(
            f"\n=== {report.system} ===\n"
            f"  questions:       {report.questions}\n"
            f"  -- indexing (last operation) --\n"
            f"  index_documents: {idx.documents}\n"
            f"  index_passages:  {idx.passages}\n"
            f"  index_time:      {idx.time_s}s\n"
            f"  index_tokens:    {idx.tokens}\n"
            f"  index_cost:      ${idx.estimated_cost}\n"
            f"  -- query --\n"
            f"  mean_score:      {agg.mean_score} / 5\n"
            f"  grades:          {counts}\n"
            f"  query_latency:   {agg.total_latency_s}s\n"
            f"  query_tokens:    {agg.total_tokens}\n"
            f"  query_cost:      ${agg.estimated_cost}"
        )

    if reports:
        # One report per model combination, named after the models used, e.g.
        # outputs/reports/gpt-4o-mini_text-embedding-3-small.md. When the reported
        # systems share one combo (the usual case) that slug names the file; if they
        # span several combos, fall back to the global-defaults slug.
        slugs = {config.model_subdir(r.system) for r in reports}
        slug = slugs.pop() if len(slugs) == 1 else config.defaults_model_subdir()
        reports_dir = Path(config.global_.paths.outputs_dir) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"{slug}.md"
        write_markdown_report(report_path, reports, config)
        lines.append(f"\nComparison table written to {report_path}")
    return lines
