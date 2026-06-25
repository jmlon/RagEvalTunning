"""Read/write per-system results.json atomically."""
from __future__ import annotations

import json
from pathlib import Path

from ragbench.models import (
    Aggregate,
    GradeLabel,
    IndexingReport,
    QueryResult,
    SCORES,
    SystemReport,
)


def build_report(
    system: str,
    index_meta: dict,
    results: list[QueryResult],
) -> SystemReport:
    n = len(results)
    total_score = sum(r.score for r in results)
    grade_counts = {label.value: 0 for label in GradeLabel}
    for r in results:
        grade_counts[r.grade.value] += 1

    aggregate = Aggregate(
        mean_score=round(total_score / n, 4) if n else 0.0,
        grade_counts=grade_counts,
        total_latency_s=round(sum(r.latency_s for r in results), 4),
        total_tokens=sum(r.tokens for r in results),
        estimated_cost=round(sum(r.estimated_cost for r in results), 6),
    )
    indexing = IndexingReport.model_validate(index_meta) if index_meta else IndexingReport()
    return SystemReport(
        system=system,
        questions=n,
        indexing=indexing,
        aggregate=aggregate,
        results=results,
    )


def results_path(output_dir: Path) -> Path:
    return Path(output_dir) / "results.json"


def write_report(output_dir: Path, report: SystemReport) -> Path:
    path = results_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def read_report(output_dir: Path) -> SystemReport:
    path = results_path(output_dir)
    if not path.exists():
        raise FileNotFoundError(f"No results at {path}; run `run` first.")
    return SystemReport.model_validate_json(path.read_text(encoding="utf-8"))


# System-config keys that are not tunable params (excluded from the per-system
# parameters listing — models are shown separately, the rest is bookkeeping).
_NON_PARAM_KEYS = {"enabled", "llm_model", "embedding_model", "tuning"}


def _resolve_models(report: SystemReport, config) -> tuple[str, str]:
    """(llm, embedding) for a report — authoritative from config when available,
    else the models recorded in the indexing metadata."""
    if config is not None and report.system in config.systems:
        return config.resolve_models(report.system)
    idx = report.indexing
    return idx.llm_model or "-", idx.embedding_model or "-"


def build_markdown_report(reports: list[SystemReport], config=None) -> str:
    """Render a comparison table (rows = systems, columns = indicators).

    When `config` is supplied, the table gains LLM/Embedding columns and a
    per-system Configuration section listing the resolved models and parameters.
    """
    grade_labels = [g.value for g in GradeLabel]
    headers = (
        ["System"]
        + (["LLM", "Embedding"] if config is not None else [])
        + ["Questions", "Mean /5"]
        + grade_labels
        + [
            "Query latency (s)", "Query tokens", "Query cost ($)",
            "Index time (s)", "Index tokens", "Index cost ($)",
        ]
    )

    def row(r: SystemReport) -> list[str]:
        agg, idx = r.aggregate, r.indexing
        models = [*_resolve_models(r, config)] if config is not None else []
        return [
            r.system,
            *models,
            str(r.questions),
            f"{agg.mean_score:.2f}",
            *[str(agg.grade_counts.get(g, 0)) for g in grade_labels],
            f"{agg.total_latency_s:.2f}",
            str(agg.total_tokens),
            f"{agg.estimated_cost:.6f}",
            f"{idx.time_s:.2f}",
            str(idx.tokens),
            f"{idx.estimated_cost:.6f}",
        ]

    lines = ["# RAG Benchmark Report", ""]
    if config is not None:
        combos = sorted({"`%s` / `%s`" % _resolve_models(r, config) for r in reports})
        lines.append("**Models (LLM / embedding):** " + ", ".join(combos))
        lines.append("")
    lines += [
        "Scores: perfect=5, good=4, partial=2, poor=1, wrong=0, no answer=0.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for r in reports:
        lines.append("| " + " | ".join(row(r)) + " |")
    lines.append("")

    if config is not None:
        lines += ["## Configuration", ""]
        for r in reports:
            llm, emb = _resolve_models(r, config)
            lines += [f"### {r.system}", "", f"- LLM: `{llm}`", f"- Embedding: `{emb}`"]
            syscfg = config.systems.get(r.system)
            params = {} if syscfg is None else syscfg.model_dump(exclude=_NON_PARAM_KEYS)
            for k in sorted(params):
                lines.append(f"- {k}: `{params[k]}`")
            lines.append("")

    return "\n".join(lines)


def write_markdown_report(path: Path, reports: list[SystemReport], config=None) -> Path:
    path = Path(path)
    path.write_text(build_markdown_report(reports, config), encoding="utf-8")
    return path


# Re-exported for callers that build QueryResult rows.
__all__ = [
    "build_report", "write_report", "read_report", "results_path", "SCORES",
    "build_markdown_report", "write_markdown_report",
]
