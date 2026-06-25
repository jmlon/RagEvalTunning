"""Typer CLI: index / run / report. Subcommands are strictly separate."""
from __future__ import annotations

import logging
import os

import typer
from dotenv import load_dotenv

from ragbench.config import load_config
from ragbench.runner import run_eval, run_index, run_report
from ragbench.tuning import run_tune

app = typer.Typer(help="Comparative RAG benchmark harness.", no_args_is_help=True)

SystemOpt = typer.Option(
    None,
    "--system",
    "-s",
    help="System name(s) to target. Repeatable. Default: all enabled.",
)
ConfigOpt = typer.Option("config.yaml", "--config", "-c", help="Path to config.yaml.")


def _bootstrap(config_path: str):
    load_dotenv()
    # Chroma's posthog telemetry is noisy and buggy in this version; disable it.
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "none")
    # Third-party libraries (httpx, chromadb, openai, ...) log at WARNING+;
    # keep INFO only for this application's own loggers.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("ragbench").setLevel(logging.INFO)
    # The telemetry env vars above don't stop chromadb's posthog client from
    # initializing and failing with a capture() signature mismatch; silence its
    # logger so those harmless ERROR lines don't appear.
    logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
    cfg = load_config(config_path)
    return cfg


@app.command(help="Ingest the corpus and incrementally (re)build each system's index.")
def index(
    system: list[str] = SystemOpt,
    config: str = ConfigOpt,
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Ignore the registry and reindex every document from scratch.",
    ),
):
    cfg = _bootstrap(config)
    run_index(cfg, system or None, force=force)


@app.command(help="Answer the question set with each system and grade the answers.")
def run(
    system: list[str] = SystemOpt,
    config: str = ConfigOpt,
):
    cfg = _bootstrap(config)
    run_eval(cfg, system or None)


@app.command(help="Print a per-system summary from stored results.json files.")
def report(
    system: list[str] = SystemOpt,
    config: str = ConfigOpt,
):
    cfg = _bootstrap(config)
    for line in run_report(cfg, system or None):
        typer.echo(line)


@app.command(help="Grid-search a system's `tuning` params; pick the best and update config.yaml.")
def tune(
    system: str = typer.Option(..., "--system", "-s", help="The single system to tune."),
    config: str = ConfigOpt,
):
    cfg = _bootstrap(config)
    summary = run_tune(cfg, system, config_path=config)
    best = summary["best"]
    typer.echo(f"\n=== tuning {summary['system']} ({summary['objective']}) ===")
    typer.echo(f"  {'params':<28} {'mean/5':>7} {'q_cost($)':>10} {'q_tokens':>9}")
    for t in summary["trials"]:
        params = ", ".join(f"{k}={v}" for k, v in t["params"].items())
        marker = "  <- best" if t["params"] == best["params"] else ""
        typer.echo(f"  {params:<28} {t['mean_score']:>7} {t['query_cost']:>10} {t['query_tokens']:>9}{marker}")
    win = ", ".join(f"{k}={v}" for k, v in best["params"].items())
    typer.echo(f"\nBest: {win} (mean_score={best['mean_score']}). config.yaml updated.")


if __name__ == "__main__":
    app()
