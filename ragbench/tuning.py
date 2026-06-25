"""Parameter grid-search tuning for a single RAG system.

`run_tune` sweeps the discrete value lists declared in a system's `tuning:`
config block (cartesian product), evaluates each combo with the normal
index -> run -> judge pipeline, picks the best by mean judge score (tie-break:
lower query cost), writes the winning params back into config.yaml (preserving
comments), and reconciles the persisted outputs to the winner.
"""
from __future__ import annotations

import itertools
import json
import logging
import re
import shutil
from pathlib import Path

from ragbench.config import BenchmarkConfig
from ragbench.results import read_report, results_path
from ragbench.runner import _output_dir, run_eval, run_index

logger = logging.getLogger(__name__)


def _combos(grid: dict[str, list]) -> list[dict]:
    """Cartesian product of {param: [values]} -> list of {param: value} dicts."""
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def _combo_label(combo: dict) -> str:
    return "_".join(f"{k}={v}" for k, v in combo.items())


def _parse_tuning(tuning: dict) -> tuple[dict, set, list[list[str]]]:
    """Normalize a system's `tuning` block into (grid, index_params, normalize_groups).

    Two accepted shapes:
      - structured:  {grid: {param: [...]}, index_params: [...], normalize: [...]}
      - flat (legacy): {param: [...]}  -> every param is treated as index-affecting

    `index_params` names the grid params whose change requires re-indexing
    (e.g. chunk_size). Query-only params (e.g. top_k) reuse the built index.

    `normalize` names parameter groups that must sum to 1 when the winner is
    written back (e.g. ensemble weights bm25_weight + dense_weight). Accepts a
    flat list (one group) or a list of lists (several groups).
    """
    if "grid" in tuning:
        grid = dict(tuning["grid"])
        index_params = set(tuning.get("index_params", list(grid))) & set(grid)
        normalize = tuning.get("normalize") or []
    else:
        grid = dict(tuning)
        index_params = set(grid)  # safe default: re-index for every param
        normalize = []
    # Coerce a flat list of names into a single group.
    if normalize and isinstance(normalize[0], str):
        normalize = [normalize]
    return grid, index_params, normalize


def update_config_param(path: str | Path, system_name: str, params: dict) -> None:
    """Replace `param: value` lines inside a system's config block in place.

    Comment- and format-preserving: only the matched value tokens change;
    indentation and any trailing inline comment are kept. A param absent from
    the block is appended to it.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()

    # Locate the `  <system_name>:` line and the bounds of its block.
    sys_re = re.compile(rf"^(\s+){re.escape(system_name)}:\s*$")
    start = next((i for i, ln in enumerate(lines) if sys_re.match(ln)), None)
    if start is None:
        raise KeyError(f"System '{system_name}' not found in {path}")
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = len(lines)
    for i in range(start + 1, len(lines)):
        ln = lines[i]
        if ln.strip() and (len(ln) - len(ln.lstrip())) <= indent:
            end = i
            break

    for param, value in params.items():
        pat = re.compile(rf"^(\s+{re.escape(param)}:\s*)\S.*?(\s*#.*)?$")
        for i in range(start + 1, end):
            m = pat.match(lines[i])
            if m:
                comment = m.group(2) or ""
                lines[i] = f"{m.group(1)}{value}{comment}"
                break
        else:
            # Not present: append into the block (indent = system indent + 2).
            lines.insert(end, f"{' ' * (indent + 2)}{param}: {value}")
            end += 1

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_tune(config: BenchmarkConfig, system_name: str, config_path: str) -> dict:
    if system_name not in config.systems:
        raise KeyError(f"System '{system_name}' not in config.")
    tuning = config.systems[system_name].get("tuning")
    if not tuning:
        raise ValueError(
            f"System '{system_name}' has no `tuning` block in config.yaml; "
            "add e.g. `tuning: {{grid: {{top_k: [3, 4, 6]}}}}`."
        )
    grid, index_params, normalize_groups = _parse_tuning(tuning)

    # The real index/results dir may be a param-keyed variant sub-path that
    # shifts as index params change between groups, so resolve it per group via
    # _output_dir (below). The archive must stay put across all of them: anchor
    # it at the outputs root (variant-independent) so `run_index(force=True)`,
    # which wipes a variant dir each trial, never deletes the archived results.
    archive = (
        Path(config.global_.paths.outputs_dir)
        / config.model_subdir(system_name)
        / f"{system_name}__tuning"
    )
    archive.mkdir(parents=True, exist_ok=True)
    syscfg = config.systems[system_name]

    # Group combos by their index-affecting params, so we re-index once per
    # distinct index config and reuse the persisted index for query-only sweeps.
    groups: dict[tuple, list[dict]] = {}
    for combo in _combos(grid):
        key = tuple((k, combo[k]) for k in grid if k in index_params)
        groups.setdefault(key, []).append(combo)
    logger.info("[tune] %d combo(s) in %d index group(s) (index_params=%s)",
                sum(len(v) for v in groups.values()), len(groups), sorted(index_params))

    trials: list[dict] = []
    for gkey, combos in groups.items():
        # Re-index once for this index config (use the group's shared index params).
        for k, v in gkey:
            setattr(syscfg, k, v)
        # Index params are now set, so the variant dir is settled for this group.
        output_dir = _output_dir(config, system_name)
        try:
            run_index(config, [system_name], force=True)  # force wipes output_dir
        except Exception:  # noqa: BLE001 — a bad index config skips its whole group
            logger.exception("[tune] indexing failed for %s; skipping group.", dict(gkey))
            continue

        for combo in combos:
            label = _combo_label(combo)
            try:
                for k, v in combo.items():  # set query params (index params unchanged)
                    setattr(syscfg, k, v)
                run_eval(config, [system_name])  # reuses the persisted index
                report = read_report(output_dir)
                archive.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(results_path(output_dir), archive / f"results_{label}.json")
                trials.append({
                    "params": combo,
                    "mean_score": report.aggregate.mean_score,
                    "grade_counts": report.aggregate.grade_counts,
                    "query_cost": report.aggregate.estimated_cost,
                    "query_tokens": report.aggregate.total_tokens,
                    "query_latency_s": report.aggregate.total_latency_s,
                    "index_time_s": report.indexing.time_s,
                })
                logger.info("[tune] %s -> mean_score=%.2f query_cost=$%s",
                            label, report.aggregate.mean_score, report.aggregate.estimated_cost)
            except Exception:  # noqa: BLE001 — one bad trial shouldn't abort the sweep
                logger.exception("[tune] trial %s failed; skipping.", label)

    if not trials:
        raise RuntimeError(f"[tune] all trials failed for '{system_name}'.")

    # Best: highest mean score, tie-break by lower query cost.
    best = max(trials, key=lambda t: (t["mean_score"], -t["query_cost"]))

    # Build the params to persist; normalize declared weight groups to sum to 1
    # (e.g. bm25_weight + dense_weight). Only the ratio affects retrieval, so this
    # is purely cosmetic — but keeps the written config readable.
    write_params = dict(best["params"])
    for group in normalize_groups:
        vals = {p: write_params.get(p, syscfg.get(p)) for p in group}
        total = sum(float(v) for v in vals.values() if v is not None)
        if total > 0:
            for p, v in vals.items():
                if v is not None:
                    write_params[p] = round(float(v) / total, 4)

    # Persist winner to config.yaml and reconcile in-memory + on-disk state.
    update_config_param(config_path, system_name, write_params)
    for k, v in write_params.items():
        setattr(syscfg, k, v)
    output_dir = _output_dir(config, system_name)  # winner's params may shift the variant dir
    run_index(config, [system_name], force=True)  # rebuild to match the winner (wipes output_dir)
    shutil.copyfile(archive / f"results_{_combo_label(best['params'])}.json",
                    results_path(output_dir))

    # output_dir now exists again (re-indexed); write the summary there, then clean archive.
    summary = {"system": system_name, "objective": "max mean_score, tie-break min query_cost",
               "best": best, "trials": trials}
    (output_dir / "tuning.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    shutil.rmtree(archive, ignore_errors=True)
    return summary
