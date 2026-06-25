"""Per-system registry of indexed documents (outputs/<system>/documents.json).

Tracks when each source document was indexed so the `index` command can skip
documents that are already indexed and unchanged. A document is (re)indexed
when it is absent from the registry or its source file is newer than the
recorded indexing.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def registry_path(output_dir: Path) -> Path:
    return Path(output_dir) / "documents.json"


def load_registry(output_dir: Path) -> dict:
    path = registry_path(output_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_registry(output_dir: Path, registry: dict) -> None:
    path = registry_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def needs_index(source_path: Path, entry: dict | None) -> bool:
    """True if the document is not yet indexed or has changed since indexing."""
    if entry is None:
        return True
    # Re-index when the source file is newer than what we last indexed.
    return source_path.stat().st_mtime > entry.get("source_mtime", 0.0)


def make_entry(source_path: Path) -> dict:
    mtime = source_path.stat().st_mtime
    return {
        "indexed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_mtime": mtime,
    }
