"""Chroma open helper that avoids stale-client errors.

chromadb caches its client/SQLite connection globally, keyed by persist path.
When a persist directory is removed and recreated within the same process —
e.g. the `tune` command's `index --force` loop, which `rmtree`s and rebuilds
`outputs/<system>/` per trial — a new `Chroma(...)` at the same path can be
handed the stale cached connection, which points at the deleted-then-recreated
SQLite file and fails writes with `sqlite3.OperationalError: attempt to write a
readonly database`.

Clearing chromadb's system cache before constructing the client forces a fresh
connection bound to the current on-disk state. Already-open client objects keep
their own references, so this is safe to call on every open.
"""
from __future__ import annotations

from chromadb.api.shared_system_client import SharedSystemClient
from langchain_chroma import Chroma


def open_chroma(collection_name: str, embedding_function, persist_directory) -> Chroma:
    SharedSystemClient.clear_system_cache()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=str(persist_directory),
    )
