"""LLM-Wiki: Karpathy's LLM Wiki pattern (pure form, no vector DB).

Indexing compiles the corpus into a structured markdown wiki: an LLM extracts
concepts from every chunk, and same-named concepts are merged across documents
into cross-document concept pages under `outputs/llm-wiki/wiki/`, plus an
`index.md` table of contents. Per-document concept extractions are persisted
under `outputs/llm-wiki/extractions/` so only changed documents are re-extracted
(the expensive LLM step); the wiki pages are re-rendered from all extractions.

Retrieval is the pure pattern — no vector search. The query starts from
`index.md`: the LLM selects relevant concept pages, optionally follows
`[[wikilinks]]` to neighbours, reads them, and answers.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from ragbench.llm import usage_from_message
from ragbench.models import Document, IndexStats, QueryOutput
from ragbench.systems.base import RAGSystem

logger = logging.getLogger(__name__)

_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


class Concept(BaseModel):
    title: str = Field(description="Canonical concept/entity/topic name, e.g. 'Self-Attention'.")
    summary: str = Field(description="One or two sentence summary of the concept.")
    key_points: list[str] = Field(default_factory=list, description="Salient points.")
    facts: list[str] = Field(default_factory=list, description="Specific factual claims, incl. numbers.")
    related: list[str] = Field(default_factory=list, description="Titles of related concepts.")


class ConceptList(BaseModel):
    concepts: list[Concept] = Field(default_factory=list)


class PageSelection(BaseModel):
    pages: list[str] = Field(
        default_factory=list,
        description="Exact titles of the wiki pages relevant to the question.",
    )


_EXTRACT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a knowledge compiler building a wiki. From the passage, "
            "extract the key concepts, entities, methods, and topics. For each, "
            "give a canonical title, a concise summary, key points, specific "
            "facts (include numbers/metrics), and related concept titles. Only "
            "extract what is supported by the passage.",
        ),
        ("human", "Passage:\n{passage}"),
    ]
)

_ROUTE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You navigate a markdown wiki to answer a question. Given the wiki "
            "INDEX (a list of concept pages with summaries), return the exact "
            "titles of the pages most relevant to the question. Choose only "
            "pages that appear in the index.",
        ),
        ("human", "QUESTION:\n{question}\n\nWIKI INDEX:\n{index}"),
    ]
)

_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the question using ONLY the "
            "provided wiki pages. If the answer is not in them, say you don't "
            "know. Be concise.",
        ),
        ("human", "Wiki pages:\n{context}\n\nQuestion: {question}"),
    ]
)


def _norm(title: str) -> str:
    return " ".join(title.strip().lower().split())


def _slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.strip().lower()).strip("-")
    return s or "concept"


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(s.strip() for s in items if s and s.strip()))


class LLMWikiSystem(RAGSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wiki_dir = self.output_dir / "wiki"
        self.extractions_dir = self.output_dir / "extractions"
        self.chunk_size = int(self.system_cfg.get("chunk_size", 3000))
        self.chunk_overlap = int(self.system_cfg.get("chunk_overlap", 300))
        self.route_max_pages = int(self.system_cfg.get("route_max_pages", 8))
        self.link_hops = int(self.system_cfg.get("link_hops", 1))
        self.temperature = float(self.system_cfg.get("temperature", 0.0))

    # ---- indexing ---------------------------------------------------------

    def _extract(self, passage: str) -> tuple[list[Concept], int, int]:
        chat = self.llm.chat(self.llm_model, temperature=self.temperature)
        structured = chat.with_structured_output(ConceptList, include_raw=True)
        res = (_EXTRACT_PROMPT | structured).invoke({"passage": passage})
        parsed: ConceptList | None = res.get("parsed")
        in_tok, out_tok = usage_from_message(res.get("raw"))
        return (parsed.concepts if parsed else []), in_tok, out_tok

    def index(self, documents: list[Document]) -> IndexStats:
        self.extractions_dir.mkdir(parents=True, exist_ok=True)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        total_in = total_out = 0
        for doc in documents:
            entries: list[dict] = []
            for chunk in splitter.split_text(doc.text):
                concepts, in_tok, out_tok = self._extract(chunk)
                total_in += in_tok
                total_out += out_tok
                for c in concepts:
                    entries.append({**c.model_dump(), "source": doc.source})
            (self.extractions_dir / f"{_slug(doc.id)}.json").write_text(
                json.dumps({"doc_id": doc.id, "source": doc.source, "concepts": entries},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("[%s] extracted %d concept mention(s) from %s",
                        self.name, len(entries), doc.source)

        pages = self._aggregate()
        self._render(pages)
        cost = self.llm.cost(self.llm_model, total_in, total_out)
        logger.info(
            "[%s] compiled %d concept page(s) (%d extraction tokens).",
            self.name, len(pages), total_in + total_out,
        )
        return IndexStats(passages=len(pages), tokens=total_in + total_out, cost=cost)

    def _aggregate(self) -> dict[str, dict]:
        """Merge concept entries across all persisted extractions, keyed by title."""
        pages: dict[str, dict] = {}
        for path in sorted(self.extractions_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            for c in data.get("concepts", []):
                title = (c.get("title") or "").strip()
                if not title:
                    continue
                acc = pages.setdefault(_norm(title), {
                    "title": title, "summaries": [], "key_points": [],
                    "facts": [], "related": [], "sources": [],
                })
                if len(title) > len(acc["title"]):
                    acc["title"] = title  # prefer the more descriptive surface form
                if c.get("summary"):
                    acc["summaries"].append(c["summary"].strip())
                acc["key_points"] += c.get("key_points", [])
                acc["facts"] += c.get("facts", [])
                acc["related"] += c.get("related", [])
                acc["sources"].append(c.get("source", ""))
        return pages

    def _render(self, pages: dict[str, dict]) -> None:
        # Fresh render of the whole wiki from current extractions.
        if self.wiki_dir.exists():
            for old in self.wiki_dir.glob("*.md"):
                old.unlink()
        self.wiki_dir.mkdir(parents=True, exist_ok=True)

        index_lines = ["# Wiki Index", ""]
        for key in sorted(pages):
            p = pages[key]
            title = p["title"]
            summaries = _dedupe(p["summaries"])
            summary = max(summaries, key=len) if summaries else ""
            key_points = _dedupe(p["key_points"])
            facts = _dedupe(p["facts"])
            related = _dedupe(p["related"])
            sources = _dedupe(p["sources"])

            md = [f"# {title}", "", "## Summary", summary or "(none)", ""]
            if key_points:
                md += ["## Key points", *[f"- {x}" for x in key_points], ""]
            if facts:
                md += ["## Facts", *[f"- {x}" for x in facts], ""]
            if related:
                md += ["## Related", *[f"- [[{x}]]" for x in related], ""]
            if sources:
                md += ["## Sources", *[f"- {x}" for x in sources], ""]
            (self.wiki_dir / f"{_slug(title)}.md").write_text("\n".join(md), encoding="utf-8")

            one_liner = summary.splitlines()[0] if summary else ""
            index_lines.append(f"- [[{title}]] — {one_liner}")
        (self.wiki_dir / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    # ---- retrieval (pure, index.md-driven) --------------------------------

    def load(self) -> None:
        if not (self.wiki_dir / "index.md").exists():
            raise FileNotFoundError(
                f"[{self.name}] no wiki at {self.wiki_dir}/index.md; run `index -s {self.name}` first."
            )

    def _page_map(self) -> dict[str, Path]:
        """Map normalized concept title -> page file (from the page H1)."""
        out: dict[str, Path] = {}
        for path in self.wiki_dir.glob("*.md"):
            if path.name == "index.md":
                continue
            first = path.read_text(encoding="utf-8").splitlines()[:1]
            if first and first[0].startswith("# "):
                out[_norm(first[0][2:])] = path
        return out

    def query(self, question: str) -> QueryOutput:
        start = time.perf_counter()
        try:
            index_md = (self.wiki_dir / "index.md").read_text(encoding="utf-8")
            page_map = self._page_map()

            # Route: LLM selects relevant page titles from the index.
            chat = self.llm.chat(self.llm_model, temperature=self.temperature)
            route = chat.with_structured_output(PageSelection, include_raw=True)
            r = (_ROUTE_PROMPT | route).invoke({"question": question, "index": index_md})
            sel: PageSelection | None = r.get("parsed")
            in_tok, out_tok = usage_from_message(r.get("raw"))

            titles = (sel.pages[: self.route_max_pages] if sel else [])
            selected = [page_map[_norm(t)] for t in titles if _norm(t) in page_map]

            # Navigate: expand via [[wikilinks]] up to link_hops.
            gathered: dict[Path, str] = {}
            frontier = list(dict.fromkeys(selected))
            for _ in range(self.link_hops + 1):
                next_frontier: list[Path] = []
                for path in frontier:
                    if path in gathered:
                        continue
                    text = path.read_text(encoding="utf-8")
                    gathered[path] = text
                    for link in _WIKILINK_RE.findall(text):
                        nb = page_map.get(_norm(link))
                        if nb and nb not in gathered:
                            next_frontier.append(nb)
                frontier = next_frontier
                if not frontier:
                    break

            page_texts = list(gathered.values())
            context = "\n\n---\n\n".join(page_texts) if page_texts else "(no relevant pages)"

            msg = (_ANSWER_PROMPT | chat).invoke({"context": context, "question": question})
            a_in, a_out = usage_from_message(msg)
            tot_in, tot_out = in_tok + a_in, out_tok + a_out
            return QueryOutput(
                answer=msg.content,
                contexts=page_texts,
                latency_s=time.perf_counter() - start,
                tokens=tot_in + tot_out,
                cost=self.llm.cost(self.llm_model, tot_in, tot_out),
            )
        except Exception:  # noqa: BLE001 — must not abort the run (FR-1.4)
            logger.exception("[%s] query failed for: %s", self.name, question)
            return QueryOutput(answer=None, latency_s=time.perf_counter() - start)
