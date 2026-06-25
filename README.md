# Comparative RAG Benchmark

A pluggable harness that benchmarks multiple RAG systems (Simple-RAG, Hybrid-RAG, Simple-RAG-Reranker, Hybrid-RAG-Reranker, Graph-RAG, LLM-Wiki)
over a shared PDF corpus and a shared question set, grading answers against
ground truth with an LLM judge.

## Setup

- Put source PDFs and associated questions in `./inputs/`. E.g., `doc1.pdf` and `doc1.json`.
- Configure systems/models in `./config.yaml`.
- Create your `.env` from the template and set the API key:

  ```sh
  cp .env-example .env
  # then edit .env and replace sk-or-XXXXX with your real OpenRouter API key
  ```

  Set `OPENROUTER_API_KEY` (and optionally `OPENROUTER_ENDPOINT`) in `.env`.
- For the rerankers, set `COHERE_API_KEY` in `.env`.
- For `graph-rag`, start a local Neo4j first: `bash scripts/neo4j.sh up` (data persists in a Docker volume; `down` to stop, `reset` to wipe). Override the password with `NEO4J_LOCAL_PASSWORD`.

## Usage

```sh
uv run python -m ragbench.cli index     # ingest PDFs, build each system's index
uv run python -m ragbench.cli run       # answer + judge every question
uv run python -m ragbench.cli report    # per-system summary + outputs/reports/ table

# Target a single system with -s / --system (repeatable), -f / --force to force reindexing
uv run python -m ragbench.cli index -s simple-rag --force
uv run python -m ragbench.cli index -s llm-wiki --force

# Running individual analysis
uv run python -m ragbench.cli run -s simple-rag
uv run python -m ragbench.cli run -s hybrid-rag
uv run python -m ragbench.cli run -s llm-wiki
uv run python -m ragbench.cli run -s graph-rag

# Tune one system's params (grid-search the values in its config `tuning:` block);
# picks the best (max score, tie-break lower query cost) and updates config.yaml:
uv run python -m ragbench.cli tune -s simple-rag
uv run python -m ragbench.cli tune -s hybrid-rag
```

Outputs are grouped by model combination. Per-system results are written to
`./outputs/<llm>_<embedding>/<system>/results.json`. The `report` command also
writes a cross-system comparison table to `./outputs/reports/<llm>_<embedding>.md`
(rows = systems, columns = performance indicators).
`index` is incremental (skips unchanged docs); use `--force` to rebuild.

## Docker

The image bundles the code and dependencies; the corpus (`inputs/`) and all
generated indexes/results (`outputs/`) stay on the host via bind-mounts, so they
persist across runs and never get baked into the image.

Build the image once:

```sh
docker build -t ragbench .
```

The image's entrypoint **is** the CLI, so `docker run ... ragbench <subcommand>`
maps directly to the `uv run python -m ragbench.cli <subcommand>` calls above.
API keys are read from `.env`; `inputs/` and `outputs/` are mounted into the
container at `/app/inputs` and `/app/outputs`.

The `--network`/`-e` flags below let the container reach Neo4j for `graph-rag`
(see the Neo4j note at the end); they're harmless for the other systems, so the
examples include them on every command for copy-paste safety. `report` only
reads files and never connects to Neo4j, so it omits them.

```sh
# Index: ingest PDFs from ./inputs and build each system's index under ./outputs
docker run --rm --env-file .env \
  --network ragbench-net \
  -e NEO4J_LOCAL_URI=bolt://ragbench-neo4j:7687 \
  -v "$(pwd)/inputs:/app/inputs" \
  -v "$(pwd)/outputs:/app/outputs" \
  ragbench index

# Evaluate: answer + judge every question
docker run --rm --env-file .env \
  --network ragbench-net \
  -e NEO4J_LOCAL_URI=bolt://ragbench-neo4j:7687 \
  -v "$(pwd)/inputs:/app/inputs" \
  -v "$(pwd)/outputs:/app/outputs" \
  ragbench run

# Report: per-system summary + cross-system table under ./outputs/reports/
docker run --rm --env-file .env \
  -v "$(pwd)/inputs:/app/inputs" \
  -v "$(pwd)/outputs:/app/outputs" \
  ragbench report
```

All CLI flags pass straight through, e.g. target a single system or force a
rebuild:

```sh
docker run --rm --env-file .env \
  -v "$(pwd)/inputs:/app/inputs" -v "$(pwd)/outputs:/app/outputs" \
  ragbench index -s simple-rag --force
```

Notes:

- `config.yaml` is baked into the image at build time. To iterate on it without
  rebuilding, mount it over the copy in the image:
  `-v "$(pwd)/config.yaml:/app/config.yaml"`.
- `graph-rag` needs a reachable Neo4j. The bundled `scripts/neo4j.sh` starts one
  and attaches it to the user-defined network `ragbench-net`. The `docker run`
  examples above join that network (`--network ragbench-net`) and point the URI
  at the container name (`-e NEO4J_LOCAL_URI=bolt://ragbench-neo4j:7687`).
  `NEO4J_LOCAL_URI` overrides `neo4j_uri` in `config.yaml`, which stays
  `bolt://localhost:7687` for host runs (`uv run …`). Do **not** put
  `NEO4J_LOCAL_URI` in `.env`: it's loaded on host runs too and would break them.
  On Linux, `--network host` with the default localhost URI also works.
