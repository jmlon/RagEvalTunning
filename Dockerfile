# Comparative RAG Benchmark — container image.
# Build:  docker build -t ragbench .
# The corpus (inputs/) and generated indexes/results (outputs/) live on the host
# and are bind-mounted at runtime, so they stay outside the image.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# - bytecode compile for faster cold starts
# - copy (not hardlink) from the uv cache: the cache is on a different mount
# - keep the project venv at a stable, on-PATH location
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# 1) Install third-party dependencies first as a cached layer: this only busts
#    when pyproject.toml / uv.lock change, not on every source edit.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# 2) Copy the application and install the project itself (README.md is required
#    because pyproject.toml declares it as the package readme).
COPY README.md config.yaml ./
COPY ragbench ./ragbench
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# inputs/ (corpus + per-doc questions) and outputs/ (indexes + results) are
# provided by host bind-mounts at runtime; declare them so they're never baked in.
VOLUME ["/app/inputs", "/app/outputs"]

# The image IS the CLI: `docker run ... ragbench <subcommand> [flags]`.
ENTRYPOINT ["python", "-m", "ragbench.cli"]
CMD ["--help"]
