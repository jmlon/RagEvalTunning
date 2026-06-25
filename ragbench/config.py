"""Configuration models and loader for the RAG benchmark.

Secrets (API keys) are never stored in config.yaml; only the *names* of the
environment variables that hold them. The endpoint and key are resolved from
the environment at load time.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


def _model_short(model: str) -> str:
    """Filesystem-friendly short name for a model id: drop the provider prefix.

    e.g. 'openai/gpt-4o-mini' -> 'gpt-4o-mini'. Used to build the model-combination
    sub-path under outputs/ and the per-combination report filename.
    """
    return model.rsplit("/", 1)[-1]


class Paths(BaseModel):
    inputs_dir: Path = Path("./inputs")
    outputs_dir: Path = Path("./outputs")
    questions_file: Path = Path("./test_questions.json")


class ModelPricing(BaseModel):
    input: float = 0.0   # USD per 1M input tokens
    output: float = 0.0  # USD per 1M output tokens


class Defaults(BaseModel):
    llm_model: str = "openai/gpt-4o-mini"
    embedding_model: str = "openai/text-embedding-3-small"


class JudgeConfig(BaseModel):
    model: str = "openai/gpt-4o-mini"
    temperature: float = 0.0


class GlobalConfig(BaseModel):
    paths: Paths = Field(default_factory=Paths)
    endpoint_env: str = "OPENROUTER_ENDPOINT"
    api_key_env: str = "OPENROUTER_API_KEY"
    defaults: Defaults = Field(default_factory=Defaults)
    pricing: dict[str, ModelPricing] = Field(default_factory=dict)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    # Seed for randomizing aggregated question order. None => non-reproducible.
    seed: int | None = None
    # Parallel worker threads for `run` (questions answered+judged concurrently). 1 = serial.
    workers: int = 4

    # Resolved at load time from the environment.
    endpoint: str = "https://openrouter.ai/api/v1"
    api_key: str = ""

    def resolve_env(self) -> None:
        self.endpoint = os.environ.get(self.endpoint_env, self.endpoint)
        self.api_key = os.environ.get(self.api_key_env, "")
        if not self.api_key:
            raise RuntimeError(
                f"API key env var '{self.api_key_env}' is not set. "
                "Set it in your environment / .env file."
            )


class SystemConfig(BaseModel):
    """Per-system config. Known fields are explicit; the rest live in `extra`."""

    model_config = {"extra": "allow"}

    enabled: bool = True
    llm_model: str | None = None
    embedding_model: str | None = None

    def get(self, key: str, default=None):
        return getattr(self, key, default)


class BenchmarkConfig(BaseModel):
    global_: GlobalConfig = Field(default_factory=GlobalConfig, alias="global")
    systems: dict[str, SystemConfig] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}

    def enabled_systems(self) -> list[str]:
        return [name for name, cfg in self.systems.items() if cfg.enabled]

    def resolve_models(self, name: str) -> tuple[str, str]:
        """Return (llm_model, embedding_model) for a system, applying defaults."""
        cfg = self.systems[name]
        llm = cfg.llm_model or self.global_.defaults.llm_model
        emb = cfg.embedding_model or self.global_.defaults.embedding_model
        return llm, emb

    def model_subdir(self, name: str) -> str:
        """Outputs sub-path segment identifying a system's model combination.

        Single source of truth for the `<llm>_<emb>` level inserted into the
        outputs tree (and reused for source-system cross-references, tuning
        archives, and report filenames), so distinct model choices keep their
        indexes/results side by side instead of overwriting each other.
        """
        llm, emb = self.resolve_models(name)
        return f"{_model_short(llm)}_{_model_short(emb)}"

    def defaults_model_subdir(self) -> str:
        """`model_subdir` for the global default models (no per-system override)."""
        d = self.global_.defaults
        return f"{_model_short(d.llm_model)}_{_model_short(d.embedding_model)}"


def load_config(path: str | Path = "config.yaml") -> BenchmarkConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = BenchmarkConfig.model_validate(raw)
    cfg.global_.resolve_env()
    return cfg
