"""LangChain client factories (OpenRouter-backed) and token/cost accounting."""
from __future__ import annotations

import logging

import tiktoken
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragbench.config import GlobalConfig

logger = logging.getLogger(__name__)

_warned_missing_pricing: set[str] = set()

# OpenAI embedding/chat models use the cl100k_base encoding; exact for billing.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(texts: list[str]) -> int:
    """Exact token count for OpenAI models (used for embedding-cost estimation)."""
    return sum(len(_ENCODING.encode(t)) for t in texts)


class LLMFactory:
    """Builds OpenRouter-backed LangChain clients and prices token usage."""

    def __init__(self, global_cfg: GlobalConfig):
        self.cfg = global_cfg

    def chat(self, model: str, temperature: float = 0.0) -> ChatOpenAI:
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url=self.cfg.endpoint,
            api_key=self.cfg.api_key,
        )

    def embeddings(self, model: str) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            model=model,
            base_url=self.cfg.endpoint,
            api_key=self.cfg.api_key,
            # OpenRouter slugs (e.g. "openai/text-embedding-3-small") are not in
            # tiktoken's registry; skip LangChain's token-based pre-chunking.
            check_embedding_ctx_length=False,
        )

    def cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate USD cost from the configured pricing table (per 1M tokens)."""
        price = self.cfg.pricing.get(model)
        if price is None:
            if model not in _warned_missing_pricing:
                logger.warning("No pricing configured for model '%s'; cost=0.0", model)
                _warned_missing_pricing.add(model)
            return 0.0
        return (input_tokens * price.input + output_tokens * price.output) / 1_000_000

    def embedding_cost(self, model: str, tokens: int) -> float:
        """Estimate embedding cost from the pricing table's input rate."""
        return self.cost(model, tokens, 0)


def usage_from_message(message) -> tuple[int, int]:
    """Extract (input_tokens, output_tokens) from a LangChain AIMessage."""
    meta = getattr(message, "usage_metadata", None) or {}
    return int(meta.get("input_tokens", 0)), int(meta.get("output_tokens", 0))
