"""LLM provider configuration (Groq via OpenAI-compatible API)."""

from __future__ import annotations

import os
from langchain_openai import ChatOpenAI


def get_llm(
    model: str = "gpt-oss-20b",
    temperature: float = 0.3,
    base_url: str = "https://api.groq.com/openai/v1",
    api_key: str | None = None,
) -> ChatOpenAI:
    """Return a ChatOpenAI instance pointed at the Groq provider."""
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key or os.environ.get("GROQ_API_KEY", ""),
        temperature=temperature,
    )
