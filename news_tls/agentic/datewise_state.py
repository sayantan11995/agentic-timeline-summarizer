"""LangGraph state definitions for the agentic datewise pipeline."""

from __future__ import annotations
from typing import TypedDict


class DatewiseState(TypedDict, total=False):
    """
    Shared state that flows through the datewise date-ranking graph.

    Sections
    --------
    Inputs   – set once before the graph is invoked.
    Query    – enriched by the *expand_query* node.
    Ranking  – written by rank_dates_batch, refined by verify_dates.
    Output   – final ranked dates produced by *select_dates*.
    """

    # --- inputs (set before graph invocation) ---
    date_contexts: list       # List[dict] with keys: date, stats, sentences
    keywords: list            # List[str]
    max_dates: int

    # --- enriched query ---
    topic_description: str
    topic_type: str
    timeline_focus: str

    # --- ranking intermediates ---
    llm_ranked_dates: list    # List[dict] from LLM batch ranking
    verified_dates: list      # List[dict] after self-reflection verification

    # --- output ---
    ranked_dates: list        # List[str] – final date strings in ranked order
