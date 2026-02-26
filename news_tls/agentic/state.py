"""LangGraph state definitions for the agentic timeline pipeline."""

from __future__ import annotations
from typing import TypedDict


class TimelineState(TypedDict, total=False):
    """
    Shared state that flows through every node of the timeline graph.

    Sections
    --------
    Inputs   – set once before the graph is invoked.
    Query    – enriched by the *expand_query* node.
    Pipeline – written / read by generate → judge → verify.
    Output   – final timeline produced by *finalize*.
    """

    # --- inputs (set before graph invocation) ---
    clusters: list            # List[dict] – one dict per cluster
    keywords: list            # List[str]
    max_dates: int
    max_summary_sents: int

    # --- enriched query ---
    topic_description: str
    topic_type: str
    timeline_focus: str

    # --- pipeline intermediates ---
    candidates: list          # generated candidate entries
    judged_entries: list      # scored / accepted entries
    verified_entries: list    # entries that survived verification

    # --- output ---
    timeline: list            # List[dict] with keys "date", "summary"
