"""Build and compile the LangGraph workflow for agentic datewise date ranking.

Graph topology
--------------
    START
      |
      v
  expand_query     <- enrich keywords into topic description  (1 LLM call)
      |
      v
  rank_dates_batch <- batch-rank ~30 candidate dates          (1 LLM call)
      |
      v
  verify_dates     <- self-reflection / critique selection    (1 LLM call)
      |
      v
  select_dates     <- deterministic merge + temporal diversity (0 LLM calls)
      |
      v
     END

Total: 3 LLM calls per topic.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from news_tls.agentic.datewise_state import DatewiseState
from news_tls.agentic.nodes import expand_query
from news_tls.agentic.datewise_nodes import (
    rank_dates_batch,
    verify_dates,
    select_dates,
)


def build_datewise_graph(llm):
    """Return a **compiled** LangGraph app for datewise date ranking."""

    # Thin closures that capture *llm* so every node gets the same instance.
    async def _expand(state):
        return await expand_query(state, llm)

    async def _rank(state):
        return await rank_dates_batch(state, llm)

    async def _verify(state):
        return await verify_dates(state, llm)

    async def _select(state):
        return await select_dates(state, llm)

    g = StateGraph(DatewiseState)
    g.add_node("expand_query", _expand)
    g.add_node("rank_dates_batch", _rank)
    g.add_node("verify_dates", _verify)
    g.add_node("select_dates", _select)

    g.add_edge(START, "expand_query")
    g.add_edge("expand_query", "rank_dates_batch")
    g.add_edge("rank_dates_batch", "verify_dates")
    g.add_edge("verify_dates", "select_dates")
    g.add_edge("select_dates", END)

    return g.compile()
