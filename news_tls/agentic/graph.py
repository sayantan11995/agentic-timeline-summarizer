"""Build and compile the LangGraph workflow for agentic timeline construction.

Graph topology
--------------
    START
      │
      ▼
  expand_query     ← enrich keywords into topic description
      │
      ▼
   generate        ← produce candidate summaries (parallel async)
      │
      ▼
    judge           ← score & filter candidates    (parallel async)
      │
      ▼
   verify           ← coherence / redundancy check
      │
      ▼
  finalize          ← deduplicate, enforce limits
      │
      ▼
     END

The graph is intentionally linear for v1 but uses conditional-edge-ready
node signatures so that future iterations can add loops (e.g. verify →
regenerate) without changing the node implementations.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from news_tls.agentic.state import TimelineState
from news_tls.agentic import nodes as _n


def build_timeline_graph(llm):
    """Return a **compiled** LangGraph app bound to *llm*."""

    # Thin closures that capture *llm* so every node gets the same instance.
    async def _expand(state):
        return await _n.expand_query(state, llm)

    async def _generate(state):
        return await _n.generate_candidates(state, llm)

    async def _judge(state):
        return await _n.judge_candidates(state, llm)

    async def _verify(state):
        return await _n.verify_timeline(state, llm)

    async def _finalize(state):
        return await _n.finalize_timeline(state, llm)

    g = StateGraph(TimelineState)
    g.add_node("expand_query", _expand)
    g.add_node("generate", _generate)
    g.add_node("judge", _judge)
    g.add_node("verify", _verify)
    g.add_node("finalize", _finalize)

    g.add_edge(START, "expand_query")
    g.add_edge("expand_query", "generate")
    g.add_edge("generate", "judge")
    g.add_edge("judge", "verify")
    g.add_edge("verify", "finalize")
    g.add_edge("finalize", END)

    return g.compile()
