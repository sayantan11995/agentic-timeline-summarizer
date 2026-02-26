"""Async node functions for the agentic timeline LangGraph.

Each public function has the signature ``(state, llm) -> dict`` and returns a
*partial* state update that LangGraph merges into the running state.
"""

from __future__ import annotations

import asyncio
import json
import re

from news_tls.agentic.prompts import (
    EXPAND_QUERY_PROMPT,
    GENERATE_CANDIDATE_PROMPT,
    JUDGE_CANDIDATE_PROMPT,
    VERIFY_TIMELINE_PROMPT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict | None:
    """Best-effort JSON extraction from an LLM response."""
    # 1. Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    # 2. Markdown code-block
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # 3. First top-level JSON object
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Node: expand_query
# ---------------------------------------------------------------------------

async def expand_query(state: dict, llm) -> dict:
    """Enrich raw keywords into a structured topic description."""
    keywords = state["keywords"]
    prompt = EXPAND_QUERY_PROMPT.format(keywords=", ".join(keywords))

    resp = await llm.ainvoke(prompt)
    parsed = _parse_json(resp.content)

    if parsed:
        return {
            "topic_description": parsed.get(
                "topic_description", ", ".join(keywords)
            ),
            "topic_type": parsed.get("topic_type", "other"),
            "timeline_focus": parsed.get(
                "timeline_focus", "key events and developments"
            ),
        }
    # Fallback when parsing fails
    return {
        "topic_description": f"News topic about: {', '.join(keywords)}",
        "topic_type": "other",
        "timeline_focus": "key events and developments",
    }


# ---------------------------------------------------------------------------
# Node: generate_candidates  (fan-out async)
# ---------------------------------------------------------------------------

async def _generate_single(
    llm, cluster, topic_desc, topic_type, timeline_focus, k, sem
):
    async with sem:
        sents_text = "\n".join(
            f"- {s}" for s in cluster["sentences"][:30]
        )
        prompt = GENERATE_CANDIDATE_PROMPT.format(
            topic_description=topic_desc,
            topic_type=topic_type,
            date=cluster["date"],
            timeline_focus=timeline_focus,
            sentences=sents_text,
            k=k,
        )
        try:
            resp = await llm.ainvoke(prompt)
            parsed = _parse_json(resp.content)
            if parsed and "summary" in parsed:
                return {
                    "cluster_id": cluster["cluster_id"],
                    "date": cluster["date"],
                    "summary": parsed["summary"][:k],
                    "reasoning": parsed.get("reasoning", ""),
                }
        except Exception as exc:
            print(f"  [generate] cluster {cluster['cluster_id']} failed: {exc}")
    return None


async def generate_candidates(state: dict, llm) -> dict:
    """Generate a candidate summary for every pre-filtered cluster."""
    sem = asyncio.Semaphore(10)
    tasks = [
        _generate_single(
            llm,
            c,
            state["topic_description"],
            state["topic_type"],
            state["timeline_focus"],
            state["max_summary_sents"],
            sem,
        )
        for c in state["clusters"]
    ]
    results = await asyncio.gather(*tasks)
    candidates = [r for r in results if r is not None]
    print(f"  [generate] {len(candidates)}/{len(state['clusters'])} candidates produced")
    return {"candidates": candidates}


# ---------------------------------------------------------------------------
# Node: judge_candidates  (fan-out async)
# ---------------------------------------------------------------------------

async def _judge_single(llm, cand, topic_desc, topic_type, timeline_focus, sem):
    async with sem:
        prompt = JUDGE_CANDIDATE_PROMPT.format(
            topic_description=topic_desc,
            topic_type=topic_type,
            date=cand["date"],
            summary="; ".join(cand["summary"]),
            timeline_focus=timeline_focus,
        )
        try:
            resp = await llm.ainvoke(prompt)
            parsed = _parse_json(resp.content)
            if parsed:
                return {
                    **cand,
                    "relevance_score": float(parsed.get("relevance_score", 0)),
                    "importance_score": float(parsed.get("importance_score", 0)),
                    "quality_score": float(parsed.get("quality_score", 0)),
                    "judge_reasoning": parsed.get("reasoning", ""),
                    "accepted": bool(parsed.get("accepted", False)),
                }
        except Exception as exc:
            print(f"  [judge] candidate {cand['date']} failed: {exc}")
    return None


async def judge_candidates(state: dict, llm) -> dict:
    """Score every candidate and keep the accepted ones."""
    sem = asyncio.Semaphore(10)
    tasks = [
        _judge_single(
            llm,
            c,
            state["topic_description"],
            state["topic_type"],
            state["timeline_focus"],
            sem,
        )
        for c in state["candidates"]
    ]
    results = await asyncio.gather(*tasks)
    judged = [r for r in results if r is not None]

    def _score(entry):
        return entry.get("importance_score", 0) + entry.get("relevance_score", 0)

    accepted = sorted(
        [j for j in judged if j.get("accepted")],
        key=_score,
        reverse=True,
    )

    # Fallback: if nothing was accepted, keep the highest-scored candidates
    if not accepted and judged:
        accepted = sorted(judged, key=_score, reverse=True)[
            : state.get("max_dates", 10)
        ]

    print(f"  [judge] {len(accepted)}/{len(judged)} candidates accepted")
    return {"judged_entries": accepted}


# ---------------------------------------------------------------------------
# Node: verify_timeline
# ---------------------------------------------------------------------------

async def verify_timeline(state: dict, llm) -> dict:
    """Check overall coherence / redundancy and prune if needed."""
    entries = state.get("judged_entries", [])
    # Feed the verifier more entries than max_dates so it can prune
    entries = entries[: state["max_dates"] * 2]

    if not entries:
        return {"verified_entries": []}

    tl_text = "\n".join(
        f"[{i}] {e['date']}: {'; '.join(e['summary'])} "
        f"(importance={e.get('importance_score', '?')})"
        for i, e in enumerate(entries)
    )
    prompt = VERIFY_TIMELINE_PROMPT.format(
        topic_description=state["topic_description"],
        topic_type=state["topic_type"],
        timeline_focus=state["timeline_focus"],
        timeline_entries=tl_text,
    )

    try:
        resp = await llm.ainvoke(prompt)
        parsed = _parse_json(resp.content)
        if parsed and "keep_indices" in parsed:
            keep = [int(i) for i in parsed["keep_indices"]]
            verified = [entries[i] for i in keep if 0 <= i < len(entries)]
            if verified:
                removed = len(entries) - len(verified)
                print(f"  [verify] kept {len(verified)}, removed {removed}")
                return {"verified_entries": verified}
    except Exception as exc:
        print(f"  [verify] failed: {exc}")

    # Fallback: keep everything
    return {"verified_entries": entries}


# ---------------------------------------------------------------------------
# Node: finalize_timeline
# ---------------------------------------------------------------------------

async def finalize_timeline(state: dict, llm) -> dict:
    """Deduplicate by date, enforce *max_dates*, and produce the output list."""
    entries = state.get("verified_entries", [])
    max_dates = state["max_dates"]

    # Keep the highest-scored entry per date
    by_date: dict[str, tuple[dict, float]] = {}
    for e in entries:
        d = e["date"]
        score = e.get("importance_score", 0) + e.get("relevance_score", 0)
        if d not in by_date or score > by_date[d][1]:
            by_date[d] = (e, score)

    sorted_entries = sorted(by_date.values(), key=lambda x: x[0]["date"])
    timeline = [
        {"date": e["date"], "summary": e["summary"]}
        for e, _ in sorted_entries[:max_dates]
    ]
    print(f"  [finalize] timeline has {len(timeline)} entries")
    return {"timeline": timeline}
