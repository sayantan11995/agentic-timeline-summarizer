"""Async node functions for the agentic timeline LangGraph.

Each public function has the signature ``(state, llm) -> dict`` and returns a
*partial* state update that LangGraph merges into the running state.
"""

from __future__ import annotations

import asyncio
import datetime
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
                    "article_count": cluster.get("article_count", 0),
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

_ARTICLE_COUNT_WEIGHT = 2.0   # max bonus from article density (on a 0-20 scale)
_ADJACENCY_PENALTY = 0.5     # score multiplier for dates within ±1 day of a pick
_ADJACENCY_DAYS = 1          # how many days count as "adjacent"


async def finalize_timeline(state: dict, llm) -> dict:
    """Select the *max_dates* most important entries with temporal diversity.

    Algorithm
    ---------
    1. Deduplicate by date (keep highest LLM-scored entry per date).
    2. Compute a composite score per entry:
         ``(importance + relevance) + article_count_bonus``
       where ``article_count_bonus`` scales linearly up to
       ``_ARTICLE_COUNT_WEIGHT``.
    3. Greedy iterative selection (like MMR for temporal spread):
       - Pick the highest-scoring entry.
       - Penalise remaining entries whose dates fall within
         ``±_ADJACENCY_DAYS`` of any already-selected date.
       - Repeat until *max_dates* entries are selected.
    4. Sort the selected entries chronologically for the final output.
    """
    entries = state.get("verified_entries", [])
    max_dates = state["max_dates"]

    if not entries:
        print("  [finalize] no entries to finalize")
        return {"timeline": []}

    # -- Step 1: deduplicate by date (keep best per date) ----------------
    by_date: dict[str, dict] = {}
    for e in entries:
        d = e["date"]
        score = e.get("importance_score", 0) + e.get("relevance_score", 0)
        if d not in by_date:
            by_date[d] = e
        else:
            prev = by_date[d]
            prev_score = prev.get("importance_score", 0) + prev.get("relevance_score", 0)
            if score > prev_score:
                by_date[d] = e

    pool = list(by_date.values())

    # -- Step 2: composite scores ----------------------------------------
    max_ac = max((e.get("article_count", 0) for e in pool), default=1) or 1

    scores: dict[str, float] = {}
    for e in pool:
        llm_score = e.get("importance_score", 0) + e.get("relevance_score", 0)
        ac_bonus = (e.get("article_count", 0) / max_ac) * _ARTICLE_COUNT_WEIGHT
        scores[e["date"]] = llm_score + ac_bonus

    # -- Step 3: greedy selection with adjacency penalty -----------------
    selected: list[dict] = []
    selected_dates: list[datetime.date] = []

    def _parse_date(ds: str) -> datetime.date:
        return datetime.datetime.strptime(ds, "%Y-%m-%d").date()

    while pool and len(selected) < max_dates:
        # pick the entry with the highest (possibly penalised) score
        best_idx = max(range(len(pool)), key=lambda i: scores[pool[i]["date"]])
        best = pool.pop(best_idx)
        selected.append(best)
        selected_dates.append(_parse_date(best["date"]))

        # penalise remaining entries that are temporally adjacent
        for e in pool:
            e_date = _parse_date(e["date"])
            for sd in selected_dates:
                if abs((e_date - sd).days) <= _ADJACENCY_DAYS:
                    scores[e["date"]] *= _ADJACENCY_PENALTY
                    break  # one penalty per round is enough

    # -- Step 4: sort chronologically ------------------------------------
    selected.sort(key=lambda e: e["date"])
    timeline = [{"date": e["date"], "summary": e["summary"]} for e in selected]
    print(f"  [finalize] timeline has {len(timeline)} entries")
    return {"timeline": timeline}
