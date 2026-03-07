"""Async node functions for the agentic datewise date-ranking pipeline.

Nodes
-----
- rank_dates_batch : single LLM call to rank ~30 candidate dates
- verify_dates     : self-reflection LLM call to critique & correct selection
- select_dates     : deterministic merge of LLM + heuristic scores
"""

from __future__ import annotations

import datetime

from news_tls.agentic.nodes import _parse_json
from news_tls.agentic.prompts import RANK_DATES_PROMPT, VERIFY_DATES_PROMPT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_date_entries(date_contexts: list[dict]) -> str:
    """Format date context dicts into a compact text block for the LLM."""
    lines = []
    for dc in date_contexts:
        stats = dc.get("stats", {})
        header = (
            f"{dc['date']} | "
            f"{stats.get('mentions', 0)} mentions, "
            f"{stats.get('docs', 0)} docs, "
            f"{stats.get('published', 0)} published"
        )
        lines.append(header)
        for sent in dc.get("sentences", [])[:5]:
            lines.append(f"  - \"{sent}\"")
        lines.append("")
    return "\n".join(lines)


def _parse_date(ds: str) -> datetime.date:
    return datetime.datetime.strptime(ds, "%Y-%m-%d").date()


# ---------------------------------------------------------------------------
# Node: rank_dates_batch  (1 LLM call)
# ---------------------------------------------------------------------------

async def rank_dates_batch(state: dict, llm) -> dict:
    """Batch-rank all candidate dates in a single LLM call."""
    date_contexts = state["date_contexts"]
    date_entries_text = _format_date_entries(date_contexts)

    prompt = RANK_DATES_PROMPT.format(
        topic_description=state.get("topic_description", "news topic"),
        topic_type=state.get("topic_type", "other"),
        timeline_focus=state.get("timeline_focus", "key events"),
        max_dates=state["max_dates"],
        date_entries=date_entries_text,
    )

    try:
        resp = await llm.ainvoke(prompt)
        parsed = _parse_json(resp.content)
        if parsed and "selected_dates" in parsed:
            llm_ranked = parsed["selected_dates"]
            # Normalise: accept both list-of-dicts and list-of-strings
            normalised = []
            for item in llm_ranked:
                if isinstance(item, dict):
                    normalised.append(item)
                elif isinstance(item, str):
                    normalised.append({"date": item, "importance": 5, "reasoning": ""})
            if normalised:
                print(f"  [rank_dates] LLM selected {len(normalised)} dates")
                return {"llm_ranked_dates": normalised}
    except Exception as exc:
        print(f"  [rank_dates] LLM call failed: {exc}")

    # Fallback: return candidates in original pre-filtered order
    fallback = [
        {"date": dc["date"], "importance": 5, "reasoning": "fallback"}
        for dc in date_contexts[: state["max_dates"]]
    ]
    print(f"  [rank_dates] using fallback ordering ({len(fallback)} dates)")
    return {"llm_ranked_dates": fallback}


# ---------------------------------------------------------------------------
# Node: verify_dates  (1 LLM call — self-reflection)
# ---------------------------------------------------------------------------

async def verify_dates(state: dict, llm) -> dict:
    """Self-reflection: critique the LLM's date selection and correct if needed."""
    llm_ranked = state.get("llm_ranked_dates", [])
    date_contexts = state["date_contexts"]

    if not llm_ranked:
        return {"verified_dates": []}

    # Format selected dates
    selected_lines = []
    for i, entry in enumerate(llm_ranked):
        d = entry.get("date", "?")
        imp = entry.get("importance", "?")
        reason = entry.get("reasoning", "")
        selected_lines.append(f"  [{i+1}] {d} (importance={imp}): {reason}")
    selected_text = "\n".join(selected_lines)

    # Format all candidates (compact)
    candidate_lines = []
    for dc in date_contexts:
        stats = dc.get("stats", {})
        candidate_lines.append(
            f"  {dc['date']} | "
            f"{stats.get('mentions', 0)} mentions, "
            f"{stats.get('docs', 0)} docs"
        )
    all_candidates_text = "\n".join(candidate_lines)

    prompt = VERIFY_DATES_PROMPT.format(
        topic_description=state.get("topic_description", "news topic"),
        topic_type=state.get("topic_type", "other"),
        timeline_focus=state.get("timeline_focus", "key events"),
        max_dates=state["max_dates"],
        selected_entries=selected_text,
        all_candidates=all_candidates_text,
    )

    try:
        resp = await llm.ainvoke(prompt)
        parsed = _parse_json(resp.content)
        if parsed:
            reflection = parsed.get("reflection", "")
            if reflection:
                print(f"  [verify_dates] reflection: {reflection}")

            if parsed.get("needs_correction") and parsed.get("corrected_dates"):
                corrected = parsed["corrected_dates"]
                # Validate: corrected dates must be from candidate set
                candidate_date_strs = {dc["date"] for dc in date_contexts}
                valid_corrected = [
                    e for e in corrected
                    if isinstance(e, dict) and e.get("date") in candidate_date_strs
                ]
                if valid_corrected:
                    print(f"  [verify_dates] corrected to {len(valid_corrected)} dates")
                    return {"verified_dates": valid_corrected}
                else:
                    print("  [verify_dates] corrected dates invalid, keeping original")
            else:
                print("  [verify_dates] selection approved, no correction needed")
    except Exception as exc:
        print(f"  [verify_dates] LLM call failed: {exc}")

    # Fallback: keep original LLM ranking
    return {"verified_dates": llm_ranked}


# ---------------------------------------------------------------------------
# Node: select_dates  (deterministic — no LLM)
# ---------------------------------------------------------------------------

_ADJACENCY_PENALTY = 0.5     # score multiplier for dates within ±1 day
_ADJACENCY_DAYS = 1
_ALPHA = 0.7                  # LLM weight in composite score


async def select_dates(state: dict, llm) -> dict:
    """Merge LLM ranking with heuristic scores; apply temporal diversity."""
    verified = state.get("verified_dates", [])
    date_contexts = state["date_contexts"]
    max_dates = state["max_dates"]

    if not verified:
        # Pure fallback: use pre-filter order
        return {"ranked_dates": [dc["date"] for dc in date_contexts[:max_dates]]}

    # Build heuristic rank scores from pre-filter order (1.0 for rank 1, decaying)
    n_candidates = len(date_contexts)
    heuristic_scores = {}
    for i, dc in enumerate(date_contexts):
        heuristic_scores[dc["date"]] = 1.0 - (i / max(n_candidates, 1))

    # Build LLM importance scores (normalised to 0-1)
    max_imp = max((e.get("importance", 0) for e in verified), default=1) or 1
    llm_scores = {}
    for e in verified:
        d = e.get("date", "")
        llm_scores[d] = e.get("importance", 0) / max_imp

    # Composite scores for all verified dates
    candidate_set = {dc["date"] for dc in date_contexts}
    pool = []
    for e in verified:
        d = e.get("date", "")
        if d not in candidate_set:
            continue
        llm_s = llm_scores.get(d, 0)
        heur_s = heuristic_scores.get(d, 0)
        composite = _ALPHA * llm_s + (1 - _ALPHA) * heur_s
        pool.append({"date": d, "score": composite})

    # Greedy selection with adjacency penalty
    selected = []
    selected_dates = []

    while pool and len(selected) < max_dates:
        # Pick highest-scoring
        best_idx = max(range(len(pool)), key=lambda i: pool[i]["score"])
        best = pool.pop(best_idx)
        selected.append(best["date"])
        try:
            selected_dates.append(_parse_date(best["date"]))
        except ValueError:
            continue

        # Penalise adjacent dates
        for entry in pool:
            try:
                e_date = _parse_date(entry["date"])
                for sd in selected_dates:
                    if abs((e_date - sd).days) <= _ADJACENCY_DAYS:
                        entry["score"] *= _ADJACENCY_PENALTY
                        break
            except ValueError:
                pass

    # If we still need more dates (LLM returned fewer than max_dates),
    # fill from pre-filter order
    if len(selected) < max_dates:
        selected_set = set(selected)
        for dc in date_contexts:
            if len(selected) >= max_dates:
                break
            if dc["date"] not in selected_set:
                selected.append(dc["date"])
                selected_set.add(dc["date"])

    print(f"  [select_dates] final selection: {len(selected)} dates")
    return {"ranked_dates": selected}
