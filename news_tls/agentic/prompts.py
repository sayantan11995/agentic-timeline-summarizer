"""Prompt templates for each agentic node.

Every template uses double-braces ``{{`` / ``}}`` for literal JSON braces
so that Python's ``str.format`` does not treat them as placeholders.
"""

EXPAND_QUERY_PROMPT = """\
You are a news-analysis expert.  Given the following keywords that describe a
news topic, produce a rich characterisation so that downstream agents can build
a high-quality timeline.

Keywords: {keywords}

Respond with **only** the JSON object below (no markdown fences):
{{
    "topic_description": "A 2-3 sentence description of the topic",
    "topic_type": "One of: disaster, political, organizational, scientific, conflict, economic, social, legal, other",
    "timeline_focus": "What aspects to prioritise in the timeline for this topic type"
}}"""

GENERATE_CANDIDATE_PROMPT = """\
You are a timeline summariser.  Given news sentences from a cluster of related
articles, generate a concise summary for the indicated date.

Topic        : {topic_description}
Topic type   : {topic_type}
Date         : {date}
Focus        : {timeline_focus}

Sentences (from articles on / around this date):
{sentences}

Requirements:
- Generate exactly {k} summary sentence(s).
- Use only facts present in the sentences above.
- For a **{topic_type}** topic, prioritise: {timeline_focus}.
- Each sentence must be self-contained and informative.

Respond with **only** the JSON object below (no markdown fences):
{{
    "summary": ["sentence1"],
    "reasoning": "Why this event is notable for the timeline"
}}"""

JUDGE_CANDIDATE_PROMPT = """\
You are a timeline quality judge.  Decide whether the entry below belongs in a
**{topic_type}** timeline about: "{topic_description}"

Date    : {date}
Summary : {summary}
Focus   : {timeline_focus}

Rate each aspect 0-10:
1. **Relevance**  – How relevant is this entry to the specific topic?
2. **Importance** – How significant is this event in the overall narrative?
3. **Quality**    – How clear, factual, and informative is the summary?

Accept the entry when **all three** scores are >= 5.

Respond with **only** the JSON object below (no markdown fences):
{{
    "relevance_score": 7,
    "importance_score": 8,
    "quality_score": 7,
    "reasoning": "Brief justification",
    "accepted": true
}}"""

VERIFY_TIMELINE_PROMPT = """\
You are a timeline coherence verifier.  Review the proposed timeline for a
**{topic_type}** topic about: "{topic_description}"

Focus: {timeline_focus}

Proposed entries:
{timeline_entries}

Check for:
1. **Redundancy** – entries that describe essentially the same event (keep the
   better one).
2. **Coherence**  – the timeline should read as a logical narrative.
3. **Relevance**  – every entry must genuinely relate to this topic type.

Return the 0-based indices of entries to **keep**.

Respond with **only** the JSON object below (no markdown fences):
{{
    "keep_indices": [0, 1, 3],
    "removed_reasons": {{"2": "Redundant with entry 1"}},
    "feedback": "Overall assessment"
}}"""
