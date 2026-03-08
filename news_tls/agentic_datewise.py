"""Agentic Datewise Timeline Generator.

Drop-in replacement for ``DatewiseTimelineGenerator`` that keeps the
datewise pipeline structure (date ranking → sentence collection →
extractive summarisation) but delegates **date ranking** to a LangGraph
pipeline (expand_query → rank_dates_batch → verify_dates → select_dates).

Total LLM calls: 3 per topic (vs. ~62 for agentic_clust).
"""

from __future__ import annotations

import asyncio
import collections
import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

from news_tls import data, utils, summarizers
from news_tls.datewise import (
    DateRanker,
    MentionCountDateRanker,
    DatewiseTimelineGenerator,
    PM_Mean_SentenceCollector,
    detect_knee_point,
)
from news_tls.agentic.llm import get_llm
from news_tls.agentic.datewise_graph import build_datewise_graph


class AgenticDateRanker(DateRanker):
    """Date ranker that uses a LangGraph pipeline for LLM-based re-ranking.

    Parameters
    ----------
    base_ranker : DateRanker, optional
        Pre-filter ranker (default: ``MentionCountDateRanker``).
    candidate_multiplier : int
        ``max_dates * candidate_multiplier`` dates are pre-filtered before
        LLM ranking.
    llm : ChatOpenAI, optional
        LLM instance; built from ``get_llm()`` when *None*.
    """

    def __init__(self, base_ranker=None, candidate_multiplier=3, llm=None):
        self.base_ranker = base_ranker or MentionCountDateRanker()
        self.candidate_multiplier = candidate_multiplier
        self._llm = llm or get_llm()
        self._graph = build_datewise_graph(self._llm)
        # Topic context — populated after the LangGraph pipeline runs
        self.topic_description = None
        self.topic_type = None
        self.timeline_focus = None

    # Forward model property to base_ranker (for SupervisedDateRanker compat)
    @property
    def model(self):
        return getattr(self.base_ranker, "model", None)

    @model.setter
    def model(self, value):
        self.base_ranker.model = value

    def rank_dates(self, collection, **kwargs):
        """Rank dates using pre-filter + LLM batch ranking + self-reflection.

        Parameters
        ----------
        collection : data.Collection
            The news article collection.
        max_dates : int, optional
            Target number of dates (passed from predict()).
        """
        max_dates = kwargs.get("max_dates", 10)

        # --- Step 1: Pre-filter with base ranker ---
        print("  [agentic_date] pre-filtering with base ranker...")
        base_ranked = self.base_ranker.rank_dates(collection)
        top_n = max_dates * self.candidate_multiplier
        candidates = base_ranked[:top_n]

        if not candidates:
            return []

        # --- Step 2: Build date contexts ---
        print(f"  [agentic_date] building contexts for {len(candidates)} candidate dates...")
        date_contexts = self._build_contexts(candidates, collection)

        # --- Step 3: Run agentic LangGraph pipeline ---
        print("  [agentic_date] running LLM date ranking pipeline...")
        initial_state = {
            "date_contexts": date_contexts,
            "keywords": list(collection.keywords),
            "max_dates": max_dates,
        }
        result = asyncio.run(self._graph.ainvoke(initial_state))

        # Store topic context so the summarizer can use it later
        self.topic_description = result.get("topic_description", "")
        self.topic_type = result.get("topic_type", "other")
        self.timeline_focus = result.get("timeline_focus", "key events and developments")

        # --- Step 4: Parse ranked dates back to date objects ---
        ranked_strs = result.get("ranked_dates", [])
        ranked_dates = []
        seen = set()
        for ds in ranked_strs:
            try:
                d = datetime.datetime.strptime(ds, "%Y-%m-%d").date()
                if d not in seen:
                    ranked_dates.append(d)
                    seen.add(d)
            except ValueError:
                continue

        # Append remaining candidates not selected by LLM (for downstream
        # truncation by DatewiseTimelineGenerator.predict())
        for d in candidates:
            if d not in seen:
                ranked_dates.append(d)
                seen.add(d)

        print(f"  [agentic_date] returning {len(ranked_dates)} ranked dates "
              f"(top {min(max_dates, len(ranked_dates))} will be used)")
        return ranked_dates

    def _build_contexts(self, candidates, collection):
        """Build compact context dicts for each candidate date."""
        # Gather date statistics
        date_to_stats = self._extract_date_stats(collection)

        # Gather sentences per date (mention-based + publication-based)
        date_to_ment = collections.defaultdict(list)
        date_to_pub = collections.defaultdict(list)
        for a in collection.articles():
            pub_date = a.time.date()
            for s in a.sentences[:5]:
                for k in range(2):  # pub_end=2
                    pub_date2 = pub_date - datetime.timedelta(days=k)
                    date_to_pub[pub_date2].append(s)
            for s in a.sentences:
                ment_date = s.get_date()
                if ment_date:
                    date_to_ment[ment_date].append(s)

        contexts = []
        for d in candidates:
            stats = date_to_stats.get(d, {})
            # Merge and keyword-filter sentences
            all_sents = date_to_ment.get(d, []) + date_to_pub.get(d, [])
            filtered = []
            seen_raw = set()
            for s in all_sents:
                lower = s.raw.lower()
                if any(kw in lower for kw in collection.keywords):
                    if s.raw not in seen_raw:
                        filtered.append(s.raw)
                        seen_raw.add(s.raw)
                if len(filtered) >= 5:
                    break

            if not filtered:
                # Take any sentences if keyword filter yields nothing
                for s in all_sents[:5]:
                    if s.raw not in seen_raw:
                        filtered.append(s.raw)
                        seen_raw.add(s.raw)

            contexts.append({
                "date": str(d),
                "stats": {
                    "mentions": stats.get("sents_total", 0),
                    "docs": stats.get("docs_total", 0),
                    "published": stats.get("docs_published", 0),
                },
                "sentences": filtered[:5],
            })

        return contexts

    def _extract_date_stats(self, collection):
        """Extract per-date statistics (lightweight version)."""
        default = lambda: {
            "sents_total": 0,
            "docs_total": 0,
            "docs_published": 0,
        }
        date_to_feats = collections.defaultdict(default)
        for a in collection.articles():
            pub_date = a.time.date()
            mentioned_dates = []
            for s in a.sentences:
                if s.time and s.time_level == "d":
                    d = s.time.date()
                    date_to_feats[d]["sents_total"] += 1
                    mentioned_dates.append(d)
            for d in sorted(set(mentioned_dates)):
                date_to_feats[d]["docs_total"] += 1
            date_to_feats[pub_date]["docs_published"] += 1
        return date_to_feats


class AgenticDatewiseTimelineGenerator(DatewiseTimelineGenerator):
    """DatewiseTimelineGenerator with agentic LLM-based date ranking **and**
    agentic LLM-based summarization (default).

    By default the summarizer is ``AgenticSummarizer`` which produces
    abstractive summaries via LLM.  Pass ``summarizer=CentroidOpt()`` to
    fall back to the legacy extractive summarizer.
    """

    def __init__(
        self,
        date_ranker=None,
        summarizer=None,
        sent_collector=None,
        clip_sents=5,
        pub_end=2,
        key_to_model=None,
        llm=None,
    ):
        _llm = llm or get_llm()
        if date_ranker is None:
            date_ranker = AgenticDateRanker(llm=_llm)
        if summarizer is None:
            summarizer = summarizers.AgenticSummarizer(llm=_llm)
        super().__init__(
            date_ranker=date_ranker,
            summarizer=summarizer,
            sent_collector=sent_collector or PM_Mean_SentenceCollector(
                clip_sents, pub_end
            ),
            key_to_model=key_to_model,
        )

    def predict(
        self,
        collection,
        max_dates=10,
        max_summary_sents=1,
        ref_tl=None,
        input_titles=False,
        output_titles=False,
        output_body_sents=True,
    ):
        print('vectorizer...')
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        vectorizer.fit([s.raw for a in collection.articles() for s in a.sentences])

        print('date ranking...')
        ranked_dates = self.date_ranker.rank_dates(collection, max_dates=max_dates)

        # --- Transfer topic context from ranker to summarizer ---
        if (hasattr(self.date_ranker, 'topic_description') and
                hasattr(self.summarizer, 'set_topic_context') and
                self.date_ranker.topic_description is not None):
            self.summarizer.set_topic_context(
                self.date_ranker.topic_description,
                self.date_ranker.topic_type,
                self.date_ranker.timeline_focus,
            )
            print('  [agentic] topic context transferred to summarizer')

        start = collection.start.date()
        end = collection.end.date()
        ranked_dates = [d for d in ranked_dates if start <= d <= end]

        print('candidates & summarization...')
        dates_with_sents = self.sent_collector.collect_sents(
            ranked_dates,
            collection,
            vectorizer,
            include_titles=input_titles,
        )

        def sent_filter(sent):
            lower = sent.raw.lower()
            if not any([kw in lower for kw in collection.keywords]):
                return False
            elif not output_titles and sent.is_title:
                return False
            elif not output_body_sents and not sent.is_sent:
                return False
            return True

        timeline = []
        l = 0
        for i, (d, d_sents) in enumerate(dates_with_sents):
            if l >= max_dates:
                break

            # Set the current date on the summarizer so it can include it
            # in the LLM prompt (AgenticSummarizer uses this; CentroidOpt
            # ignores it via duck typing).
            if hasattr(self.summarizer, 'current_date'):
                self.summarizer.current_date = d

            summary = self.summarizer.summarize(
                d_sents,
                k=max_summary_sents,
                vectorizer=vectorizer,
                filter=sent_filter,
            )
            if summary:
                time = datetime.datetime(d.year, d.month, d.day)
                timeline.append((time, summary))
                l += 1

        timeline.sort(key=lambda x: x[0])
        return data.Timeline(timeline)
