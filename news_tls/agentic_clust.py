"""Agentic Clustering Timeline Generator.

Drop-in replacement for ``ClusteringTimelineGenerator`` that keeps the
Markov-clustering event-detection phase but delegates **cluster ranking**
and **summarisation** to a LangGraph multi-agent pipeline
(expand → generate → judge → verify → finalise).
"""

from __future__ import annotations

import asyncio
import datetime
import collections

from sklearn.feature_extraction.text import TfidfVectorizer

from news_tls import data, utils
from news_tls.clust import (
    TemporalMarkovClusterer,
    ClusterDateMentionCountRanker,
)
from news_tls.agentic.llm import get_llm
from news_tls.agentic.graph import build_timeline_graph


class AgenticClusteringTimelineGenerator:
    """Timeline generator: Markov clustering + LangGraph agentic pipeline.

    Parameters
    ----------
    clusterer : Clusterer, optional
        Article clusterer (default: ``TemporalMarkovClusterer``).
    cluster_ranker : ClusterRanker, optional
        Used only as a **pre-filter** to limit the number of clusters
        sent to the (more expensive) LLM pipeline.
    llm : ChatOpenAI, optional
        LLM instance; built from ``get_llm()`` when *None*.
    clip_sents : int
        Max sentences per article to consider.
    unique_dates : bool
        Enforce one summary per calendar date.
    candidate_multiplier : int
        ``max_dates * candidate_multiplier`` clusters are sent to the
        agentic pipeline.  Increase for better recall, decrease for speed.
    """

    def __init__(
        self,
        clusterer=None,
        cluster_ranker=None,
        llm=None,
        clip_sents=5,
        unique_dates=True,
        candidate_multiplier=3,
    ):
        self.clusterer = clusterer or TemporalMarkovClusterer()
        self.cluster_ranker = cluster_ranker or ClusterDateMentionCountRanker()
        self.clip_sents = clip_sents
        self.unique_dates = unique_dates
        self.candidate_multiplier = candidate_multiplier

        self.llm = llm or get_llm()
        self.graph = build_timeline_graph(self.llm)

    # ------------------------------------------------------------------
    # Public API (same signature as ClusteringTimelineGenerator)
    # ------------------------------------------------------------------

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
        # ---- Phase 1: Markov Clustering (unchanged) ----
        print("clustering articles...")
        doc_vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
        clusters = self.clusterer.cluster(collection, doc_vectorizer)

        print("assigning cluster times...")
        for c in clusters:
            c.time = c.most_mentioned_time()
            if c.time is None:
                c.time = c.earliest_pub_time()

        # ---- Phase 2: Pre-rank to limit LLM calls ----
        print("pre-ranking clusters...")
        ranked_clusters = self.cluster_ranker.rank(clusters, collection)
        top_n = max_dates * self.candidate_multiplier
        top_clusters = ranked_clusters[:top_n]

        # ---- Phase 3: Build cluster contexts ----
        print("preparing cluster contexts...")
        cluster_infos = []
        for idx, c in enumerate(top_clusters):
            sents = self._collect_cluster_sents(
                c, collection.keywords, output_titles, output_body_sents
            )
            if not sents:
                continue
            cluster_infos.append(
                {
                    "cluster_id": idx,
                    "date": str(c.time.date()),
                    "sentences": [s.raw for s in sents][:30],
                    "article_count": len(c.articles),
                }
            )

        # ---- Phase 4: Agentic pipeline ----
        print("running agentic pipeline...")
        initial_state = {
            "clusters": cluster_infos,
            "keywords": list(collection.keywords),
            "max_dates": max_dates,
            "max_summary_sents": max_summary_sents,
        }
        result = asyncio.run(self.graph.ainvoke(initial_state))

        # ---- Phase 5: Assemble Timeline ----
        print("building timeline...")
        timeline = []
        for entry in result.get("timeline", []):
            try:
                d = datetime.datetime.strptime(entry["date"], "%Y-%m-%d")
            except ValueError:
                continue
            timeline.append((d, entry["summary"]))

        timeline.sort(key=lambda x: x[0])
        return data.Timeline(timeline)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_cluster_sents(
        self, cluster, keywords, output_titles, output_body_sents
    ):
        """Return keyword-filtered sentences from a cluster."""
        sents = []
        for a in cluster.articles:
            for s in a.sentences[: self.clip_sents]:
                lower = s.raw.lower()
                if not any(kw in lower for kw in keywords):
                    continue
                if not output_titles and s.is_title:
                    continue
                if not output_body_sents and not s.is_title:
                    continue
                sents.append(s)
        return sents

    def load(self, ignored_topics):
        """No-op: clustering models don't need topic-specific loading."""
        pass
