# News Timeline Summarization
Data & code for the ACL 2020 paper Examining the State-of-the-Art in News Timeline Summarization ([paper](https://www.aclweb.org/anthology/2020.acl-main.122.pdf),  [slides](acl20-slides.pdf)).

### Updates
Available
* all datasets
* methods & evaluation code
* preprocessing instructions for new datasets

Planned
* instructions to train date ranking models
* more user-friendly fast TLS version to run on unpreprocessed data

### Datasets

All datasets used in our experiments are [available here](https://drive.google.com/drive/folders/1gDAF5QZyCWnF_hYKbxIzOyjT6MSkbQXu?usp=sharing), including:
* T17
* Crisis
* Entities
### Library installation
The `news-tls` library contains tools for loading TLS datasets and running TLS methods.
To install, run:
```
pip install -r requirements.txt
pip install -e .
```
[Tilse](https://github.com/smartschat/tilse) also needs to be installed for evaluation and some TLS-specific data classes.

### Loading a dataset
Check out [news_tls/explore_dataset.py](news_tls/explore_dataset.py) to see how to load the provided datasets.

### Running methods & evaluation
Check out [experiments here](experiments).

#### Original clustering method
```bash
python experiments/evaluate.py \
    --dataset $DATASETS/t17 \
    --method clust \
    --output $RESULTS/t17.clust.json
```

#### Agentic clustering method (LangGraph + Groq)

The `agentic_clust` method replaces cluster ranking and summarisation with
a multi-agent LLM pipeline built on [LangGraph](https://github.com/langchain-ai/langgraph).
The pipeline runs five stages:

1. **Expand Query** -- enriches the collection keywords into a structured
   topic description (topic type, focus areas) so that downstream agents
   produce context-aware judgements.
2. **Generate** -- produces a candidate summary for each pre-filtered
   cluster (async, parallel).
3. **Judge** -- scores every candidate on relevance, importance, and quality
   relative to the topic type (async, parallel).
4. **Verify** -- checks the assembled timeline for redundancy and coherence.
5. **Finalise** -- deduplicates by date and enforces `max_dates`.

```bash
# 1. Set your Groq API key
export GROQ_API_KEY="gsk_..."

# 2. Run the agentic method
python experiments/evaluate.py \
    --dataset $DATASETS/t17 \
    --method agentic_clust \
    --output $RESULTS/t17.agentic_clust.json
```

By default the pipeline uses the **Groq** provider with model `gpt-oss-20b`.
To change the model or provider, pass a custom LLM when constructing
`AgenticClusteringTimelineGenerator` (see `news_tls/agentic/llm.py`).

##### Architecture overview

```
news_tls/agentic/
  state.py    -- LangGraph TypedDict state
  llm.py      -- LLM provider setup (Groq via OpenAI-compat API)
  prompts.py  -- Prompt templates for each agent
  nodes.py    -- Async node implementations
  graph.py    -- Graph builder (StateGraph wiring)

news_tls/agentic_clust.py
  AgenticClusteringTimelineGenerator  -- drop-in replacement for
      ClusteringTimelineGenerator with the same predict() interface
```

The design is extensible: each node is an independent async function,
new nodes can be inserted into the graph (e.g., a multi-agent debate step
for key-date extraction), and the `TimelineState` TypedDict can be extended
with additional fields without breaking existing nodes.

#### Agentic datewise method (LangGraph + Groq)

The `agentic_datewise` method enhances the datewise pipeline with LLM-powered
date ranking while keeping traditional extractive summarisation.  It uses only
**3 LLM calls** per topic (vs ~62 for `agentic_clust`), making it a lightweight
but effective hybrid approach.

```bash
# 1. Set your Groq API key
export GROQ_API_KEY="gsk_..."

# 2. Run the agentic datewise method
python experiments/evaluate.py \
    --dataset $DATASETS/entities \
    --method agentic_datewise \
    --output $RESULTS/entities.agentic_datewise.json

# Optional: use supervised date ranker as pre-filter (needs trained models)
python experiments/evaluate.py \
    --dataset $DATASETS/entities \
    --method agentic_datewise \
    --resources $RESOURCES \
    --output $RESULTS/entities.agentic_datewise.json
```

##### Pipeline overview

```
                    ┌─────────────────────────────────────────────┐
   News Articles    │          agentic_datewise pipeline          │
   + Keywords       │                                             │
        │           │  ┌───────────────────────────────────────┐  │
        ▼           │  │     Pre-Filter (no LLM)               │  │
  ┌───────────┐     │  │                                       │  │
  │ TF-IDF    │────▶│  │  MentionCountDateRanker               │  │
  │ Vectorizer│     │  │  → top 3×N candidate dates (~30)      │  │
  └───────────┘     │  │  → build date cards (stats + sents)   │  │
                    │  └──────────────┬────────────────────────┘  │
                    │                 │                            │
                    │  ┌──────────────▼────────────────────────┐  │
                    │  │     LangGraph (3 LLM calls)           │  │
                    │  │                                       │  │
                    │  │  ┌─────────────┐  LLM call #1         │  │
                    │  │  │expand_query │  keywords → topic     │  │
                    │  │  │             │  description, type,   │  │
                    │  │  │             │  focus areas           │  │
                    │  │  └──────┬──────┘                       │  │
                    │  │         ▼                               │  │
                    │  │  ┌─────────────┐  LLM call #2         │  │
                    │  │  │rank_dates   │  ALL ~30 date cards   │  │
                    │  │  │  _batch     │  in ONE call → ranked │  │
                    │  │  │             │  list + importance     │  │
                    │  │  └──────┬──────┘                       │  │
                    │  │         ▼                               │  │
                    │  │  ┌─────────────┐  LLM call #3         │  │
                    │  │  │verify_dates │  Self-reflection:     │  │
                    │  │  │             │  missing events?       │  │
                    │  │  │             │  temporal gaps?         │  │
                    │  │  │             │  narrative complete?    │  │
                    │  │  └──────┬──────┘                       │  │
                    │  │         ▼                               │  │
                    │  │  ┌─────────────┐  deterministic        │  │
                    │  │  │select_dates │  0.7×LLM + 0.3×heur  │  │
                    │  │  │             │  + adjacency penalty   │  │
                    │  │  │             │  → top N dates         │  │
                    │  │  └─────────────┘                       │  │
                    │  └──────────────┬────────────────────────┘  │
                    │                 │                            │
                    │  ┌──────────────▼────────────────────────┐  │
                    │  │     Downstream (no LLM, unchanged)    │  │
                    │  │                                       │  │
                    │  │  PM_Mean_SentenceCollector             │  │
                    │  │  → CentroidOpt extractive summariser  │  │
                    │  │  → Timeline                           │  │
                    │  └───────────────────────────────────────┘  │
                    └─────────────────────────────────────────────┘
```

##### Method comparison

```
┌─────────────────────┬────────────┬───────────────────────┬──────────────────┐
│                     │  datewise  │  agentic_datewise     │  agentic_clust   │
├─────────────────────┼────────────┼───────────────────────┼──────────────────┤
│ LLM Calls           │     0      │         3             │      ~62         │
│ Date Ranking        │ Statistical│ LLM batch + verify    │ LLM per-cluster  │
│ Summarisation       │ Extractive │ Extractive            │ LLM abstractive  │
│ Self-Reflection     │     No     │        Yes            │      Yes         │
│ Topic Awareness     │     No     │        Yes            │      Yes         │
│ Temporal Diversity  │     No     │ Adjacency penalty     │ Adjacency penalty│
└─────────────────────┴────────────┴───────────────────────┴──────────────────┘
```

##### Architecture overview

```
news_tls/agentic/
  datewise_state.py  -- DatewiseState TypedDict
  datewise_nodes.py  -- rank_dates_batch, verify_dates, select_dates
  datewise_graph.py  -- 4-node LangGraph builder
  prompts.py         -- RANK_DATES_PROMPT, VERIFY_DATES_PROMPT (added)

news_tls/agentic_datewise.py
  AgenticDateRanker                    -- LLM-based date ranker (plugs into
      DatewiseTimelineGenerator)
  AgenticDatewiseTimelineGenerator     -- convenience subclass
```

### Format & preprocess your own dataset
If you have a new dataset yourself and want to use preprocess it as the datasets above, check out the [preprocessing steps here](preprocessing).

### Citation
```
@inproceedings{gholipour-ghalandari-ifrim-2020-examining,
    title = "Examining the State-of-the-Art in News Timeline Summarization",
    author = "Gholipour Ghalandari, Demian  and
      Ifrim, Georgiana",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.122",
    pages = "1322--1334",
}
```
