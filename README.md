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
