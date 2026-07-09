# 🧭 Multi-source Agentic RAG

An **agentic RAG system** built with [LangGraph](https://github.com/langchain-ai/langgraph) that doesn't just retrieve-and-generate — it **decides where to look, judges its own evidence, and corrects itself** before answering.

**100% open source. No API keys. Runs fully local on CPU.**

- 🧠 LLM: any [Ollama](https://ollama.com) model (default: `qwen2.5:3b`, ~2 GB, CPU-friendly)
- 🔤 Embeddings: multilingual `sentence-transformers` (works in English and Spanish)
- 🗂️ Vector store: [Chroma](https://www.trychroma.com/) (embedded, zero config)
- 🌐 Web search: DuckDuckGo (no API key)
- 📊 Evaluation: [RAGAS](https://docs.ragas.io/) — faithfulness, answer relevancy, context precision & recall

## How it works

Each question flows through a LangGraph state machine that makes explicit decisions instead of following a fixed pipeline:

```mermaid
flowchart TD
    START([question]) --> route[🧭 route]
    route -->|domain docs| retrieve[📚 retrieve<br/>Chroma]
    route -->|recent facts| web[🌐 web_search<br/>DuckDuckGo]
    route -->|structured data| sql[🗄️ query_sql<br/>SQLite]
    retrieve --> grade[🔍 grade_documents]
    grade -->|relevant docs| generate[💬 generate]
    grade -->|nothing relevant,<br/>retries left| rewrite[✏️ rewrite_query]
    grade -->|retry budget<br/>exhausted| web
    rewrite --> retrieve
    web --> generate
    sql --> generate
    generate --> check{grounded in<br/>evidence?}
    check -->|yes| END([answer + sources])
    check -->|no, retries left| rewrite
    check -->|budget exhausted| END
```

1. **Adaptive routing** — a structured-output LLM call sends the question to the vector store, a SQL database, or web search.
2. **Retrieval grading** — every retrieved chunk is judged for relevance; irrelevant ones are discarded before generation.
3. **Query rewriting** — if nothing relevant survives, the query is reformulated and retried (bounded), then falls back to web search.
4. **Hallucination check** — the answer is verified against the evidence before being returned; ungrounded answers trigger a self-correction loop.
5. **Guaranteed termination** — every loop is bounded by a retry budget (`MAX_RETRIES`, default 2).

## Quickstart

Requirements: Python **3.10+** and [Ollama](https://ollama.com/download).

```bash
# 1. Pull the local model (~2 GB)
ollama pull qwen2.5:3b

# 2. Install
git clone https://github.com/Slahnia/agentic-rag.git
cd agentic-rag
python -m venv .venv && .venv\Scripts\activate    # Windows
pip install -e ".[ui,eval,dev]"

# 3. Create the sample SQL database and index the sample docs
python scripts/create_sample_db.py
agentic-rag-ingest

# 4. Chat!
streamlit run app.py        # web UI with live agent steps
# or
agentic-rag -v "What is retrieval grading?"        # CLI
```

Try questions that exercise each route:

| Question | Route taken |
|---|---|
| "What is retrieval grading and why is it useful?" | 📚 vectorstore |
| "Which product had the most sales in Spain?" | 🗄️ SQL |
| "What happened in the news today?" | 🌐 web search |

To use **your own documents**, drop `.md`/`.txt`/`.pdf` files into `data/documents/`, run `agentic-rag-ingest`, and update `KB_DESCRIPTION` in `.env` so the router knows what the knowledge base contains.

## Evaluation

The eval harness runs the full agent over [`evaluation/dataset.json`](evaluation/dataset.json) and scores it with four RAGAS metrics:

```bash
python evaluation/run_evaluation.py
```

| Metric | What it diagnoses |
|---|---|
| **Faithfulness** | Is every claim in the answer supported by the evidence? (hallucination) |
| **Response relevancy** | Does the answer actually address the question? |
| **Context precision** | Are the relevant chunks ranked at the top? (retriever ranking) |
| **Context recall** | Does the retrieved context contain everything needed? (ingestion/retrieval) |

Per-question scores are written to `evaluation/results.csv`. Reading the metrics **together** localises the failure: low recall + high faithfulness → retrieval is the bottleneck; high recall + low faithfulness → the generator ignores its evidence.

### The judge matters (measured, not assumed)

The judge defaults to the same 3B model the agent uses, and that turned out to be measurably unreliable: it scored **faithfulness 0.0** on an answer that was near-verbatim from the source document (full context retrieved, recall 1.0). Faithfulness requires decomposing an answer into claims and verifying each one — too hard for a 3B model. Answer relevancy, context precision and recall are simpler judgments and stayed consistent.

Fix: keep the small model for the agent, use a larger one only as judge (`EVAL_MODEL=qwen2.5:7b`). Evaluation is offline, so the extra latency doesn't matter.

Results on this repo's sample KB and [dataset](evaluation/dataset.json), agent = `qwen2.5:3b` on CPU:

| Metric | 3B judge | 7B judge |
|---|---|---|
| Faithfulness | 0.500 ⚠️ | **0.833** |
| Response relevancy | 0.893 | 0.870 |
| Context precision | 0.812 | 1.000 |
| Context recall | 0.783 | 0.746 |

Same agent, same answers — faithfulness jumps from 0.50 to 0.83 just by upgrading the judge, confirming the 3B score was judge noise rather than hallucination. The simpler embedding-based relevancy metric barely moves, as expected.

## Design decisions

- **Graders reason before they judge.** Forcing a 3B model to emit an immediate structured yes/no made it reject obviously relevant chunks (0/4 relevant with the answer verbatim in the KB). Letting it think briefly and parsing a trailing `VERDICT: yes|no` fixed grading with the same model — chain-of-thought is not optional at this size. Parse failures default to the *safe* verdict so grading noise can never discard all evidence or trap the graph in a loop.
- **Every self-correction loop is bounded.** Rewrite attempts are capped and the vectorstore path falls back to web search, so the graph always terminates — no runaway LLM-call loops.
- **CPU latency shaped the architecture.** Each grading step is an extra LLM call (seconds on CPU). The system keeps the two highest-value checks (document relevance, answer grounding) and skips a separate answer-usefulness grader.
- **SQL is read-only by construction.** Generated queries are validated to be single `SELECT`/`WITH` statements with no DDL/DML keywords before execution ([`tools/sql.py`](src/agentic_rag/tools/sql.py)).
- **Honest failure over confident nonsense.** If a source returns nothing, the generator is instructed to say "I don't know" rather than fill gaps from parametric memory — and the eval dataset includes unanswerable questions to verify it.
- **Testable control flow.** Routing decisions are pure functions over the graph state, unit-tested without any model running (`pytest tests/`).

## Project structure

```
agentic-rag/
├── app.py                      # Streamlit chat UI (live agent steps)
├── src/agentic_rag/
│   ├── config.py               # settings via env vars / .env
│   ├── ingestion.py            # load → chunk → embed → Chroma
│   ├── cli.py                  # terminal interface
│   ├── graph/
│   │   ├── state.py            # shared GraphState
│   │   ├── chains.py           # router, graders, rewriter, generator
│   │   ├── nodes.py            # node functions + routing decisions
│   │   └── build.py            # graph assembly
│   └── tools/
│       ├── web_search.py       # DuckDuckGo source
│       └── sql.py              # text-to-SQL source (read-only guard)
├── evaluation/
│   ├── dataset.json            # eval questions + ground truths
│   └── run_evaluation.py       # RAGAS harness
├── data/documents/             # knowledge base (sample docs included)
├── scripts/create_sample_db.py # sample SQLite database
└── tests/                      # control-flow & safety unit tests
```

## Docker

```bash
docker compose up -d
docker compose exec ollama ollama pull qwen2.5:3b   # first time only
# open http://localhost:8501
```

## Roadmap

- [ ] Hybrid search (BM25 + dense) with reranking
- [ ] Conversation memory (multi-turn) via LangGraph checkpoints
- [ ] Observability with self-hosted Langfuse
- [ ] Synthetic eval-set generation with RAGAS `TestsetGenerator`

## License

[MIT](LICENSE)
