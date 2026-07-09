# Evaluating RAG Systems with RAGAS

RAGAS (Retrieval-Augmented Generation Assessment) is an open-source framework
for evaluating RAG pipelines without requiring large hand-labelled datasets.
It uses an LLM as a judge to score different aspects of the system, so the
quality of the judge model matters: small models produce noisier scores than
large ones.

## Core metrics

- **Faithfulness** measures whether the generated answer is supported by the
  retrieved context. It decomposes the answer into individual claims and
  verifies each one against the context. Low faithfulness means the system
  hallucinates.
- **Answer relevancy** (response relevancy) measures whether the answer
  actually addresses the user's question, regardless of correctness. It is
  computed by generating synthetic questions from the answer and comparing
  their embeddings with the original question.
- **Context precision** measures whether the relevant chunks are ranked at
  the top of the retrieved results. It penalises retrievers that fetch the
  right document buried under irrelevant ones.
- **Context recall** measures whether the retrieved context contains all the
  information needed to produce the reference answer. Low recall points to
  ingestion or retrieval problems, not generation problems.

## Interpreting the metrics together

The metrics diagnose different components. Low context recall with high
faithfulness means retrieval is the bottleneck: the model honestly answers
from incomplete evidence. High recall with low faithfulness means retrieval
works but the generator ignores the evidence. This decomposition is the main
advantage of RAGAS over a single end-to-end accuracy number.

## Building an evaluation dataset

A RAGAS evaluation needs a set of questions with reference answers (ground
truth). A good starting dataset covers: questions answerable from a single
chunk, questions requiring information from multiple documents, and questions
that are NOT answerable from the knowledge base — to verify the system says
"I don't know" instead of hallucinating. Even 10–20 well-chosen questions
reveal most regressions when run consistently after each change.
