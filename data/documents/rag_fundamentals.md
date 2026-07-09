# RAG Fundamentals

Retrieval-Augmented Generation (RAG) is a technique that grounds the answers
of a large language model in external knowledge. Instead of relying only on
what the model memorised during training, a RAG system first retrieves
relevant documents from a knowledge base and then asks the model to answer
using that evidence.

## The basic pipeline

A classic RAG pipeline has three stages:

1. **Ingestion**: documents are split into chunks, converted into vector
   embeddings with an embedding model, and stored in a vector database such
   as Chroma or Qdrant.
2. **Retrieval**: at question time, the user query is embedded with the same
   model and the most similar chunks are fetched (typically top-k similarity
   search).
3. **Generation**: the retrieved chunks are injected into the prompt as
   context and the LLM writes an answer grounded in them.

## Chunking

Chunking strategy has a large impact on retrieval quality. Chunks that are
too small lose context; chunks that are too large dilute the relevant signal
and waste the model's context window. A common starting point is 500–1000
characters with an overlap of 10–20% so that sentences cut at a boundary
still appear complete in one of the chunks. Recursive character splitting,
which tries to break on paragraphs first, then sentences, then words, is the
default strategy in most frameworks.

## Why RAG instead of fine-tuning

RAG is usually preferred over fine-tuning when the knowledge changes
frequently, when sources must be citable, or when data cannot be baked into
model weights for privacy reasons. Updating a RAG system means re-indexing
documents, which takes minutes, while fine-tuning requires training runs and
careful evaluation. The two techniques are complementary: fine-tuning shapes
behaviour and style, RAG supplies fresh knowledge.

## Known failure modes

The most common RAG failure modes are: retrieval misses (the relevant chunk
is not in the top-k), context dilution (relevant and irrelevant chunks mixed
together), and hallucination (the model answers from its parametric memory,
ignoring or contradicting the retrieved evidence). Advanced RAG architectures
add explicit safeguards against each of these failure modes.
