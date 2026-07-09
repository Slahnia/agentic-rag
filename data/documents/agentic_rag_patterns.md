# Agentic RAG Patterns

Agentic RAG replaces the fixed retrieve-then-generate pipeline with an agent
that makes decisions at each step: where to look for information, whether the
retrieved evidence is good enough, and whether its own answer is trustworthy.
Frameworks like LangGraph model this as a state machine where nodes are
actions and edges are decisions.

## Adaptive routing

A router examines the incoming question and sends it to the most appropriate
data source: a vector store for domain documentation, a SQL database for
structured or numerical questions, or web search for recent events. Routing
is usually implemented with a small LLM call that returns a structured output
constrained to the available sources. Good routing avoids wasting retrieval
budget on sources that cannot contain the answer.

## Retrieval grading

After retrieval, a grader LLM inspects each retrieved chunk and scores its
relevance to the question with a binary yes/no judgment. Irrelevant chunks
are discarded before generation, which reduces context dilution and
hallucinations. If no chunk survives grading, the system knows retrieval
failed and can react instead of generating a poor answer.

## Query rewriting and self-correction

When retrieval fails, an agentic system rewrites the query — expanding
acronyms, adding synonyms, making the intent explicit — and retries. This is
called query rewriting or query transformation. Systems typically bound the
number of rewrite attempts (for example, two retries) and then fall back to
an alternative source such as web search, so the loop always terminates.

## Hallucination checking

After generation, a grader compares the answer against the retrieved
evidence and verifies every claim is supported. If the answer is not
grounded, the system can regenerate, rewrite the query for better evidence,
or return an honest "I don't know". This post-generation check is what
separates self-correcting RAG from a standard pipeline: the system evaluates
its own output before showing it to the user.

## Cost trade-offs

Every grading step is an extra LLM call, which adds latency — especially on
CPU inference. Practical systems choose which checks matter most: document
grading and hallucination checking give the most value per call, while
separate answer-usefulness grading is often skipped in latency-sensitive
deployments.
