"""LLM chains used by the graph: router, graders, rewriter and generator.

Every chain is built lazily (lru_cache) so importing this module never
requires a running Ollama server — important for tests and tooling.
"""

from functools import lru_cache
from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from ..config import settings


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
    )


# --------------------------------------------------------------------------
# Router: pick the best data source for the question
# --------------------------------------------------------------------------
class RouteQuery(BaseModel):
    """Route a user question to the most appropriate data source."""

    datasource: Literal["vectorstore", "web_search", "sql"] = Field(
        description="The data source best suited to answer the question."
    )


ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at routing a user question to one data source:\n"
            '- "vectorstore": {kb_description}\n'
            '- "sql": {sql_description}\n'
            '- "web_search": current events, recent facts, or anything not '
            "covered by the other two sources.\n"
            "Pick the single best datasource for the question.",
        ),
        ("human", "{question}"),
    ]
).partial(kb_description=settings.kb_description, sql_description=settings.sql_description)


@lru_cache(maxsize=1)
def get_router():
    return ROUTER_PROMPT | get_llm().with_structured_output(RouteQuery)


# --------------------------------------------------------------------------
# Document grader: is this retrieved chunk relevant to the question?
# --------------------------------------------------------------------------
class GradeDocument(BaseModel):
    """Binary relevance score for a retrieved document."""

    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the document is relevant to the question."
    )


DOC_GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader assessing the relevance of a retrieved document "
            "to a user question. The document does not need to fully answer "
            "the question: grade 'yes' if it contains keywords or semantic "
            "meaning related to the question, otherwise 'no'.",
        ),
        ("human", "Document:\n{document}\n\nQuestion: {question}"),
    ]
)


@lru_cache(maxsize=1)
def get_document_grader():
    return DOC_GRADER_PROMPT | get_llm().with_structured_output(GradeDocument)


# --------------------------------------------------------------------------
# Hallucination grader: is the answer grounded in the evidence?
# --------------------------------------------------------------------------
class GradeGrounding(BaseModel):
    """Binary score: is the answer grounded in the provided facts?"""

    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the answer is supported by the facts."
    )


GROUNDING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader assessing whether an answer is grounded in the "
            "provided facts. Grade 'yes' if every claim in the answer is "
            "supported by the facts, 'no' if the answer invents information.",
        ),
        ("human", "Facts:\n{documents}\n\nAnswer: {generation}"),
    ]
)


@lru_cache(maxsize=1)
def get_grounding_grader():
    return GROUNDING_PROMPT | get_llm().with_structured_output(GradeGrounding)


# --------------------------------------------------------------------------
# Question rewriter: improve the query when retrieval fails
# --------------------------------------------------------------------------
REWRITER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite questions into better search queries. The previous "
            "query did not retrieve relevant results. Reformulate it: expand "
            "acronyms, add synonyms or make the intent explicit. Return ONLY "
            "the rewritten query, in the same language as the original.",
        ),
        ("human", "Original question: {original_question}\nPrevious query: {question}"),
    ]
)


@lru_cache(maxsize=1)
def get_rewriter():
    return REWRITER_PROMPT | get_llm() | StrOutputParser()


# --------------------------------------------------------------------------
# Generator: answer strictly from the gathered evidence
# --------------------------------------------------------------------------
GENERATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant for question answering. Use ONLY the context "
            "below to answer. If the context does not contain the answer, say "
            "you don't know — never invent information. Be concise, mention "
            "the source names when useful, and answer in the same language as "
            "the question.\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


@lru_cache(maxsize=1)
def get_generator():
    return GENERATOR_PROMPT | get_llm() | StrOutputParser()
