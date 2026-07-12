"""Central configuration, overridable through environment variables / .env."""

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is a soft dependency
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path(env_var: str, default: str) -> str:
    """Resolve a path setting relative to the project root."""
    raw = os.getenv(env_var, default)
    path = Path(raw)
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


@dataclass(frozen=True)
class Settings:
    # Models
    llm_model: str = os.getenv("LLM_MODEL", "qwen2.5:3b")
    eval_model: str = os.getenv("EVAL_MODEL", os.getenv("LLM_MODEL", "qwen2.5:3b"))
    # Text-to-SQL is the hardest task in the pipeline for a small model;
    # point this to a larger one (e.g. qwen2.5:7b) if your RAM allows.
    sql_model: str = os.getenv("SQL_MODEL", os.getenv("LLM_MODEL", "qwen2.5:3b"))
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0"))
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    # Retrieval
    chroma_dir: str = field(default_factory=lambda: _path("CHROMA_DIR", ".chroma"))
    collection_name: str = os.getenv("CHROMA_COLLECTION", "knowledge_base")
    documents_dir: str = field(
        default_factory=lambda: _path("DOCUMENTS_DIR", "data/documents")
    )
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "4"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "2"))

    # SQL source
    sqlite_path: str = field(default_factory=lambda: _path("SQLITE_PATH", "data/sample.db"))
    tables_dir: str = field(default_factory=lambda: _path("TABLES_DIR", "data/tables"))

    # Source descriptions used by the router prompt
    kb_description: str = os.getenv(
        "KB_DESCRIPTION",
        "Documentation about Retrieval-Augmented Generation (RAG), agentic RAG "
        "patterns with LangGraph, and LLM evaluation with RAGAS.",
    )
    sql_description: str = os.getenv(
        "SQL_DESCRIPTION",
        "A SQLite database of a fictional online store with three tables: "
        "products (name, category, price), sales (product, quantity, date, "
        "country) and product_reviews (product, rating, country, review_date, "
        "comment).",
    )


settings = Settings()
