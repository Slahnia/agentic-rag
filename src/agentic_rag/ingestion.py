"""Document ingestion: load files, split into chunks and index them in Chroma."""

import logging
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings

logger = logging.getLogger(__name__)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TEXT_SUFFIXES = {".md", ".txt"}


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def load_documents(directory: str | None = None) -> list[Document]:
    """Load .md, .txt and .pdf files from a directory tree."""
    base = Path(directory or settings.documents_dir)
    documents: list[Document] = []
    for path in sorted(base.rglob("*")):
        if path.suffix.lower() in TEXT_SUFFIXES:
            documents.append(
                Document(
                    page_content=path.read_text(encoding="utf-8"),
                    metadata={"source": path.name},
                )
            )
        elif path.suffix.lower() == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader

            documents.extend(PyPDFLoader(str(path)).load())
    logger.info("Loaded %d documents from %s", len(documents), base)
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        persist_directory=settings.chroma_dir,
        embedding_function=get_embeddings(),
    )


def ingest(directory: str | None = None) -> int:
    """(Re)index every document found in the documents directory."""
    chunks = split_documents(load_documents(directory))
    store = get_vectorstore()
    store.reset_collection()
    if chunks:
        store.add_documents(chunks)
    logger.info("Indexed %d chunks into '%s'", len(chunks), settings.collection_name)
    return len(chunks)


def ensure_index() -> None:
    """Ingest the sample documents on first run so the demo works out of the box."""
    if get_vectorstore()._collection.count() == 0:
        ingest()


def get_retriever(k: int | None = None):
    return get_vectorstore().as_retriever(search_kwargs={"k": k or settings.retrieval_k})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    count = ingest()
    print(f"Done. {count} chunks indexed in {settings.chroma_dir}")


if __name__ == "__main__":
    main()
