"""Command-line interface: ask one question or start an interactive session."""

import argparse
import logging

from .graph.build import build_graph
from .ingestion import ensure_index


def ask(graph, question: str) -> None:
    result = graph.invoke({"question": question})
    print(f"\n{result['generation']}\n")
    sources = {
        doc.metadata.get("source", "unknown") for doc in result.get("documents", [])
    }
    if sources:
        print(f"[route: {result.get('datasource', '?')} | sources: {', '.join(sorted(sources))}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-source agentic RAG")
    parser.add_argument("question", nargs="*", help="Question to ask (omit for interactive mode)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show agent reasoning steps")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="  %(message)s",
    )

    ensure_index()
    graph = build_graph()

    if args.question:
        ask(graph, " ".join(args.question))
        return

    print("Agentic RAG — interactive mode (Ctrl+C or 'exit' to quit)")
    while True:
        try:
            question = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not question or question.lower() in {"exit", "quit"}:
            break
        ask(graph, question)


if __name__ == "__main__":
    main()
