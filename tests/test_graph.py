"""The graph must compile without a running Ollama server or existing index."""

from agentic_rag.graph.build import build_graph

EXPECTED_NODES = {
    "route",
    "retrieve",
    "grade_documents",
    "rewrite_query",
    "web_search",
    "query_sql",
    "generate",
}


def test_graph_compiles_with_expected_nodes():
    graph = build_graph()
    nodes = set(graph.get_graph().nodes)
    assert EXPECTED_NODES <= nodes
