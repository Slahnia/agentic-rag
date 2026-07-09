"""Assemble the agentic RAG graph.

                 START
                   |
                 route
          /        |        \\
     retrieve   web_search   query_sql
         |          \\          /
   grade_documents   \\        /
    /    |     \\      \\      /
generate rewrite web    generate
    |       \\___________/  |
grade_generation <----------
    |        \\
   END     rewrite_query -> (retrieve | web_search)
"""

from langgraph.graph import END, START, StateGraph

from . import nodes
from .state import GraphState


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("route", nodes.route)
    workflow.add_node("retrieve", nodes.retrieve)
    workflow.add_node("grade_documents", nodes.grade_documents)
    workflow.add_node("rewrite_query", nodes.rewrite_query)
    workflow.add_node("web_search", nodes.web_search)
    workflow.add_node("query_sql", nodes.query_sql)
    workflow.add_node("generate", nodes.generate)

    workflow.add_edge(START, "route")
    workflow.add_conditional_edges(
        "route",
        nodes.select_datasource,
        {"vectorstore": "retrieve", "web_search": "web_search", "sql": "query_sql"},
    )
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        nodes.decide_after_grading,
        {"generate": "generate", "rewrite": "rewrite_query", "web_search": "web_search"},
    )
    workflow.add_conditional_edges(
        "rewrite_query",
        nodes.route_after_rewrite,
        {"retrieve": "retrieve", "web_search": "web_search"},
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("query_sql", "generate")
    workflow.add_conditional_edges(
        "generate",
        nodes.grade_generation,
        {"useful": END, "rewrite": "rewrite_query"},
    )

    return workflow.compile()
