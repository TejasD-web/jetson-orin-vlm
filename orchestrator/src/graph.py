"""
LangGraph orchestrator — agentic reasoning and routing.

Determines whether a query goes to edge (simple/fast) or cloud (complex),
and orchestrates multi-step reasoning chains via RAG.
"""

from __future__ import annotations


def classify_complexity(messages: list[dict]) -> str:
    """
    Classify query complexity to decide routing.
    Returns "edge" or "cloud".

    TODO P3: implement complexity scoring based on:
    - token count / context length
    - multimodal content presence
    - multi-step reasoning requirements
    - model capability requirements
    """
    return "edge"


def build_graph():
    """
    Build the LangGraph state graph.

    TODO P3: define nodes:
    - classify: complexity routing
    - edge_invoke: call Jetson endpoint
    - cloud_invoke: call Triton endpoint
    - rag_retrieve: document retrieval
    - synthesize: combine results
    """
    pass
