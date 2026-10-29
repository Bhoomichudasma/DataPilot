"""LangGraph workflow: parallel quality and statistics branches, then insight."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
from langgraph.graph import END, START, StateGraph

from src.agents import (
    deterministic_insight_node,
    llm_insight_node,
    merge_node,
    quality_node,
    route_after_merge,
    statistics_node,
)
from src.data_tools import build_dataset_summary, register_dataframe
from src.report_validation import validate_report
from src.state import AnalysisState


def validate_node(state: AnalysisState) -> dict[str, Any]:
    validation = validate_report(
        state.get("report") or {},
        state.get("quality_result") or {},
        state.get("stats_result") or {},
        state.get("branch_status") or {},
    )
    return {"validation": validation}


def build_graph():
    graph = StateGraph(AnalysisState)
    graph.add_node("quality", quality_node)
    graph.add_node("statistics", statistics_node)
    graph.add_node("merge", merge_node)
    graph.add_node("llm_insight", llm_insight_node)
    graph.add_node("deterministic_insight", deterministic_insight_node)
    graph.add_node("validate", validate_node)
    graph.add_edge(START, "quality")
    graph.add_edge(START, "statistics")
    graph.add_edge(["quality", "statistics"], "merge")
    graph.add_conditional_edges(
        "merge",
        route_after_merge,
        {
            "llm_insight": "llm_insight",
            "deterministic_insight": "deterministic_insight",
        },
    )
    graph.add_edge("llm_insight", "validate")
    graph.add_edge("deterministic_insight", "validate")
    graph.add_edge("validate", END)
    return graph.compile()


_COMPILED = None


def get_graph():
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = build_graph()
    return _COMPILED


def run_analysis(
    df: pd.DataFrame,
    question: str = "",
    *,
    force_deterministic: bool = False,
) -> dict[str, Any]:
    """Run the multi-agent graph. End-to-end time includes LLM network time when Groq is called."""
    dataset_id = register_dataframe(df)
    summary = build_dataset_summary(df)
    started = time.perf_counter()
    state = get_graph().invoke(
        {
            "dataset_id": dataset_id,
            "question": question or "",
            "force_deterministic": force_deterministic,
            "dataset_summary": summary,
            "tool_logs": [],
            "errors": [],
        }
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    state["dataset_summary"] = summary
    state["end_to_end_ms"] = elapsed_ms
    state["includes_llm_network_time"] = state.get("insight_source") == "groq"
    state["timing_note"] = (
        "This is one workflow measurement, not a stable benchmark. "
        + (
            "It includes time waiting for the Groq API."
            if state["includes_llm_network_time"]
            else "It does not include Groq network time."
        )
    )
    return state
