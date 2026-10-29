"""Tests for the LangGraph workflow."""

import json

from src import agents
from src.graph import build_graph, run_analysis
from benchmarks.fixtures import sales_frame


def test_parallel_branches_overlap_and_merge(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    agents.NODE_DELAY_SECONDS = 0.6
    state = run_analysis(sales_frame(), "How many rows are there?", force_deterministic=True)
    quality = state["quality_timing"]
    stats = state["stats_timing"]
    assert quality["start"] < stats["end"]
    assert stats["start"] < quality["end"]
    assert state["end_to_end_ms"] < 1100
    tool_names = {log["tool_name"] for log in state["tool_logs"]}
    assert "profile_dataset" in tool_names
    assert "calculate_statistics" in tool_names
    assert state["branch_status"]["quality"] == "success"
    assert state["branch_status"]["statistics"] == "success"
    assert state["quality_result"]["profile"]["row_count"] == 6
    assert "units" in state["stats_result"]["statistics"]["numeric"]
    assert state["insight_source"] == "deterministic"
    assert state["validation"]["passed"] is True
    assert state["includes_llm_network_time"] is False


def test_graph_has_parallel_join_and_conditional_route():
    compiled = build_graph()
    drawing = compiled.get_graph()
    node_names = set(drawing.nodes)
    assert {"quality", "statistics", "merge", "llm_insight", "deterministic_insight", "validate"} <= node_names


def test_missing_api_key_uses_deterministic_report(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    state = run_analysis(sales_frame(), "What is the mean of units?")
    assert state["insight_source"] == "deterministic"
    assert state["llm_usage"]["llm_calls"] == 0
    assert state["llm_usage"]["input_tokens"] is None
    report_text = json.dumps(state["report"])
    assert "units mean=26.666667" in report_text
    assert "GROQ_API_KEY is not configured" in report_text


def test_llm_failure_falls_back_without_leaking_the_key(monkeypatch):
    secret = "sk-test-secret-value"
    monkeypatch.setenv("GROQ_API_KEY", secret)

    def explode(_context):
        raise RuntimeError(f"request failed for key {secret}")

    monkeypatch.setattr("src.llm.generate_insight", explode)
    state = run_analysis(sales_frame(), "Summarize the table.")
    report_text = json.dumps(state["report"])
    assert state["insight_source"] == "deterministic_fallback"
    assert "row_count=6" in report_text
    assert secret not in report_text
    assert "[REDACTED]" in report_text
    assert state["validation"]["branch_disclosure_ok"] is True
