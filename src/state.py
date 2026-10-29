"""Typed state shared by the LangGraph workflow."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AnalysisState(TypedDict, total=False):
    """Graph state.

    Parallel branches write their own result keys. Shared lists use reducers
    so LangGraph can merge both branches instead of rejecting the update.
    """

    dataset_id: str
    question: str
    force_deterministic: bool
    dataset_summary: dict[str, Any]
    quality_result: dict[str, Any]
    stats_result: dict[str, Any]
    quality_timing: dict[str, float]
    stats_timing: dict[str, float]
    branch_status: dict[str, Any]
    report: dict[str, Any]
    validation: dict[str, Any]
    tool_logs: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]
    llm_usage: dict[str, Any]
    llm_latency_ms: float | None
    insight_source: str
