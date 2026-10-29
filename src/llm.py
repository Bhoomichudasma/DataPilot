"""Groq insight generation with a JSON fallback.

Token counts and cost come only from response metadata and configured prices.
"""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import BaseModel, Field

from src.config import groq_api_key, groq_model, groq_prices_per_million, redact_secrets


class FindingModel(BaseModel):
    claim: str = Field(description="One statement using only numbers present in the tool results.")
    evidence: str = Field(
        description="Tool name and metric, for example calculate_statistics units mean=26.666667."
    )
    interpretation: str = Field(description="What the evidence means. Do not claim causation.")
    recommendation: str = Field(description="A next step. This is advice, not a measured fact.")
    kind: str = Field(description="observation, hypothesis, or recommendation")


class InsightModel(BaseModel):
    answer: str = Field(description="Direct answer to the user question.")
    findings: list[FindingModel] = Field(description="Up to 5 findings.")
    limitations: list[str]
    cannot_conclude: list[str]
    data_quality_notes: list[str]


SYSTEM_PROMPT = """You are the insight agent for DataPilot AI.
Use only the tool results in the user message. Do not invent metrics, correlations, rows, or trends.
If a number is not in the tool results, do not use it.
Separate three kinds of statement:
- observation: a value the tools calculated
- hypothesis: a possible explanation that the data does not prove
- recommendation: an action a person could take
Correlation does not prove causation. Never say that one column causes another.
Mention data-quality limitations from the quality results.
Say what cannot be concluded from this dataset.
If an analysis branch failed, say so plainly.
Keep the answer concise.
"""


def empty_usage() -> dict[str, Any]:
    return {
        "llm_calls": 0,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "estimated_cost_usd": None,
        "cost_note": "N/A",
    }


def estimate_cost(usage: dict[str, Any]) -> dict[str, Any]:
    input_price, output_price = groq_prices_per_million()
    in_tokens = usage.get("input_tokens")
    out_tokens = usage.get("output_tokens")
    if in_tokens is None or out_tokens is None or input_price is None or output_price is None:
        usage["estimated_cost_usd"] = None
        usage["cost_note"] = "N/A"
        return usage
    cost = (in_tokens / 1_000_000) * input_price + (out_tokens / 1_000_000) * output_price
    usage["estimated_cost_usd"] = round(cost, 8)
    usage["cost_note"] = "Estimated from configured prices and reported token counts."
    return usage


def _usage_from_message(message: Any, calls: int) -> dict[str, Any]:
    metadata = getattr(message, "usage_metadata", None) or {}
    usage = empty_usage()
    usage["llm_calls"] = calls
    if metadata:
        usage["input_tokens"] = metadata.get("input_tokens")
        usage["output_tokens"] = metadata.get("output_tokens")
        usage["total_tokens"] = metadata.get("total_tokens")
    return estimate_cost(usage)


def _context_text(context: dict[str, Any]) -> str:
    text = json.dumps(context, default=str)
    if len(text) > 14000:
        text = text[:14000] + "...[truncated]"
    return text


def generate_insight(context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], float]:
    """Return the report, usage metadata, and LLM latency in milliseconds."""
    key = groq_api_key()
    if not key:
        raise RuntimeError("GROQ_API_KEY is not configured.")
    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:
        raise RuntimeError("The langchain-groq package is not installed.") from exc

    llm = ChatGroq(
        model=groq_model(),
        temperature=0,
        api_key=key,
        max_retries=1,
        timeout=60,
        max_tokens=1400,
    )
    messages = [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Question:\n"
            + str(context.get("question") or "Provide an overall summary.")
            + "\n\nTool results:\n"
            + _context_text(context.get("tool_results") or {}),
        ),
    ]
    started = time.perf_counter()
    calls = 1
    try:
        structured = llm.with_structured_output(InsightModel, method="function_calling", include_raw=True)
        result = structured.invoke(messages)
        parsed = result["parsed"]
        if parsed is None:
            raise ValueError(result.get("parsing_error") or "Structured output was empty.")
        report = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
        usage = _usage_from_message(result.get("raw"), calls)
    except Exception:
        calls = 2
        fallback = llm.invoke(
            messages
            + [
                (
                    "human",
                    "Return one JSON object with keys answer, findings, limitations, "
                    "cannot_conclude, and data_quality_notes. findings items need claim, "
                    "evidence, interpretation, recommendation, and kind. No markdown.",
                )
            ]
        )
        report = _parse_fallback(getattr(fallback, "content", ""))
        usage = _usage_from_message(fallback, calls)
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    report["source"] = "groq"
    report["model"] = groq_model()
    return _normalize_report(report), usage, latency_ms


def _parse_fallback(content: Any) -> dict[str, Any]:
    text = content if isinstance(content, str) else json.dumps(content)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("The model did not return JSON.")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise RuntimeError("The model returned JSON that could not be parsed.") from exc
    return InsightModel.model_validate(payload).model_dump()


def _normalize_report(report: dict[str, Any]) -> dict[str, Any]:
    findings = []
    for item in (report.get("findings") or [])[:5]:
        findings.append(
            {
                "claim": str(item.get("claim", "")).strip(),
                "evidence": str(item.get("evidence", "")).strip(),
                "interpretation": str(item.get("interpretation", "")).strip(),
                "recommendation": str(item.get("recommendation", "")).strip(),
                "kind": str(item.get("kind", "observation")).strip() or "observation",
            }
        )
    report["findings"] = findings
    for key in ("limitations", "cannot_conclude", "data_quality_notes"):
        report[key] = [str(item) for item in (report.get(key) or [])]
    report["answer"] = str(report.get("answer", "")).strip()
    return report


def safe_llm_error(exc: Exception) -> str:
    return redact_secrets(str(exc)) or "The language model request failed."
