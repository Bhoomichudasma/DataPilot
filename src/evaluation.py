"""Evaluation metrics for DataPilot AI.

Tool latency uses only Python tool durations. End-to-end latency is reported
separately and may include Groq network time.
"""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.benchmark_cases import BENCHMARK_CASES, EVIDENCE_REVIEWS
from benchmarks.fixtures import (
    constant_frame,
    empty_frame,
    invalid_frame,
    outlier_frame,
    sales_frame,
)
from src.data_tools import (
    analyze_group_by_frame,
    analyze_trend_frame,
    calculate_statistics_frame,
    find_correlations_frame,
    profile_dataset_frame,
)
from src.report_validation import _causal_flag, numbers_match

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

NUMERICAL_SCOPE = (
    "Twenty assertions on fixed in-memory fixtures: sales shape, missing revenue, "
    "duplicate rows, units and revenue mean/median/std, Pearson correlation, "
    "region and category group means, the units trend direction, a constant column, "
    "one IQR outlier, one invalid numeric token, and an empty table. "
    "The denominator is the number of these assertions. Failed assertions are not excluded."
)


def tool_success_rate(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    successful = sum(1 for record in records if record.get("success"))
    if total == 0:
        return {"successful": 0, "total": 0, "rate_percent": None, "display": "N/A"}
    rate = 100 * successful / total
    return {
        "successful": successful,
        "total": total,
        "rate_percent": round(rate, 3),
        "display": f"{rate:.1f}%",
    }


def _percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (percent / 100)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def tool_latency_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [float(record["duration_ms"]) for record in records if "duration_ms" in record]
    if not durations:
        return {
            "average_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "per_tool": [],
            "call_count": 0,
            "note": "N/A. No Python tool calls were recorded. LLM time is not included here.",
        }
    grouped: dict[str, list[float]] = {}
    for record in records:
        grouped.setdefault(record["tool_name"], []).append(float(record["duration_ms"]))
    per_tool = []
    for name, values in sorted(grouped.items()):
        per_tool.append(
            {
                "tool_name": name,
                "calls": len(values),
                "average_ms": round(statistics.fmean(values), 3),
                "median_ms": round(statistics.median(values), 3),
            }
        )
    return {
        "average_ms": round(statistics.fmean(durations), 3),
        "median_ms": round(statistics.median(durations), 3),
        "p95_ms": round(_percentile(durations, 95), 3),
        "per_tool": per_tool,
        "call_count": len(durations),
        "note": "Python tool execution only. Time spent waiting for an LLM response is excluded.",
    }


def end_to_end_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [float(run["duration_ms"]) for run in runs]
    includes_llm = any(run.get("includes_llm_network_time") for run in runs)
    if not durations:
        return {
            "average_ms": None,
            "median_ms": None,
            "run_count": 0,
            "includes_llm_network_time": False,
            "note": "N/A. No workflow runs were measured.",
        }
    note = (
        "End-to-end time includes Groq network time."
        if includes_llm
        else "End-to-end time does not include Groq network time."
    )
    note += " A single run or a small fixed suite is not a stable performance benchmark."
    return {
        "average_ms": round(statistics.fmean(durations), 3),
        "median_ms": round(statistics.median(durations), 3),
        "run_count": len(durations),
        "includes_llm_network_time": includes_llm,
        "note": note,
    }


def _check(checks: list[dict[str, Any]], name: str, actual: Any, expected: Any) -> None:
    if isinstance(expected, float):
        passed = actual is not None and abs(float(actual) - expected) <= 1e-6
    else:
        passed = actual == expected
    checks.append({"name": name, "passed": passed, "expected": expected, "actual": actual})


def run_numerical_checks() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    sales = sales_frame()
    profile = profile_dataset_frame(sales)
    stats = calculate_statistics_frame(sales)
    correlations = find_correlations_frame(sales)
    by_region = analyze_group_by_frame(sales, question="average revenue by region")
    by_category = analyze_group_by_frame(sales, question="average units by category")
    trend = analyze_trend_frame(sales)
    revenue_column = next(column for column in profile["columns"] if column["name"] == "revenue")
    region_means = {
        row["group"]: row["metrics"]["revenue"]["mean"]
        for row in by_region["groups"]
    }
    category_means = {
        row["group"]: row["metrics"]["units"]["mean"]
        for row in by_category["groups"]
    }
    units_direction = next(item["direction"] for item in trend["directions"] if item["column"] == "units")
    pair = correlations["pairs"][0]

    _check(checks, "sales_row_count", profile["row_count"], 6)
    _check(checks, "sales_column_count", profile["column_count"], 5)
    _check(checks, "revenue_missing_count", revenue_column["missing_count"], 1)
    _check(checks, "duplicate_row_count", profile["duplicate_row_count"], 1)
    _check(checks, "units_mean", stats["numeric"]["units"]["mean"], 26.666667)
    _check(checks, "units_median", stats["numeric"]["units"]["median"], 25)
    _check(checks, "units_std", stats["numeric"]["units"]["std"], 16.329932)
    _check(checks, "revenue_mean", stats["numeric"]["revenue"]["mean"], 360)
    _check(checks, "revenue_median", stats["numeric"]["revenue"]["median"], 300)
    _check(checks, "units_revenue_correlation", pair["correlation"], 0.97049)
    _check(
        checks,
        "correlation_pair_columns",
        sorted([pair["left"], pair["right"]]),
        ["revenue", "units"],
    )
    _check(checks, "east_revenue_mean", region_means.get("East"), 650)
    _check(checks, "widget_units_mean", category_means.get("Widget"), 20)
    _check(checks, "gadget_units_mean", category_means.get("Gadget"), 33.333333)
    _check(checks, "units_trend_direction", units_direction, "increasing")

    constant = profile_dataset_frame(constant_frame())
    _check(checks, "status_is_constant", "status" in constant["constant_columns"], True)

    outliers = profile_dataset_frame(outlier_frame())["outliers"]
    outlier_count = outliers[0]["outlier_count"] if outliers else None
    _check(checks, "iqr_outlier_count", outlier_count, 1)

    invalid = profile_dataset_frame(invalid_frame())["invalid_types"]
    invalid_count = invalid[0]["invalid_count"] if invalid else None
    _check(checks, "invalid_numeric_count", invalid_count, 1)

    empty = profile_dataset_frame(empty_frame())
    _check(checks, "empty_row_count", empty["row_count"], 0)
    _check(checks, "empty_column_count", empty["column_count"], 2)

    passed = sum(1 for check in checks if check["passed"])
    total = len(checks)
    return {
        "passed": passed,
        "total": total,
        "rate_percent": round(100 * passed / total, 3) if total else None,
        "checks": checks,
        "scope": NUMERICAL_SCOPE,
        "denominator": "all assertions in the numerical fixture suite",
    }


def _lookup_metric(bundle: dict[str, Any], binding: dict[str, Any]) -> Any:
    source = binding.get("source")
    if source == "profile":
        value: Any = bundle["profile"]
        for key in binding.get("path") or []:
            value = value[key]
        return value
    if source == "statistic":
        return bundle["statistics"]["numeric"][binding["column"]][binding["stat"]]
    if source == "correlation":
        wanted = {binding["left"], binding["right"]}
        for pair in bundle["correlations"].get("pairs") or []:
            if {pair["left"], pair["right"]} == wanted:
                return pair["correlation"]
        return None
    return None


def score_evidence_reviews(
    reviews: list[dict[str, Any]] | None = None,
    bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score a fixed sample of claims. This is not a universal hallucination rate."""
    reviews = EVIDENCE_REVIEWS if reviews is None else reviews
    if bundle is None:
        frame = sales_frame()
        bundle = {
            "profile": profile_dataset_frame(frame),
            "statistics": calculate_statistics_frame(frame),
            "correlations": find_correlations_frame(frame),
        }
    scored = []
    for review in reviews:
        actual = _lookup_metric(bundle, review["binding"])
        stated = review["binding"].get("stated")
        cites_metric = review["binding"].get("source") != "missing" and actual is not None
        evidence_text = review.get("evidence") or ""
        cites_tool = any(
            token in evidence_text
            for token in (
                "profile_dataset",
                "calculate_statistics",
                "find_correlations",
                "analyze_group_by",
                "analyze_trend",
                "mean",
                "row_count",
                "duplicate_row_count",
                "correlation",
            )
        )
        supporting = bool(cites_metric and cites_tool)
        numerical = bool(cites_metric and stated is not None and numbers_match(float(stated), {float(actual)}))
        interpretation = review.get("interpretation") or ""
        interpretation_numbers_ok = True
        if actual is not None:
            from src.report_validation import extract_claim_numbers

            for number in extract_claim_numbers(interpretation):
                if not numbers_match(number, {float(actual)}):
                    interpretation_numbers_ok = False
        interpretation_follows = interpretation_numbers_ok and not _causal_flag(interpretation) and not _causal_flag(
            review.get("recommendation") or ""
        )
        passed = supporting and numerical and interpretation_follows
        scored.append(
            {
                "id": review["id"],
                "claim": review["claim"],
                "supporting_evidence": supporting,
                "numerical_correct": numerical,
                "interpretation_follows": interpretation_follows,
                "passed": passed,
            }
        )
    total = len(scored)
    passed_count = sum(1 for item in scored if item["passed"])
    return {
        "passed": passed_count,
        "total": total,
        "rate_percent": round(100 * passed_count / total, 3) if total else None,
        "reviews": scored,
        "denominator": "claims in the fixed reviewed sample",
        "note": (
            f"This score covers {total} hand-written claims about the sales fixture. "
            "It is a limited sample, not a universal hallucination rate. "
            "No LLM judge is used. Supporting evidence means the cited metric exists in the "
            "tool output and the evidence names a tool or metric. Numerical correctness compares "
            "the stated value with that metric. Interpretation follows when it adds no conflicting "
            "number and does not state a causal conclusion."
        ),
    }


def run_task_benchmark() -> dict[str, Any]:
    from src.graph import run_analysis

    case_results = []
    runs = []
    logs: list[dict[str, Any]] = []
    for case in BENCHMARK_CASES:
        try:
            state = run_analysis(case["loader"](), case["question"], force_deterministic=True)
            report_text = json.dumps(state.get("report") or {})
            tool_ok = all(
                any(log.get("tool_name") == name and log.get("success") for log in state.get("tool_logs") or [])
                for name in case["required_tools"]
            )
            missing = [token for token in case["required_evidence"] if token not in report_text]
            completed = tool_ok and not missing
            error = None
            logs.extend(state.get("tool_logs") or [])
            runs.append(
                {
                    "duration_ms": state.get("end_to_end_ms"),
                    "includes_llm_network_time": state.get("includes_llm_network_time"),
                }
            )
            validation_passed = bool((state.get("validation") or {}).get("passed"))
        except Exception as exc:
            completed = False
            tool_ok = False
            missing = list(case["required_evidence"])
            error = type(exc).__name__
            validation_passed = False
        case_results.append(
            {
                "id": case["id"],
                "fixture": case["fixture"],
                "question": case["question"],
                "acceptance": case["acceptance"],
                "completed": completed,
                "required_tools_succeeded": tool_ok,
                "missing_evidence": missing,
                "validation_passed": validation_passed,
                "error": error,
            }
        )
    total = len(BENCHMARK_CASES)
    passed = sum(1 for case in case_results if case["completed"])
    return {
        "passed": passed,
        "failed": total - passed,
        "total": total,
        "rate_percent": round(100 * passed / total, 3) if total else None,
        "cases": case_results,
        "tool_logs": logs,
        "runs": runs,
        "denominator": "all benchmark cases; failed and errored cases count as not completed",
        "method": (
            "Each case runs the LangGraph workflow with force_deterministic=True so the "
            "result does not depend on Groq. A case is completed only when every required "
            "tool invocation succeeds and the report contains every required evidence token."
        ),
    }


def run_benchmark(save: bool = True) -> dict[str, Any]:
    numerical = run_numerical_checks()
    tasks = run_task_benchmark()
    evidence = score_evidence_reviews()
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "numerical_correctness": {key: value for key, value in numerical.items() if key != "checks"},
        "numerical_failures": [check for check in numerical["checks"] if not check["passed"]],
        "task_completion": {key: value for key, value in tasks.items() if key not in {"tool_logs", "runs"}},
        "tool_success_rate": tool_success_rate(tasks["tool_logs"]),
        "tool_latency": tool_latency_summary(tasks["tool_logs"]),
        "end_to_end": end_to_end_summary(tasks["runs"]),
        "evidence_review": evidence,
        "llm_usage": {
            "llm_calls": 0,
            "input_tokens": None,
            "output_tokens": None,
            "estimated_cost_usd": None,
            "display": "N/A",
            "note": "The built-in benchmark does not call Groq, so token counts and cost are N/A.",
        },
    }
    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = RESULTS_DIR / f"benchmark_{stamp}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["saved_to"] = str(path)
    return payload


def summarize_run(state: dict[str, Any]) -> dict[str, Any]:
    usage = state.get("llm_usage") or {}
    tokens_known = usage.get("input_tokens") is not None or usage.get("output_tokens") is not None
    return {
        "tool_success_rate": tool_success_rate(state.get("tool_logs") or []),
        "tool_latency": tool_latency_summary(state.get("tool_logs") or []),
        "end_to_end": end_to_end_summary(
            [
                {
                    "duration_ms": state.get("end_to_end_ms") or 0,
                    "includes_llm_network_time": state.get("includes_llm_network_time"),
                }
            ]
        )
        if state.get("end_to_end_ms") is not None
        else end_to_end_summary([]),
        "llm_usage": {
            "llm_calls": usage.get("llm_calls"),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "estimated_cost_usd": usage.get("estimated_cost_usd"),
            "cost_note": usage.get("cost_note", "N/A"),
            "tokens_display": "reported" if tokens_known else "N/A",
        },
        "validation_passed": (state.get("validation") or {}).get("passed"),
    }
