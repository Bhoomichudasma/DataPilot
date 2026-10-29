"""Specialized agents for quality, statistics, and insight generation."""

from __future__ import annotations

import time
from typing import Any

from src import llm
from src.config import groq_configured
from src.data_tools import fmt_num, invoke_tool
from src.state import AnalysisState

# Tests set this so the two branch nodes overlap in wall-clock time.
NODE_DELAY_SECONDS = 0.0


def _pause(started: float) -> float:
    if NODE_DELAY_SECONDS:
        time.sleep(NODE_DELAY_SECONDS)
    return started


def quality_node(state: AnalysisState) -> dict[str, Any]:
    started = _pause(time.perf_counter())
    logs: list[dict[str, Any]] = []
    try:
        profile, record = invoke_tool("profile_dataset", {"dataset_id": state["dataset_id"]})
        logs.append(record)
        success = bool(record["success"])
        result = {
            "success": success,
            "profile": profile if success else {},
            "issues": _quality_issues(profile) if success else [],
            "charts": _missing_chart(profile) if success else [],
            "error": None if success else profile.get("error", "Data quality profiling failed."),
        }
        errors = [] if success else [f"Data Quality Agent failed: {result['error']}"]
    except Exception as exc:
        result = {"success": False, "profile": {}, "issues": [], "charts": [], "error": str(exc)}
        errors = ["Data Quality Agent failed."]
    return {
        "quality_result": result,
        "quality_timing": _timing(started),
        "tool_logs": logs,
        "errors": errors,
    }


def statistics_node(state: AnalysisState) -> dict[str, Any]:
    started = _pause(time.perf_counter())
    logs: list[dict[str, Any]] = []
    question = state.get("question") or ""
    summary = state.get("dataset_summary") or {}
    planned = plan_stat_tools(summary, question)
    outputs: dict[str, Any] = {}
    failures: list[str] = []
    try:
        for tool_name in planned:
            tool_input = {"dataset_id": state["dataset_id"]}
            if tool_name == "analyze_group_by":
                tool_input["question"] = question
            payload, record = invoke_tool(tool_name, tool_input)
            logs.append(record)
            outputs[tool_name] = payload
            if not record["success"]:
                failures.append(tool_name)
        core_ok = "calculate_statistics" in outputs and "calculate_statistics" not in failures
        statistics = outputs.get("calculate_statistics") or {}
        correlations = outputs.get("find_correlations") or {}
        group_by = outputs.get("analyze_group_by") or {}
        trend = outputs.get("analyze_trend") or {}
        result = {
            "success": core_ok and "calculate_statistics" in planned,
            "planned_tools": planned,
            "statistics": statistics if "calculate_statistics" not in failures else {},
            "correlations": correlations if "find_correlations" not in failures else {},
            "group_by": group_by if "analyze_group_by" not in failures else {},
            "trend": trend if "analyze_trend" not in failures else {},
            "highlights": _stat_highlights(statistics, correlations, group_by, trend),
            "charts": _stat_charts(statistics, correlations, group_by, trend),
            "tool_failures": failures,
            "error": None if not failures else "One or more statistical tools failed.",
        }
        errors = [f"Statistical tool failed: {name}" for name in failures]
        if not result["success"]:
            errors.append("Statistical Analysis Agent failed.")
    except Exception:
        result = {
            "success": False,
            "planned_tools": planned,
            "statistics": {},
            "correlations": {},
            "group_by": {},
            "trend": {},
            "highlights": [],
            "charts": [],
            "tool_failures": failures,
            "error": "Statistical analysis failed.",
        }
        errors = ["Statistical Analysis Agent failed."]
    return {
        "stats_result": result,
        "stats_timing": _timing(started),
        "tool_logs": logs,
        "errors": errors,
    }


def plan_stat_tools(summary: dict[str, Any], question: str) -> list[str]:
    """Pick tools from column types and the question. The tools still decide if a calculation applies."""
    text = f" {question.lower()} "
    tools = ["calculate_statistics"]
    numeric = summary.get("numeric_columns") or []
    rows = int(summary.get("row_count") or 0)
    if "correlat" in text or (len(numeric) >= 2 and rows >= 3):
        tools.append("find_correlations")
    group_words = (" by ", " per ", "group", "compare", "average", "highest", "lowest", "which ")
    if summary.get("categorical_columns") or any(word in text for word in group_words):
        tools.append("analyze_group_by")
    trend_words = ("trend", "over time", "monthly", "daily", "change")
    if summary.get("date_columns") or any(word in text for word in trend_words):
        tools.append("analyze_trend")
    return tools


def merge_node(state: AnalysisState) -> dict[str, Any]:
    quality_ok = bool((state.get("quality_result") or {}).get("success"))
    stats_ok = bool((state.get("stats_result") or {}).get("success"))
    failed = []
    if not quality_ok:
        failed.append("quality")
    if not stats_ok:
        failed.append("statistics")
    return {
        "branch_status": {
            "quality": "success" if quality_ok else "failed",
            "statistics": "success" if stats_ok else "failed",
            "failed_branches": failed,
        }
    }


def route_after_merge(state: AnalysisState) -> str:
    """Skip the network call when the benchmark or a missing API key asks for tool results only."""
    if state.get("force_deterministic") or not groq_configured():
        return "deterministic_insight"
    return "llm_insight"


def deterministic_insight_node(state: AnalysisState) -> dict[str, Any]:
    reason = None
    if state.get("force_deterministic"):
        reason = "This run used the deterministic report so the result does not depend on a language model."
    elif not groq_configured():
        reason = "GROQ_API_KEY is not configured. The report below is built only from tool results."
    report = build_deterministic_report(
        state.get("question") or "",
        state.get("quality_result") or {},
        state.get("stats_result") or {},
        state.get("branch_status") or {},
        notice=reason,
    )
    return {
        "report": report,
        "insight_source": "deterministic",
        "llm_usage": llm.empty_usage(),
        "llm_latency_ms": None,
    }


def llm_insight_node(state: AnalysisState) -> dict[str, Any]:
    context = {
        "question": state.get("question") or "",
        "dataset_summary": state.get("dataset_summary") or {},
        "branch_status": state.get("branch_status") or {},
        "tool_results": _compact_results(state.get("quality_result") or {}, state.get("stats_result") or {}),
    }
    try:
        report, usage, latency_ms = llm.generate_insight(context)
        report = _disclose_failed_branches(report, state.get("branch_status") or {})
        source = "groq"
    except Exception as exc:
        report = build_deterministic_report(
            state.get("question") or "",
            state.get("quality_result") or {},
            state.get("stats_result") or {},
            state.get("branch_status") or {},
            notice="The Groq request failed. Deterministic tool results are shown instead.",
            llm_error=llm.safe_llm_error(exc),
        )
        usage = llm.empty_usage()
        latency_ms = None
        source = "deterministic_fallback"
    return {
        "report": report,
        "insight_source": source,
        "llm_usage": usage,
        "llm_latency_ms": latency_ms,
        "errors": [report["llm_error"]] if report.get("llm_error") else [],
    }


def _disclose_failed_branches(report: dict[str, Any], branch_status: dict[str, Any]) -> dict[str, Any]:
    limitations = list(report.get("limitations") or [])
    blob = " ".join(limitations).lower()
    phrases = {
        "quality": "Data Quality Agent failed.",
        "statistics": "Statistical Analysis Agent failed.",
    }
    for branch in branch_status.get("failed_branches") or []:
        phrase = phrases.get(branch)
        if phrase and phrase.lower() not in blob:
            limitations.append(phrase)
    report["limitations"] = limitations
    return report


def build_deterministic_report(
    question: str,
    quality: dict[str, Any],
    stats: dict[str, Any],
    branch_status: dict[str, Any],
    notice: str | None = None,
    llm_error: str | None = None,
) -> dict[str, Any]:
    lines = canonical_evidence(quality, stats)
    findings = _findings_from_results(quality, stats)
    limitations = []
    if notice:
        limitations.append(notice)
    if llm_error:
        limitations.append(llm_error)
    for branch in branch_status.get("failed_branches") or []:
        if branch == "quality":
            limitations.append("Data Quality Agent failed.")
        if branch == "statistics":
            limitations.append("Statistical Analysis Agent failed.")
    for issue in quality.get("issues") or []:
        if issue.get("severity") in {"high", "medium"}:
            limitations.append(issue["issue"])
    if not limitations:
        limitations.append("The checks cover missing values, duplicates, constants, invalid types, and numeric outliers only.")
    cannot = [
        "These results describe the uploaded rows only.",
        "The tools do not prove causation.",
    ]
    if (stats.get("correlations") or {}).get("pairs"):
        cannot.append("Correlation does not prove causation.")
    if (stats.get("trend") or {}).get("directions"):
        cannot.append("A change from the first period to the last period is not a forecast.")
    if (quality.get("profile") or {}).get("row_count") == 0:
        cannot.append("No numeric average can be calculated when the dataset has no rows.")
    answer_lines = ["Deterministic analysis for the question: " + (question.strip() or "Provide an overall summary.")]
    answer_lines.extend(lines)
    if (stats.get("correlations") or {}).get("pairs"):
        answer_lines.append("Correlation does not prove causation.")
    return {
        "source": "deterministic",
        "model": None,
        "answer": "\n".join(answer_lines),
        "findings": findings,
        "limitations": limitations,
        "cannot_conclude": cannot,
        "data_quality_notes": [issue["issue"] for issue in quality.get("issues") or []],
        "llm_error": llm_error,
    }


def canonical_evidence(quality: dict[str, Any], stats: dict[str, Any]) -> list[str]:
    """Stable evidence lines. Benchmarks look for these exact phrases."""
    lines: list[str] = []
    profile = quality.get("profile") or {}
    if profile:
        lines.append(f"row_count={profile.get('row_count')}")
        lines.append(f"column_count={profile.get('column_count')}")
        lines.append(f"duplicate_row_count={profile.get('duplicate_row_count')}")
        constants = profile.get("constant_columns") or []
        lines.append("constant_columns=" + (",".join(constants) if constants else "none"))
        for column in profile.get("columns") or []:
            if column.get("missing_count"):
                lines.append(f"{column['name']} missing_count={column['missing_count']}")
        for item in profile.get("invalid_types") or []:
            lines.append(f"{item['column']} invalid_count={item['invalid_count']}")
        for item in profile.get("outliers") or []:
            lines.append(f"{item['column']} outlier_count={item['outlier_count']}")
    for name, metrics in ((stats.get("statistics") or {}).get("numeric") or {}).items():
        if metrics.get("mean") is not None:
            lines.append(f"{name} mean={fmt_num(metrics['mean'])}")
        if metrics.get("median") is not None:
            lines.append(f"{name} median={fmt_num(metrics['median'])}")
    for pair in ((stats.get("correlations") or {}).get("pairs") or [])[:3]:
        lines.append(
            f"correlation={fmt_num(pair.get('correlation'))} between {pair.get('left')} and {pair.get('right')}"
        )
    group = stats.get("group_by") or {}
    group_column = group.get("group_column")
    for row in (group.get("groups") or [])[:8]:
        for value_name, metric in (row.get("metrics") or {}).items():
            if metric.get("mean") is not None:
                lines.append(f"{group_column}={row['group']} {value_name} mean={fmt_num(metric['mean'])}")
    for period in ((stats.get("trend") or {}).get("periods") or []):
        lines.append(f"period={period['period']} row_count={period['row_count']}")
    for direction in ((stats.get("trend") or {}).get("directions") or []):
        lines.append(
            f"{direction['column']} direction={direction['direction']} "
            f"from {direction['first_period']} to {direction['last_period']}"
        )
    return lines


def _findings_from_results(quality: dict[str, Any], stats: dict[str, Any]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    profile = quality.get("profile") or {}
    if profile:
        findings.append(
            _finding(
                f"The dataset has {profile.get('row_count')} rows and {profile.get('column_count')} columns.",
                f"profile_dataset row_count={profile.get('row_count')} column_count={profile.get('column_count')}",
                "These counts describe the table after it was loaded.",
                "Confirm that this extract is the table you meant to analyze.",
            )
        )
    duplicates = profile.get("duplicate_row_count")
    if duplicates:
        findings.append(
            _finding(
                f"The duplicate-row check found {duplicates} extra duplicate rows.",
                f"profile_dataset duplicate_row_count={duplicates}",
                "Repeated rows can inflate counts and averages.",
                "Decide whether those repeated rows are real events or reload them after removing duplicates.",
            )
        )
    numeric = (stats.get("statistics") or {}).get("numeric") or {}
    for name, metrics in list(numeric.items())[:2]:
        if metrics.get("mean") is None and metrics.get("median") is None:
            continue
        findings.append(
            _finding(
                f"For {name}, the mean is {fmt_num(metrics.get('mean'))} and the median is {fmt_num(metrics.get('median'))}.",
                f"calculate_statistics {name} mean={fmt_num(metrics.get('mean'))} {name} median={fmt_num(metrics.get('median'))}",
                "The mean and median summarize the observed numeric values. They are not a prediction.",
                "Compare the mean with the median before treating the mean as a typical value.",
            )
        )
    pairs = (stats.get("correlations") or {}).get("pairs") or []
    if pairs:
        pair = pairs[0]
        findings.append(
            _finding(
                f"The Pearson correlation between {pair.get('left')} and {pair.get('right')} is {fmt_num(pair.get('correlation'))}.",
                f"find_correlations correlation={fmt_num(pair.get('correlation'))} between {pair.get('left')} and {pair.get('right')}",
                "This is a linear association in the observed rows. It does not prove causation.",
                "Treat the association as a clue and test it with more evidence before acting.",
            )
        )
    group = stats.get("group_by") or {}
    if group.get("groups") and group.get("value_columns"):
        value_name = group["value_columns"][0]
        ranked = [
            row
            for row in group["groups"]
            if (row.get("metrics") or {}).get(value_name, {}).get("mean") is not None
        ]
        if ranked:
            best = max(ranked, key=lambda row: row["metrics"][value_name]["mean"])
            mean_value = best["metrics"][value_name]["mean"]
            findings.append(
                _finding(
                    f"The highest observed mean of {value_name} is {fmt_num(mean_value)} in {best['group']}.",
                    f"analyze_group_by {group.get('group_column')}={best['group']} {value_name} mean={fmt_num(mean_value)}",
                    "This compares group averages in the uploaded file. It does not explain why the groups differ.",
                    "Review the group sizes before treating the highest average as a decision by itself.",
                )
            )
    directions = (stats.get("trend") or {}).get("directions") or []
    if directions:
        direction = directions[0]
        findings.append(
            _finding(
                f"From {direction['first_period']} to {direction['last_period']}, {direction['column']} is {direction['direction']}.",
                (
                    f"analyze_trend {direction['column']} direction={direction['direction']} "
                    f"from {direction['first_period']} to {direction['last_period']}"
                ),
                "This compares the first and last period means only. It is not a forecast.",
                "Look at the period table before describing the series as a lasting trend.",
            )
        )
    if not findings:
        findings.append(
            _finding(
                "The available tools did not produce numeric findings.",
                "profile_dataset returned no usable profile.",
                "There is not enough successful tool output to support a numeric claim.",
                "Check the file contents and run the analysis again.",
            )
        )
    return findings


def _finding(
    claim: str,
    evidence: str,
    interpretation: str,
    recommendation: str,
    kind: str = "observation",
) -> dict[str, str]:
    return {
        "claim": claim,
        "evidence": evidence,
        "interpretation": interpretation,
        "recommendation": recommendation,
        "kind": kind,
    }


def _quality_issues(profile: dict[str, Any]) -> list[dict[str, str]]:
    issues = []
    for column in profile.get("columns") or []:
        missing = column.get("missing_count") or 0
        if not missing:
            continue
        percent = column.get("missing_percent")
        severity = "high" if (percent or 0) >= 20 else "medium"
        issues.append(
            {
                "severity": severity,
                "issue": f"{column['name']} has {missing} missing values ({fmt_num(percent)}%).",
                "evidence": f"{column['name']} missing_count={missing}",
            }
        )
    duplicates = profile.get("duplicate_row_count") or 0
    if duplicates:
        issues.append(
            {
                "severity": "medium",
                "issue": f"There are {duplicates} extra duplicate rows.",
                "evidence": f"duplicate_row_count={duplicates}",
            }
        )
    for name in profile.get("constant_columns") or []:
        issues.append(
            {
                "severity": "low",
                "issue": f"{name} is constant.",
                "evidence": f"constant_columns={name}",
            }
        )
    for item in profile.get("invalid_types") or []:
        issues.append(
            {
                "severity": "high",
                "issue": f"{item['column']} has {item['invalid_count']} values that are not numbers.",
                "evidence": f"{item['column']} invalid_count={item['invalid_count']}",
            }
        )
    for item in profile.get("outliers") or []:
        issues.append(
            {
                "severity": "medium",
                "issue": (
                    f"{item['column']} has {item['outlier_count']} potential outliers "
                    f"by the Tukey IQR rule."
                ),
                "evidence": f"{item['column']} outlier_count={item['outlier_count']}",
            }
        )
    if not issues:
        issues.append(
            {
                "severity": "low",
                "issue": "These checks did not flag missing values, duplicates, constant columns, invalid types, or IQR outliers.",
                "evidence": "profile_dataset",
            }
        )
    return issues


def _missing_chart(profile: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [column for column in profile.get("columns") or [] if column.get("missing_count")]
    if not rows:
        return []
    return [
        {
            "type": "bar",
            "title": "Missing values by column",
            "x": [column["name"] for column in rows],
            "y": [column["missing_count"] for column in rows],
            "x_label": "Column",
            "y_label": "Missing count",
        }
    ]


def _stat_highlights(statistics: dict, correlations: dict, group_by: dict, trend: dict) -> list[str]:
    highlights = []
    for name, metrics in (statistics.get("numeric") or {}).items():
        highlights.append(f"{name} mean={fmt_num(metrics.get('mean'))}, median={fmt_num(metrics.get('median'))}.")
    for pair in (correlations.get("pairs") or [])[:1]:
        highlights.append(
            f"Pearson correlation between {pair['left']} and {pair['right']} is {fmt_num(pair['correlation'])}. "
            "This does not prove causation."
        )
    if correlations.get("skipped_reason"):
        highlights.append(str(correlations["skipped_reason"]))
    directions = trend.get("directions") or []
    if directions:
        item = directions[0]
        highlights.append(
            f"{item['column']} is {item['direction']} from {item['first_period']} to {item['last_period']}."
        )
    elif trend.get("skipped_reason"):
        highlights.append(str(trend["skipped_reason"]))
    if group_by.get("skipped_reason"):
        highlights.append(str(group_by["skipped_reason"]))
    return highlights


def _stat_charts(statistics: dict, correlations: dict, group_by: dict, trend: dict) -> list[dict[str, Any]]:
    charts: list[dict[str, Any]] = []
    histograms = statistics.get("histograms") or {}
    for name, histogram in histograms.items():
        charts.append(
            {
                "type": "bar",
                "title": f"Distribution of {name}",
                "x": histogram.get("bin_labels") or [],
                "y": histogram.get("counts") or [],
                "x_label": name,
                "y_label": "Rows",
            }
        )
        break
    groups = group_by.get("groups") or []
    value_columns = group_by.get("value_columns") or []
    if groups and value_columns:
        value_name = value_columns[0]
        charts.append(
            {
                "type": "bar",
                "title": f"Mean {value_name} by {group_by.get('group_column')}",
                "x": [row["group"] for row in groups],
                "y": [(row.get("metrics") or {}).get(value_name, {}).get("mean") or 0 for row in groups],
                "x_label": str(group_by.get("group_column")),
                "y_label": f"Mean {value_name}",
            }
        )
    periods = trend.get("periods") or []
    directions = trend.get("directions") or []
    if periods and directions:
        column = directions[0]["column"]
        charts.append(
            {
                "type": "line",
                "title": f"Mean {column} by {trend.get('frequency')}",
                "x": [period["period"] for period in periods],
                "y": [(period.get("means") or {}).get(column) or 0 for period in periods],
                "x_label": "Period",
                "y_label": f"Mean {column}",
            }
        )
    labels = []
    for pair in correlations.get("pairs") or []:
        for name in (pair.get("left"), pair.get("right")):
            if name not in labels:
                labels.append(name)
    if 1 < len(labels) <= 6:
        index = {name: position for position, name in enumerate(labels)}
        matrix = [[1 if row == col else None for col in range(len(labels))] for row in range(len(labels))]
        for pair in correlations.get("pairs") or []:
            left = index[pair["left"]]
            right = index[pair["right"]]
            matrix[left][right] = pair["correlation"]
            matrix[right][left] = pair["correlation"]
        charts.append(
            {
                "type": "heatmap",
                "title": "Pearson correlations",
                "labels": labels,
                "matrix": matrix,
            }
        )
    return charts[:4]


def _compact_results(quality: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    statistics = dict(stats.get("statistics") or {})
    statistics.pop("histograms", None)
    return {
        "quality_issues": quality.get("issues") or [],
        "profile": quality.get("profile") or {},
        "statistics": statistics,
        "correlations": stats.get("correlations") or {},
        "group_by": stats.get("group_by") or {},
        "trend": stats.get("trend") or {},
        "quality_error": quality.get("error"),
        "statistics_error": stats.get("error"),
    }


def _timing(started: float) -> dict[str, float]:
    ended = time.perf_counter()
    return {"start": started, "end": ended, "duration_ms": round((ended - started) * 1000, 3)}
