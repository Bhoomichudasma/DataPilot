"""Checks a report against tool results.

The validator matches quantitative tokens and a short list of causal phrases.
It does not decide whether a sentence is semantically true.
"""

from __future__ import annotations

import re
from typing import Any

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
DATE_TOKEN_RE = re.compile(r"\b\d{4}-\d{1,2}(?:-\d{1,2})?\b")
TOOL_OR_METRIC_RE = re.compile(
    r"profile_dataset|calculate_statistics|find_correlations|analyze_group_by|analyze_trend|"
    r"row_count|column_count|missing_count|duplicate_row_count|constant_columns|"
    r"\bmean\b|\bmedian\b|\bcorrelation\b|\bdirection\b|\bperiod\b",
    re.IGNORECASE,
)
SAFE_CAUSAL_PHRASES = (
    "does not prove causation",
    "do not prove causation",
    "not prove causation",
    "not evidence of causation",
    "no causation",
    "does not show causation",
    "correlation is not causation",
)


def collect_numbers(payload: Any) -> set[float]:
    found: set[float] = set()

    def walk(value: Any) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            found.add(float(value))
            return
        if isinstance(value, str):
            return
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(payload)
    return found


def _ignore_number(value: float) -> bool:
    return value.is_integer() and 1900 <= int(value) <= 2100


def extract_claim_numbers(text: str) -> list[float]:
    numbers = []
    cleaned = DATE_TOKEN_RE.sub(" ", text or "")
    for match in NUMBER_RE.findall(cleaned):
        value = float(match)
        if _ignore_number(value):
            continue
        numbers.append(value)
    return numbers


def numbers_match(claimed: float, available: set[float]) -> bool:
    for candidate in available:
        if claimed == candidate:
            return True
        claimed_whole = float(claimed).is_integer()
        candidate_whole = float(candidate).is_integer()
        if claimed_whole and candidate_whole:
            if int(claimed) == int(candidate):
                return True
            continue
        difference = abs(claimed - candidate)
        scale = max(abs(claimed), abs(candidate), 1e-9)
        if difference <= 0.015 or difference / scale <= 0.005:
            return True
    return False


def _causal_flag(text: str) -> bool:
    cleaned = text or ""
    lowered = cleaned.lower()
    for phrase in SAFE_CAUSAL_PHRASES:
        lowered = lowered.replace(phrase, "")
    return bool(re.search(r"\b(causes|caused|causing|causation|proves that)\b", lowered))


def validate_report(
    report: dict[str, Any],
    quality_result: dict[str, Any],
    stats_result: dict[str, Any],
    branch_status: dict[str, Any],
) -> dict[str, Any]:
    available = collect_numbers({"quality": quality_result, "statistics": stats_result})
    finding_rows = []
    unsupported = []
    causal_flags = []
    texts = [
        str(report.get("answer", "")),
        " ".join(report.get("limitations") or []),
        " ".join(report.get("cannot_conclude") or []),
        " ".join(report.get("data_quality_notes") or []),
    ]
    for index, finding in enumerate(report.get("findings") or []):
        claim = str(finding.get("claim", ""))
        evidence = str(finding.get("evidence", ""))
        interpretation = str(finding.get("interpretation", ""))
        recommendation = str(finding.get("recommendation", ""))
        texts.extend([claim, evidence, interpretation, recommendation])
        problems = []
        for field in ("claim", "evidence", "interpretation", "recommendation"):
            if not str(finding.get(field, "")).strip():
                problems.append(f"missing_{field}")
        numbers = extract_claim_numbers(f"{claim} {evidence}")
        if numbers and not TOOL_OR_METRIC_RE.search(evidence):
            problems.append("evidence_does_not_cite_a_tool_or_metric")
        missing_numbers = [value for value in numbers if not numbers_match(value, available)]
        if missing_numbers:
            problems.append("number_not_in_tool_results")
        if any(_causal_flag(part) for part in (claim, interpretation, recommendation)):
            problems.append("causal_wording")
            causal_flags.append({"index": index, "claim": claim})
        if not numbers:
            status = "not_quantitative"
        elif problems:
            status = "unsupported"
            unsupported.append({"index": index, "claim": claim, "problems": problems})
        else:
            status = "supported"
        finding_rows.append({"index": index, "status": status, "problems": problems})

    for text in texts:
        if _causal_flag(text):
            if not any(flag.get("claim") == text for flag in causal_flags):
                causal_flags.append({"index": None, "claim": text[:240]})

    failed = list(branch_status.get("failed_branches") or [])
    disclosure_blob = " ".join(
        [
            str(report.get("answer", "")),
            " ".join(report.get("limitations") or []),
            " ".join(report.get("data_quality_notes") or []),
        ]
    ).lower()
    disclosure_ok = True
    disclosure_problems = []
    expected_phrases = {
        "quality": "data quality agent failed",
        "statistics": "statistical analysis agent failed",
    }
    for branch in failed:
        phrase = expected_phrases.get(branch)
        if phrase and phrase not in disclosure_blob:
            disclosure_ok = False
            disclosure_problems.append(branch)

    quantitative = [row for row in finding_rows if row["status"] != "not_quantitative"]
    supported = [row for row in quantitative if row["status"] == "supported"]
    passed = not unsupported and disclosure_ok and not causal_flags
    return {
        "method": (
            "Each finding must include claim, evidence, interpretation, and recommendation. "
            "Numbers in the claim and evidence are compared with numbers present in the tool "
            "results. Integers must match exactly. Other numbers match within 0.015 absolute "
            "or 0.5% relative. Calendar years from 1900 to 2100 are ignored. Evidence for a "
            "quantitative finding must cite a tool or metric name. Causal phrases are flagged "
            "unless the text explicitly says causation is not shown. Failed branches must be "
            "named in the report."
        ),
        "does_not_prove": (
            "These checks do not prove that a sentence is semantically correct, or that a "
            "matching number is the right metric for the claim. A value can appear in the "
            "tool output for a different reason."
        ),
        "findings": finding_rows,
        "unsupported_claims": unsupported,
        "causal_flags": causal_flags,
        "branch_disclosure_ok": disclosure_ok,
        "disclosure_problems": disclosure_problems,
        "failed_branches": failed,
        "quantitative_claims": len(quantitative),
        "supported_quantitative_claims": len(supported),
        "passed": passed,
    }
