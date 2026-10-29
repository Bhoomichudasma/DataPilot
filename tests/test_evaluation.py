"""Tests for metrics, validation, and the offline benchmark."""

from src.evaluation import (
    end_to_end_summary,
    run_benchmark,
    run_numerical_checks,
    score_evidence_reviews,
    tool_latency_summary,
    tool_success_rate,
)
from src.report_validation import validate_report


def test_tool_success_rate_and_empty_rate():
    records = [
        {"success": True},
        {"success": False},
        {"success": True},
        {"success": True},
    ]
    summary = tool_success_rate(records)
    assert summary["successful"] == 3
    assert summary["total"] == 4
    assert summary["rate_percent"] == 75
    assert tool_success_rate([])["display"] == "N/A"
    assert tool_success_rate([])["rate_percent"] is None


def test_tool_latency_excludes_nothing_but_uses_tool_durations_only():
    records = [
        {"tool_name": "profile_dataset", "duration_ms": 10},
        {"tool_name": "profile_dataset", "duration_ms": 30},
        {"tool_name": "calculate_statistics", "duration_ms": 20},
    ]
    summary = tool_latency_summary(records)
    assert summary["average_ms"] == 20
    assert summary["median_ms"] == 20
    assert summary["p95_ms"] == 29
    assert summary["per_tool"][1]["tool_name"] == "profile_dataset"
    assert "LLM" in summary["note"]
    assert tool_latency_summary([])["average_ms"] is None


def test_end_to_end_labels_llm_time_and_small_samples():
    summary = end_to_end_summary(
        [
            {"duration_ms": 100, "includes_llm_network_time": False},
            {"duration_ms": 300, "includes_llm_network_time": False},
        ]
    )
    assert summary["average_ms"] == 200
    assert summary["median_ms"] == 200
    assert summary["includes_llm_network_time"] is False
    assert "does not include Groq" in summary["note"]
    assert "not a stable" in summary["note"]
    with_llm = end_to_end_summary([{"duration_ms": 50, "includes_llm_network_time": True}])
    assert "includes Groq" in with_llm["note"]


def test_evidence_validator_flags_unknown_numbers_and_causation():
    quality = {"profile": {"row_count": 6, "column_count": 5, "duplicate_row_count": 0, "columns": []}}
    stats = {"statistics": {"numeric": {"units": {"mean": 26.666667, "median": 25}}}}
    good = {
        "answer": "The mean is grounded in the tool output.",
        "findings": [
            {
                "claim": "The mean of units is 26.666667.",
                "evidence": "calculate_statistics units mean=26.666667",
                "interpretation": "This is the observed arithmetic mean.",
                "recommendation": "Compare it with the median.",
            }
        ],
        "limitations": [],
        "cannot_conclude": ["Correlation does not prove causation."],
        "data_quality_notes": [],
    }
    assert validate_report(good, quality, stats, {"failed_branches": []})["passed"] is True

    bad = {
        "answer": "Units causes revenue to rise.",
        "findings": [
            {
                "claim": "The mean of units is 100.",
                "evidence": "calculate_statistics units mean=100",
                "interpretation": "This causes a higher result.",
                "recommendation": "Act on the mean.",
            }
        ],
        "limitations": [],
        "cannot_conclude": [],
        "data_quality_notes": [],
    }
    result = validate_report(bad, quality, stats, {"failed_branches": ["quality"]})
    assert result["passed"] is False
    assert result["unsupported_claims"]
    assert result["causal_flags"]
    assert result["branch_disclosure_ok"] is False
    assert "does not prove" in result["does_not_prove"].lower() or "semantically" in result["does_not_prove"]


def test_numerical_and_evidence_sample_denominators():
    numerical = run_numerical_checks()
    assert numerical["total"] == 20
    assert numerical["passed"] == 20
    assert numerical["rate_percent"] == 100
    evidence = score_evidence_reviews()
    assert evidence["total"] == 6
    by_id = {item["id"]: item for item in evidence["reviews"]}
    assert by_id["er01_rows"]["passed"] is True
    assert by_id["er02_units_mean"]["passed"] is True
    assert by_id["er03_wrong_mean"]["numerical_correct"] is False
    assert by_id["er03_wrong_mean"]["passed"] is False
    assert by_id["er05_causal"]["numerical_correct"] is True
    assert by_id["er05_causal"]["interpretation_follows"] is False
    assert by_id["er06_invented"]["supporting_evidence"] is False
    assert evidence["passed"] == 3
    assert "not a universal" in evidence["note"]


def test_full_benchmark_completes_without_network(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("src.evaluation.RESULTS_DIR", tmp_path)
    payload = run_benchmark(save=True)
    assert payload["task_completion"]["total"] == 10
    assert payload["task_completion"]["passed"] == 10
    assert payload["task_completion"]["failed"] == 0
    assert payload["numerical_correctness"]["passed"] == 20
    assert payload["tool_success_rate"]["rate_percent"] == 100
    assert payload["tool_latency"]["average_ms"] is not None
    assert payload["end_to_end"]["includes_llm_network_time"] is False
    assert payload["llm_usage"]["input_tokens"] is None
    saved = list(tmp_path.glob("benchmark_*.json"))
    assert len(saved) == 1
    text = saved[0].read_text(encoding="utf-8")
    assert "Widget" not in text
    assert "sk-" not in text
