"""Tests for deterministic dataset tools."""

import pytest

from src.data_tools import (
    analyze_group_by_frame,
    analyze_trend_frame,
    calculate_statistics_frame,
    find_correlations_frame,
    invoke_tool,
    load_csv_bytes,
    profile_dataset_frame,
    register_dataframe,
)
from benchmarks.fixtures import (
    constant_frame,
    empty_frame,
    invalid_frame,
    outlier_frame,
    sales_frame,
)


def test_profile_counts_missing_and_duplicates():
    profile = profile_dataset_frame(sales_frame())
    revenue = next(column for column in profile["columns"] if column["name"] == "revenue")
    assert profile["row_count"] == 6
    assert profile["column_count"] == 5
    assert revenue["missing_count"] == 1
    assert profile["duplicate_row_count"] == 1
    assert profile["constant_columns"] == []


def test_numeric_summaries_and_correlation():
    stats = calculate_statistics_frame(sales_frame())
    assert stats["numeric"]["units"]["mean"] == 26.666667
    assert stats["numeric"]["units"]["median"] == 25
    assert stats["numeric"]["units"]["std"] == 16.329932
    assert stats["numeric"]["revenue"]["mean"] == 360
    assert stats["numeric"]["revenue"]["median"] == 300
    correlations = find_correlations_frame(sales_frame())
    pair = correlations["pairs"][0]
    assert {pair["left"], pair["right"]} == {"units", "revenue"}
    assert pair["correlation"] == 0.97049
    assert "does not prove causation" in correlations["note"]


def test_group_means_and_trend():
    by_region = analyze_group_by_frame(sales_frame(), question="Which region has the highest average revenue?")
    means = {row["group"]: row["metrics"]["revenue"]["mean"] for row in by_region["groups"]}
    assert by_region["group_column"] == "region"
    assert means["East"] == 650
    assert means["South"] == 300
    by_category = analyze_group_by_frame(sales_frame(), question="What is the average units by category?")
    category_means = {row["group"]: row["metrics"]["units"]["mean"] for row in by_category["groups"]}
    assert category_means["Widget"] == 20
    assert category_means["Gadget"] == 33.333333
    trend = analyze_trend_frame(sales_frame())
    units = next(item for item in trend["directions"] if item["column"] == "units")
    assert units["direction"] == "increasing"
    assert units["first_period"] == "2024-01"
    assert units["last_period"] == "2024-03"


def test_constant_missing_outlier_and_invalid_columns():
    constant = profile_dataset_frame(constant_frame())
    assert "status" in constant["constant_columns"]
    assert "amount" in constant["constant_columns"]
    outliers = profile_dataset_frame(outlier_frame())["outliers"]
    assert outliers[0]["column"] == "score"
    assert outliers[0]["method"] == "tukey_iqr"
    assert outliers[0]["outlier_count"] == 1
    invalid = profile_dataset_frame(invalid_frame())["invalid_types"]
    assert invalid[0]["invalid_count"] == 1


def test_empty_and_malformed_csv(monkeypatch):
    profile = profile_dataset_frame(empty_frame())
    stats = calculate_statistics_frame(empty_frame())
    correlations = find_correlations_frame(empty_frame())
    grouped = analyze_group_by_frame(empty_frame(), question="average units by region")
    trend = analyze_trend_frame(empty_frame())
    assert profile["row_count"] == 0
    assert stats["numeric"] == {}
    assert stats["skipped_reason"]
    assert correlations["pairs"] == []
    assert grouped["groups"] == []
    assert trend["periods"] == []

    with pytest.raises(Exception, match="empty"):
        load_csv_bytes(b"", "empty.csv")

    with pytest.raises(Exception, match="UTF-8|CSV|csv"):
        load_csv_bytes(b"\xff\xfe\xfd", "broken.csv")

    monkeypatch.setattr("src.data_tools.MAX_ROWS", 1)
    try:
        load_csv_bytes(b"units,region\n1,North\n2,South\n", "too_big.csv")
        assert False
    except Exception as exc:
        assert "20000" not in str(exc)
        assert "1" in str(exc)


def test_too_few_rows_skips_correlation_safely():
    frame = sales_frame().head(2)
    result = find_correlations_frame(frame)
    assert result["pairs"] == []
    assert result["skipped_reason"]


def test_tool_invocation_records_timing_without_cell_values():
    dataset_id = register_dataframe(sales_frame())
    payload, record = invoke_tool("profile_dataset", {"dataset_id": dataset_id})
    assert payload["row_count"] == 6
    assert record["tool_name"] == "profile_dataset"
    assert record["success"] is True
    assert record["error_category"] is None
    assert record["duration_ms"] >= 0
    assert record["start_time"]
    assert record["end_time"]
    blob = record["input_summary"] + record["output_summary"]
    assert "Widget" not in blob
    assert "800" not in blob

    failed, failed_record = invoke_tool("calculate_statistics", {"dataset_id": "missing-dataset"})
    assert "error" in failed
    assert failed_record["success"] is False
    assert failed_record["error_category"] == "dataset_not_found"
    assert failed_record["duration_ms"] >= 0
