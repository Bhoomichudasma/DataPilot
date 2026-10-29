"""Reproducible benchmark questions.

Task completion uses these cases, the deterministic report, and the synthetic
fixtures. A case passes only when every required tool succeeds and every
evidence token appears in the report.
"""

from __future__ import annotations

from benchmarks.fixtures import constant_frame, empty_frame, sales_frame

BENCHMARK_CASES = [
    {
        "id": "bq01_shape",
        "fixture": "sales",
        "loader": sales_frame,
        "question": "How many rows and columns does this dataset have?",
        "required_tools": ["profile_dataset"],
        "required_evidence": ["row_count=6", "column_count=5"],
        "acceptance": "The report states 6 rows and 5 columns from the profiler.",
    },
    {
        "id": "bq02_missing",
        "fixture": "sales",
        "loader": sales_frame,
        "question": "How many revenue values are missing?",
        "required_tools": ["profile_dataset"],
        "required_evidence": ["revenue missing_count=1"],
        "acceptance": "The report states that revenue has 1 missing value.",
    },
    {
        "id": "bq03_duplicates",
        "fixture": "sales",
        "loader": sales_frame,
        "question": "Are there duplicate rows?",
        "required_tools": ["profile_dataset"],
        "required_evidence": ["duplicate_row_count=1"],
        "acceptance": "The report states 1 extra duplicate row, using pandas keep='first' counting.",
    },
    {
        "id": "bq04_units_center",
        "fixture": "sales",
        "loader": sales_frame,
        "question": "What are the mean and median of units?",
        "required_tools": ["calculate_statistics"],
        "required_evidence": ["units mean=26.666667", "units median=25"],
        "acceptance": "The report states the mean and median of units.",
    },
    {
        "id": "bq05_region_revenue",
        "fixture": "sales",
        "loader": sales_frame,
        "question": "Which region has the highest average revenue?",
        "required_tools": ["analyze_group_by"],
        "required_evidence": ["region=East", "revenue mean=650"],
        "acceptance": "The report shows East with mean revenue 650.",
    },
    {
        "id": "bq06_correlation",
        "fixture": "sales",
        "loader": sales_frame,
        "question": "Is there a correlation between units and revenue?",
        "required_tools": ["find_correlations"],
        "required_evidence": ["correlation=0.97049", "does not prove causation"],
        "acceptance": "The report states the Pearson correlation and says it does not prove causation.",
    },
    {
        "id": "bq07_trend",
        "fixture": "sales",
        "loader": sales_frame,
        "question": "How do units change over time?",
        "required_tools": ["analyze_trend"],
        "required_evidence": ["period=2024-01", "period=2024-03", "units direction=increasing"],
        "acceptance": "The report compares the first and last month and calls units increasing.",
    },
    {
        "id": "bq08_constant",
        "fixture": "constant",
        "loader": constant_frame,
        "question": "Are any columns constant?",
        "required_tools": ["profile_dataset"],
        "required_evidence": ["constant_columns=status"],
        "acceptance": "The report names status as a constant column.",
    },
    {
        "id": "bq09_empty",
        "fixture": "empty",
        "loader": empty_frame,
        "question": "What is the average of units?",
        "required_tools": ["profile_dataset"],
        "required_evidence": ["row_count=0"],
        "acceptance": "The report states that the dataset has 0 rows, so no average is available.",
    },
    {
        "id": "bq10_category_units",
        "fixture": "sales",
        "loader": sales_frame,
        "question": "What is the average units by category?",
        "required_tools": ["analyze_group_by"],
        "required_evidence": ["category=Widget units mean=20", "category=Gadget units mean=33.333333"],
        "acceptance": "The report states both category means for units.",
    },
]


# Hand-authored claims for the evidence-grounding sample.
# The scorer compares each stated number with the sales fixture tool output.
EVIDENCE_REVIEWS = [
    {
        "id": "er01_rows",
        "claim": "The dataset has 6 rows.",
        "evidence": "profile_dataset row_count=6",
        "interpretation": "Six records were loaded from this fixture.",
        "recommendation": "Limit the conclusion to this file.",
        "binding": {"source": "profile", "path": ["row_count"], "stated": 6},
    },
    {
        "id": "er02_units_mean",
        "claim": "The mean of units is 26.666667.",
        "evidence": "calculate_statistics units mean=26.666667",
        "interpretation": "This is the arithmetic mean of the observed units.",
        "recommendation": "Also look at the median before calling the mean typical.",
        "binding": {"source": "statistic", "column": "units", "stat": "mean", "stated": 26.666667},
    },
    {
        "id": "er03_wrong_mean",
        "claim": "The mean of units is 100.",
        "evidence": "calculate_statistics units mean=100",
        "interpretation": "A typical row has 100 units.",
        "recommendation": "Recheck the mean before acting on it.",
        "binding": {"source": "statistic", "column": "units", "stat": "mean", "stated": 100},
    },
    {
        "id": "er04_duplicates",
        "claim": "There is 1 extra duplicate row.",
        "evidence": "profile_dataset duplicate_row_count=1",
        "interpretation": "One repeated row was detected with the default duplicate count.",
        "recommendation": "Inspect repeated rows before aggregating.",
        "binding": {"source": "profile", "path": ["duplicate_row_count"], "stated": 1},
    },
    {
        "id": "er05_causal",
        "claim": "The Pearson correlation between units and revenue is 0.97049.",
        "evidence": "find_correlations correlation=0.97049 between units and revenue",
        "interpretation": "Higher units causes higher revenue.",
        "recommendation": "Increase units to cause a revenue increase.",
        "binding": {
            "source": "correlation",
            "left": "units",
            "right": "revenue",
            "stated": 0.97049,
        },
    },
    {
        "id": "er06_invented",
        "claim": "The churn rate is 40.",
        "evidence": "churn_rate=40",
        "interpretation": "Many customers left.",
        "recommendation": "Start a retention program.",
        "binding": {"source": "missing", "stated": 40},
    },
]
