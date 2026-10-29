"""Deterministic dataset tools.

Agents call these through LangChain tool objects. Calculations use pandas and
NumPy only, so the same CSV always produces the same numbers.
"""

from __future__ import annotations

import json
import time
import uuid
import warnings
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import BaseTool, tool

from src.config import MAX_COLUMNS, MAX_ROWS, MAX_UPLOAD_BYTES

_DATASETS: dict[str, pd.DataFrame] = {}


class DataPilotError(Exception):
    """Expected application error with a stable category name."""

    def __init__(self, message: str, category: str = "computation_error") -> None:
        super().__init__(message)
        self.category = category


class DatasetValidationError(DataPilotError):
    def __init__(self, message: str) -> None:
        super().__init__(message, "invalid_input")


def clear_datasets() -> None:
    _DATASETS.clear()


def register_dataframe(df: pd.DataFrame) -> str:
    dataset_id = uuid.uuid4().hex
    _DATASETS[dataset_id] = df.copy()
    return dataset_id


def release_dataset(dataset_id: str) -> None:
    _DATASETS.pop(dataset_id, None)


def get_dataframe(dataset_id: str) -> pd.DataFrame:
    if dataset_id not in _DATASETS:
        raise DataPilotError(
            "The dataset is no longer available. Upload the CSV and run the analysis again.",
            "dataset_not_found",
        )
    return _DATASETS[dataset_id]


def num(value: Any) -> int | float | None:
    """Convert numpy values to JSON-safe rounded numbers."""
    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number) or np.isinf(number):
        return None
    rounded = round(number, 6)
    if rounded.is_integer():
        return int(rounded)
    return rounded


def fmt_num(value: Any) -> str:
    converted = num(value)
    if converted is None:
        return "n/a"
    if isinstance(converted, int):
        return str(converted)
    return f"{converted:.6f}".rstrip("0").rstrip(".")


def load_csv_bytes(payload: bytes, filename: str = "upload.csv") -> pd.DataFrame:
    """Read a user CSV without executing any code inside the file."""
    if not filename.lower().endswith(".csv"):
        raise DatasetValidationError("Upload a .csv file.")
    if not payload:
        raise DatasetValidationError("The file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES / (1024 * 1024)
        raise DatasetValidationError(f"The file is larger than the {limit_mb:.0f} MB limit.")
    try:
        frame = pd.read_csv(BytesIO(payload), low_memory=False)
    except pd.errors.EmptyDataError as exc:
        raise DatasetValidationError("The CSV has no header or rows.") from exc
    except pd.errors.ParserError as exc:
        raise DatasetValidationError("The file could not be parsed as a CSV.") from exc
    except UnicodeDecodeError as exc:
        raise DatasetValidationError("The file is not valid UTF-8 or ASCII text.") from exc
    except Exception as exc:
        raise DatasetValidationError("The file could not be read as a CSV.") from exc
    return _validate_frame(frame)


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.shape[1] == 0:
        raise DatasetValidationError("The CSV has no columns.")
    if frame.shape[1] > MAX_COLUMNS:
        raise DatasetValidationError(f"The CSV has more than {MAX_COLUMNS} columns.")
    if len(frame) > MAX_ROWS:
        raise DatasetValidationError(f"The CSV has more than {MAX_ROWS} rows.")
    cleaned = frame.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    if cleaned.columns.duplicated().any():
        raise DatasetValidationError("The CSV has duplicate column names.")
    return cleaned


def numeric_columns(df: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in df.columns:
        if pd.api.types.is_bool_dtype(df[column]):
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            columns.append(str(column))
    return columns


def _parse_dates(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", format="%Y-%m-%d")
    observed = int(series.notna().sum())
    if observed and int(parsed.notna().sum()) >= 0.8 * observed:
        return parsed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            return pd.to_datetime(series, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            return pd.to_datetime(series, errors="coerce")


def date_columns(df: pd.DataFrame) -> list[str]:
    found: list[str] = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            found.append(str(column))
            continue
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        non_null = series.dropna()
        if len(non_null) < 2:
            continue
        parsed = _parse_dates(non_null)
        if parsed.notna().mean() >= 0.8:
            found.append(str(column))
    return found


def categorical_columns(df: pd.DataFrame) -> list[str]:
    dates = set(date_columns(df))
    found: list[str] = []
    for column in df.columns:
        name = str(column)
        if name in dates or name in numeric_columns(df):
            continue
        series = df[column]
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_bool_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            continue
        unique = int(series.nunique(dropna=True))
        if 1 < unique <= 30:
            found.append(name)
    return found


def build_dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Compact schema summary. It does not include cell values."""
    preview_types = {str(column): str(df[column].dtype) for column in df.columns}
    return {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "column_names": [str(column) for column in df.columns],
        "dtypes": preview_types,
        "numeric_columns": numeric_columns(df),
        "categorical_columns": categorical_columns(df),
        "date_columns": date_columns(df),
    }


def profile_dataset_frame(df: pd.DataFrame) -> dict[str, Any]:
    columns = []
    constant_columns: list[str] = []
    for column in df.columns:
        series = df[column]
        missing = int(series.isna().sum())
        unique = int(series.nunique(dropna=True))
        constant = unique <= 1
        if constant:
            constant_columns.append(str(column))
        columns.append(
            {
                "name": str(column),
                "dtype": str(series.dtype),
                "missing_count": missing,
                "missing_percent": num(100 * missing / len(df)) if len(df) else 0,
                "unique_count": unique,
                "constant": constant,
            }
        )
    return {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "columns": columns,
        "duplicate_row_count": int(df.duplicated().sum()) if len(df) else 0,
        "constant_columns": constant_columns,
        "invalid_types": _invalid_types(df),
        "outliers": _outliers(df),
        "outlier_method": "tukey_iqr",
    }


def _invalid_types(df: pd.DataFrame) -> list[dict[str, Any]]:
    findings = []
    for column in df.columns:
        series = df[column]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        non_null = series.dropna().astype(str).str.strip()
        non_null = non_null[non_null != ""]
        if len(non_null) < 3:
            continue
        parsed = pd.to_numeric(non_null, errors="coerce")
        numeric_ratio = float(parsed.notna().mean())
        if 0.5 <= numeric_ratio < 1:
            invalid = int(parsed.isna().sum())
            findings.append(
                {
                    "column": str(column),
                    "expected": "number",
                    "invalid_count": invalid,
                    "invalid_percent": num(100 * invalid / len(non_null)),
                }
            )
    return findings


def _outliers(df: pd.DataFrame) -> list[dict[str, Any]]:
    findings = []
    for column in numeric_columns(df):
        series = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(series) < 4:
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())
        if count == 0:
            continue
        findings.append(
            {
                "column": column,
                "method": "tukey_iqr",
                "lower_bound": num(lower),
                "upper_bound": num(upper),
                "outlier_count": count,
                "outlier_percent": num(100 * count / len(series)),
            }
        )
    return findings


def calculate_statistics_frame(df: pd.DataFrame) -> dict[str, Any]:
    if len(df) == 0:
        return {
            "row_count": 0,
            "numeric": {},
            "categorical": {},
            "histograms": {},
            "skipped_reason": "The dataset has no rows, so summary statistics were not calculated.",
        }
    numeric: dict[str, Any] = {}
    histograms: dict[str, Any] = {}
    for column in numeric_columns(df):
        series = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            continue
        numeric[column] = {
            "count": int(series.count()),
            "mean": num(series.mean()),
            "std": num(series.std(ddof=1)) if len(series) >= 2 else None,
            "min": num(series.min()),
            "q1": num(series.quantile(0.25)),
            "median": num(series.median()),
            "q3": num(series.quantile(0.75)),
            "max": num(series.max()),
        }
        if len(series) >= 4 and len(histograms) < 2:
            histograms[column] = _histogram(series)
    categorical: dict[str, Any] = {}
    for column in categorical_columns(df):
        counts = df[column].fillna("(missing)").astype(str).value_counts()
        top = counts.head(8)
        categorical[column] = {
            "unique_count": int(df[column].nunique(dropna=True)),
            "top_values": [
                {
                    "value": str(index),
                    "count": int(count),
                    "percent": num(100 * count / len(df)),
                }
                for index, count in top.items()
            ],
            "truncated": bool(len(counts) > 8),
        }
    return {
        "row_count": int(len(df)),
        "numeric": numeric,
        "categorical": categorical,
        "histograms": histograms,
        "skipped_reason": None if numeric or categorical else "No numeric or categorical columns were available.",
    }


def _histogram(series: pd.Series, bins: int = 8) -> dict[str, Any]:
    bin_count = min(bins, max(int(series.nunique()), 1))
    counts, edges = np.histogram(series.to_numpy(dtype=float), bins=bin_count)
    labels = [f"{fmt_num(edges[i])} to {fmt_num(edges[i + 1])}" for i in range(len(counts))]
    return {"bin_labels": labels, "counts": [int(count) for count in counts]}


def find_correlations_frame(df: pd.DataFrame) -> dict[str, Any]:
    note = "Pearson correlation measures linear association and does not prove causation."
    columns = []
    for column in numeric_columns(df):
        series = pd.to_numeric(df[column], errors="coerce")
        if series.nunique(dropna=True) > 1:
            columns.append(column)
    if len(df) < 3:
        return {
            "method": "pearson",
            "pairs": [],
            "note": note,
            "skipped_reason": "At least 3 rows are required for a correlation.",
        }
    if len(columns) < 2:
        return {
            "method": "pearson",
            "pairs": [],
            "note": note,
            "skipped_reason": "At least two non-constant numeric columns are required.",
        }
    pairs = []
    for index, left in enumerate(columns):
        for right in columns[index + 1 :]:
            subset = df[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(subset) < 3:
                continue
            if subset[left].nunique() <= 1 or subset[right].nunique() <= 1:
                continue
            correlation = subset[left].corr(subset[right], method="pearson")
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "correlation": num(correlation),
                    "pairwise_count": int(len(subset)),
                }
            )
    pairs.sort(key=lambda item: abs(item["correlation"] or 0), reverse=True)
    return {
        "method": "pearson",
        "pairs": pairs[:15],
        "note": note,
        "skipped_reason": None if pairs else "No numeric pair had enough varying values.",
    }


def _question_mentions(question: str, column: str) -> bool:
    return column.lower() in question.lower()


def choose_group_column(df: pd.DataFrame, question: str = "", explicit: str = "") -> tuple[str | None, str]:
    if explicit and explicit in df.columns:
        return explicit, "The caller named the grouping column."
    candidates = categorical_columns(df)
    for column in candidates:
        if _question_mentions(question, column):
            return column, "The question names this column."
    if candidates:
        return candidates[0], "This is the first low-cardinality categorical column."
    return None, "No suitable categorical column was found."


def choose_value_columns(df: pd.DataFrame, question: str = "", explicit: str = "") -> list[str]:
    if explicit and explicit in numeric_columns(df):
        return [explicit]
    named = [column for column in numeric_columns(df) if _question_mentions(question, column)]
    varying = []
    for column in named or numeric_columns(df):
        series = pd.to_numeric(df[column], errors="coerce")
        if series.nunique(dropna=True) > 1 or series.notna().any():
            varying.append(column)
    return varying[:3]


def analyze_group_by_frame(
    df: pd.DataFrame,
    question: str = "",
    group_column: str = "",
    value_column: str = "",
) -> dict[str, Any]:
    if len(df) == 0:
        return {
            "group_column": None,
            "value_columns": [],
            "groups": [],
            "selection_reason": "The dataset has no rows.",
            "skipped_reason": "The dataset has no rows, so group comparisons were not calculated.",
        }
    selected, reason = choose_group_column(df, question, group_column)
    values = choose_value_columns(df, question, value_column)
    if selected is None:
        return {
            "group_column": None,
            "value_columns": values,
            "groups": [],
            "selection_reason": reason,
            "skipped_reason": reason,
        }
    if not values:
        return {
            "group_column": selected,
            "value_columns": [],
            "groups": [],
            "selection_reason": reason,
            "skipped_reason": "No numeric value column was available to aggregate.",
        }
    unique = int(df[selected].nunique(dropna=True))
    if unique < 2:
        return {
            "group_column": selected,
            "value_columns": values,
            "groups": [],
            "selection_reason": reason,
            "skipped_reason": "The grouping column has fewer than 2 distinct values.",
        }
    grouped = df.groupby(df[selected].fillna("(missing)").astype(str), dropna=False)
    records = []
    for name, part in grouped:
        metrics = {}
        for column in values:
            series = pd.to_numeric(part[column], errors="coerce").dropna()
            metrics[column] = {
                "count": int(series.count()),
                "mean": num(series.mean()) if len(series) else None,
                "median": num(series.median()) if len(series) else None,
            }
        records.append({"group": str(name), "count": int(len(part)), "metrics": metrics})
    records.sort(key=lambda item: item["count"], reverse=True)
    truncated = len(records) > 12
    return {
        "group_column": selected,
        "value_columns": values,
        "groups": records[:12],
        "selection_reason": reason,
        "skipped_reason": None,
        "truncated": truncated,
    }


def analyze_trend_frame(df: pd.DataFrame) -> dict[str, Any]:
    empty = {
        "date_column": None,
        "frequency": None,
        "periods": [],
        "directions": [],
        "skipped_reason": "No date column was detected.",
    }
    if len(df) == 0:
        empty["skipped_reason"] = "The dataset has no rows, so no trend was calculated."
        return empty
    dates = date_columns(df)
    if not dates:
        return empty
    column = dates[0]
    parsed = _parse_dates(df[column])
    valid = df.loc[parsed.notna()].copy()
    valid["_period_dt"] = parsed.loc[parsed.notna()]
    if len(valid) < 2:
        empty["date_column"] = column
        empty["skipped_reason"] = "Fewer than 2 valid dates were available."
        return empty
    span_days = (valid["_period_dt"].max() - valid["_period_dt"].min()).days
    if span_days > 62:
        valid["_period"] = valid["_period_dt"].dt.to_period("M").astype(str)
        frequency = "month"
    else:
        valid["_period"] = valid["_period_dt"].dt.strftime("%Y-%m-%d")
        frequency = "day"
    value_columns = [name for name in numeric_columns(df) if valid[name].nunique(dropna=True) >= 1][:3]
    periods = []
    for period, part in valid.groupby("_period", sort=True):
        means = {}
        for name in value_columns:
            series = pd.to_numeric(part[name], errors="coerce").dropna()
            means[name] = num(series.mean()) if len(series) else None
        periods.append({"period": str(period), "row_count": int(len(part)), "means": means})
    directions = []
    if len(periods) >= 2:
        first, last = periods[0], periods[-1]
        for name in value_columns:
            start = first["means"].get(name)
            end = last["means"].get(name)
            directions.append(
                {
                    "column": name,
                    "first_period": first["period"],
                    "last_period": last["period"],
                    "first_mean": start,
                    "last_mean": end,
                    "direction": _direction(start, end),
                    "note": "Compares the first and last period means only. This is not a forecast.",
                }
            )
    return {
        "date_column": column,
        "frequency": frequency,
        "periods": periods,
        "directions": directions,
        "skipped_reason": None if periods else "Dates could not be grouped into periods.",
    }


def _direction(start: Any, end: Any) -> str:
    if start is None or end is None:
        return "insufficient"
    start_f = float(start)
    end_f = float(end)
    if start_f == 0:
        if end_f == 0:
            return "flat"
        return "increasing" if end_f > 0 else "decreasing"
    change = (end_f - start_f) / abs(start_f)
    if change > 0.05:
        return "increasing"
    if change < -0.05:
        return "decreasing"
    return "flat"


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, default=str)


def _frame_for_tool(dataset_id: str) -> pd.DataFrame:
    return get_dataframe(dataset_id)


@tool
def profile_dataset(dataset_id: str) -> str:
    """Profile row counts, types, missing values, duplicates, constant columns, invalid types, and IQR outliers."""
    return _dump(profile_dataset_frame(_frame_for_tool(dataset_id)))


@tool
def calculate_statistics(dataset_id: str) -> str:
    """Calculate numeric summary statistics and compact categorical distributions."""
    return _dump(calculate_statistics_frame(_frame_for_tool(dataset_id)))


@tool
def find_correlations(dataset_id: str) -> str:
    """Calculate Pearson correlations between non-constant numeric columns. Correlation is not causation."""
    return _dump(find_correlations_frame(_frame_for_tool(dataset_id)))


@tool
def analyze_group_by(
    dataset_id: str,
    question: str = "",
    group_column: str = "",
    value_column: str = "",
) -> str:
    """Aggregate numeric columns by a categorical column chosen from the question and schema."""
    frame = _frame_for_tool(dataset_id)
    return _dump(
        analyze_group_by_frame(
            frame,
            question=question,
            group_column=group_column,
            value_column=value_column,
        )
    )


@tool
def analyze_trend(dataset_id: str) -> str:
    """Describe period means when a date column exists. This is not a forecast."""
    return _dump(analyze_trend_frame(_frame_for_tool(dataset_id)))


TOOLS: dict[str, BaseTool] = {
    "profile_dataset": profile_dataset,
    "calculate_statistics": calculate_statistics,
    "find_correlations": find_correlations,
    "analyze_group_by": analyze_group_by,
    "analyze_trend": analyze_trend,
}


def _input_summary(tool_name: str, tool_input: dict[str, Any]) -> str:
    dataset_id = str(tool_input.get("dataset_id", ""))
    parts = [f"tool={tool_name}", f"dataset_id={dataset_id[:8]}"]
    question = str(tool_input.get("question", ""))
    if question:
        parts.append(f"question_length={len(question)}")
    if tool_input.get("group_column"):
        parts.append("group_column=set")
    if tool_input.get("value_column"):
        parts.append("value_column=set")
    return ", ".join(parts)


def _output_summary(payload: Any) -> str:
    if not isinstance(payload, dict):
        return "non_dict_output"
    parts: list[str] = []
    for key in ("row_count", "column_count", "skipped_reason", "group_column", "date_column"):
        if payload.get(key) not in (None, ""):
            value = payload.get(key)
            if key == "skipped_reason":
                parts.append("skipped=yes")
            else:
                parts.append(f"{key}={value}")
    if "pairs" in payload:
        parts.append(f"pairs={len(payload['pairs'])}")
    if "groups" in payload:
        parts.append(f"groups={len(payload['groups'])}")
    if "periods" in payload:
        parts.append(f"periods={len(payload['periods'])}")
    if "columns" in payload and isinstance(payload["columns"], list):
        parts.append(f"profiled_columns={len(payload['columns'])}")
    if "numeric" in payload and isinstance(payload["numeric"], dict):
        parts.append(f"numeric_columns={len(payload['numeric'])}")
    if "error" in payload:
        parts.append("error=yes")
    return ", ".join(parts)[:300] or "empty_summary"


def invoke_tool(tool_name: str, tool_input: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call a LangChain tool and record timing separately from any LLM call."""
    tool = TOOLS[tool_name]
    started = time.perf_counter()
    start_iso = datetime.now(timezone.utc).isoformat()
    success = False
    error_category = None
    payload: dict[str, Any]
    try:
        raw = tool.invoke(tool_input)
        payload = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(payload, dict):
            raise DataPilotError("The tool returned an unexpected payload.")
        success = True
    except DataPilotError as exc:
        error_category = exc.category
        payload = {"error": str(exc)}
    except Exception:
        error_category = "computation_error"
        payload = {"error": "The tool failed before it could return a result."}
    ended = time.perf_counter()
    record = {
        "tool_name": tool_name,
        "start_time": start_iso,
        "end_time": datetime.now(timezone.utc).isoformat(),
        "duration_ms": round((ended - started) * 1000, 3),
        "success": success,
        "error_category": error_category,
        "input_summary": _input_summary(tool_name, tool_input),
        "output_summary": _output_summary(payload),
    }
    return payload, record
