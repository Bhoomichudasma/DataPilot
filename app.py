"""DataPilot AI dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import MAX_COLUMNS, MAX_ROWS, MAX_UPLOAD_BYTES, groq_configured, redact_secrets
from src.data_tools import DatasetValidationError, load_csv_bytes
from src.evaluation import run_benchmark, summarize_run
from src.graph import run_analysis

SAMPLE_PATH = Path(__file__).resolve().parent / "sample_data" / "sample_sales.csv"

st.set_page_config(page_title="DataPilot AI", page_icon="DP", layout="wide")
st.title("DataPilot AI")
st.caption("Upload a CSV, ask a question, and review a multi-agent report with evaluation metrics.")

with st.sidebar:
    st.header("Project overview")
    st.markdown(
        """
DataPilot AI profiles a CSV with two agents at the same time:

- **Data Quality Agent** checks missing values, duplicates, constants, invalid types, and IQR outliers.
- **Statistical Analysis Agent** calculates summaries, correlations, group comparisons, and date trends.
- **Insight Agent** writes the final report with Groq when an API key is configured.

Evidence checks compare numbers in the report with tool results. They do not prove that a sentence is semantically correct.
        """
    )
    if groq_configured():
        st.success("Groq API key detected.")
    else:
        st.warning("No Groq API key. Deterministic tool results still run.")
    st.caption(
        f"Limits: {MAX_UPLOAD_BYTES // (1024 * 1024)} MB, {MAX_ROWS} rows, {MAX_COLUMNS} columns."
    )


def load_sample() -> pd.DataFrame:
    return load_csv_bytes(SAMPLE_PATH.read_bytes(), SAMPLE_PATH.name)


def render_chart(spec: dict) -> None:
    title = spec.get("title", "Chart")
    if spec.get("type") == "heatmap":
        matrix = [
            [float("nan") if value is None else value for value in row]
            for row in spec.get("matrix") or []
        ]
        figure = px.imshow(
            matrix,
            x=spec.get("labels"),
            y=spec.get("labels"),
            text_auto=True,
            color_continuous_scale="Blues",
            title=title,
            aspect="auto",
        )
    elif spec.get("type") == "line":
        figure = px.line(
            x=spec.get("x"),
            y=spec.get("y"),
            markers=True,
            title=title,
            labels={"x": spec.get("x_label", ""), "y": spec.get("y_label", "")},
        )
    else:
        figure = px.bar(
            x=spec.get("x"),
            y=spec.get("y"),
            title=title,
            labels={"x": spec.get("x_label", ""), "y": spec.get("y_label", "")},
        )
    figure.update_layout(margin=dict(l=10, r=10, t=48, b=10))
    st.plotly_chart(figure, width="stretch")


def metric_or_na(value, suffix=""):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.1f}{suffix}"
    return f"{value}{suffix}"


left, right = st.columns([1.1, 1])
with left:
    st.subheader("CSV upload and preview")
    uploaded = st.file_uploader("Upload a CSV", type=["csv"])
    if st.button("Use sample sales data"):
        st.session_state["sample_mode"] = True
    if uploaded is not None:
        st.session_state["sample_mode"] = False
with right:
    st.subheader("Question")
    question = st.text_area(
        "Optional natural-language question",
        placeholder="Which region has the highest average revenue?",
        height=120,
    )
    run_clicked = st.button("Run analysis", type="primary")

frame = None
source_name = None
load_error = None
try:
    if uploaded is not None:
        frame = load_csv_bytes(uploaded.getvalue(), uploaded.name)
        source_name = uploaded.name
    elif st.session_state.get("sample_mode"):
        frame = load_sample()
        source_name = SAMPLE_PATH.name
except DatasetValidationError as exc:
    load_error = str(exc)

if load_error:
    st.error(load_error)
elif frame is not None:
    st.caption(f"Loaded `{source_name}` with {len(frame)} rows and {frame.shape[1]} columns.")
    st.dataframe(frame.head(15), width="stretch")
else:
    st.info("Upload a CSV or use the sample sales file to begin.")

if run_clicked:
    if frame is None:
        st.error("Load a CSV before running the analysis.")
    else:
        with st.spinner("Running the quality and statistics agents in parallel, then building the report..."):
            try:
                st.session_state["analysis"] = run_analysis(frame, question)
                st.session_state["analysis_question"] = question
                st.session_state["analysis_source"] = source_name
                st.session_state["analysis_rows"] = int(len(frame))
            except Exception as exc:
                st.session_state["analysis"] = None
                st.error("The analysis could not be completed. " + redact_secrets(str(exc)))

analysis = st.session_state.get("analysis")
if analysis:
    profile = (analysis.get("quality_result") or {}).get("profile") or {}
    stats = analysis.get("stats_result") or {}
    report = analysis.get("report") or {}
    run_metrics = summarize_run(analysis)

    if (
        question != st.session_state.get("analysis_question")
        or source_name != st.session_state.get("analysis_source")
        or (frame is not None and int(len(frame)) != st.session_state.get("analysis_rows"))
    ):
        st.warning("The report below is from the previous run. Click Run analysis to refresh it.")

    st.subheader("Dataset summary")
    cards = st.columns(4)
    cards[0].metric("Rows", profile.get("row_count", "N/A"))
    cards[1].metric("Columns", profile.get("column_count", "N/A"))
    cards[2].metric("Duplicate rows", profile.get("duplicate_row_count", "N/A"))
    missing_total = sum(column.get("missing_count", 0) for column in profile.get("columns") or [])
    cards[3].metric("Missing values", missing_total)
    st.caption(f"Question: {st.session_state.get('analysis_question') or 'No question was entered.'}")

    st.subheader("Data-quality findings")
    issues = (analysis.get("quality_result") or {}).get("issues") or []
    if issues:
        st.dataframe(pd.DataFrame(issues), width="stretch")
    elif (analysis.get("quality_result") or {}).get("error"):
        st.error("Data Quality Agent failed. " + str((analysis.get("quality_result") or {}).get("error")))
    else:
        st.write("No quality findings were returned.")

    st.subheader("Interactive charts")
    charts = list((analysis.get("quality_result") or {}).get("charts") or [])
    charts.extend(stats.get("charts") or [])
    if charts:
        chart_columns = st.columns(2)
        for index, spec in enumerate(charts):
            with chart_columns[index % 2]:
                render_chart(spec)
    else:
        st.write("No charts were produced for this dataset.")

    st.subheader("Statistical analysis")
    numeric = (stats.get("statistics") or {}).get("numeric") or {}
    if numeric:
        st.markdown("**Numeric summaries**")
        st.dataframe(pd.DataFrame(numeric).T, width="stretch")
    correlations = (stats.get("correlations") or {}).get("pairs") or []
    if correlations:
        st.markdown("**Correlations**")
        st.caption((stats.get("correlations") or {}).get("note"))
        st.dataframe(pd.DataFrame(correlations), width="stretch")
    elif (stats.get("correlations") or {}).get("skipped_reason"):
        st.info(stats["correlations"]["skipped_reason"])
    groups = (stats.get("group_by") or {}).get("groups") or []
    if groups:
        st.markdown(f"**Group comparison: `{stats['group_by'].get('group_column')}`**")
        flat_groups = []
        for row in groups:
            flat = {"group": row["group"], "count": row["count"]}
            for name, metric in (row.get("metrics") or {}).items():
                flat[f"{name} mean"] = metric.get("mean")
                flat[f"{name} median"] = metric.get("median")
            flat_groups.append(flat)
        st.dataframe(pd.DataFrame(flat_groups), width="stretch")
    elif (stats.get("group_by") or {}).get("skipped_reason"):
        st.info(stats["group_by"]["skipped_reason"])
    periods = (stats.get("trend") or {}).get("periods") or []
    if periods:
        st.markdown("**Trend periods**")
        st.caption("First-to-last period comparison. This is not a forecast.")
        st.dataframe(pd.DataFrame(periods), width="stretch")
    for highlight in stats.get("highlights") or []:
        st.write("- " + highlight)

    st.subheader("Final report")
    source = analysis.get("insight_source")
    if source == "groq":
        st.success(f"Insight source: Groq ({report.get('model')}).")
    elif source == "deterministic_fallback":
        st.error("The Groq request failed. The report below uses deterministic tool results.")
    else:
        st.info("Insight source: deterministic tool results. No Groq response was requested for this run.")
    st.write(report.get("answer") or "No answer was produced.")
    findings = report.get("findings") or []
    if findings:
        st.dataframe(pd.DataFrame(findings), width="stretch")

    st.subheader("Agent execution and tool calls")
    branch = analysis.get("branch_status") or {}
    exec_cols = st.columns(4)
    exec_cols[0].metric("Quality branch", branch.get("quality", "N/A"))
    exec_cols[1].metric("Statistics branch", branch.get("statistics", "N/A"))
    exec_cols[2].metric(
        "Quality node ms",
        metric_or_na((analysis.get("quality_timing") or {}).get("duration_ms")),
    )
    exec_cols[3].metric(
        "Statistics node ms",
        metric_or_na((analysis.get("stats_timing") or {}).get("duration_ms")),
    )
    if analysis.get("errors"):
        for message in analysis["errors"]:
            st.warning(redact_secrets(str(message)))
    logs = analysis.get("tool_logs") or []
    if logs:
        st.dataframe(pd.DataFrame(logs), width="stretch")
    else:
        st.write("No tool calls were recorded.")

    st.subheader("Evaluation metrics for this run")
    st.caption(analysis.get("timing_note"))
    success = run_metrics["tool_success_rate"]
    latency = run_metrics["tool_latency"]
    end_to_end = run_metrics["end_to_end"]
    usage = run_metrics["llm_usage"]
    metric_cols = st.columns(4)
    metric_cols[0].metric("Tool success rate", success["display"])
    metric_cols[1].metric("Average tool latency", metric_or_na(latency["average_ms"], " ms"))
    metric_cols[2].metric("Median tool latency", metric_or_na(latency["median_ms"], " ms"))
    metric_cols[3].metric("P95 tool latency", metric_or_na(latency["p95_ms"], " ms"))
    st.caption(latency["note"])
    end_cols = st.columns(3)
    end_cols[0].metric("End-to-end latency", metric_or_na(end_to_end["average_ms"], " ms"))
    end_cols[1].metric("LLM calls", usage["llm_calls"] if usage["llm_calls"] is not None else "N/A")
    token_text = "N/A"
    if usage["tokens_display"] == "reported":
        token_text = f"in {usage['input_tokens']} / out {usage['output_tokens']}"
    end_cols[2].metric("Tokens", token_text)
    cost = usage["estimated_cost_usd"]
    st.caption("Estimated cost: " + (f"${cost}" if cost is not None else "N/A. " + str(usage["cost_note"])))
    st.caption(end_to_end["note"])
    if latency.get("per_tool"):
        latency_frame = pd.DataFrame(latency["per_tool"])
        figure = px.bar(
            latency_frame,
            x="tool_name",
            y="average_ms",
            title="Average Python tool latency",
            labels={"tool_name": "Tool", "average_ms": "Milliseconds"},
        )
        st.plotly_chart(figure, width="stretch")
        st.caption("LLM response time is not included in this chart.")

st.divider()
st.subheader("Benchmark")
st.write(
    "The benchmark runs 10 fixed questions on synthetic tables, 20 numerical assertions, "
    "and 6 reviewed claims. It does not call Groq and it does not save uploaded data."
)
if st.button("Run built-in benchmark"):
    with st.spinner("Running the reproducible benchmark..."):
        try:
            st.session_state["benchmark"] = run_benchmark(save=True)
        except Exception as exc:
            st.error("The benchmark could not be completed. " + redact_secrets(str(exc)))

benchmark = st.session_state.get("benchmark")
if benchmark:
    tasks = benchmark["task_completion"]
    numerical = benchmark["numerical_correctness"]
    evidence = benchmark["evidence_review"]
    success = benchmark["tool_success_rate"]
    latency = benchmark["tool_latency"]
    end_to_end = benchmark["end_to_end"]
    bench_cols = st.columns(4)
    bench_cols[0].metric("Cases passed", f"{tasks['passed']} / {tasks['total']}")
    bench_cols[1].metric("Cases failed", tasks["failed"])
    bench_cols[2].metric("Numerical pass rate", f"{numerical['rate_percent']:.1f}%")
    bench_cols[3].metric("Task completion", f"{tasks['rate_percent']:.1f}%")
    more_cols = st.columns(4)
    more_cols[0].metric("Tool success", success["display"])
    more_cols[1].metric("Tool latency avg", metric_or_na(latency["average_ms"], " ms"))
    more_cols[2].metric("Tool latency p95", metric_or_na(latency["p95_ms"], " ms"))
    more_cols[3].metric("End-to-end median", metric_or_na(end_to_end["median_ms"], " ms"))
    st.caption(tasks["method"])
    st.caption(numerical["scope"])
    st.caption(end_to_end["note"])
    st.caption(evidence["note"])
    st.caption(
        f"Evidence review: {evidence['passed']} of {evidence['total']} claims passed "
        f"({evidence['rate_percent']}%). LLM tokens and cost for this benchmark: N/A."
    )
    if benchmark.get("saved_to"):
        st.caption("Saved results: " + benchmark["saved_to"])
    case_frame = pd.DataFrame(tasks["cases"])
    case_frame["passed_flag"] = case_frame["completed"].map({True: 1, False: 0})
    st.dataframe(
        case_frame[["id", "completed", "required_tools_succeeded", "missing_evidence", "validation_passed"]],
        width="stretch",
    )
    figure = px.bar(
        case_frame,
        x="id",
        y="passed_flag",
        title="Benchmark cases",
        labels={"passed_flag": "Passed (1) or failed (0)", "id": "Case"},
    )
    st.plotly_chart(figure, width="stretch")
    st.markdown("**Reviewed claims**")
    st.dataframe(pd.DataFrame(evidence["reviews"]), width="stretch")
    if latency.get("per_tool"):
        figure = px.bar(
            pd.DataFrame(latency["per_tool"]),
            x="tool_name",
            y="average_ms",
            title="Benchmark tool latency",
            labels={"tool_name": "Tool", "average_ms": "Milliseconds"},
        )
        st.plotly_chart(figure, width="stretch")
