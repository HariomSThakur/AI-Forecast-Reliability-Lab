"""Guided Streamlit app for the M5 experiment and uploaded time-series CSVs."""

from __future__ import annotations

import hashlib
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retail_forecast.demo import make_demo_data
from retail_forecast.evaluation import regression_metrics
from retail_forecast.upload_forecast import (
    FREQUENCIES,
    prepare_time_series,
    read_csv_bytes,
    run_forecast,
)

ARTIFACTS = ROOT / "artifacts"


@st.cache_data(show_spinner=False)
def _load_artifacts() -> tuple[pd.DataFrame, dict, bool]:
    predictions_path = ARTIFACTS / "predictions.parquet"
    metrics_path = ARTIFACTS / "metrics.json"
    if predictions_path.exists() and metrics_path.exists():
        predictions = pd.read_parquet(predictions_path)
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        return predictions, metrics, False
    return make_demo_data(), {}, True


def main() -> None:
    st.set_page_config(page_title="Retail Forecast Lab", page_icon="📈", layout="wide")
    _apply_visual_theme()
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand"><span>Forecast Lab</span><small>RETAIL</small></div>',
            unsafe_allow_html=True,
        )
        section = st.radio(
            "Dashboard section",
            ["Case study", "Your CSV"],
            label_visibility="collapsed",
            format_func=lambda value: "◌   Case study" if value == "Case study" else "↥   Your CSV",
            key="dashboard_section",
        )
        st.markdown(
            '<div class="sidebar-note">A practice test on historical demand — not a live retailer forecast. '
            "Uploaded CSVs are processed in memory; avoid sensitive data on a public deployment.</div>",
            unsafe_allow_html=True,
        )
    if section == "Case study":
        _show_m5_experiment()
    else:
        _show_upload_forecast()


def _show_m5_experiment() -> None:
    st.markdown('<div class="section-kicker">01 / HISTORICAL CASE STUDY</div>', unsafe_allow_html=True)
    st.markdown(
        '<h1 class="page-title">When does a model earn more than last week’s pattern?</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="page-lead">We hide the last 28 days of sampled M5 sales, ask a weekday-repeating baseline '
        "and a LightGBM model to predict them, then compare both with what actually sold. This is a practice "
        "test on past data — not a live retailer forecast.</p>",
        unsafe_allow_html=True,
    )
    with st.expander("How does this comparison work?", expanded=False):
        st.markdown(
            "- **Simple baseline:** repeats the recent pattern for each weekday.\n"
            "- **LightGBM:** learns from recent and year-ago sales, dates, events, and prices. Its training length is selected using earlier development data, not the final test.\n"
            "- **Selected model:** uses development-period results to choose one of those methods for a demand group."
        )
    st.caption("M5 includes anonymous product codes and state/store codes, not product titles or city names. The dashboard keeps that distinction clear.")

    if st.button("Refresh saved M5 results", help="Use this after the data pipeline finishes while the app is open."):
        _load_artifacts.clear()
        st.rerun()
    predictions, report, is_demo = _load_artifacts()
    if is_demo:
        st.warning(
            "This is a generated preview so you can explore the app. These numbers are not results from the M5 data. "
            "Run the pipeline and refresh this page to load the real experiment."
        )
    else:
        fold = int(report.get("final", {}).get("fold", predictions["fold"].max()))
        predictions = predictions[predictions["fold"].eq(fold)].copy()
        predictions["date"] = pd.to_datetime(predictions["date"])
        naive_wape = regression_metrics(predictions, "seasonal_naive_forecast").get("wape")
        ml_wape = regression_metrics(predictions, "lightgbm_forecast").get("wape")
        selected_wape = regression_metrics(predictions, "selected_forecast").get("wape")
        coverage = regression_metrics(predictions, "selected_forecast").get("interval_coverage")
        relative_change = (
            (naive_wape - ml_wape) / naive_wape
            if naive_wape is not None and ml_wape is not None and naive_wape != 0
            else None
        )
        metric_columns = st.columns(4)
        metric_columns[0].metric("Simple forecast error", _format_percent(naive_wape), help="WAPE: lower means the total forecast error was smaller.")
        metric_columns[1].metric(
            "ML forecast error",
            _format_percent(ml_wape),
            delta=f"{relative_change:+.1%} vs simple" if relative_change is not None else None,
            help="Positive means relative error is lower than the simple forecast; lower WAPE is better.",
        )
        metric_columns[2].metric("Chosen forecast error", _format_percent(selected_wape), help="This uses the model-selection rule learned from development folds.")
        metric_columns[3].metric("Actuals inside 80% range", _format_percent(coverage), help="Across the test rows, how often sales fell inside the estimated range.")
        if relative_change is not None:
            if relative_change > 0:
                _insight_callout(f"On this historical test, machine learning had about {relative_change:.1%} less error than the simple forecast.", "good")
            elif relative_change < 0:
                _insight_callout(f"On this historical test, the simple forecast did better; machine learning had {abs(relative_change):.1%} more error.", "warning")
            else:
                _insight_callout("The two forecasts had the same WAPE on this historical test.", "neutral")
        st.caption(
            f"This test covers {predictions['series_id'].nunique():,} sampled product-store series and "
            f"{len(predictions):,} product-days. WAPE is an error score, not an accuracy percentage."
        )

        st.subheader("Where does each approach work better?")
        st.caption("Choose a group below. Compare the two WAPE columns; the smaller number is better for that group.")
        segment_column = st.selectbox(
            "Group the comparison by",
            options=["demand_group", "store_id", "dept_id", "event_day"],
            format_func=lambda value: {
                "demand_group": "How often products sell",
                "store_id": "Store location (state-level only)",
                "dept_id": "Product group",
                "event_day": "Event vs regular day",
            }[value],
            key="m5_segment",
        )
        comparison_rows = []
        for segment_value, group in predictions.groupby(segment_column, dropna=False, sort=True):
            comparison_rows.append(
                {
                    "Group": (
                        _store_label(predictions, str(segment_value))
                        if segment_column == "store_id"
                        else _department_label(str(segment_value))
                        if segment_column == "dept_id"
                        else _segment_label(segment_column, segment_value)
                    ),
                    "Product-location series": int(group["series_id"].nunique()),
                    "Simple forecast WAPE (lower is better)": regression_metrics(group, "seasonal_naive_forecast").get("wape"),
                    "ML forecast WAPE (lower is better)": regression_metrics(group, "lightgbm_forecast").get("wape"),
                    "Model used": str(group["selected_model"].mode().iloc[0]).replace("_", " ").title(),
                }
            )
        _show_table(pd.DataFrame(comparison_rows))

        st.subheader("Look at one product in one location")
        st.caption("The controls only filter the saved test results; they do not change or retrain the model.")
        left, middle, right = st.columns(3)
        stores = sorted(predictions["store_id"].astype(str).unique())
        store = left.selectbox(
            "Location (state and anonymized store)",
            stores,
            format_func=lambda value: _store_label(predictions, value),
            key="m5_store",
        )
        store_rows = predictions[predictions["store_id"].astype(str).eq(store)]
        departments = sorted(store_rows["dept_id"].astype(str).unique())
        department = middle.selectbox(
            "Product group",
            departments,
            format_func=_department_label,
            key="m5_department",
        )
        department_rows = store_rows[store_rows["dept_id"].astype(str).eq(department)]
        items = sorted(department_rows["item_id"].astype(str).unique())
        item = right.selectbox("Product code (friendly label)", items, format_func=_item_label, key="m5_item")
        selected = department_rows[department_rows["item_id"].astype(str).eq(item)].sort_values("date")
        st.caption(
            "M5 hides real product titles and city names. These friendly labels explain its anonymous codes; "
            "they do not identify a particular item like milk or a particular city."
        )

        selected_model = str(selected["selected_model"].iloc[0]).replace("_", " ").title()
        selected_wape = regression_metrics(selected, "selected_forecast").get("wape")
        baseline_wape = regression_metrics(selected, "seasonal_naive_forecast").get("wape")
        metric_columns = st.columns(3)
        metric_columns[0].metric("Model chosen for this group", selected_model)
        metric_columns[1].metric("Chosen model error", _format_percent(selected_wape), help="Lower WAPE means the forecast was closer overall to the real sales.")
        metric_columns[2].metric("Simple forecast error", _format_percent(baseline_wape), help="Compare this with the chosen model's error; lower is better.")
        _forecast_line_chart(
            selected,
            x_column="date",
            series_columns={
                "What actually sold": "actual_units",
                "Chosen forecast": "selected_forecast",
                "Simple forecast": "seasonal_naive_forecast",
                "ML forecast": "lightgbm_forecast",
            },
            y_title="Units sold",
        )
        st.caption("Compare each forecast line with what actually sold. Estimated 80% range bounds are listed in the detailed values below.")
        st.subheader("Does the error change further into the future?")
        horizon_rows = []
        for horizon, group in selected.groupby("horizon", sort=True):
            horizon_rows.append(
                {
                    "Days ahead": int(horizon),
                    "Chosen forecast WAPE": regression_metrics(group, "selected_forecast").get("wape"),
                    "Simple forecast WAPE": regression_metrics(group, "seasonal_naive_forecast").get("wape"),
                    "ML forecast WAPE": regression_metrics(group, "lightgbm_forecast").get("wape"),
                }
            )
        _show_table(pd.DataFrame(horizon_rows))
        with st.expander("Daily forecast values and project limitations"):
            visible_columns = ["date", "horizon", "actual_units", "seasonal_naive_forecast", "lightgbm_forecast", "selected_model", "selected_forecast", "lower80", "upper80"]
            visible = selected[visible_columns].copy()
            _show_table(visible)
            st.download_button("Download these results as CSV", data=visible.to_csv(index=False).encode("utf-8"), file_name=f"forecast-{item}.csv", mime="text/csv")
            st.write(
                "The 80% range is estimated from earlier forecast errors; it is not a guarantee. "
                "M5 records sales, which can be lower than true demand when a product is out of stock. "
                "The historical test does not prove the same result will hold for every retailer or future period."
            )
        with st.expander("How could a future experiment reduce the error?"):
            st.markdown(
                "1. **Add stronger business signals:** promotions, stock availability, and planned price changes can explain sales spikes that past sales alone miss.\n"
                "2. **Use more history and series:** rerun the same time-based folds with a larger sample, then check whether gains hold across stores and product groups. More data helps confidence, but does not guarantee a lower score.\n"
                "3. **Investigate hard segments:** `HOBBIES_2` is a case where the development-selected LightGBM model lost slightly to the simple baseline on the final test. More development windows could show whether that is a stable pattern.\n"
                "4. **Protect the final test:** choose features and model settings on development folds only; use the final fold once to report the result."
            )


def _show_upload_forecast() -> None:
    st.markdown('<div class="section-kicker">02 / BRING YOUR OWN DATA</div>', unsafe_allow_html=True)
    st.markdown(
        '<h1 class="page-title">Try the same comparison on a CSV you already have.</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="page-lead">Upload a table with a date and a numeric quantity, such as sales, orders, or '
        "website visits. If you track several products or cities, map those ID columns so each combination "
        "is forecast on its own.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="stepper">
          <div class="step-card"><span>01</span><strong>Upload</strong><small>Add a CSV</small></div>
          <div class="step-card"><span>02</span><strong>Map columns</strong><small>Pick date and quantity</small></div>
          <div class="step-card"><span>03</span><strong>Choose a series</strong><small>Select product/location</small></div>
          <div class="step-card"><span>04</span><strong>Compare</strong><small>Test, then forecast</small></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    sample = _sample_csv()
    upload_actions = st.columns([1, 1])
    upload_actions[0].button(
        "Load sample sales",
        on_click=_activate_sample_csv,
        help="Explore the workflow immediately with invented grocery and city data.",
        key="load_sample_csv",
    )
    upload_actions[1].download_button(
        "Download sample CSV",
        sample,
        file_name="sample_sales.csv",
        mime="text/csv",
    )
    st.caption("The sample contains invented sales for Milk, Apples, Tomatoes, Bananas, and Bread in Mumbai, Pune, and Delhi. It is practice data, not the M5 retailer data.")
    st.caption("Required columns: a date and a nonnegative number. Optional: an ID such as product, store, or location. Dates are processed in memory and are not saved by this app; avoid uploading sensitive data to a public deployment.")
    upload_version = st.session_state.get("csv_upload_version", 0)
    uploaded = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key=f"user_csv_{upload_version}",
        on_change=_deactivate_sample_csv,
    )
    sample_active = st.session_state.get("sample_csv_active", False)
    if uploaded is None and not sample_active:
        st.info("Choose a CSV or load the sample sales to explore the workflow.")
        with st.expander("What should my CSV look like?"):
            st.code("date,sales,product,city\n2025-01-01,12,Milk,Mumbai\n2025-01-02,15,Milk,Mumbai\n2025-01-01,7,Bread,Pune", language="csv")
            st.write("If one product has many dates, repeat its product name on each row. The app can combine duplicate dates using sum or mean.")
        return

    if sample_active:
        raw = sample
        st.caption("Sample sales loaded. These made-up values are only for learning the workflow.")
    else:
        raw = uploaded.getvalue()
    try:
        raw_frame = read_csv_bytes(raw)
    except ValueError as exc:
        st.error(str(exc))
        return
    if len(raw_frame.columns) < 2:
        st.error("This CSV needs at least two columns: one date and one quantity.")
        return

    st.write("**Step 1 — Check your columns.** Here is a small preview of the uploaded file:")
    _show_table(raw_frame.head(8))
    columns = list(raw_frame.columns)
    date_default = _guess_column(columns, ("date", "day", "time", "ds"), 0)
    value_default = _guess_column(columns, ("sales", "demand", "quantity", "units", "value", "target", "y"), 1 if date_default == 0 else 0)
    controls = st.columns(3)
    date_col = controls[0].selectbox("Which column contains dates?", columns, index=date_default, key="csv_date")
    value_col = controls[1].selectbox("Which column contains the quantity?", columns, index=value_default, key="csv_value")
    id_options = ["No ID (one series)"] + [column for column in columns if column not in {date_col, value_col}]
    series_default_index = _guess_column(id_options, ("series", "series_id", "product_location", "product_store", "item_id"), 0)
    series_defaults = [id_options[series_default_index]] if series_default_index > 0 else []
    series_choice = controls[2].multiselect(
        "Which columns identify separate series? (optional; choose product + city if needed)",
        id_options[1:],
        default=series_defaults,
        key="csv_series_columns",
    )
    series_cols = list(series_choice) or None

    settings = st.columns(3)
    frequency = settings[0].selectbox("How often is each measurement?", list(FREQUENCIES), key="csv_frequency")
    aggregation = settings[1].selectbox("If several rows fall in one period, combine them by", ["Sum", "Mean"], help="Sum is common for sales/orders; mean is common for rates or measurements.", key="csv_aggregation")
    missing_periods = settings[2].selectbox("If a date is missing", ["Fill with zero", "Carry last value"], help="For sales, a missing row often means zero sales. Choose carry-forward for measurements where a missing row means no new reading.", key="csv_missing")

    try:
        panel = prepare_time_series(raw_frame, date_col, value_col, series_cols, frequency, aggregation, missing_periods)
    except ValueError as exc:
        st.error(str(exc))
        return
    series_names = sorted(panel["series"].astype(str).unique())
    chosen_series = st.selectbox("Which product or series do you want to forecast?", series_names, key="csv_chosen_series")
    series_frame = panel[panel["series"].astype(str).eq(chosen_series)].sort_values("date").reset_index(drop=True)
    max_horizon = max(1, min(90, (len(series_frame) - 1) // 3))
    preferred = {"Daily": 28, "Weekly": 8, "Monthly": 6}[frequency]
    default_horizon = min(preferred, max_horizon)
    horizon = st.slider("How many periods ahead should the forecast go?", 1, max_horizon, default_horizon, key="csv_horizon")
    st.caption(
        f"Ready: **{len(series_frame):,} {frequency.lower()} data points** for '{chosen_series}'. "
        "The app holds back the latest periods as a practice test, then forecasts the next periods."
    )
    signature = hashlib.sha256(
        raw + repr((date_col, value_col, series_cols, frequency, aggregation, missing_periods, chosen_series, horizon)).encode("utf-8")
    ).hexdigest()
    if st.button("Run my forecast", type="primary", key="run_csv_forecast"):
        with st.spinner("Checking the recent test period and building your forecast…"):
            try:
                result = run_forecast(series_frame, frequency, int(horizon))
            except Exception as exc:
                st.error(f"I couldn't build this forecast: {exc}")
                result = None
        if result is not None:
            st.session_state["upload_forecast_result"] = result
            st.session_state["upload_forecast_signature"] = signature

    if st.session_state.get("upload_forecast_signature") == signature:
        result = st.session_state["upload_forecast_result"]
        _show_uploaded_results(series_frame, result, frequency, chosen_series)
    else:
        st.info("Your file is ready. Press **Run my forecast** to see how the methods perform.")


def _show_uploaded_results(series_frame: pd.DataFrame, result: dict[str, object], frequency: str, series_name: str) -> None:
    st.header("Your results")
    ml_wape = result["lightgbm_wape"]
    baseline_wape = result["baseline_wape"]
    relative_change = (
        (baseline_wape - ml_wape) / baseline_wape
        if baseline_wape is not None and ml_wape is not None and baseline_wape != 0
        else None
    )
    metrics = st.columns(3)
    metrics[0].metric("Simple forecast error", _format_percent(baseline_wape), help="WAPE on the recent test period. Lower is better.")
    metrics[1].metric(
        "ML forecast error",
        _format_percent(ml_wape),
        delta=f"{relative_change:+.1%} vs simple" if relative_change is not None else None,
        help="Positive means relative error is lower than the simple forecast; lower WAPE is better.",
    )
    metrics[2].metric("Method used for future", str(result["selected_model"]))
    if ml_wape is None:
        _insight_callout(str(result["ml_status"]) + " The simple forecast is still available.", "neutral")
    elif result["selected_model"] == "LightGBM":
        _insight_callout("LightGBM had lower error on this recent test period, so the app uses it for the forecast below.", "good")
    else:
        _insight_callout("The simple forecast had lower error on this recent test period, so the app uses it below.", "neutral")
    st.caption("WAPE is a way to compare total forecast error with total actual quantity. Lower is better; it is not an accuracy percentage.")

    st.subheader("1. Did it predict the recent past well?")
    st.write(f"These are the last {result['horizon']} periods where actual values are already known. The model did not see those values when it made this test forecast.")
    backtest = result["backtest"].rename(
        columns={"actual": "What actually happened", "seasonal_baseline": "Simple forecast", "lightgbm": "LightGBM forecast"}
    )
    _forecast_line_chart(
        backtest,
        x_column="date",
        series_columns={
            "What actually happened": "What actually happened",
            "Simple forecast": "Simple forecast",
            "LightGBM forecast": "LightGBM forecast",
        },
        y_title="Quantity",
    )
    st.caption("A forecast line closer to 'What actually happened' is better. WAPE above compares the total errors over this test.")

    st.subheader(f"2. What could happen over the next {result['horizon']} {frequency.lower()} periods?")
    future = result["future"].copy()
    _forecast_line_chart(
        future,
        x_column="date",
        series_columns={
            "Forecast chosen by backtest": "selected_forecast",
            "Simple forecast": "seasonal_baseline",
            "LightGBM forecast": "lightgbm",
        },
        y_title="Forecast quantity",
    )
    st.caption(str(result["interval_note"]))
    visible = future.rename(
        columns={
            "date": "date",
            "selected_forecast": "forecast",
            "seasonal_baseline": "simple_forecast",
            "lightgbm": "lightgbm_forecast",
            "lower80": "estimated_low",
            "upper80": "estimated_high",
        }
    )
    _show_table(visible)
    st.download_button(
        "Download these future forecasts as CSV",
        visible.to_csv(index=False).encode("utf-8"),
        file_name=f"forecast-{_safe_filename(series_name)}.csv",
        mime="text/csv",
    )
    with st.expander("How does the app make this forecast?"):
        st.markdown(
            "- **Simple forecast:** repeats the latest seasonal pattern (weekly for daily data, yearly for weekly/monthly data).\n"
            "- **LightGBM:** learns from earlier values, recent averages, and calendar patterns, then predicts forward one period at a time.\n"
            "- **Method used:** chooses the method with lower WAPE on the recent test window. This is a useful quick comparison, "
            "but one test window cannot guarantee which model will win in the future.\n"
            "- **Estimated range:** uses the errors from the recent test. It is an approximate range, not a promise."
        )
        st.write(f"Series: {series_name}. {result['future_ml_status']}")
    with st.expander("How can I improve this uploaded-data forecast?"):
        st.markdown(
            "1. **Check the setup:** choose the right daily/weekly/monthly frequency, combine duplicate rows correctly, and decide whether a missing date means zero or a value carried forward.\n"
            "2. **Give it enough history:** multiple seasonal cycles usually help the model learn repeating patterns.\n"
            "3. **Add useful business signals:** this CSV version currently learns from dates and past quantities. It does not yet use extra columns like price, promotions, holidays, or stock availability as predictors; the app needs a feature-mapping change to use them.\n"
            "4. **Compare several time windows:** a single recent test can be unusually easy or hard. Multiple chronological tests give a more reliable estimate."
        )


def _forecast_line_chart(
    frame: pd.DataFrame,
    x_column: str,
    series_columns: dict[str, str],
    y_title: str,
) -> None:
    """Render the original, straightforward Streamlit line-chart comparison."""
    data = frame.copy()
    if x_column not in data.columns and data.index.name == x_column:
        data = data.reset_index()
    data[x_column] = pd.to_datetime(data[x_column])
    chart_data = data[[x_column, *series_columns.values()]].copy().set_index(x_column)
    chart_data.columns = list(series_columns)
    st.line_chart(chart_data, y_label=y_title, use_container_width=True)


def _show_table(frame: pd.DataFrame) -> None:
    """Format percentages and add gentle striping to tabular results."""
    numeric_columns = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    formats = {}
    for column in numeric_columns:
        label = str(column).lower()
        if "wape" in label or "coverage" in label:
            formats[column] = "{:.1%}"
        elif pd.api.types.is_integer_dtype(frame[column]):
            formats[column] = "{:,.0f}"
        else:
            formats[column] = "{:,.2f}"
    styled = (
        frame.style.format(formats, na_rep="—")
        .set_table_styles(
            [
                {"selector": "thead th", "props": [("background-color", "#202a36"), ("color", "#dce5ee"), ("font-weight", "700")]},
                {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#111923")]},
                {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", "#151f2b")]},
                {"selector": "tbody td", "props": [("border-bottom", "1px solid #2b3542")]},
            ]
        )
        .set_properties(subset=numeric_columns, **{"text-align": "right", "color": "#eef2f6"})
    )
    st.dataframe(styled, width="stretch", hide_index=True)


def _insight_callout(message: str, tone: str = "good") -> None:
    headings = {"good": ("⚡", "What the test says"), "warning": ("↗", "Worth investigating"), "neutral": ("ℹ", "Forecast note")}
    icon, heading = headings.get(tone, headings["neutral"])
    st.markdown(
        f'<div class="insight-card {tone}"><span class="insight-icon">{icon}</span>'
        f'<div><strong>{heading}</strong><p>{html.escape(message)}</p></div></div>',
        unsafe_allow_html=True,
    )


def _sample_csv() -> bytes:
    rng = np.random.default_rng(12)
    dates = pd.date_range("2025-01-01", periods=180, freq="D")
    rows = []
    products = {"Milk": 32, "Apples": 15, "Tomatoes": 18, "Bananas": 13, "Bread": 24}
    cities = {"Mumbai": 1.15, "Pune": 0.92, "Delhi": 1.08}
    for city, city_scale in cities.items():
        for product, level in products.items():
            weekend = np.asarray([2.0 if day >= 5 else 0.0 for day in dates.dayofweek])
            seasonal = 2.5 * np.sin(np.arange(len(dates)) * 2 * np.pi / 7)
            trend = np.linspace(0, 1.5, len(dates))
            noise = rng.normal(0, max(1.0, level * 0.08), len(dates))
            values = np.maximum(0, np.round(city_scale * level + weekend + seasonal + trend + noise))
            series = f"{city} | {product}"
            rows.extend(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "sales": int(value),
                    "product": product,
                    "city": city,
                    "series": series,
                }
                for date, value in zip(dates, values)
            )
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def _activate_sample_csv() -> None:
    """Switch the upload workflow to its built-in, clearly synthetic sample."""
    st.session_state["sample_csv_active"] = True
    st.session_state["csv_upload_version"] = st.session_state.get("csv_upload_version", 0) + 1


def _deactivate_sample_csv() -> None:
    """Prefer a newly uploaded file over the previous built-in sample."""
    st.session_state["sample_csv_active"] = False


def _apply_visual_theme() -> None:
    st.markdown(
        """
        <style>
          :root { color-scheme: dark; --ink: #edf2f7; --muted: #a0acbb; --subtle: #748195; --primary: #35c9a2; --line: rgba(180,194,210,.15); --paper: #0a0f16; --surface: #121a24; --surface-2: #1b2633; }
          body, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--paper); color: var(--ink); }
          [data-testid="stHeader"] { background: transparent; }
          .block-container { max-width: 1180px; padding-top: 2.8rem; padding-bottom: 4rem; }
          section[data-testid="stSidebar"] { width: 238px !important; min-width: 238px !important; background: #080d13; border-right: 1px solid var(--line); }
          section[data-testid="stSidebar"] > div { background: transparent; }
          [data-testid="stSidebar"] .block-container { padding: 1.6rem 1.1rem 1.4rem; }
          .sidebar-brand { display: flex; align-items: baseline; gap: 9px; margin: .15rem 0 2.5rem; white-space: nowrap; }
          .sidebar-brand span { color: var(--ink); font: 500 1.35rem/1 Georgia, "Palatino Linotype", serif; letter-spacing: -.045em; }
          .sidebar-brand small { color: var(--muted); font-size: .62rem; letter-spacing: .16em; }
          .sidebar-note { position: fixed; bottom: 1.4rem; width: 190px; color: var(--subtle); font-size: .72rem; line-height: 1.65; }
          [data-testid="stSidebar"] [data-testid="stRadio"] > label { display: none; }
          [data-testid="stSidebar"] [role="radiogroup"] { gap: .35rem; }
          [data-testid="stSidebar"] [role="radiogroup"] > label { min-height: 42px; padding: .65rem .8rem; border-radius: 11px; color: var(--muted); transition: background .15s ease, color .15s ease; }
          [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) { background: #1c4338; color: #d9fff2; }
          [data-testid="stSidebar"] [role="radiogroup"] > label:hover { background: var(--surface-2); color: var(--ink); }
          [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked):hover { background: #1c4338; color: #d9fff2; }
          .section-kicker { color: #72d7b6; font-size: .66rem; font-weight: 600; letter-spacing: .2em; margin: .15rem 0 1rem; text-transform: uppercase; }
          .page-title { max-width: 850px; color: var(--ink); font: 400 clamp(2.55rem, 5vw, 3.9rem)/1.08 Georgia, "Palatino Linotype", serif; letter-spacing: -.052em; margin: 0 0 1.2rem; }
          .page-lead { max-width: 790px; color: var(--muted); font-size: 1.02rem; line-height: 1.75; margin: 0 0 1.8rem; }
          h2, h3 { color: var(--ink); font-family: Georgia, "Palatino Linotype", serif; font-weight: 400 !important; letter-spacing: -.035em; }
          [data-testid="stMetric"] { min-height: 120px; background: var(--surface); border: 1px solid var(--line); border-radius: 22px; padding: 17px 19px; box-shadow: 0 10px 30px -20px rgba(0,0,0,.7); }
          [data-testid="stMetricLabel"] { color: var(--muted); font-size: .64rem; font-weight: 600; letter-spacing: .15em; text-transform: uppercase; }
          [data-testid="stMetricValue"] { color: var(--ink); font: 500 2rem/1.1 Georgia, "Palatino Linotype", serif; letter-spacing: -.045em; }
          [data-testid="stMetricDelta"] { font-size: .72rem; }
          .stepper { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; position: relative; margin: .8rem 0 1.5rem; }
          .step-card { min-height: 104px; padding: 16px; background: var(--surface); border: 1px solid var(--line); border-radius: 22px; box-shadow: 0 10px 28px -20px rgba(0,0,0,.7); position: relative; z-index: 1; }
          .step-card span { display: block; color: var(--muted); font: 500 .62rem/1.2 ui-monospace, Consolas, monospace; letter-spacing: .16em; margin-bottom: 12px; }
          .step-card strong { display: block; color: var(--ink); font-size: .9rem; font-weight: 600; }
          .step-card small { display: block; color: var(--muted); margin-top: 4px; font-size: .78rem; }
          [data-testid="stDataFrame"] { background: var(--surface); border: 1px solid var(--line); border-radius: 18px; overflow: hidden; }
          [data-testid="stAlert"] { background: var(--surface); border-radius: 18px; border-color: var(--line); color: var(--ink); }
          [data-testid="stExpander"] { background: var(--surface); border: 1px solid var(--line); border-radius: 18px; }
          .stButton > button, .stDownloadButton > button { min-height: 42px; border-radius: 11px; font-weight: 600; }
          .stButton > button[kind="primary"] { background: #1c6a54; border-color: #25876b; color: #effff9; }
          section[data-testid="stFileUploaderDropzone"] { border: 1.5px dashed rgba(180,194,210,.28) !important; border-radius: 22px !important; background: rgba(18,26,36,.8) !important; min-height: 150px; }
          section[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--primary) !important; background: var(--surface) !important; }
          .insight-card { display: flex; gap: 13px; align-items: flex-start; background: var(--surface); color: var(--ink); border: 1px solid var(--line); border-left: 3px solid #41c99c; border-radius: 18px; padding: 15px 17px; margin: .7rem 0 1rem; box-shadow: 0 10px 28px -20px rgba(0,0,0,.7); }
          .insight-icon { display: none; }
          .insight-card strong { color: var(--muted); font-size: .64rem; font-weight: 600; letter-spacing: .15em; text-transform: uppercase; }
          .insight-card p { color: var(--ink); margin: 4px 0 0; font-size: .88rem; }
          .insight-card.warning { background: var(--surface); color: var(--ink); border-color: var(--line); border-left-color: #f0a35e; box-shadow: 0 10px 28px -20px rgba(0,0,0,.7); }
          .insight-card.warning strong, .insight-card.warning p { color: var(--ink); }
          .insight-card.neutral { background: var(--surface); color: var(--ink); border-color: var(--line); border-left-color: #7ca9ff; box-shadow: 0 10px 28px -20px rgba(0,0,0,.7); }
          .insight-card.neutral strong, .insight-card.neutral p { color: var(--ink); }
          [data-baseweb="select"] > div, [data-baseweb="input"] > div { background: var(--surface); border-color: rgba(180,194,210,.2); border-radius: 11px; }
          [data-testid="stCaptionContainer"] { color: var(--muted); }
          hr { border-color: var(--line); }
          @media (max-width: 760px) { .block-container { padding-top: 1.6rem; } .stepper { grid-template-columns: repeat(2,1fr); } .sidebar-note { display: none; } .page-title { font-size: 2.45rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _guess_column(columns: list[str], terms: tuple[str, ...], fallback: int) -> int:
    lowered = [str(column).strip().lower() for column in columns]
    for term in terms:
        if term in lowered:
            return lowered.index(term)
    return min(fallback, len(columns) - 1)


def _segment_label(column: str, value: object) -> str:
    if column == "event_day":
        return "Event date" if bool(value) else "Regular date"
    if column == "demand_group":
        return {"low": "Sells infrequently", "medium": "Sells sometimes", "high": "Sells often"}.get(str(value), str(value))
    return str(value)


def _store_label(predictions: pd.DataFrame, store_id: str) -> str:
    state = predictions.loc[predictions["store_id"].astype(str).eq(str(store_id)), "state_id"]
    state_code = str(state.iloc[0]) if not state.empty else ""
    state_name = {"CA": "California", "TX": "Texas", "WI": "Wisconsin"}.get(state_code, state_code)
    store_number = str(store_id).rsplit("_", 1)[-1]
    return f"{state_name} location {store_number} ({store_id})"


def _department_label(department_id: str) -> str:
    family, _, number = str(department_id).partition("_")
    family_name = {"FOODS": "Food", "HOBBIES": "Hobby", "HOUSEHOLD": "Household"}.get(family, family.title())
    return f"{family_name} product group {number} ({department_id})"


def _item_label(item_id: str) -> str:
    parts = str(item_id).split("_")
    family_name = {"FOODS": "Food", "HOBBIES": "Hobby", "HOUSEHOLD": "Household"}.get(parts[0], parts[0].title())
    code = parts[-1] if parts else str(item_id)
    return f"{family_name} item {code} ({item_id})"


def _safe_filename(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    return cleaned.strip("-")[:60] or "series"


def _format_percent(value: float | None) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.1%}"


if __name__ == "__main__":
    main()
