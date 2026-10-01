"""Small, explainable forecasting workflow for user-uploaded CSV time series."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd


FREQUENCIES = {
    "Daily": {"rule": "D", "season": 7},
    "Weekly": {"rule": "W-SUN", "season": 52},
    "Monthly": {"rule": "MS", "season": 12},
}


def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    """Read a CSV with a helpful error for common encoding issues."""
    try:
        return pd.read_csv(BytesIO(raw))
    except Exception as exc:  # pandas raises several parser/encoding exceptions
        raise ValueError(f"I could not read this CSV. Check that it is a valid, comma-separated file. Details: {exc}") from exc


def prepare_time_series(
    frame: pd.DataFrame,
    date_column: str,
    value_column: str,
    series_columns: str | list[str] | None,
    frequency: str,
    aggregation: str = "Sum",
    missing_periods: str = "Fill with zero",
) -> pd.DataFrame:
    """Convert a tidy CSV to one regular row per date and series."""
    if frequency not in FREQUENCIES:
        raise ValueError("Choose Daily, Weekly, or Monthly data frequency.")
    if date_column == value_column:
        raise ValueError("Choose different columns for the date and quantity.")
    if date_column not in frame or value_column not in frame:
        raise ValueError("The selected date or quantity column is not in the uploaded file.")
    if isinstance(series_columns, str):
        series_columns = [series_columns]
    series_columns = list(series_columns or [])
    if any(column not in frame for column in series_columns):
        raise ValueError("One of the selected ID columns is not in the uploaded file.")
    if date_column in series_columns or value_column in series_columns:
        raise ValueError("ID columns must be different from the date and quantity columns.")

    work = pd.DataFrame(index=frame.index)
    work["date"] = pd.to_datetime(frame[date_column], errors="coerce")
    work["value"] = pd.to_numeric(frame[value_column], errors="coerce")
    invalid = work["date"].isna() | work["value"].isna() | ~np.isfinite(work["value"])
    if invalid.any():
        raise ValueError(
            f"{int(invalid.sum())} row(s) have a date that cannot be read or a non-numeric quantity. "
            "Fix or remove those rows, then upload the CSV again."
        )
    if (work["value"] < 0).any():
        raise ValueError("This version forecasts nonnegative quantities such as sales, demand, or visits. Negative values were found.")
    if series_columns:
        work["series"] = frame[series_columns].fillna("Unknown").astype(str).agg(" | ".join, axis=1).to_numpy()
    else:
        work["series"] = "All data"

    agg = "mean" if aggregation == "Mean" else "sum"
    rule = FREQUENCIES[frequency]["rule"]
    records: list[pd.DataFrame] = []
    for series_name, group in work.groupby("series", sort=True):
        values = group.set_index("date")["value"].sort_index().resample(rule).agg(agg)
        if missing_periods == "Carry last value":
            values = values.ffill().fillna(0.0)
        else:
            values = values.fillna(0.0)
        records.append(
            pd.DataFrame(
                {
                    "series": str(series_name),
                    "date": values.index,
                    "actual": values.to_numpy(dtype=float),
                }
            )
        )
    if not records:
        raise ValueError("No usable time-series rows were found in the CSV.")
    return pd.concat(records, ignore_index=True).sort_values(["series", "date"]).reset_index(drop=True)


def _feature_row(history: np.ndarray, date: pd.Timestamp, period: int, frequency: str) -> dict[str, float]:
    date = pd.Timestamp(date)
    row = {
        "lag_1": float(history[-1]),
        "lag_season": float(history[-period]),
        "rolling_season": float(np.mean(history[-period:])),
        "rolling_long": float(np.mean(history[-min(4 * period, len(history)) :])),
        "time_index": float(len(history)),
        "month_sin": float(np.sin(2 * np.pi * (date.month - 1) / 12)),
        "month_cos": float(np.cos(2 * np.pi * (date.month - 1) / 12)),
    }
    if frequency == "Daily":
        row["weekday_sin"] = float(np.sin(2 * np.pi * date.dayofweek / 7))
        row["weekday_cos"] = float(np.cos(2 * np.pi * date.dayofweek / 7))
    elif frequency == "Weekly":
        week = int(date.isocalendar().week)
        row["year_week_sin"] = float(np.sin(2 * np.pi * (week - 1) / 52))
        row["year_week_cos"] = float(np.cos(2 * np.pi * (week - 1) / 52))
    return row


def _fit_recursive_lightgbm(
    history: np.ndarray,
    dates: pd.DatetimeIndex,
    future_dates: pd.DatetimeIndex,
    frequency: str,
) -> tuple[np.ndarray | None, str]:
    """Fit a one-step model and roll it forward without seeing future actuals."""
    period = FREQUENCIES[frequency]["season"]
    x_rows: list[dict[str, float]] = []
    targets: list[float] = []
    for target_index in range(period, len(history)):
        x_rows.append(_feature_row(history[:target_index], dates[target_index], period, frequency))
        targets.append(float(history[target_index]))
    minimum_rows = max(30, period)
    if len(x_rows) < minimum_rows:
        return None, f"LightGBM needs at least {minimum_rows} usable training rows; this series has {len(x_rows)}."

    try:
        import lightgbm as lgb
    except ImportError:
        return None, "LightGBM is unavailable in this installation, so only the simple seasonal forecast can be shown."

    train_x = pd.DataFrame(x_rows)
    dataset = lgb.Dataset(train_x, label=np.asarray(targets), free_raw_data=False)
    model = lgb.train(
        {
            "objective": "regression_l2",
            "metric": "l1",
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_data_in_leaf": 8,
            "feature_fraction": 0.9,
            "lambda_l2": 2.0,
            "verbosity": -1,
            "num_threads": 2,
            "seed": 42,
            "deterministic": True,
            "force_col_wise": True,
        },
        dataset,
        num_boost_round=150,
    )

    rolling_history = history.astype(float).tolist()
    predictions = []
    for date in future_dates:
        features = pd.DataFrame([_feature_row(np.asarray(rolling_history), date, period, frequency)])
        prediction = max(0.0, float(model.predict(features)[0]))
        predictions.append(prediction)
        rolling_history.append(prediction)
    return np.asarray(predictions, dtype=float), f"Trained on {len(x_rows):,} earlier observations."


def _seasonal_naive(history: np.ndarray, horizon: int, period: int) -> np.ndarray:
    if len(history) >= period:
        pattern = history[-period:]
    else:
        pattern = history[-1:]
    return np.maximum(0.0, np.resize(pattern, horizon).astype(float))


def _wape(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    denominator = float(np.abs(actual).sum())
    if denominator == 0:
        return None
    return float(np.abs(actual - predicted).sum() / denominator)


def _future_dates(last_date: pd.Timestamp, frequency: str, horizon: int) -> pd.DatetimeIndex:
    rule = FREQUENCIES[frequency]["rule"]
    return pd.date_range(start=pd.Timestamp(last_date), periods=horizon + 1, freq=rule)[1:]


def run_forecast(series_frame: pd.DataFrame, frequency: str, horizon: int) -> dict[str, object]:
    """Backtest on the most recent horizon, then forecast the next horizon."""
    frame = series_frame.sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise ValueError("This series has duplicate dates. Check your date mapping and aggregation setting.")
    values = frame["actual"].to_numpy(dtype=float)
    if horizon < 1 or len(values) <= horizon:
        raise ValueError("The forecast horizon must be shorter than the available series history.")
    period = FREQUENCIES[frequency]["season"]
    split = len(values) - horizon
    train_values = values[:split]
    train_dates = pd.DatetimeIndex(frame.loc[: split - 1, "date"])
    test_dates = pd.DatetimeIndex(frame.loc[split:, "date"])

    baseline_test = _seasonal_naive(train_values, horizon, period)
    ml_test, ml_status = _fit_recursive_lightgbm(train_values, train_dates, test_dates, frequency)
    actual_test = values[split:]
    wape_baseline = _wape(actual_test, baseline_test)
    wape_ml = _wape(actual_test, ml_test) if ml_test is not None else None
    selected_model = "LightGBM" if wape_ml is not None and (wape_baseline is None or wape_ml < wape_baseline) else "Seasonal baseline"
    selected_test = ml_test if selected_model == "LightGBM" else baseline_test

    residuals = actual_test - selected_test
    lower_residual, upper_residual = np.quantile(residuals, [0.10, 0.90])
    backtest = pd.DataFrame(
        {
            "date": test_dates,
            "actual": actual_test,
            "seasonal_baseline": baseline_test,
            "lightgbm": ml_test if ml_test is not None else np.full(horizon, np.nan),
        }
    )

    next_dates = _future_dates(frame["date"].iloc[-1], frequency, horizon)
    baseline_future = _seasonal_naive(values, horizon, period)
    ml_future, full_ml_status = _fit_recursive_lightgbm(
        values, pd.DatetimeIndex(frame["date"]), next_dates, frequency
    )
    # Keep the held-out model choice unchanged for the future display.
    if selected_model == "LightGBM" and ml_future is not None:
        selected_future = ml_future
    else:
        selected_model = "Seasonal baseline"
        selected_future = baseline_future
    future = pd.DataFrame(
        {
            "date": next_dates,
            "seasonal_baseline": baseline_future,
            "lightgbm": ml_future if ml_future is not None else np.full(horizon, np.nan),
            "selected_forecast": selected_future,
        }
    )
    future["lower80"] = np.maximum(0.0, future["selected_forecast"] + lower_residual)
    future["upper80"] = np.maximum(future["lower80"], future["selected_forecast"] + upper_residual)
    return {
        "backtest": backtest,
        "future": future,
        "selected_model": selected_model,
        "baseline_wape": wape_baseline,
        "lightgbm_wape": wape_ml,
        "selected_wape": _wape(actual_test, selected_test),
        "baseline_mae": float(np.mean(np.abs(actual_test - baseline_test))),
        "lightgbm_mae": float(np.mean(np.abs(actual_test - ml_test))) if ml_test is not None else None,
        "ml_status": ml_status,
        "future_ml_status": full_ml_status,
        "interval_note": "The range uses errors from the recent backtest. It is a rough guide, not a guarantee.",
        "horizon": horizon,
    }
