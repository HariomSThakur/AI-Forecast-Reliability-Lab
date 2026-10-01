"""Metrics, development-only model routing, and empirical interval calibration."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

MODEL_COLUMNS = {
    "seasonal_naive": "seasonal_naive_forecast",
    "lightgbm": "lightgbm_forecast",
}


def safe_wape(actual: Iterable[float], predicted: Iterable[float]) -> float | None:
    actual_values = np.asarray(list(actual), dtype=float)
    predicted_values = np.asarray(list(predicted), dtype=float)
    denominator = float(np.abs(actual_values).sum())
    if denominator == 0:
        return None
    return float(np.abs(actual_values - predicted_values).sum() / denominator)


def regression_metrics(
    frame: pd.DataFrame,
    forecast_column: str,
    include_intervals: bool | None = None,
) -> dict[str, Any]:
    actual = frame["actual_units"].to_numpy(dtype=float)
    predicted = frame[forecast_column].to_numpy(dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]
    if not len(actual):
        return {"rows": 0, "wape": None, "mae": None, "bias": None}
    wape = safe_wape(actual, predicted)
    result: dict[str, Any] = {
        "rows": int(len(actual)),
        "wape": wape,
        "mae": float(np.mean(np.abs(actual - predicted))),
        "bias": float(np.mean(predicted - actual)),
    }
    if include_intervals is None:
        include_intervals = forecast_column == "selected_forecast"
    if include_intervals and {"lower80", "upper80"}.issubset(frame.columns):
        lower = frame.loc[valid, "lower80"].to_numpy(dtype=float)
        upper = frame.loc[valid, "upper80"].to_numpy(dtype=float)
        interval_valid = np.isfinite(lower) & np.isfinite(upper)
        if interval_valid.any():
            result["interval_coverage"] = float(
                np.mean((actual[interval_valid] >= lower[interval_valid]) & (actual[interval_valid] <= upper[interval_valid]))
            )
            result["mean_interval_width"] = float(np.mean(upper[interval_valid] - lower[interval_valid]))
    return result


def grouped_metrics(
    frame: pd.DataFrame,
    forecast_column: str,
    group_columns: list[str],
) -> list[dict[str, Any]]:
    results = []
    for key, group in frame.groupby(group_columns, dropna=False, sort=True, observed=True):
        keys = key if isinstance(key, tuple) else (key,)
        row = {column: _json_scalar(value) for column, value in zip(group_columns, keys)}
        row.update(regression_metrics(group, forecast_column))
        results.append(row)
    return results


def choose_model_by_demand_group(development_predictions: pd.DataFrame) -> dict[str, str]:
    """Choose on mean fold-level WAPE from folds 1-2; ties prefer the baseline."""
    choices: dict[str, str] = {}
    for demand_group, group in development_predictions.groupby("demand_group", sort=True):
        fold_groups = group.groupby("fold", sort=True) if "fold" in group.columns else [(None, group)]
        naive_scores = []
        lightgbm_scores = []
        for _, fold_group in fold_groups:
            naive = safe_wape(fold_group["actual_units"], fold_group[MODEL_COLUMNS["seasonal_naive"]])
            lightgbm = safe_wape(fold_group["actual_units"], fold_group[MODEL_COLUMNS["lightgbm"]])
            if naive is not None:
                naive_scores.append(naive)
            if lightgbm is not None:
                lightgbm_scores.append(lightgbm)
        naive_wape = float(np.mean(naive_scores)) if naive_scores else None
        lightgbm_wape = float(np.mean(lightgbm_scores)) if lightgbm_scores else None
        choices[str(demand_group)] = (
            "lightgbm" if lightgbm_wape is not None and (naive_wape is None or lightgbm_wape < naive_wape) else "seasonal_naive"
        )
    return choices


def apply_model_choices(frame: pd.DataFrame, choices: dict[str, str]) -> pd.DataFrame:
    result = frame.copy()
    result["selected_model"] = result["demand_group"].astype(str).map(choices).fillna("seasonal_naive")
    result["selected_forecast"] = np.where(
        result["selected_model"].eq("lightgbm"),
        result[MODEL_COLUMNS["lightgbm"]],
        result[MODEL_COLUMNS["seasonal_naive"]],
    )
    return result


def calibrate_residual_intervals(
    development_selected: pd.DataFrame,
    coverage: float = 0.80,
    minimum_rows: int = 20,
) -> dict[str, dict[str, float | int | str]]:
    """Calibrate asymmetric residual bands by demand group and horizon.

    Sparse cells fall back to a demand-group pool, then a horizon pool, then
    the global pool. These are empirical intervals, not formal guarantees.
    """
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between 0 and 1")
    alpha = 1.0 - coverage
    residuals = development_selected["actual_units"].to_numpy(dtype=float) - development_selected[
        "selected_forecast"
    ].to_numpy(dtype=float)
    work = development_selected[["demand_group", "horizon"]].copy()
    work["residual"] = residuals
    global_values = work["residual"].to_numpy(dtype=float)
    calibration: dict[str, dict[str, float | int | str]] = {}
    for (group, horizon), cell in work.groupby(["demand_group", "horizon"], sort=True):
        source = cell["residual"].to_numpy(dtype=float)
        source_label = "demand_group_and_horizon"
        if len(source) < minimum_rows:
            group_values = work.loc[work["demand_group"].eq(group), "residual"].to_numpy(dtype=float)
            if len(group_values) >= minimum_rows:
                source = group_values
                source_label = "demand_group"
            else:
                horizon_values = work.loc[work["horizon"].eq(horizon), "residual"].to_numpy(dtype=float)
                if len(horizon_values) >= minimum_rows:
                    source = horizon_values
                    source_label = "horizon"
                else:
                    source = global_values
                    source_label = "global"
        key = f"{group}|{int(horizon)}"
        calibration[key] = {
            "lower_residual": float(np.quantile(source, alpha / 2)),
            "upper_residual": float(np.quantile(source, 1 - alpha / 2)),
            "calibration_rows": int(len(source)),
            "calibration_pool": source_label,
        }
    return calibration


def apply_residual_intervals(
    frame: pd.DataFrame,
    calibration: dict[str, dict[str, float | int | str]],
) -> pd.DataFrame:
    result = frame.copy()
    lower = []
    upper = []
    for row in result.itertuples(index=False):
        key = f"{row.demand_group}|{int(row.horizon)}"
        params = calibration.get(key)
        if params is None:
            lower.append(np.nan)
            upper.append(np.nan)
            continue
        lower_value = max(0.0, float(row.selected_forecast) + float(params["lower_residual"]))
        upper_value = max(lower_value, float(row.selected_forecast) + float(params["upper_residual"]))
        lower.append(lower_value)
        upper.append(upper_value)
    result["lower80"] = lower
    result["upper80"] = upper
    return result


def _json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value.item() if isinstance(value, np.generic) else value
