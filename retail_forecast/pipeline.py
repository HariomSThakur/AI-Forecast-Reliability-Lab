"""CLI pipeline: M5 files -> rolling-origin forecasts -> compact artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .data import load_m5_dataset
from .evaluation import (
    apply_model_choices,
    apply_residual_intervals,
    calibrate_residual_intervals,
    choose_model_by_demand_group,
    grouped_metrics,
    regression_metrics,
)
from .features import build_forecast_examples, build_training_examples
from .folds import make_backtest_folds
from .model import fit_lightgbm_predict, weekly_seasonal_naive


def _run(config: PipelineConfig, data_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    daily, calendar = load_m5_dataset(data_dir, sample_size=config.sample_size, seed=config.seed)
    dates = pd.DatetimeIndex(sorted(daily["date"].unique()))
    folds = make_backtest_folds(
        dates,
        horizon=config.horizon,
        fold_count=config.fold_count,
        min_train_days=config.min_train_days,
    )
    sales_panel = daily.pivot(index="id", columns="date", values="actual_units").reindex(columns=dates)
    ids = sorted(daily["id"].astype(str).unique())
    sales_panel.index = sales_panel.index.astype(str)
    actuals = daily[["id", "date", "actual_units"]].rename(columns={"id": "series_id"}).copy()
    predictions = []

    for fold in folds:
        print(f"Fold {fold.number}/{len(folds)}: building training examples")
        training = build_training_examples(
            daily,
            calendar,
            dates,
            fold,
            horizon=config.horizon,
            min_train_days=config.min_train_days,
        )
        future = build_forecast_examples(daily, calendar, dates, fold, horizon=config.horizon)
        future["lightgbm_forecast"] = fit_lightgbm_predict(training, future, seed=config.seed)

        naive_values = []
        for series_id in future["series_id"].drop_duplicates().tolist():
            history = sales_panel.loc[series_id].iloc[: fold.train_end_idx + 1].to_numpy(dtype=float)
            naive_values.extend(weekly_seasonal_naive(history, horizon=config.horizon))
        future["seasonal_naive_forecast"] = np.asarray(naive_values, dtype=float)
        future = future.merge(actuals, on=["series_id", "date"], how="left", validate="one_to_one")
        if future["actual_units"].isna().any():
            raise ValueError(f"Fold {fold.number} has missing actuals for its validation dates.")
        future["fold"] = fold.number
        future["train_end_date"] = dates[fold.train_end_idx]
        predictions.append(future)

    all_predictions = pd.concat(predictions, ignore_index=True)
    development = all_predictions[all_predictions["fold"].lt(config.fold_count)].copy()
    choices = choose_model_by_demand_group(development)
    all_predictions = apply_model_choices(all_predictions, choices)

    development_selected = all_predictions[all_predictions["fold"].lt(config.fold_count)].copy()
    calibration = calibrate_residual_intervals(
        development_selected,
        coverage=config.interval_coverage,
        minimum_rows=config.minimum_calibration_rows,
    )
    all_predictions["lower80"] = np.nan
    all_predictions["upper80"] = np.nan
    final_mask = all_predictions["fold"].eq(config.fold_count)
    all_predictions.loc[final_mask, ["lower80", "upper80"]] = apply_residual_intervals(
        all_predictions.loc[final_mask], calibration
    )[["lower80", "upper80"]].to_numpy()

    development_metrics = {
        model: regression_metrics(development_selected, column)
        for model, column in {
            "seasonal_naive": "seasonal_naive_forecast",
            "lightgbm": "lightgbm_forecast",
            "selected": "selected_forecast",
        }.items()
    }
    final = all_predictions.loc[final_mask].copy()
    comparison_columns = {
        "seasonal_naive": "seasonal_naive_forecast",
        "lightgbm": "lightgbm_forecast",
        "selected": "selected_forecast",
    }
    segment_dimensions = {
        "horizon": ["horizon"],
        "store": ["store_id"],
        "department": ["dept_id"],
        "demand_group": ["demand_group"],
        "event_day": ["event_day"],
    }
    final_metrics = {
        model: regression_metrics(final, column)
        for model, column in comparison_columns.items()
    }
    for model in ("seasonal_naive", "lightgbm"):
        final_metrics[model].pop("interval_coverage", None)
        final_metrics[model].pop("mean_interval_width", None)
    final_metrics["selected"].update(_interval_metrics(final))
    report = {
        "dataset": "M5 Forecasting - Accuracy",
        "config": {
            "sample_size": config.sample_size,
            "seed": config.seed,
            "horizon_days": config.horizon,
            "fold_count": config.fold_count,
            "min_train_days": config.min_train_days,
            "interval_target_coverage": config.interval_coverage,
            "interval_method": "development-fold residual quantiles, grouped by demand frequency and horizon",
            "feature_price_policy": "latest sell price known on or before the forecast origin",
            "lightgbm_training": "up to 800 rounds with early stopping on the latest up to three chronological training origins, then refit on the full fold training data",
            "additional_sales_features": ["target-date-aligned lag 56", "target-date-aligned lag 364 (52 weeks)", "56-day rolling mean", "364-day rolling mean"],
        },
        "selection_by_demand_group": choices,
        "development": {
            "folds": [fold.number for fold in folds if fold.number < config.fold_count],
            "overall": development_metrics,
            "by_demand_group": {
                model: grouped_metrics(development_selected, column, ["demand_group"])
                for model, column in {
                    "seasonal_naive": "seasonal_naive_forecast",
                    "lightgbm": "lightgbm_forecast",
                    "selected": "selected_forecast",
                }.items()
            },
        },
        "final": {
            "fold": config.fold_count,
            "train_end_date": str(final["train_end_date"].iloc[0].date()),
            "overall": final_metrics,
            "by_horizon": grouped_metrics(final, "selected_forecast", ["horizon"]),
            "by_store": grouped_metrics(final, "selected_forecast", ["store_id"]),
            "by_department": grouped_metrics(final, "selected_forecast", ["dept_id"]),
            "by_demand_group": grouped_metrics(final, "selected_forecast", ["demand_group"]),
            "by_event_day": grouped_metrics(final, "selected_forecast", ["event_day"]),
            "by_model_and_segment": {
                model: {
                    f"by_{dimension}": grouped_metrics(final, column, group_columns)
                    for dimension, group_columns in segment_dimensions.items()
                }
                for model, column in comparison_columns.items()
            },
        },
        "interval_calibration": {
            "target_coverage": config.interval_coverage,
            "calibration_cells": len(calibration),
            "observed_final_coverage": _interval_metrics(final).get("interval_coverage"),
            "mean_final_interval_width": _interval_metrics(final).get("mean_interval_width"),
            "guarantee": "Empirical coverage only; no formal distribution-free guarantee is claimed.",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.parquet"
    metrics_path = output_dir / "metrics.json"
    try:
        all_predictions.to_parquet(predictions_path, index=False)
    except ImportError as exc:
        raise RuntimeError("Parquet output requires pyarrow. Install project dependencies with `pip install -r requirements.txt`.") from exc
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report), handle, indent=2, ensure_ascii=False)
    return all_predictions, report


def _interval_metrics(frame: pd.DataFrame) -> dict[str, float | None]:
    available = frame.dropna(subset=["lower80", "upper80"])
    if available.empty:
        return {"interval_coverage": None, "mean_interval_width": None}
    return {
        "interval_coverage": float(
            ((available["actual_units"] >= available["lower80"]) & (available["actual_units"] <= available["upper80"])).mean()
        ),
        "mean_interval_width": float((available["upper80"] - available["lower80"]).mean()),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"), help="Folder containing the M5 CSV files.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"), help="Folder for compact app artifacts.")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = PipelineConfig(sample_size=args.sample_size, seed=args.seed)
    _, report = _run(config, args.data_dir, args.output_dir)
    print(f"Wrote {args.output_dir / 'predictions.parquet'}")
    print(f"Wrote {args.output_dir / 'metrics.json'}")
    print(json.dumps(report["final"]["overall"], indent=2))


if __name__ == "__main__":
    main()
