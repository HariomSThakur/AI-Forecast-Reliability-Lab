"""Direct multi-horizon features built strictly from each forecast origin."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .folds import BacktestFold

CATEGORICAL_FEATURES = [
    "item_id",
    "store_id",
    "dept_id",
    "cat_id",
    "state_id",
    "weekday",
    "month",
]
NUMERIC_FEATURES = [
    "horizon",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "lag_56",
    "lag_364",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_mean_56",
    "rolling_mean_364",
    "event_day",
    "snap",
    "price_at_origin",
    "price_change_28d",
    "price_known",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
METADATA_COLUMNS = ["series_id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "demand_group"]


def _aligned_panel(
    daily_sales: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
) -> tuple[list[str], pd.DataFrame, np.ndarray, np.ndarray]:
    dates = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    series_ids = sorted(daily_sales["id"].astype(str).unique().tolist())
    metadata = (
        daily_sales.sort_values("date")
        .groupby("id", sort=True)
        .first()[["item_id", "dept_id", "cat_id", "store_id", "state_id", "demand_group"]]
        .reindex(series_ids)
    )
    sales = daily_sales.pivot(index="id", columns="date", values="actual_units").reindex(
        index=series_ids, columns=dates
    )
    if sales.isna().any().any():
        raise ValueError("Every sampled series must have one sales value for every observed date.")
    prices = daily_sales.pivot(index="id", columns="date", values="sell_price").reindex(
        index=series_ids, columns=dates
    )
    sales_values = sales.to_numpy(dtype=float)
    price_values = prices.to_numpy(dtype=float)

    # Forward-fill price only along time. At an origin, the feature builder
    # reads the filled value at that origin, never a later price observation.
    positions = np.broadcast_to(np.arange(price_values.shape[1]), price_values.shape)
    last_price_idx = np.maximum.accumulate(np.where(np.isfinite(price_values), positions, -1), axis=1)
    safe_idx = np.maximum(last_price_idx, 0)
    filled_prices = price_values[np.arange(len(series_ids))[:, None], safe_idx]
    filled_prices[last_price_idx < 0] = np.nan
    return series_ids, metadata, sales_values, filled_prices


def _calendar_arrays(calendar: pd.DataFrame, dates: Sequence[pd.Timestamp]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    indexed = calendar.copy()
    indexed["date"] = pd.to_datetime(indexed["date"])
    indexed = indexed.drop_duplicates("date").set_index("date").reindex(pd.DatetimeIndex(dates))
    if indexed["event_day"].isna().any():
        raise ValueError("Calendar is missing an observed or forecast date.")
    snap_by_state: dict[str, np.ndarray] = {}
    for column in [name for name in indexed.columns if name.startswith("snap_")]:
        snap_by_state[column.removeprefix("snap_")] = pd.to_numeric(indexed[column], errors="coerce").fillna(0).to_numpy(float)
    return indexed, snap_by_state


def _base_features(
    series_ids: list[str],
    metadata: pd.DataFrame,
    sales_values: np.ndarray,
    filled_prices: np.ndarray,
    calendar_indexed: pd.DataFrame,
    snap_by_state: dict[str, np.ndarray],
    dates: pd.DatetimeIndex,
    origin_idx: int,
    horizon: int,
) -> pd.DataFrame:
    series_count = len(series_ids)
    horizon_values = np.arange(1, horizon + 1, dtype=int)
    target_indices = origin_idx + horizon_values
    if target_indices.max() >= len(dates):
        raise ValueError("Forecast horizon extends beyond the supplied dates.")

    current_price = filled_prices[:, origin_idx].copy()
    previous_index = max(0, origin_idx - 27)
    previous_price = filled_prices[:, previous_index].copy()
    price_known = np.isfinite(current_price)
    current_price = np.nan_to_num(current_price, nan=0.0)
    previous_price = np.nan_to_num(previous_price, nan=0.0)
    price_change = np.divide(
        current_price - previous_price,
        previous_price,
        out=np.zeros_like(current_price),
        where=previous_price > 0,
    )

    def target_date_lag(lag: int) -> np.ndarray:
        indices = target_indices - lag
        values = np.full((series_count, horizon), np.nan, dtype=float)
        available = indices >= 0
        if available.any():
            values[:, available] = sales_values[:, indices[available]]
        return values.reshape(-1)

    rolling_56_start = max(0, origin_idx - 55)
    rolling_364_start = max(0, origin_idx - 363)

    rows: dict[str, np.ndarray] = {
        "series_id": np.repeat(np.asarray(series_ids, dtype=object), horizon),
        "date": np.tile(dates[target_indices].to_numpy(), series_count),
        "horizon": np.tile(horizon_values, series_count),
        "lag_1": np.repeat(sales_values[:, origin_idx], horizon),
        "lag_7": np.repeat(sales_values[:, origin_idx - 6], horizon),
        "lag_14": np.repeat(sales_values[:, origin_idx - 13], horizon),
        "lag_28": np.repeat(sales_values[:, origin_idx - 27], horizon),
        # Target-date-aligned seasonal lags. Since every forecast horizon is
        # shorter than 364 days, these values are known at the forecast origin.
        "lag_56": target_date_lag(56),
        "lag_364": target_date_lag(364),
        "rolling_mean_7": np.repeat(sales_values[:, origin_idx - 6 : origin_idx + 1].mean(axis=1), horizon),
        "rolling_mean_28": np.repeat(sales_values[:, origin_idx - 27 : origin_idx + 1].mean(axis=1), horizon),
        "rolling_mean_56": np.repeat(sales_values[:, rolling_56_start : origin_idx + 1].mean(axis=1), horizon),
        "rolling_mean_364": np.repeat(sales_values[:, rolling_364_start : origin_idx + 1].mean(axis=1), horizon),
        "event_day": np.tile(calendar_indexed["event_day"].to_numpy(dtype=bool)[target_indices].astype(int), series_count),
        "price_at_origin": np.repeat(current_price, horizon),
        "price_change_28d": np.repeat(price_change, horizon),
        "price_known": np.repeat(price_known.astype(int), horizon),
        "demand_group": np.repeat(metadata["demand_group"].astype(str).to_numpy(), horizon),
    }

    states = metadata["state_id"].astype(str).to_numpy()
    repeated_states = np.repeat(states, horizon)
    snap_values = np.zeros(series_count * horizon, dtype=float)
    tiled_targets = np.tile(target_indices, series_count)
    for state in np.unique(states):
        mask = repeated_states == state
        state_snap = snap_by_state.get(state)
        if state_snap is not None:
            snap_values[mask] = state_snap[tiled_targets[mask]]
    rows["snap"] = snap_values

    forecast_dates = dates[target_indices]
    rows["weekday"] = np.tile(forecast_dates.dayofweek.astype(str).to_numpy(), series_count)
    rows["month"] = np.tile(forecast_dates.month.astype(str).to_numpy(), series_count)
    for column in ("item_id", "dept_id", "cat_id", "store_id", "state_id"):
        rows[column] = np.repeat(metadata[column].astype(str).to_numpy(), horizon)
    return pd.DataFrame(rows)


def build_training_examples(
    daily_sales: pd.DataFrame,
    calendar: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
    fold: BacktestFold,
    horizon: int = 28,
    min_train_days: int = 365,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    series_ids, metadata, sales_values, prices = _aligned_panel(daily_sales, dates)
    calendar_indexed, snap_by_state = _calendar_arrays(calendar, dates)
    first_origin = min_train_days - 1
    last_origin = fold.train_end_idx - horizon
    if first_origin < 27:
        raise ValueError("min_train_days must leave at least 28 days for lag features.")
    origins = range(first_origin, last_origin + 1, horizon)
    frames = []
    for origin in origins:
        features = _base_features(
            series_ids, metadata, sales_values, prices, calendar_indexed, snap_by_state, dates, origin, horizon
        )
        target_indices = origin + np.arange(1, horizon + 1)
        features["target"] = sales_values[:, target_indices].reshape(-1)
        features["origin_date"] = dates[origin]
        frames.append(features)
    if not frames:
        raise ValueError("No training examples were generated; reduce min_train_days or check the data length.")
    return pd.concat(frames, ignore_index=True)


def build_forecast_examples(
    daily_sales: pd.DataFrame,
    calendar: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
    fold: BacktestFold,
    horizon: int = 28,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    series_ids, metadata, sales_values, prices = _aligned_panel(daily_sales, dates)
    calendar_indexed, snap_by_state = _calendar_arrays(calendar, dates)
    features = _base_features(
        series_ids,
        metadata,
        sales_values,
        prices,
        calendar_indexed,
        snap_by_state,
        dates,
        fold.train_end_idx,
        horizon,
    )
    features["origin_date"] = dates[fold.train_end_idx]
    features["fold"] = fold.number
    return features


def align_categorical_features(
    train_features: pd.DataFrame,
    predict_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Give train/predict frames identical pandas category dictionaries."""
    train = train_features[FEATURE_COLUMNS].copy()
    predict = predict_features[FEATURE_COLUMNS].copy()
    for column in CATEGORICAL_FEATURES:
        levels = sorted(set(train[column].astype(str)) | set(predict[column].astype(str)))
        dtype = pd.CategoricalDtype(categories=levels)
        train[column] = train[column].astype(str).astype(dtype)
        predict[column] = predict[column].astype(str).astype(dtype)
    return train, predict
