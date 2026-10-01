"""Leakage-safe chronological backtest fold construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class BacktestFold:
    number: int
    train_end_idx: int
    validation_start_idx: int
    validation_end_idx: int


def make_backtest_folds(
    dates: Sequence[pd.Timestamp],
    horizon: int = 28,
    fold_count: int = 3,
    min_train_days: int = 365,
) -> list[BacktestFold]:
    ordered = pd.DatetimeIndex(pd.to_datetime(list(dates))).sort_values().unique()
    if len(ordered) != len(dates):
        raise ValueError("dates must be unique")
    required = min_train_days + horizon * fold_count
    if len(ordered) < required:
        raise ValueError(
            f"Need at least {required} observed dates ({min_train_days} training + "
            f"{fold_count} x {horizon}-day folds); found {len(ordered)}."
        )
    first_validation = len(ordered) - horizon * fold_count
    folds = []
    for offset in range(fold_count):
        start = first_validation + offset * horizon
        folds.append(
            BacktestFold(
                number=offset + 1,
                train_end_idx=start - 1,
                validation_start_idx=start,
                validation_end_idx=start + horizon - 1,
            )
        )
    return folds

