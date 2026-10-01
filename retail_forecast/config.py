"""Shared, explicit defaults for the forecasting experiment."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    sample_size: int = 500
    seed: int = 42
    horizon: int = 28
    fold_count: int = 3
    min_train_days: int = 365
    interval_coverage: float = 0.80
    minimum_calibration_rows: int = 20

