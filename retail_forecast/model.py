"""LightGBM training and the seasonal-naive comparison forecast."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, align_categorical_features


def weekly_seasonal_naive(history: np.ndarray, horizon: int = 28) -> np.ndarray:
    """Repeat the latest observed seven-day pattern recursively."""
    history = np.asarray(history, dtype=float)
    if len(history) < 7:
        raise ValueError("At least seven historical observations are required for the baseline.")
    predictions = np.zeros(horizon, dtype=float)
    for step in range(horizon):
        predictions[step] = history[-7 + step] if step < 7 else predictions[step - 7]
    return np.maximum(predictions, 0.0)


def fit_lightgbm_predict(
    training_examples: pd.DataFrame,
    prediction_examples: pd.DataFrame,
    seed: int = 42,
) -> np.ndarray:
    """Fit a pooled Poisson LightGBM regressor and return nonnegative forecasts."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError(
            "LightGBM is required to train the forecasting model. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    x_train, x_predict = align_categorical_features(training_examples, prediction_examples)
    params = {
        "objective": "poisson",
        "metric": "l1",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_bin": 127,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "seed": seed,
        "verbosity": -1,
        "num_threads": -1,
        "deterministic": True,
        "force_col_wise": True,
    }
    target = training_examples["target"].to_numpy(dtype=float)

    # Training examples are grouped by forecast origin. Keep the latest few
    # origins inside this fold as a chronological validation window, choose the
    # boosting length there, then refit on the complete fold history. The outer
    # final holdout is never used for this choice.
    origins = sorted(pd.to_datetime(training_examples["origin_date"]).unique())
    if len(origins) >= 5:
        validation_count = min(3, max(1, len(origins) // 5))
        validation_origins = set(origins[-validation_count:])
        validation_mask = pd.to_datetime(training_examples["origin_date"]).isin(validation_origins).to_numpy()
        fit_mask = ~validation_mask
        fit_data = lgb.Dataset(
            x_train.loc[fit_mask],
            label=target[fit_mask],
            categorical_feature="auto",
            free_raw_data=False,
        )
        validation_data = lgb.Dataset(
            x_train.loc[validation_mask],
            label=target[validation_mask],
            categorical_feature="auto",
            reference=fit_data,
            free_raw_data=False,
        )
        selection_model = lgb.train(
            params,
            fit_data,
            num_boost_round=800,
            valid_sets=[validation_data],
            valid_names=["recent_development"],
            callbacks=[lgb.early_stopping(60, first_metric_only=True, verbose=False)],
        )
        best_rounds = max(1, int(selection_model.best_iteration or 800))
        training_data = lgb.Dataset(
            x_train,
            label=target,
            categorical_feature="auto",
            free_raw_data=False,
        )
        model = lgb.train(params, train_set=training_data, num_boost_round=best_rounds)
    else:
        training_data = lgb.Dataset(
            x_train,
            label=target,
            categorical_feature="auto",
            free_raw_data=False,
        )
        model = lgb.train(params, train_set=training_data, num_boost_round=300)
    return np.maximum(model.predict(x_predict), 0.0)
