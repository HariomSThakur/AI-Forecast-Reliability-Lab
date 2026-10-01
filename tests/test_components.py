from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from retail_forecast.data import assign_demand_groups, load_m5_dataset, stratified_sample
from retail_forecast.evaluation import (
    apply_model_choices,
    apply_residual_intervals,
    calibrate_residual_intervals,
    choose_model_by_demand_group,
    safe_wape,
)
from retail_forecast.features import build_forecast_examples, build_training_examples
from retail_forecast.folds import BacktestFold, make_backtest_folds
from retail_forecast.model import weekly_seasonal_naive


class SamplingTests(unittest.TestCase):
    def test_demand_groups_and_stratified_sampling_are_reproducible(self):
        rows = []
        for store in ("S1", "S2"):
            for item in range(60):
                daily = [0 if item % 3 == 0 else (1 if day % (item % 6 + 1) == 0 else 0) for day in range(70)]
                rows.append({"id": f"{store}_{item:03}", "store_id": store, **{f"d_{i + 1}": v for i, v in enumerate(daily)}})
        frame = pd.DataFrame(rows)
        frame["demand_group"] = assign_demand_groups(frame, [f"d_{i}" for i in range(1, 71)])
        first = stratified_sample(frame, sample_size=30, seed=42)
        second = stratified_sample(frame, sample_size=30, seed=42)
        self.assertEqual(first["id"].tolist(), second["id"].tolist())
        self.assertEqual(len(first), 30)
        self.assertEqual(set(first["demand_group"]), {"low", "medium", "high"})
        self.assertEqual(set(first["store_id"]), {"S1", "S2"})

    def test_m5_loader_joins_calendar_and_prices(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            days = 70
            day_columns = [f"d_{day}" for day in range(1, days + 1)]
            sales_rows = []
            price_rows = []
            for store_index, store in enumerate(("CA_1", "TX_1")):
                state = store.split("_")[0]
                for item_index in range(6):
                    row = {
                        "id": f"ITEM_{item_index}_{store}",
                        "item_id": f"ITEM_{item_index}",
                        "dept_id": f"DEPT_{item_index % 2}",
                        "cat_id": "FOODS",
                        "store_id": store,
                        "state_id": state,
                    }
                    for day in range(1, days + 1):
                        row[f"d_{day}"] = int((day + item_index + store_index) % (item_index + 2) == 0)
                    sales_rows.append(row)
                    for week in range((days + 6) // 7):
                        price_rows.append(
                            {"item_id": f"ITEM_{item_index}", "store_id": store, "wm_yr_wk": week, "sell_price": 2.5 + item_index}
                        )
            calendar = pd.DataFrame(
                {
                    "d": day_columns,
                    "date": pd.date_range("2021-01-01", periods=days),
                    "wm_yr_wk": [(day - 1) // 7 for day in range(1, days + 1)],
                    "event_name_1": ["Holiday" if day == 14 else np.nan for day in range(1, days + 1)],
                    "event_name_2": [np.nan] * days,
                    "snap_CA": [0] * days,
                    "snap_TX": [1] * days,
                }
            )
            pd.DataFrame(sales_rows).to_csv(root / "sales_train_validation.csv", index=False)
            calendar.to_csv(root / "calendar.csv", index=False)
            pd.DataFrame(price_rows).to_csv(root / "sell_prices.csv", index=False)

            daily, loaded_calendar = load_m5_dataset(root, sample_size=6, seed=7)
            self.assertEqual(daily["id"].nunique(), 6)
            self.assertEqual(len(daily), 6 * days)
            self.assertEqual(daily["date"].nunique(), days)
            self.assertTrue(daily["sell_price"].notna().all())
            self.assertTrue(daily.loc[daily["state_id"].eq("TX"), "snap"].eq(1).all())
            self.assertEqual(int(loaded_calendar["event_day"].sum()), 1)


class FoldTests(unittest.TestCase):
    def test_folds_are_chronological_fixed_horizon_blocks(self):
        dates = pd.date_range("2020-01-01", periods=200)
        folds = make_backtest_folds(dates, horizon=28, fold_count=3, min_train_days=100)
        self.assertEqual(len(folds), 3)
        for fold in folds:
            self.assertEqual(fold.validation_end_idx - fold.validation_start_idx + 1, 28)
            self.assertEqual(fold.train_end_idx + 1, fold.validation_start_idx)
        self.assertLess(folds[0].validation_end_idx, folds[1].validation_start_idx)
        self.assertLess(folds[1].validation_end_idx, folds[2].validation_start_idx)

    def test_short_history_fails_with_actionable_message(self):
        with self.assertRaisesRegex(ValueError, "Need at least"):
            make_backtest_folds(pd.date_range("2020-01-01", periods=120), min_train_days=100)


class BaselineTests(unittest.TestCase):
    def test_weekly_naive_repeats_last_observed_week(self):
        result = weekly_seasonal_naive(np.arange(1, 15), horizon=10)
        np.testing.assert_array_equal(result, [8, 9, 10, 11, 12, 13, 14, 8, 9, 10])

    def test_weekly_naive_rejects_insufficient_history(self):
        with self.assertRaisesRegex(ValueError, "seven"):
            weekly_seasonal_naive([1, 2, 3])


class EvaluationTests(unittest.TestCase):
    def test_wape_returns_none_when_actual_total_is_zero(self):
        self.assertIsNone(safe_wape([0, 0], [1, 2]))
        self.assertAlmostEqual(safe_wape([2, 2], [1, 3]), 0.5)

    def test_model_selection_uses_only_development_rows(self):
        rows = []
        for fold in (1, 2):
            rows.extend(
                [
                    {"fold": fold, "demand_group": "regular", "actual_units": 10.0, "seasonal_naive_forecast": 8.0, "lightgbm_forecast": 10.0},
                    {"fold": fold, "demand_group": "sparse", "actual_units": 0.0, "seasonal_naive_forecast": 0.0, "lightgbm_forecast": 2.0},
                ]
            )
        development = pd.DataFrame(rows)
        choices = choose_model_by_demand_group(development)
        self.assertEqual(choices, {"regular": "lightgbm", "sparse": "seasonal_naive"})
        final = pd.DataFrame(
            [{"demand_group": "regular", "seasonal_naive_forecast": 1.0, "lightgbm_forecast": 2.0}]
        )
        selected = apply_model_choices(final, choices)
        self.assertEqual(selected.loc[0, "selected_forecast"], 2.0)

    def test_model_selector_averages_fold_wape_instead_of_pooling_rows(self):
        development = pd.DataFrame(
            [
                {"fold": 1, "demand_group": "regular", "actual_units": 100.0, "seasonal_naive_forecast": 50.0, "lightgbm_forecast": 80.0},
                {"fold": 2, "demand_group": "regular", "actual_units": 1.0, "seasonal_naive_forecast": 1.0, "lightgbm_forecast": 0.0},
            ]
        )
        self.assertEqual(choose_model_by_demand_group(development)["regular"], "seasonal_naive")

    def test_intervals_are_calibrated_and_nonnegative(self):
        development = pd.DataFrame(
            {
                "demand_group": ["low"] * 30,
                "horizon": [1] * 30,
                "actual_units": np.arange(30, dtype=float),
                "selected_forecast": np.arange(30, dtype=float) + np.tile([-2, -1, 0, 1, 2], 6),
            }
        )
        calibration = calibrate_residual_intervals(development, coverage=0.8, minimum_rows=20)
        final = pd.DataFrame(
            {"demand_group": ["low"], "horizon": [1], "selected_forecast": [0.5]}
        )
        result = apply_residual_intervals(final, calibration)
        self.assertGreaterEqual(result.loc[0, "lower80"], 0)
        self.assertLessEqual(result.loc[0, "lower80"], result.loc[0, "selected_forecast"])
        self.assertGreaterEqual(result.loc[0, "upper80"], result.loc[0, "selected_forecast"])


class LeakageTests(unittest.TestCase):
    @staticmethod
    def _fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
        dates = pd.date_range("2022-01-01", periods=140)
        rows = []
        for series_index in range(3):
            state = ("CA", "TX", "WI")[series_index]
            sales = (np.arange(len(dates)) + series_index) % (series_index + 4)
            prices = np.where(np.arange(len(dates)) % 7 == 0, 2.0 + series_index, np.nan)
            for date, actual, price in zip(dates, sales, prices):
                rows.append(
                    {
                        "id": f"item_{series_index}",
                        "series_id": f"item_{series_index}",
                        "item_id": f"item_{series_index}",
                        "dept_id": "dept_1",
                        "cat_id": "foods",
                        "store_id": f"store_{series_index}",
                        "state_id": state,
                        "demand_group": ("low", "medium", "high")[series_index],
                        "date": date,
                        "actual_units": float(actual),
                        "sell_price": price,
                    }
                )
        calendar = pd.DataFrame(
            {
                "date": dates,
                "event_day": [False] * len(dates),
                "snap_CA": [0] * len(dates),
                "snap_TX": [0] * len(dates),
                "snap_WI": [0] * len(dates),
            }
        )
        return pd.DataFrame(rows), calendar, dates

    def test_training_features_do_not_read_sales_after_training_cutoff(self):
        daily, calendar, dates = self._fixture()
        fold = BacktestFold(number=1, train_end_idx=110, validation_start_idx=111, validation_end_idx=117)
        first = build_training_examples(daily, calendar, dates, fold, horizon=7, min_train_days=35)
        changed = daily.copy()
        changed.loc[changed["date"] > dates[fold.train_end_idx], "actual_units"] += 1000
        second = build_training_examples(changed, calendar, dates, fold, horizon=7, min_train_days=35)
        pd.testing.assert_frame_equal(first, second)

    def test_forecast_examples_have_one_full_horizon_per_series(self):
        daily, calendar, dates = self._fixture()
        fold = BacktestFold(number=3, train_end_idx=83, validation_start_idx=84, validation_end_idx=111)
        first = build_forecast_examples(daily, calendar, dates, fold, horizon=28)
        changed = daily.copy()
        changed.loc[changed["date"] > dates[fold.train_end_idx], "actual_units"] += 1000
        second = build_forecast_examples(changed, calendar, dates, fold, horizon=28)
        self.assertEqual(len(first), 3 * 28)
        self.assertTrue(first.groupby("series_id").size().eq(28).all())
        self.assertTrue(first.groupby("series_id")["horizon"].apply(lambda values: sorted(values.tolist()) == list(range(1, 29))).all())
        pd.testing.assert_frame_equal(first, second)


if __name__ == "__main__":
    unittest.main()
