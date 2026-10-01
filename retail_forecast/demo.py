"""Clearly labeled deterministic synthetic data for a no-download app preview."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_demo_data() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2025-01-01", periods=28, freq="D")
    rows = []
    for store_number in range(1, 4):
        for group in ("low", "medium", "high"):
            series_id = f"DEMO_{store_number}_{group}"
            base = {"low": 1.3, "medium": 5.0, "high": 13.0}[group] * (0.8 + store_number * 0.1)
            actual = np.maximum(0, rng.poisson(base + 1.1 * np.sin(np.arange(28) * 2 * np.pi / 7)))
            naive = np.maximum(0, actual + rng.normal(0, base * 0.24 + 0.7, size=28))
            lightgbm = np.maximum(0, actual + rng.normal(0, base * 0.18 + 0.5, size=28))
            chosen = "lightgbm" if store_number != 2 else "seasonal_naive"
            selected = lightgbm if chosen == "lightgbm" else naive
            width = np.maximum(1.0, base * 0.7)
            for h, date in enumerate(dates, start=1):
                rows.append(
                    {
                        "series_id": series_id,
                        "state_id": ("CA", "TX", "WI")[store_number - 1],
                        "store_id": f"DEMO_{store_number}",
                        "dept_id": f"DEPT_{1 + (store_number - 1) % 3}",
                        "cat_id": "SYNTHETIC",
                        "item_id": series_id,
                        "demand_group": group,
                        "date": date,
                        "horizon": h,
                        "actual_units": float(actual[h - 1]),
                        "seasonal_naive_forecast": float(naive[h - 1]),
                        "lightgbm_forecast": float(lightgbm[h - 1]),
                        "selected_model": chosen,
                        "selected_forecast": float(selected[h - 1]),
                        "lower80": max(0.0, float(selected[h - 1] - width)),
                        "upper80": float(selected[h - 1] + width),
                        "event_day": bool(h in (8, 20)),
                        "fold": 3,
                    }
                )
    return pd.DataFrame(rows)

