"""Loading, validating, and sampling the M5 competition files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
SALES_CANDIDATES = ("sales_train_evaluation.csv", "sales_train_validation.csv")


def _day_key(value: str) -> int:
    return int(value[2:])


def _balanced_quotas(capacities: dict[tuple[str, str], int], target: int) -> dict[tuple[str, str], int]:
    """Allocate a sample as evenly as possible across store x demand strata."""
    quotas = {key: 0 for key in sorted(capacities)}
    remaining = min(target, sum(capacities.values()))
    while remaining:
        active = [key for key in quotas if quotas[key] < capacities[key]]
        if not active:
            break
        share, extra = divmod(remaining, len(active))
        if share == 0:
            for key in active[:remaining]:
                quotas[key] += 1
            remaining = 0
            continue
        used = 0
        for i, key in enumerate(active):
            addition = min(share + (1 if i < extra else 0), capacities[key] - quotas[key])
            quotas[key] += addition
            used += addition
        remaining -= used
    return quotas


def assign_demand_groups(frame: pd.DataFrame, history_columns: Iterable[str]) -> pd.Series:
    """Label each series low/medium/high by nonzero-day frequency within store."""
    history_columns = list(history_columns)
    if not history_columns:
        raise ValueError("At least one sales day is required to derive demand groups.")
    frequency = frame[history_columns].gt(0).mean(axis=1)
    labels = pd.Series(index=frame.index, dtype="object")
    for _, indices in frame.groupby("store_id", sort=True).groups.items():
        percentile = frequency.loc[indices].rank(method="first", pct=True)
        labels.loc[indices] = np.select(
            [percentile <= 1 / 3, percentile <= 2 / 3],
            ["low", "medium"],
            default="high",
        )
    return labels.astype(str)


def stratified_sample(frame: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """Sample evenly across store and demand-frequency strata, reproducibly."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if "demand_group" not in frame:
        raise ValueError("frame must include demand_group")
    if sample_size >= len(frame):
        return frame.sort_values("id").reset_index(drop=True).copy()

    grouped = {
        key: part
        for key, part in frame.groupby(["store_id", "demand_group"], sort=True, observed=True)
    }
    capacities = {key: len(part) for key, part in grouped.items()}
    quotas = _balanced_quotas(capacities, sample_size)
    selected = []
    for offset, (key, quota) in enumerate(quotas.items()):
        if quota:
            selected.append(grouped[key].sample(n=quota, random_state=seed + offset))
    return pd.concat(selected, ignore_index=True).sort_values("id").reset_index(drop=True)


def _sales_path(data_dir: Path) -> Path:
    for name in SALES_CANDIDATES:
        candidate = data_dir / name
        if candidate.exists():
            return candidate
    choices = ", ".join(SALES_CANDIDATES)
    raise FileNotFoundError(f"Expected one of these sales files in {data_dir}: {choices}")


def load_m5_dataset(
    data_dir: str | Path,
    sample_size: int = 500,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return sampled daily sales and a normalized calendar table.

    Expected files are an M5 sales file, calendar.csv, and sell_prices.csv. The
    future sell-price rows are retained in the source only; model features use
    the latest price known on or before each forecast origin.
    """
    data_dir = Path(data_dir)
    sales_path = _sales_path(data_dir)
    calendar_path = data_dir / "calendar.csv"
    prices_path = data_dir / "sell_prices.csv"
    for path in (calendar_path, prices_path):
        if not path.exists():
            raise FileNotFoundError(f"Required M5 file not found: {path}")

    header = pd.read_csv(sales_path, nrows=0).columns.tolist()
    day_columns = sorted((name for name in header if name.startswith("d_")), key=_day_key)
    missing_ids = sorted(set(ID_COLUMNS) - set(header))
    if missing_ids or not day_columns:
        raise ValueError(f"Sales file is missing required columns: {missing_ids or 'd_* sales columns'}")

    sales_dtypes = {name: "uint16" for name in day_columns}
    sales = pd.read_csv(sales_path, dtype=sales_dtypes)
    if sales["id"].duplicated().any():
        raise ValueError("Sales file contains duplicate series IDs.")
    history_end = max(1, int(len(day_columns) * 0.70))
    demand_groups = assign_demand_groups(sales, day_columns[:history_end]).rename("demand_group")
    sales = pd.concat([sales, demand_groups], axis=1)
    sample = stratified_sample(sales, sample_size=sample_size, seed=seed)

    calendar = pd.read_csv(calendar_path)
    calendar_required = {"d", "date", "wm_yr_wk"}
    if missing := sorted(calendar_required - set(calendar.columns)):
        raise ValueError(f"calendar.csv is missing required columns: {missing}")
    calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
    if calendar["date"].isna().any() or calendar["d"].duplicated().any():
        raise ValueError("calendar.csv has invalid dates or duplicate d keys.")
    event_columns = [column for column in ("event_name_1", "event_name_2") if column in calendar]
    calendar["event_day"] = calendar[event_columns].notna().any(axis=1) if event_columns else False
    calendar["event_day"] = calendar["event_day"].astype(bool)
    calendar["wm_yr_wk"] = pd.to_numeric(calendar["wm_yr_wk"], errors="raise").astype(int)
    snap_columns = [column for column in calendar if column.startswith("snap_")]

    prices = pd.read_csv(prices_path)
    price_required = {"item_id", "store_id", "wm_yr_wk", "sell_price"}
    if missing := sorted(price_required - set(prices.columns)):
        raise ValueError(f"sell_prices.csv is missing required columns: {missing}")
    prices["wm_yr_wk"] = pd.to_numeric(prices["wm_yr_wk"], errors="raise").astype(int)
    prices["sell_price"] = pd.to_numeric(prices["sell_price"], errors="coerce")
    if prices.duplicated(["item_id", "store_id", "wm_yr_wk"]).any():
        raise ValueError("sell_prices.csv contains duplicate item/store/week keys.")

    long = sample.melt(
        id_vars=ID_COLUMNS + ["demand_group"],
        value_vars=day_columns,
        var_name="d",
        value_name="actual_units",
    )
    long = long.merge(
        calendar[["d", "date", "wm_yr_wk", "event_day"] + snap_columns],
        on="d",
        how="left",
        validate="many_to_one",
    )
    if long["date"].isna().any():
        missing_days = long.loc[long["date"].isna(), "d"].drop_duplicates().head(5).tolist()
        raise ValueError(f"Sales days are missing from calendar.csv: {missing_days}")
    long = long.merge(
        prices[["item_id", "store_id", "wm_yr_wk", "sell_price"]],
        on=["item_id", "store_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )
    long["date"] = pd.to_datetime(long["date"])
    long["actual_units"] = pd.to_numeric(long["actual_units"], errors="raise")
    if (long["actual_units"] < 0).any() or not np.isfinite(long["actual_units"]).all():
        raise ValueError("Sales must contain finite, nonnegative unit counts.")

    long["snap"] = 0.0
    for state, indices in long.groupby("state_id", sort=False).groups.items():
        snap_column = f"snap_{state}"
        if snap_column in long:
            long.loc[indices, "snap"] = pd.to_numeric(long.loc[indices, snap_column], errors="coerce").fillna(0)
    long = long.sort_values(["id", "date"]).reset_index(drop=True)
    if long.duplicated(["id", "date"]).any():
        raise ValueError("Melted sales contain duplicate series/date rows.")

    normalized_calendar = calendar[["d", "date", "wm_yr_wk", "event_day"] + snap_columns].copy()
    return long, normalized_calendar
