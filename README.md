# Retail Forecast Reliability Lab

**Question:** When does machine learning improve on a simple forecast, and when should you trust the simpler model instead?

This project compares a weekly seasonal-naive forecast with a pooled LightGBM model on the M5 retail sales data. It uses three chronological 28-day backtests, selects a model by demand-frequency group using only the first two folds, calibrates empirical 80% prediction intervals on those development folds, and evaluates the unchanged choices once on the final fold.

The differentiator is the reliability analysis: performance is reported by forecast horizon, store, department, demand frequency, and event day. A model win is not required; the goal is to show where each approach works and how the selection rule behaves on held-out data.

## Quick start

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the unit tests:

```powershell
python -m unittest discover -s tests -v
```

## Get the M5 data

Download the files from the [M5 Forecasting - Accuracy competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data). The competition may require a Kaggle account and acceptance of its terms. Put these files in `data/raw/`:

- `calendar.csv`
- `sell_prices.csv`
- `sales_train_evaluation.csv` or `sales_train_validation.csv`

The raw folder is ignored by Git. The pipeline prefers `sales_train_evaluation.csv` when both sales files are present. It stratifies a fixed-seed sample of 500 series across store and demand-frequency groups before converting the selected rows to daily records.

Run the pipeline from the repository root:

```powershell
python -m retail_forecast.pipeline --data-dir data/raw --output-dir artifacts
```

The pipeline writes:

- `artifacts/predictions.parquet` — all three folds, actuals, baseline and LightGBM forecasts, the development-selected forecast, interval bounds on the final fold, and fold metadata.
- `artifacts/metrics.json` — development model choices and metrics plus final-fold results by horizon, store, department, demand group, and event day.

## Current run snapshot

With the default seed and 500-series sample from `sales_train_evaluation.csv`, the final 28-day holdout contains 14,000 product-day forecasts. After adding seasonal features and development-only early stopping, the seasonal-naive baseline produced **91.32% WAPE** and LightGBM produced **75.65% WAPE**, a **17.16% relative reduction** from the baseline. This is a modest improvement over the earlier fixed-round LightGBM run (75.84% WAPE), not a dramatic jump. The selected model's empirical interval coverage was **80.00%** against an 80% target. LightGBM was selected for all three demand-frequency groups. In `HOBBIES_2`, however, the baseline had **141.83% WAPE** versus LightGBM's **142.50%**; the development-fold selector preferred LightGBM, but that choice did not carry over to this final window. This is an honest example of why more training or features cannot guarantee lower error on every segment. Results are specific to this fixed sample and final time window.

The model uses daily-sales lags (including a 52-week seasonal lag), shifted rolling averages (including a year of history), target-date calendar information, and the latest sell price available on or before each forecast origin. It deliberately does not use future sell-price rows as features. LightGBM can train for up to 800 rounds; a chronological validation slice from within each training fold selects the stopping point, then the model refits on that fold's full training data. The final holdout is not used to choose training length.

M5 anonymizes item titles and provides state/store codes, not city names. In the app these are shown as readable food/hobby/household item codes and state-level store locations. They are not relabeled as actual grocery products or cities because the source data does not identify those. The uploaded-data sample uses clearly synthetic grocery products and city names for demonstration only.

## Open the dashboard

```powershell
streamlit run retail_forecast/app.py
```

The dashboard uses a dark layout with a compact left navigation, clear metric cards, straightforward line charts, and readable result tables. Until real artifacts exist, the app shows a small, deterministic **synthetic preview** and labels it clearly. Run the M5 pipeline to make it display measured results. For a public demo, push the repository and compact `artifacts/` outputs to GitHub, then deploy the app with [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## Use the app with your own CSV

The **Your CSV** section accepts a CSV with a date column and a nonnegative numeric quantity column, for example sales, orders, or visits. Select one or more optional ID columns to identify products and locations; the app combines them into a series key and lets you choose one series at a time. You can load a synthetic sample directly in the app or download it as a CSV. Forecast charts use the native Streamlit line chart, while estimated interval bounds remain available in the forecast table.

The app asks whether the data is daily, weekly, or monthly, how to combine repeated rows in a period (sum or mean), and how to fill missing periods (zero or carry forward). It tests the methods on the most recent portion of the selected series, compares seasonal-naive and LightGBM WAPE, then forecasts the next selected number of periods. For short series, it explains when LightGBM does not have enough history and still shows the simple forecast. The CSV is processed in memory and is not written into the project directory. Avoid uploading sensitive information to a public deployment.

The uploaded-data forecast is a practical demonstration, not a production forecasting service. Its one recent holdout is useful for a quick comparison but is not an independent, multi-fold benchmark. The estimated range is based on recent errors and is not a guarantee. The forecasting choices are designed for nonnegative sales or demand quantities; signed measurements and irregular event data need a different workflow.

Suggested CSV shape:

```csv
date,sales,product
2025-01-01,12,Widget A
2025-01-02,15,Widget A
2025-01-01,7,Widget B
```

## Evaluation design

- Use three sequential, non-overlapping 28-day validation folds. Each model is trained only on data through that fold's forecast origin.
- Select LightGBM for a demand-frequency group only when its mean fold-level WAPE across the development folds is lower than the seasonal-naive WAPE. Ties and undefined WAPE choose the simpler baseline.
- Calibrate asymmetric residual quantiles by demand group and horizon using development predictions. Sparse calibration cells fall back to a demand-group pool, then a horizon pool, then a global pool.
- Evaluate the selected model, both individual models, and empirical interval coverage on the final fold. Intervals are empirical estimates, not formal distribution-free guarantees.
- WAPE is reported as unavailable when the group has zero total actual demand. The report also includes MAE and mean forecast bias.

## Limitations

M5 records sales, not true unconstrained demand, inventory, stockouts, or replenishment outcomes. Price features use the last observed price at the forecast origin. The sample and historical U.S. retail data support a reproducible forecasting experiment; they do not establish business savings or generalization to another retailer.

## Resume bullet

> Compared seasonal-naive and LightGBM forecasts across 500 product-store series using three rolling 28-day backtests; seasonal features and development-only early stopping produced 75.65% final-fold WAPE versus 91.32% for the baseline, a 17.16% relative reduction, with 80.00% empirical coverage for nominal 80% intervals. Identified `HOBBIES_2` as a segment where the baseline performed slightly better on the final window.
