# Retail Forecast Reliability Lab

Can a more complex model improve a simple weekly sales forecast—and does that improvement hold up on later dates?

This project compares a weekly seasonal-naive forecast with a pooled LightGBM model on the public M5 retail dataset. It samples 500 product-store series with a fixed seed and forecasts 28 days ahead. Two chronological folds are used for model selection and interval calibration; the third fold is kept for the final evaluation.

## Result on the held-out period

The figures below are from the committed run in `artifacts/metrics.json`. The final fold was trained through April 24, 2016 and evaluated on the following 28 days (14,000 product-day forecasts).

| Forecast | WAPE | MAE |
| --- | ---: | ---: |
| Weekly seasonal-naive | 91.32% | 1.096 |
| LightGBM | **75.65%** | **0.908** |

LightGBM reduced WAPE by 17.16% relative to the baseline on this sample and period. That is a relative reduction, not a 17-point increase in accuracy. A 75.65% WAPE still means the total absolute error is substantial.

### What the scores mean

One thing I had to understand carefully was that WAPE and interval coverage describe different things. WAPE compares the total absolute forecast error with total actual sales; lower is better. Interval coverage measures how often actual sales fell inside the estimated prediction range. The nominal coverage target was 80%, and observed overall coverage on this fold was 80.00%. That does not mean the forecasts are “80% accurate,” and coverage varies across stores, products, and forecast days.

The simple model still had a useful counterexample. In department `HOBBIES_2`, seasonal-naive WAPE was 141.83%, slightly lower than LightGBM's 142.50%. The selector chose LightGBM for all three demand-frequency groups using the first two folds, so this department shows that a development-period choice does not guarantee a win in every later segment.

## How the comparison works

- Each fold predicts the next 28 days using only information available before that forecast begins.
- The baseline repeats the weekly seasonal pattern. LightGBM is trained across the sampled series using sales history, calendar and event information, and the latest price known at the forecast origin.
- The first two folds determine which model to use for each demand-frequency group and calibrate empirical 80% intervals. The final fold is used once to report held-out results.
- `artifacts/predictions.parquet` contains the daily predictions. `artifacts/metrics.json` contains overall and grouped metrics, selections, and interval coverage.

## Run it

Create an environment and install the packages:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download the M5 files from the [Kaggle competition data page](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) and place these in `data/raw/`:

- `calendar.csv`
- `sell_prices.csv`
- `sales_train_evaluation.csv` or `sales_train_validation.csv`

Raw data is excluded from Git. Run the pipeline from the repository root:

```powershell
python -m retail_forecast.pipeline --data-dir data/raw --output-dir artifacts
```

To open the dashboard:

```powershell
streamlit run retail_forecast/app.py
```

The dashboard shows the forecast comparison and supports a quick forecast from a user's CSV. That upload workflow uses a single recent holdout for a fast comparison; it is separate from the three-fold M5 experiment above.

## Check the code

Run the unit tests with:

```powershell
python -m unittest discover -s tests -v
```

## Scope and limitations

The M5 data records sales, not unconstrained demand, inventory, or stockouts. The results apply to this fixed sample and one final 28-day period; they do not show business savings or prove that the same model will work for another retailer. Prediction intervals are empirical estimates calibrated from development-fold errors, not guaranteed bounds. The upload feature is a practical demo, not a production forecasting service.
