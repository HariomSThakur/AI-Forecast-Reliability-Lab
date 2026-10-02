# Retail Forecast Reliability Lab

Can a more complex model improve a simple weekly sales forecast, and does that improvement hold up on later dates?

This project compares a weekly seasonal-naive forecast with a pooled LightGBM model on the public M5 retail dataset. It samples 500 product-store series with seed 42 and forecasts 28 days ahead. The first two chronological folds are used for model selection and interval calibration; the third is held out for evaluation.

## Held-out results

These figures come from the committed run in `artifacts/metrics.json`. The final fold was trained through April 24, 2016 and evaluated from April 25 to May 22, 2016, covering 14,000 product-day forecasts.

| Forecast | WAPE | MAE |
| --- | ---: | ---: |
| Weekly seasonal-naive | 91.32% | 1.096 |
| LightGBM | **75.65%** | **0.908** |

On this sample and period, LightGBM's WAPE was 17.16% lower than the baseline. That is a relative reduction in error; it does not mean the forecast was 17 percentage points more accurate. The final WAPE is still substantial.

### Reading the scores

I initially found WAPE and interval coverage easy to mix up. WAPE measures the total absolute forecast error relative to total actual sales; lower is better. Coverage measures how often actual sales fell inside the prediction range. Overall coverage on the held-out fold was 80.00%, matching the nominal 80% target. That does not mean the point forecasts were “80% accurate,” and coverage varies by store, department, and forecast day.

The simple model still has a useful counterexample. In `HOBBIES_2`, seasonal-naive WAPE was 141.83%, slightly lower than LightGBM's 142.50%. The selector chose LightGBM for all three demand-frequency groups using the first two folds, so that development-period choice did not win in every department on the held-out dates.

## How the comparison works

- Each fold predicts the next 28 days using information available before that forecast begins.
- The baseline repeats the sales from the same weekday one week earlier. LightGBM pools the sampled series and uses sales lags and rolling averages, calendar and event features, and the latest price known at the forecast origin.
- The first two folds determine the model choice for each demand-frequency group and calibrate empirical 80% prediction intervals. The final fold is used once for the held-out results.
- `artifacts/predictions.parquet` contains daily predictions for all folds. `artifacts/metrics.json` contains overall results and breakdowns by horizon, store, department, demand frequency, and event day.

## Run the project

Create a virtual environment and install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download the M5 files from the [Kaggle competition data page](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) and put these files in `data/raw/`:

- `calendar.csv`
- `sell_prices.csv`
- `sales_train_evaluation.csv` or `sales_train_validation.csv`

Raw data is excluded from Git. From the repository root, run the pipeline:

```powershell
python -m retail_forecast.pipeline --data-dir data/raw --output-dir artifacts
```

Start the dashboard with:

```powershell
streamlit run retail_forecast/app.py
```

The dashboard displays the forecast comparison and accepts a CSV upload for a quick forecast check. The upload workflow uses one recent holdout; it is separate from the three-fold M5 experiment.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Limitations

M5 records sales, not unconstrained demand, inventory, or stockouts. These results describe one fixed sample and one 28-day evaluation period; they do not estimate business savings or establish performance at another retailer. The prediction intervals are empirical estimates based on development-fold errors, not guaranteed bounds. The upload workflow is a demo with a single quick holdout, not a multi-fold benchmark.
