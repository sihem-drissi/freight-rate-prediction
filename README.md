# Freight Rate Prediction

Forecast the posted rate of 12,000 loads for **Nov-Dec 2025** from 48,000 labeled loads (Jan-Oct 2025), and produce a fixed December price curve for one lane.

| | |
|---|---|
| **Model** | LightGBM for load shape **+** linear terms for market level, time trend and city effects; coordinate-kNN fallback for unseen cities |
| **Validation** | Rolling-origin (forward-in-time) CV: 5 folds, 2-month horizon, everything refit inside each fold |
| **Result (mean of 5 forward folds)** | **MAE $105, MAPE 4.5%** (plain LightGBM: $120 / 5.0%; median $/mile baseline: $221 / 10.5%) |
| **Key data issue** | ~1.4% of training labels are corrupted; removed from training only, never from evaluation |

**Start with the notebook:** [`notebooks/freight_rate_solution.ipynb`](notebooks/freight_rate_solution.ipynb). It walks through the data-quality audit, EDA, validation design, model, results and decision log, with all outputs included.

## Run

Python 3.10+. Put the four  files in `data/` (see [`data/README.md`](data/README.md)), then:

```bash
./run_all.sh
```

or step by step:

```bash
pip install -r requirements.txt
python scripts/run_validation.py       # forward CV + unseen-city test      -> outputs/cv_*.csv, unseen_city_test.csv
python scripts/make_predictions.py     # fit on all labeled data, predict  -> outputs/validation_predictions.csv
                                       #                                     outputs/december_chart_predictions.csv
python score.py --predictions outputs/validation_predictions.csv \
                --december-predictions outputs/december_chart_predictions.csv \
                --output-dir outputs/scorer_results       # validates files, draws candidate_december.png

```

Runs are seeded and deterministic: a rerun reproduces identical files.

## Layout

```
notebooks/freight_rate_solution.ipynb   full narrative: EDA, validation, model, results, decisions
src/freight_rate/
    config.py        paths, constants, CV folds, LightGBM parameters
    data.py          loading + per-day market context (features only, no labels)
    model.py         FreightModel: GBM + linear backfit, label cleaning, unseen-city kNN
    validation.py    metrics, baselines, rolling-origin CV, unseen-city stress test
    submission.py    builds the two deliverable files
scripts/             run_validation.py, make_predictions.py
outputs/             validation_predictions.csv, december chart + predictions, CV results
score.py             provided scorer (unchanged)
```

## Approach in brief

- **Split:** validation is a two-month forecast, so validation is forward-in-time. A shuffled split makes the same model look ~24% better on MAE than it is on future data (measured in the notebook).
- **Data quality:** corrupted labels (spikes 2-5x, drops to ~0.3x) dropped from training only; negative weights are sign flips (absolute value); missing `market_index` filled from that day's level; missing `weight` left to LightGBM; `distance` used for pricing, coordinates only to place cities.
- **Model:** `log(rate) = GBM(log distance, |weight|, equipment) + linear(log daily market level, row deviation, daily quote level, months elapsed, city effects)`. Trees cannot extrapolate a trend past the training period, so terms that must extrapolate are linear.
- **Daily market context:** built from `market_index` / `quote_signal` feature columns of both files (features only, no labels), so each Nov-Dec date has its own level.

## Known limitations

Trend extrapolated two months; December seasonality unobservable from Jan-Oct data; lane-specific effects not modelled; loads at the 70-mile distance floor over-predicted by ~7%; model selection used the same folds as the reported CV, so CV numbers are slightly optimistic.
