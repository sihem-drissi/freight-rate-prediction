"""Forward-in-time validation, baselines and the unseen-city stress test."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import CLEAN_THRESHOLD, EQUIPMENT_CODES, FOLDS, GBM_PARAMS, TARGET
from .model import FreightModel


def metrics(y_true: np.ndarray, pred: np.ndarray) -> dict:
    """MAE/MAPE on ALL rows (labels are never filtered at evaluation), plus 'core' metrics.

    'Core' = rows whose label is plausible (within ~28% of the prediction). The corrupted
    labels dominate the all-row numbers, so showing both keeps the comparison honest.
    'bias_pct' is the median signed error, which reveals systematic drift over time.
    """
    ape = np.abs(pred - y_true) / y_true
    log_err = np.abs(np.log(y_true) - np.log(pred))
    core = log_err < CLEAN_THRESHOLD
    return dict(
        MAE=np.abs(pred - y_true).mean(),
        MAPE=100 * ape.mean(),
        MedAPE=100 * np.median(ape),
        core_MAPE=100 * ape[core].mean(),
        core_logMAE=log_err[core].mean(),
        bias_pct=100 * np.median(np.log(pred / y_true)),   # signed: <0 means under-forecasting
    )


# ---------------------------------------------------------------- comparison models
def baseline_per_mile(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Median $/mile by equipment type times distance. The 'no ML' reference point."""
    per_mile = (train[TARGET] / train["distance"]).groupby(train["equipment"]).median()
    return (test["distance"] * test["equipment"].map(per_mile)).values


def _raw_features(d: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "log_dist": np.log(d["distance"]), "weight": d["weight"].abs(),
        "equip": d["equipment"].map(EQUIPMENT_CODES),
        "plat": d["pickup_lat"], "plon": d["pickup_lon"],
        "dlat": d["delivery_lat"], "dlon": d["delivery_lon"],
        "market_index": d["market_index"], "quote_signal": d["quote_signal"],
    })


def plain_lightgbm(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """A single well-configured LightGBM on all raw features, no trend or cleaning logic."""
    model = lgb.LGBMRegressor(**{**GBM_PARAMS, "objective": "l1"})
    model.fit(_raw_features(train), np.log(train[TARGET]))
    return np.exp(model.predict(_raw_features(test)))


# ---------------------------------------------------------------- validation runs
def split_by_date(train: pd.DataFrame, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit on rows before `start`; score rows in [start, end)."""
    return train[train["date"] < start], train[(train["date"] >= start) & (train["date"] < end)]


def rolling_origin_cv(train: pd.DataFrame, context: pd.DataFrame, folds=FOLDS, verbose=True) -> pd.DataFrame:
    """Score the baseline, plain LightGBM and the final hybrid on every forward fold."""
    rows = []
    for start, end in folds:
        fit_df, test_df = split_by_date(train, start, end)
        y = test_df[TARGET].values
        preds = {
            "1. Median $/mile baseline": baseline_per_mile(fit_df, test_df),
            "2. Plain LightGBM": plain_lightgbm(fit_df, test_df),
            "3. Hybrid (final)": FreightModel(context).fit(fit_df, np.log(fit_df[TARGET].values)).predict(test_df),
        }
        for name, p in preds.items():
            rows.append(dict(model=name, fold=start[:7], n_train=len(fit_df), n_test=len(test_df), **metrics(y, p)))
        if verbose:
            print(f"fold starting {start}: done", flush=True)
    return pd.DataFrame(rows)


def random_split_cv(train: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> dict:
    """The WRONG way (shuffled K-fold), shown only to demonstrate how optimistic it is."""
    from sklearn.model_selection import KFold
    errs = []
    for fit_idx, test_idx in KFold(n_splits, shuffle=True, random_state=seed).split(train):
        fit_df, test_df = train.iloc[fit_idx], train.iloc[test_idx]
        errs.append(metrics(test_df[TARGET].values, plain_lightgbm(fit_df, test_df)))
    return pd.DataFrame(errs).mean().to_dict()


def unseen_city_test(train: pd.DataFrame, context: pd.DataFrame, n_hold: int = 8, seed: int = 7) -> pd.DataFrame:
    """Hold out `n_hold` cities, refit, and forecast Sep-Oct loads that touch them.

    Mimics validation.csv, where 8 cities never appear in training.
    """
    cities = sorted(set(train["pickup"]) | set(train["delivery"]))
    held = set(np.random.RandomState(seed).choice(cities, n_hold, replace=False))
    touches = lambda d: d["pickup"].isin(held) | d["delivery"].isin(held)
    fit_all, test = train[train["date"] < "2025-09-01"], train[train["date"] >= "2025-09-01"]
    test = test[touches(test)]
    out = {}
    for label, fit_df in (("Cities seen in training", fit_all), ("8 cities unseen (kNN fallback)", fit_all[~touches(fit_all)])):
        p = FreightModel(context).fit(fit_df, np.log(fit_df[TARGET].values)).predict(test)
        out[label] = metrics(test[TARGET].values, p)
    return pd.DataFrame(out).T
