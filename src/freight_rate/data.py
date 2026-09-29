"""Data loading and the per-day market context."""
from __future__ import annotations

import pandas as pd

from .config import DATA_DIR


def load(name: str) -> pd.DataFrame:
    """Read a CSV from data/ and parse its `date` column."""
    df = pd.read_csv(DATA_DIR / name)
    if "date" in df:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, validation, validation_template, december_chart_inputs)."""
    return (
        load("train_test.csv"),
        load("validation.csv"),
        pd.read_csv(DATA_DIR / "validation_predictions_template.csv"),
        load("december_chart_inputs.csv"),
    )


def daily_context(*frames: pd.DataFrame) -> pd.DataFrame:
    """Per-day market level built from FEATURE columns only (never labels).

    `market_index` is essentially one number per day (98% of its variance is between days),
    with random per-row noise and some missing values. The daily median is a clean estimate
    of that day's level. `quote_signal` is mostly per-load noise, so only its daily mean is
    informative.

    Pooling train + validation feature columns gives every date, including Nov-Dec, its own
    level. No target value is used, so this cannot leak labels.
    """
    x = pd.concat([f[["date", "market_index", "quote_signal"]] for f in frames])
    g = x.groupby("date")
    return pd.DataFrame({"mi": g["market_index"].median(), "qs": g["quote_signal"].mean()})
