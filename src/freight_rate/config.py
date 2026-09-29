"""Paths, constants and the validation folds used everywhere in the project."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

TARGET = "posted_rate"

# Time origin for the linear trend term (first day of the labeled data).
T0 = pd.Timestamp("2025-01-01")

EQUIPMENT_CODES = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}

# Rolling-origin validation: fit on everything BEFORE `start`, score the two months after it.
# A two-month horizon mirrors the real task (train ends Oct 31, validation is Nov-Dec).
FOLDS = [
    ("2025-05-01", "2025-07-01"),
    ("2025-06-01", "2025-08-01"),
    ("2025-07-01", "2025-09-01"),
    ("2025-08-01", "2025-10-01"),
    ("2025-09-01", "2025-11-01"),
]

# A label further than this (in log-rate units, ~28%) from the model is treated as corrupted.
CLEAN_THRESHOLD = 0.25

# LightGBM settings for the non-linear "load shape" part of the model.
GBM_PARAMS = dict(
    objective="l2",
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    verbose=-1,
    random_state=0,
)
