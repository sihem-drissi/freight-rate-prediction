"""Build the two deliverable files from a fitted model."""
from __future__ import annotations

import pandas as pd

from .model import FreightModel


def validation_submission(model: FreightModel, val: pd.DataFrame, template: pd.DataFrame) -> pd.DataFrame:
    """`load_id,predicted_rate` for all 12,000 loads, in the order of the provided template."""
    pred = pd.Series(model.predict(val), index=val["load_id"])
    out = pd.DataFrame({
        "load_id": template["load_id"],
        "predicted_rate": pred.reindex(template["load_id"]).values.round(2),
    })
    assert len(out) == 12_000, "expected 12,000 rows"
    assert out["predicted_rate"].notna().all() and (out["predicted_rate"] > 0).all()
    return out


def december_submission(model: FreightModel, december: pd.DataFrame) -> pd.DataFrame:
    """Fill the fixed December scenario (one lane, only the date changes).

    The chart file has no lat/lon, so city coordinates are taken from training. It also has no
    market_index, so each date uses its own daily market level from the validation.csv features.
    """
    feats = december.copy()
    feats["pickup_lat"], feats["pickup_lon"] = zip(*feats["pickup"].map(model.coords))
    feats["delivery_lat"], feats["delivery_lon"] = zip(*feats["delivery"].map(model.coords))
    out = december.copy()
    out["predicted_rate"] = model.predict(feats).round(2)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out
