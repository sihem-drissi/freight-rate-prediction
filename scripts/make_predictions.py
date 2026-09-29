"""Fit on ALL labeled data, then write the two deliverable prediction files.

  outputs/validation_predictions.csv      load_id,predicted_rate (12,000 rows, template order)
  outputs/december_chart_predictions.csv  the fixed December scenario, ready for score.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from freight_rate.config import OUT_DIR, TARGET
from freight_rate.data import daily_context, load_all
from freight_rate.model import FreightModel
from freight_rate.submission import december_submission, validation_submission


def main():
    train, val, template, december = load_all()
    model = FreightModel(daily_context(train, val)).fit(train, np.log(train[TARGET].values))
    print(f"dropped {model.n_dropped} corrupted training labels ({100 * model.n_dropped / len(train):.2f}%)")

    validation_submission(model, val, template).to_csv(OUT_DIR / "validation_predictions.csv", index=False)
    dec = december_submission(model, december)
    dec.to_csv(OUT_DIR / "december_chart_predictions.csv", index=False)
    print("December forecast:", dec["predicted_rate"].describe().round(1).to_dict())


if __name__ == "__main__":
    main()
