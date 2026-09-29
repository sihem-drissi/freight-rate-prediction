"""Forward-in-time validation -> outputs/cv_folds.csv, cv_summary.csv, unseen_city_test.csv."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from freight_rate.config import OUT_DIR
from freight_rate.data import daily_context, load_all
from freight_rate.validation import rolling_origin_cv, unseen_city_test


def main():
    train, val, _, _ = load_all()
    context = daily_context(train, val)

    folds = rolling_origin_cv(train, context)
    summary = folds.groupby("model")[["MAE", "MAPE", "MedAPE", "core_MAPE", "core_logMAE", "bias_pct"]].mean().round(3)
    folds.to_csv(OUT_DIR / "cv_folds.csv", index=False)
    summary.to_csv(OUT_DIR / "cv_summary.csv")
    print(summary.to_string())

    unseen = unseen_city_test(train, context).round(3)
    unseen.to_csv(OUT_DIR / "unseen_city_test.csv")
    print(unseen.to_string())


if __name__ == "__main__":
    main()
