#!/usr/bin/env bash
# Reproduces every deliverable from the raw data in data/ (about 2-3 minutes on one CPU core).
set -euo pipefail
cd "$(dirname "$0")"

python -m pip install -r requirements.txt
python scripts/run_validation.py        # forward-in-time CV + unseen-city test -> outputs/cv_*.csv
python scripts/make_predictions.py      # final fit -> outputs/validation_predictions.csv, december_chart_predictions.csv
python score.py --predictions outputs/validation_predictions.csv \
                --december-predictions outputs/december_chart_predictions.csv \
                --output-dir outputs/scorer_results
jupyter nbconvert --to notebook --execute --inplace notebooks/freight_rate_solution.ipynb   # writes split schematic
python reports/build_report.py          # -> reports/report.pdf
