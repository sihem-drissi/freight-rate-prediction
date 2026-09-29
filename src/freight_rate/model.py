"""The final model: a GBM for load shape plus a linear model for market, time and city effects.

    log(rate) =  GBM( log distance, |weight|, equipment )                    <- load shape
               + b1 * log(daily market index)                                <- market level
               + b2 * (row market_index vs. its day's level)
               + b3 * daily mean quote_signal
               + b4 * months since 2025-01-01                                <- time trend
               + city_effect(pickup) + city_effect(delivery)                 <- geography

Why a hybrid? The validation months (Nov-Dec) all lie AFTER the training data. Trees cannot
extrapolate a trend beyond the range they saw, but a linear term can. So the GBM handles the
non-linear shape of a load, and the linear part carries everything that must extrapolate.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from .config import CLEAN_THRESHOLD, EQUIPMENT_CODES, GBM_PARAMS, T0


class FreightModel:
    """Fit on log(rate); `predict` returns dollars."""

    def __init__(self, context: pd.DataFrame, clean_threshold: float = CLEAN_THRESHOLD, k_unseen: int = 4):
        self.ctx = context              # per-day market level from data.daily_context()
        self.threshold = clean_threshold
        self.k_unseen = k_unseen        # neighbours used for cities never seen in training

    # ------------------------------------------------------------------ features
    @staticmethod
    def _gbm_features(d: pd.DataFrame) -> pd.DataFrame:
        """Load-shape features for the GBM."""
        return pd.DataFrame({
            "log_dist": np.log(d["distance"]),
            "weight": d["weight"].abs(),              # negative weights are sign errors
            "equip": d["equipment"].map(EQUIPMENT_CODES),
        })                                            # missing weights stay NaN (LightGBM handles them)

    def _linear_features(self, d: pd.DataFrame) -> pd.DataFrame:
        """Market and time features for the linear part."""
        day_level = d["date"].map(self.ctx["mi"])
        row_value = d["market_index"] if "market_index" in d else pd.Series(np.nan, index=d.index)
        return pd.DataFrame({
            "log_mi": np.log(day_level),
            "mi_dev": np.log(row_value.fillna(day_level)) - np.log(day_level),  # 0 if missing
            "qs_day": d["date"].map(self.ctx["qs"]),
            "months": (d["date"] - T0).dt.days / 30.0,
        })

    def _city_matrix(self, d: pd.DataFrame) -> np.ndarray:
        """One-hot city columns; a row has +1 for its pickup city and +1 for its delivery city."""
        M = np.zeros((len(d), len(self.cities)))
        for col in ("pickup", "delivery"):
            for row, city in enumerate(d[col].values):
                if city in self.city_index:
                    M[row, self.city_index[city]] += 1
        return M

    # ------------------------------------------------------------------ fit
    def fit(self, d: pd.DataFrame, y: np.ndarray) -> "FreightModel":
        """`y` is log(posted_rate). Corrupted labels are detected and dropped (training only)."""
        self._register_cities(d)

        X_gbm = self._gbm_features(d)
        linear_cols = list(self._linear_features(d.head(1)).columns)
        Z = np.hstack([self._linear_features(d).values, self._city_matrix(d)])

        # Backfitting: alternate GBM <-> linear model, each fitting what the other left over.
        # After passes 0 and 1 we drop rows the model cannot explain (corrupted labels), so the
        # final pass (the full-size GBM) trains on clean data only.
        keep = np.ones(len(d), dtype=bool)
        linear_part = np.zeros(len(d))
        self.lr = LinearRegression()
        for it in range(3):
            n_trees = GBM_PARAMS["n_estimators"] if it == 2 else 300
            self.gbm = lgb.LGBMRegressor(**{**GBM_PARAMS, "n_estimators": n_trees})
            self.gbm.fit(X_gbm[keep], (y - linear_part)[keep])
            gbm_part = self.gbm.predict(X_gbm)
            self.lr.fit(Z[keep], (y - gbm_part)[keep])
            linear_part = self.lr.predict(Z)
            if it < 2:
                keep = np.abs(y - gbm_part - linear_part) < self.threshold
        self.n_dropped = int((~keep).sum())
        self.keep_mask = keep

        # Expose readable coefficients; centre city effects (each row carries two of them).
        n_lin = len(linear_cols)
        self.lin_coef = dict(zip(linear_cols, self.lr.coef_[:n_lin]))
        raw_city = pd.Series(self.lr.coef_[n_lin:], index=self.cities)
        self.city_effect = raw_city - raw_city.mean()
        self.lr.intercept_ = self.lr.intercept_ + 2 * raw_city.mean()

        # Fallback for cities that never appear in training: nearest known cities by coordinates.
        points = np.array([self.coords[c] for c in self.cities])
        self.knn = KNeighborsRegressor(self.k_unseen, weights="distance").fit(points, self.city_effect.values)
        return self

    def _register_cities(self, d: pd.DataFrame) -> None:
        self.cities = sorted(set(d["pickup"]) | set(d["delivery"]))
        self.city_index = {c: i for i, c in enumerate(self.cities)}
        coords = {}
        for c, la, lo in zip(d["pickup"], d["pickup_lat"], d["pickup_lon"]):
            coords[c] = (la, lo)
        for c, la, lo in zip(d["delivery"], d["delivery_lat"], d["delivery_lon"]):
            coords[c] = (la, lo)
        self.coords = coords

    # ------------------------------------------------------------------ predict
    def predict_log(self, d: pd.DataFrame) -> np.ndarray:
        n_lin = len(self.lin_coef)
        out = (self.gbm.predict(self._gbm_features(d))
               + self.lr.intercept_
               + self._linear_features(d).values @ self.lr.coef_[:n_lin])
        for col, lat, lon in (("pickup", "pickup_lat", "pickup_lon"),
                              ("delivery", "delivery_lat", "delivery_lon")):
            known = d[col].isin(self.city_effect.index).values
            effect = np.zeros(len(d))
            effect[known] = d.loc[known, col].map(self.city_effect).values
            if (~known).any():                       # unseen city -> effect of its geographic neighbours
                effect[~known] = self.knn.predict(d.loc[~known, [lat, lon]].values)
            out = out + effect
        return out

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(d))
