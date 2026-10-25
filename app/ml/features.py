"""Data cleaning + feature engineering, implemented as sklearn transformers so that the
saved model accepts RAW customer rows and applies exactly the same preprocessing at inference."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "churned"
ID_COL = "customer_id"
CATEGORICAL = ["industry", "region", "company_size", "contract_type"]
BASE_NUMERIC = ["tenure_months", "annual_contract_value", "seats_purchased", "seats_active", "logins_30d",
                "usage_trend_pct", "feature_adoption_score", "support_tickets_90d", "avg_resolution_hours",
                "sla_breaches_90d", "nps_score", "late_payments_12m", "days_sales_outstanding", "discount_pct",
                "competitor_mentions_90d", "days_to_renewal", "exec_sponsor"]
ENGINEERED = ["seat_utilization", "acv_per_seat", "sla_breach_rate", "payment_risk", "support_load",
              "usage_decline_flag", "renewal_soon", "log_acv", "low_nps_flag", "tenure_bucket_years"]
RAW_FEATURES = CATEGORICAL + BASE_NUMERIC
_CANON = {v.lower(): v for v in ["Manufacturing", "Retail", "Healthcare", "Finance", "Technology", "Logistics", "Education",
                                 "North America", "EMEA", "APAC", "LATAM", "SMB", "Mid-Market", "Enterprise",
                                 "Monthly", "Annual", "Multi-year"]}


def normalise_category(s: pd.Series) -> pd.Series:
    """Trim whitespace and map any casing variant onto the canonical label (e.g. 'RETAIL ' -> 'Retail')."""
    t = s.astype("string").str.strip()
    return t.map(lambda v: _CANON.get(v.lower(), v.title()) if isinstance(v, str) else v).astype(object)


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level cleaning used for analysis/EDA (dedupe + normalise). Stateless."""
    df = df.copy()
    if ID_COL in df:
        df = df.drop_duplicates(subset=ID_COL, keep="first")
    for c in CATEGORICAL:
        if c in df:
            df[c] = normalise_category(df[c])
    if {"seats_active", "seats_purchased"} <= set(df.columns):
        df["seats_active"] = np.minimum(df["seats_active"], df["seats_purchased"])
    return df.reset_index(drop=True)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Cleans a raw frame, winsorises outliers (limits learned in fit) and adds domain features."""

    def __init__(self, winsor_cols=("annual_contract_value",), q=0.99):
        self.winsor_cols = winsor_cols
        self.q = q

    def fit(self, X, y=None):
        X = self._clean(X)
        self.caps_ = {c: float(X[c].quantile(self.q)) for c in self.winsor_cols}
        return self

    @staticmethod
    def _clean(X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in CATEGORICAL:
            X[c] = normalise_category(X[c])
            X[c] = X[c].where(X[c].notna(), np.nan)
        X["seats_active"] = np.minimum(X["seats_active"], X["seats_purchased"])
        return X

    def transform(self, X):
        X = self._clean(X[RAW_FEATURES])
        for c, cap in self.caps_.items():
            X[c] = np.minimum(X[c], cap)
        seats = X["seats_purchased"].clip(lower=1)
        X["seat_utilization"] = X["seats_active"] / seats
        X["acv_per_seat"] = X["annual_contract_value"] / seats
        X["sla_breach_rate"] = X["sla_breaches_90d"] / X["support_tickets_90d"].clip(lower=1)
        X["payment_risk"] = X["late_payments_12m"] * X["days_sales_outstanding"]
        X["support_load"] = X["support_tickets_90d"] * X["avg_resolution_hours"]
        X["usage_decline_flag"] = (X["usage_trend_pct"] <= -20).astype(int)
        X["renewal_soon"] = (X["days_to_renewal"] <= 120).astype(int)
        X["log_acv"] = np.log1p(X["annual_contract_value"])
        X["low_nps_flag"] = (X["nps_score"] <= 5).astype(float).where(X["nps_score"].notna(), np.nan)
        X["tenure_bucket_years"] = np.minimum(X["tenure_months"] // 12, 5)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(RAW_FEATURES + ENGINEERED)


def build_preprocessor(scale: bool = True) -> Pipeline:
    num = BASE_NUMERIC + ENGINEERED
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    ct = ColumnTransformer([
        ("num", Pipeline(num_steps), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
    ], remainder="drop")
    return Pipeline([("features", FeatureEngineer()), ("columns", ct)])
