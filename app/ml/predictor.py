"""Inference wrapper: probabilities, risk tiers (per Retention Policy s.2) and local explanations."""
from __future__ import annotations

import json
from functools import cached_property

import joblib
import numpy as np
import pandas as pd

from ..config import settings
from .features import CATEGORICAL, ID_COL, RAW_FEATURES, clean_raw

HUMAN = {
    "sla_breaches_90d": "SLA breaches (90d)", "late_payments_12m": "Late payments (12m)",
    "usage_trend_pct": "Usage trend (%)", "competitor_mentions_90d": "Competitor mentions (90d)",
    "nps_score": "NPS score", "avg_resolution_hours": "Avg ticket resolution (h)", "contract_type": "Contract type",
    "industry": "Industry", "tenure_months": "Tenure (months)", "seats_active": "Active seats",
    "seats_purchased": "Purchased seats", "exec_sponsor": "Executive sponsor", "discount_pct": "Discount (%)",
}


def tier_for(p: float) -> str:
    if p >= settings.high_risk_threshold:
        return "High"
    if p >= settings.medium_risk_threshold:
        return "Medium"
    return "Low"


class ChurnPredictor:
    def __init__(self):
        path = settings.artifacts_dir / "best_model.joblib"
        if not path.exists():
            raise FileNotFoundError("No trained model found. Run `python -m app.ml.train` first.")
        self.bundle = self._load_checked(path)
        self.model = self.bundle["model"]

    @staticmethod
    def _load_checked(path):
        """Load the pickled model and prove it can predict. Pickles are tied to the scikit-learn version that
        created them; if the user's environment differs, retrain once instead of crashing."""
        try:
            bundle = joblib.load(path)
            sample = clean_raw(pd.read_csv(settings.customers_csv).head(3))
            bundle["model"].predict_proba(sample[RAW_FEATURES])
            return bundle
        except Exception as e:  # noqa: BLE001 - any unpickle / version / schema problem
            print(f"[predictor] saved model unusable ({type(e).__name__}: {e}); retraining with the installed scikit-learn ...")
            from .train import train
            train()
            return joblib.load(path)

    @property
    def info(self) -> dict:
        b = self.bundle
        return {"model": b["model_name"], "version": b["version"], "threshold": b["threshold"]}

    @cached_property
    def customers(self) -> pd.DataFrame:
        return clean_raw(pd.read_csv(settings.customers_csv))

    @cached_property
    def scores(self) -> pd.DataFrame:
        p = settings.artifacts_dir / "customer_scores.csv"
        if p.exists():
            sc = pd.read_csv(p)
        else:
            sc = self.customers[[ID_COL]].copy()
            sc["churn_probability"] = self.predict_proba(self.customers)
        return sc

    @cached_property
    def portfolio(self) -> pd.DataFrame:
        df = self.customers.merge(self.scores, on=ID_COL, how="left")
        df["risk_tier"] = df["churn_probability"].map(tier_for)
        df["expected_loss"] = df["churn_probability"] * df["annual_contract_value"]
        return df

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(df[RAW_FEATURES])[:, 1]

    def predict_record(self, record: dict) -> dict:
        df = pd.DataFrame([{k: record.get(k, np.nan) for k in RAW_FEATURES}])
        p = float(self.predict_proba(df)[0])
        return {"churn_probability": round(p, 4), "risk_tier": tier_for(p),
                "predicted_churn": bool(p >= self.bundle["threshold"])}

    def get_customer(self, customer_id: str) -> dict | None:
        row = self.customers[self.customers[ID_COL].str.upper() == customer_id.upper()]
        if row.empty:
            return None
        rec = row.iloc[0].to_dict()
        return {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in rec.items()}

    def explain(self, record: dict, top_n: int = 6) -> list[dict]:
        """Model-agnostic local explanation by occlusion: replace one raw feature at a time with the
        training median/mode and measure the change in predicted probability. Positive = pushes risk up."""
        base_row = {k: record.get(k, np.nan) for k in RAW_FEATURES}
        base_p = float(self.predict_proba(pd.DataFrame([base_row]))[0])
        groups = [[f] for f in RAW_FEATURES if f not in ("seats_active", "seats_purchased")]
        groups.append(["seats_active", "seats_purchased"])          # utilisation is a ratio: occlude jointly
        rows = []
        for g in groups:
            alt = dict(base_row)
            for f in g:
                alt[f] = self.bundle["modes"].get(f) if f in CATEGORICAL else self.bundle["medians"].get(f)
            rows.append(alt)
        p_alt = self.predict_proba(pd.DataFrame(rows))
        out = []
        for g, pa in zip(groups, p_alt):
            if g[0] == "seats_active":
                util = base_row["seats_active"] / max(base_row["seats_purchased"], 1)
                out.append({"feature": "seat_utilization", "label": "Seat utilization", "value": round(float(util), 2),
                            "impact": round(base_p - float(pa), 4)})
            else:
                out.append({"feature": g[0], "label": HUMAN.get(g[0], g[0]), "value": base_row[g[0]],
                            "impact": round(base_p - float(pa), 4)})
        out.sort(key=lambda d: -abs(d["impact"]))
        return out[:top_n]

    def predict_customer(self, customer_id: str) -> dict | None:
        rec = self.get_customer(customer_id)
        if rec is None:
            return None
        res = self.predict_record(rec)
        res.update({"customer_id": rec[ID_COL], "drivers": self.explain(rec), "profile": rec})
        return res

    @staticmethod
    def metrics() -> dict:
        p = settings.artifacts_dir / "metrics.json"
        return json.loads(p.read_text()) if p.exists() else {}
