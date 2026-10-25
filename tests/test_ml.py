import json
import unittest

import numpy as np
import pandas as pd

from tests.helpers import ensure_artifacts, platform


class TestML(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_artifacts()
        from app.ml.predictor import ChurnPredictor
        cls.pred = ChurnPredictor()

    def test_at_least_two_models_compared_and_best_selected(self):
        m = self.pred.metrics()
        models = [k for k in ("logistic_regression", "random_forest", "hist_gradient_boosting") if k in m]
        self.assertGreaterEqual(len(models), 2)
        best = max(models, key=lambda k: m[k]["cv_roc_auc"])
        self.assertEqual(m["selected"]["model"], best)
        self.assertGreater(m["selected"]["test_calibrated"]["roc_auc"], 0.7)      # far above the 0.5 baseline

    def test_beats_baseline(self):
        m = self.pred.metrics()
        self.assertGreater(m["selected"]["test_calibrated"]["roc_auc"], m["baseline_prior"]["test"]["roc_auc"] + 0.2)

    def test_pipeline_accepts_messy_raw_rows(self):
        rec = {"industry": " RETAIL ", "region": "emea", "company_size": "smb", "contract_type": "Monthly", "tenure_months": 5,
               "annual_contract_value": 9e9, "seats_purchased": 10, "seats_active": 50, "logins_30d": 3, "usage_trend_pct": -30,
               "feature_adoption_score": np.nan, "support_tickets_90d": 4, "avg_resolution_hours": np.nan, "sla_breaches_90d": 3,
               "nps_score": np.nan, "late_payments_12m": 4, "days_sales_outstanding": 60, "discount_pct": 5,
               "competitor_mentions_90d": 2, "days_to_renewal": 30, "exec_sponsor": 0}
        r = self.pred.predict_record(rec)
        self.assertTrue(0 <= r["churn_probability"] <= 1)
        self.assertEqual(r["risk_tier"], "High")

    def test_missing_columns_do_not_crash(self):
        r = self.pred.predict_record({"industry": "Finance"})
        self.assertTrue(0 <= r["churn_probability"] <= 1)

    def test_risky_profile_scores_higher_than_safe_profile(self):
        c = self.pred.customers
        safe = c.iloc[0].to_dict(); risky = dict(safe)
        risky.update({"sla_breaches_90d": 6, "late_payments_12m": 5, "competitor_mentions_90d": 4, "usage_trend_pct": -40, "contract_type": "Monthly"})
        safe.update({"sla_breaches_90d": 0, "late_payments_12m": 0, "competitor_mentions_90d": 0, "usage_trend_pct": 15, "contract_type": "Multi-year"})
        self.assertGreater(self.pred.predict_record(risky)["churn_probability"], self.pred.predict_record(safe)["churn_probability"] + 0.3)

    def test_tiers_follow_policy_thresholds(self):
        from app.ml.predictor import tier_for
        self.assertEqual([tier_for(x) for x in (0.0, 0.29, 0.30, 0.59, 0.60, 1.0)], ["Low", "Low", "Medium", "Medium", "High", "High"])

    def test_explanations_are_signed_and_sorted(self):
        p = self.pred.predict_customer(self.pred.customers.customer_id.iloc[5])
        imp = [abs(d["impact"]) for d in p["drivers"]]
        self.assertEqual(imp, sorted(imp, reverse=True))
        self.assertTrue(all({"label", "impact", "value"} <= set(d) for d in p["drivers"]))

    def test_unknown_customer(self):
        self.assertIsNone(self.pred.predict_customer("C9999"))

    def test_cleaning_dedupes_and_normalises(self):
        from app.ml.features import clean_raw
        raw = pd.read_csv(self.pred.customers.attrs.get("p", "data/raw/customers.csv"))
        clean = clean_raw(raw)
        self.assertEqual(clean.customer_id.nunique(), len(clean))
        self.assertLess(len(clean), len(raw))
        self.assertTrue(set(clean.region.unique()) <= {"North America", "EMEA", "APAC", "LATAM"})
        self.assertTrue((clean.seats_active <= clean.seats_purchased).all())

    def test_metrics_file_is_valid_json(self):
        from app.config import settings
        json.loads((settings.artifacts_dir / "metrics.json").read_text())


if __name__ == "__main__":
    unittest.main()
