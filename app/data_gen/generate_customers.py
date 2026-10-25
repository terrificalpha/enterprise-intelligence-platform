"""Generate a realistic, deliberately messy B2B customer dataset with a latent churn process."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import settings

INDUSTRIES = ["Manufacturing", "Retail", "Healthcare", "Finance", "Technology", "Logistics", "Education"]
IND_P = [0.18, 0.17, 0.14, 0.14, 0.17, 0.12, 0.08]
IND_EFFECT = {"Manufacturing": 0.55, "Retail": 0.50, "Logistics": 0.15, "Education": 0.0,
              "Healthcare": -0.25, "Finance": -0.30, "Technology": -0.1}
REGIONS = ["North America", "EMEA", "APAC", "LATAM"]
SIZES = ["SMB", "Mid-Market", "Enterprise"]
CONTRACTS = ["Monthly", "Annual", "Multi-year"]


def generate_customers(n: int = 3000, seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(settings.random_state if seed is None else seed)
    df = pd.DataFrame({"customer_id": [f"C{i:04d}" for i in range(1, n + 1)]})
    df["industry"] = rng.choice(INDUSTRIES, n, p=IND_P)
    df["region"] = rng.choice(REGIONS, n, p=[0.4, 0.28, 0.22, 0.10])
    df["company_size"] = rng.choice(SIZES, n, p=[0.5, 0.35, 0.15])
    df["contract_type"] = rng.choice(CONTRACTS, n, p=[0.2, 0.55, 0.25])
    df["tenure_months"] = np.clip(rng.gamma(3.0, 10.0, n), 1, 120).round().astype(int)

    size_mult = df["company_size"].map({"SMB": 1.0, "Mid-Market": 3.0, "Enterprise": 9.0}).to_numpy()
    df["annual_contract_value"] = (rng.lognormal(10.0, 0.45, n) * size_mult).round(-2)
    df["seats_purchased"] = np.maximum(3, (df["annual_contract_value"] / rng.uniform(450, 900, n))).round().astype(int)
    util = np.clip(rng.beta(4.5, 2.2, n), 0.05, 1.0)
    df["seats_active"] = np.round(df["seats_purchased"] * util).astype(int)
    df["logins_30d"] = rng.poisson(np.maximum(1, util * 40)).astype(int)
    df["usage_trend_pct"] = np.round(rng.normal(-2 - 12 * (0.65 - util), 14, n), 1)
    df["feature_adoption_score"] = np.clip(np.round(rng.normal(55 + 25 * (util - 0.7), 15, n), 1), 0, 100)

    df["support_tickets_90d"] = rng.poisson(np.where(df["company_size"] == "Enterprise", 6, 3.2)).astype(int)
    df["avg_resolution_hours"] = np.round(rng.gamma(3.0, 7.0, n) + df["support_tickets_90d"] * 0.8, 1)
    df["sla_breaches_90d"] = rng.binomial(df["support_tickets_90d"], np.clip(df["avg_resolution_hours"] / 120, 0.02, 0.5))
    df["nps_score"] = np.clip(np.round(rng.normal(6.8 - 0.35 * df["sla_breaches_90d"], 2.0, n), 0), 0, 10)

    df["late_payments_12m"] = rng.poisson(np.where(df["company_size"] == "SMB", 1.2, 0.6)).astype(int)
    df["days_sales_outstanding"] = np.round(np.clip(rng.normal(34 + 6 * df["late_payments_12m"], 9, n), 5, 150), 0)
    df["discount_pct"] = np.clip(np.round(rng.gamma(2.0, 4.0, n), 1), 0, 30)
    df["competitor_mentions_90d"] = rng.poisson(np.where(df["industry"].isin(["Retail", "Manufacturing"]), 1.0, 0.4)).astype(int)
    df["days_to_renewal"] = rng.integers(10, 540, n)
    df["exec_sponsor"] = ((df["annual_contract_value"] >= 100_000) & (rng.random(n) < 0.6)).astype(int)

    # ---- latent churn process (the "ground truth" the models must rediscover) ----
    seat_util = df["seats_active"] / df["seats_purchased"]
    z0 = (df["industry"].map(IND_EFFECT).to_numpy()
         + 0.40 * df["sla_breaches_90d"]
         + 0.45 * df["late_payments_12m"]
         + 0.55 * df["competitor_mentions_90d"]
         - 0.030 * df["usage_trend_pct"]
         - 1.3 * (seat_util - 0.7)
         - 0.22 * (df["nps_score"] - 6.5)
         + 0.012 * np.maximum(df["avg_resolution_hours"] - 24, 0)
         + np.where(df["contract_type"] == "Monthly", 0.7, np.where(df["contract_type"] == "Multi-year", -0.9, 0.0))
         - 0.012 * np.minimum(df["tenure_months"], 60)
         - 0.5 * df["exec_sponsor"]
         + 0.04 * df["discount_pct"] * 0  # discount has no true effect (a red herring)
         )
    z = -1.75 + 1.6 * (z0 - z0.mean()) + rng.normal(0, 0.6, n)
    p = 1 / (1 + np.exp(-z))
    df["churned"] = (rng.random(n) < p).astype(int)

    # ---- inject real-world mess so preprocessing is meaningful ----
    for col, frac in [("nps_score", 0.14), ("avg_resolution_hours", 0.06), ("feature_adoption_score", 0.08),
                      ("days_sales_outstanding", 0.05)]:
        df.loc[rng.random(n) < frac, col] = np.nan
    idx = rng.choice(n, size=int(0.02 * n), replace=False)
    df.loc[idx, "annual_contract_value"] *= 25                      # outliers
    idx = rng.choice(n, size=int(0.01 * n), replace=False)
    df.loc[idx, "seats_active"] = df.loc[idx, "seats_purchased"] + rng.integers(1, 6, len(idx))  # impossible values
    idx = rng.choice(n, size=int(0.05 * n), replace=False)
    df.loc[idx, "industry"] = df.loc[idx, "industry"].str.upper()   # inconsistent casing
    idx = rng.choice(n, size=int(0.04 * n), replace=False)
    df.loc[idx, "region"] = " " + df.loc[idx, "region"] + " "       # stray whitespace
    dups = df.sample(20, random_state=1)
    df = pd.concat([df, dups], ignore_index=True)                    # duplicate rows
    return df


def main() -> None:
    settings.ensure_dirs()
    df = generate_customers()
    df.to_csv(settings.customers_csv, index=False)
    print(f"Wrote {len(df)} rows -> {settings.customers_csv}  (churn rate {df['churned'].mean():.1%})")


if __name__ == "__main__":
    main()
