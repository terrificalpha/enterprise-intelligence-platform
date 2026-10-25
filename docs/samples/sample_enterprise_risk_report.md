# Enterprise Risk Report
*Generated 2026-10-22 18:37 by the Enterprise Intelligence Platform*

## 1. Executive summary
3,000 customers; actual churn 27.1%; High/Medium/Low = 432/600/1968; expected annual revenue loss $70,998,770 of $320,778,600 ACV.
Highest churn by industry: Retail (41.6%, 500 customers); lowest: Healthcare (16.8%).
Expected annual revenue loss $70,998,770; High-risk accounts hold $29,155,200 ACV (432 customers). Largest by industry: Manufacturing ($21,304,033).

## 2. Portfolio KPIs
```
customers: 3000
actual_churn_rate: 0.2713
avg_predicted_churn: 0.2693
total_acv: 320778600.0
expected_loss: 70998769.98753342
tier_counts: {'Low': 1968, 'Medium': 600, 'High': 432}
```

## 3. Churn by industry
| industry      |   customers |   actual_churn_rate |   avg_predicted_churn |     total_acv |
|:--------------|------------:|--------------------:|----------------------:|--------------:|
| Retail        |         500 |                0.42 |                  0.42 | 59,549,200.00 |
| Manufacturing |         557 |                0.40 |                  0.38 | 67,243,300.00 |
| Logistics     |         353 |                0.26 |                  0.26 | 28,407,900.00 |
| Education     |         256 |                0.23 |                  0.22 | 20,454,700.00 |
| Technology    |         494 |                0.18 |                  0.19 | 51,519,000.00 |
| Finance       |         417 |                0.18 |                  0.17 | 51,427,600.00 |
| Healthcare    |         423 |                0.17 |                  0.18 | 42,176,900.00 |

## 4. Revenue at risk by industry
| industry      |   customers |     total_acv |   expected_loss |
|:--------------|------------:|--------------:|----------------:|
| Manufacturing |         557 | 67,243,300.00 |   21,304,032.80 |
| Retail        |         500 | 59,549,200.00 |   19,986,631.06 |
| Technology    |         494 | 51,519,000.00 |    7,867,760.91 |
| Logistics     |         353 | 28,407,900.00 |    6,456,294.15 |
| Finance       |         417 | 51,427,600.00 |    5,897,450.23 |
| Healthcare    |         423 | 42,176,900.00 |    5,217,475.61 |
| Education     |         256 | 20,454,700.00 |    4,269,125.23 |

## 5. Top 10 at-risk accounts
| customer_id   | industry      | region        | company_size   |   annual_contract_value |   churn_probability | risk_tier   |   expected_loss |
|:--------------|:--------------|:--------------|:---------------|------------------------:|--------------------:|:------------|----------------:|
| C0085         | Manufacturing | APAC          | SMB            |               20,500.00 |                1.00 | High        |       20,400.98 |
| C1572         | Finance       | APAC          | Enterprise     |              195,300.00 |                1.00 | High        |      194,334.28 |
| C1542         | Retail        | North America | SMB            |               32,500.00 |                0.99 | High        |       32,263.29 |
| C1390         | Manufacturing | EMEA          | SMB            |               53,500.00 |                0.99 | High        |       52,882.40 |
| C1242         | Retail        | APAC          | SMB            |               10,900.00 |                0.99 | High        |       10,757.07 |
| C0571         | Retail        | APAC          | Mid-Market     |               97,300.00 |                0.98 | High        |       95,747.86 |
| C2986         | Retail        | North America | Mid-Market     |               72,100.00 |                0.98 | High        |       70,900.70 |
| C1904         | Manufacturing | APAC          | Mid-Market     |               94,400.00 |                0.98 | High        |       92,601.46 |
| C1693         | Retail        | APAC          | SMB            |               15,100.00 |                0.98 | High        |       14,788.86 |
| C0029         | Manufacturing | EMEA          | Enterprise     |              272,600.00 |                0.98 | High        |      266,616.01 |

## 6. Churn drivers (data)
Strongest correlates of churn: competitor_mentions_90d (+0.26), sla_breaches_90d (+0.23), nps_score (-0.22), late_payments_12m (+0.20)

| feature                 |   correlation_with_churn |
|:------------------------|-------------------------:|
| competitor_mentions_90d |                     0.26 |
| sla_breaches_90d        |                     0.23 |
| nps_score               |                    -0.22 |
| late_payments_12m       |                     0.20 |
| usage_trend_pct         |                    -0.18 |
| avg_resolution_hours    |                     0.13 |
| support_tickets_90d     |                     0.11 |
| days_sales_outstanding  |                     0.10 |

## 7. Policy-grounded recommendations
- High Risk accounts must receive an executive outreach call within 5 business days. A written save plan must be approved by the Regional VP within 10 business days. Medium Risk accounts require a customer success manager check-in within 15 business days. [1]
- Account Managers may approve retention discounts of up to 10 percent of annual contract value. Discounts between 10 and 20 percent require Regional VP approval. Discounts above 20 percent require CFO approval. No retention discount may exceed 30 percent under any circumstances. [2]
- Accounts with annual contract value of 100,000 USD or more must be assigned an executive sponsor. Executive sponsors hold a quarterly business review with the customer. [3]
- Three or more late payments within 12 months place the account on Credit Watch. Five or more late payments trigger a formal credit review and may require prepayment for renewals. [4]

## Model & methodology
- Selected model: **logistic_regression** (calibrated). Test ROC-AUC 0.860, F1 0.657.
- Churn probabilities are calibrated; risk tiers follow Customer Retention Policy s.2 (High >= 0.60, Medium >= 0.30).
- Evidence retrieved with hybrid search (dense + BM25, reciprocal rank fusion) and reranking.

## Limitations
- Data is synthetic; probabilities describe model risk, not certainty. Policy citations are lexically verified, not semantically proven.

## References
[1] Customer_Retention_Policy.pdf (page 1): High Risk accounts must receive an executive outreach call within 5 business days. A written save plan must be approved by the Regional VP within 10 business days. Medium Risk accounts require a custo
[2] Customer_Retention_Policy.pdf (page 1): Account Managers may approve retention discounts of up to 10 percent of annual contract value. Discounts between 10 and 20 percent require Regional VP approval. Discounts above 20 percent require CFO 
[3] Customer_Retention_Policy.pdf (page 1): Accounts with annual contract value of 100,000 USD or more must be assigned an executive sponsor. Executive sponsors hold a quarterly business review with the customer.
[4] Credit_and_Payment_Risk_Policy.pdf (page 1): Three or more late payments within 12 months place the account on Credit Watch. Five or more late payments trigger a formal credit review and may require prepayment for renewals.