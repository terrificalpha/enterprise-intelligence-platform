"""Evaluation sets. Gold facts come from the generated documents (see app/data_gen/generate_documents.py)."""
from __future__ import annotations

import pandas as pd

from ..config import settings

RET, CRED, SUP, MSA = ("Customer_Retention_Policy.pdf", "Credit_and_Payment_Risk_Policy.pdf",
                       "Support_Service_Level_Policy.pdf", "Master_Service_Agreement.docx")
COMP, Q3 = "Competitive_Landscape_Brief.docx", "Q3_Account_Review_Report.docx"

# (question, gold_source, gold_chunk_keywords (ALL must appear in relevant chunk), answer_keyword, kind)
SUPPORTED = [
    ("What churn probability makes an account High Risk?", RET, ["0.60"], "0.60", "direct"),
    ("How quickly must executive outreach happen for a high-risk account?", RET, ["5 business days"], "5 business days", "direct"),
    ("Who approves a retention discount of 15 percent?", RET, ["Regional VP"], "Regional VP", "direct"),
    ("What is the maximum retention discount allowed?", RET, ["30 percent"], "30 percent", "direct"),
    ("When must an executive sponsor be assigned?", RET, ["100,000"], "100,000", "direct"),
    ("How many days before contract end must renewal conversations begin?", RET, ["120 days"], "120 days", "direct"),
    ("What triggers Credit Watch?", CRED, ["Three or more late payments"], "late payments", "direct"),
    ("When is a credit hold applied on overdue invoices?", CRED, ["60 days overdue"], "60 days", "direct"),
    ("What late fee applies to overdue balances?", CRED, ["1.5 percent"], "1.5 percent", "direct"),
    ("What is the P1 resolution time?", SUP, ["P1", "8 hours"], "8 hours", "direct"),
    ("What is the cap on SLA breach service credits?", SUP, ["capped at 20 percent"], "20 percent", "direct"),
    ("How many SLA breaches trigger executive escalation?", SUP, ["Three or more SLA breaches"], "Three or more", "direct"),
    ("What is the notice period for non-renewal in the MSA?", MSA, ["60 days"], "60 days", "direct"),
    ("What is the liability cap in the agreement?", MSA, ["preceding 12 months"], "12 months", "direct"),
    ("How long until customer data is deleted after termination?", MSA, ["30 days after termination"], "30 days", "direct"),
    ("What monthly uptime does the service commit to?", MSA, ["99.9 percent"], "99.9", "direct"),
    ("What annual price increase applies at renewal?", MSA, ["5 percent"], "5 percent", "direct"),
    ("Which competitor bundles free support?", COMP, ["OrbitDesk"], "OrbitDesk", "direct"),
    ("Which competitor offers on-premise deployment for manufacturers?", COMP, ["Vantiq"], "Vantiq", "direct"),
    ("At what seat utilization must a training plan be offered?", RET, ["50 percent"], "50 percent", "direct"),
    ("Which customers can get Net 45 payment terms?", CRED, ["Net 45", "Enterprise"], "Enterprise", "direct"),
    ("How soon must a personal data breach be notified?", MSA, ["72 hours"], "72 hours", "direct"),
    # paraphrased: little lexical overlap with the source wording
    ("At what predicted likelihood of leaving is a customer labelled high risk?", RET, ["0.60"], "0.60", "paraphrase"),
    ("Up to what percentage can an account manager discount without needing anyone's sign-off?", RET, ["up to 10 percent"], "10 percent", "paraphrase"),
    ("What happens when an invoice is two months past due?", CRED, ["60 days overdue"], "credit hold", "paraphrase"),
    ("How fast must support resolve the most critical tickets?", SUP, ["P1", "8 hours"], "8 hours", "paraphrase"),
    ("Can either side end the contract early without cause?", MSA, ["terminate for convenience"], "90 days", "paraphrase"),
    ("Who undercuts our pricing in retail?", COMP, ["NimbusOps"], "NimbusOps", "paraphrase"),
]

UNSUPPORTED_OOD = [
    "What is the CEO's salary?", "Who won the 2022 FIFA World Cup?", "What is our parental leave policy?",
    "How do I configure Kubernetes autoscaling?", "What is the weather in Mumbai today?",
    "What is the stock price target for our company?", "Write a poem about the ocean.",
    "What is the capital of Australia?", "How do I reset my email password?",
    "What is the refund policy for hardware returns?", "What are the office opening hours?",
    "Which programming language is best for data science?",
]
UNSUPPORTED_NEAR = [   # same domain vocabulary, but the answer is NOT in the corpus
    "What is the penalty for late delivery of hardware?", "What is the maximum number of seats per license?",
    "What is the late fee for Net 90 payment terms?", "What is OrbitDesk's pricing per seat?",
    "How many employees does NimbusOps have?", "What is the SLA for P0 incidents?", "Which cloud provider hosts our service?",
]


# ---------------------------------------------------------------------------------------------
# HELD-OUT set: written BEFORE the retrieval improvements (alias expansion, identifier check) were made,
# and never used to tune them. Mostly paraphrases with little lexical overlap with the source text.
HELD_OUT_SUPPORTED = [
    ("Beyond what discount level does the CFO have to sign off?", RET, ["above 20 percent"], "20 percent", "paraphrase"),
    ("What is the highest discount anyone is ever allowed to give a customer at risk of leaving?", RET, ["30 percent"], "30 percent", "paraphrase"),
    ("How soon after identifying low adoption must we offer the customer training?", RET, ["30 days"], "30 days", "paraphrase"),
    ("What reduction in product usage prompts a health review?", RET, ["20 percent over 90 days"], "20 percent", "paraphrase"),
    ("After how many days of non-payment are new orders frozen?", CRED, ["60 days overdue"], "60 days", "paraphrase"),
    ("What monthly interest is charged on unpaid balances?", CRED, ["1.5 percent per month"], "1.5 percent", "paraphrase"),
    ("How many tardy payments put an account under financial surveillance?", CRED, ["Three or more late payments"], "Three or more", "paraphrase"),
    ("How long do we have to respond to a top-severity incident?", SUP, ["P1", "1 hour"], "1 hour", "paraphrase"),
    ("What is the maximum compensation a customer can receive for missed support targets?", SUP, ["capped at 20 percent"], "20 percent", "paraphrase"),
    ("Within how many days must a root cause analysis be delivered after repeated breaches?", SUP, ["7 days"], "7 days", "paraphrase"),
    ("How much notice is required to walk away from the agreement?", MSA, ["90 days written notice"], "90 days", "paraphrase"),
    ("What compensation is owed if availability drops under 99 percent?", MSA, ["25 percent credit"], "25 percent", "paraphrase"),
    ("Which rival is focused on factories and supports self-hosting?", COMP, ["Vantiq"], "Vantiq", "paraphrase"),
    ("What does the report say is the total value of contracts lost to churn?", Q3, ["71.8 million"], "71.8", "report"),
    ("What defensive strategy is advised instead of deep price cuts?", COMP, ["value-based selling"], "value-based", "paraphrase"),
    ("What is the minimum contract value that gets a dedicated executive sponsor?", RET, ["100,000"], "100,000", "paraphrase"),
]

HELD_OUT_UNSUPPORTED = [
    "What is the interest rate on our corporate bonds?", "Who is the head of the Berlin office?",
    "What is the SLA for P5 incidents?", "What late fee applies to Net 60 invoices?",
    "How many customers does Vantiq Cloud have?", "What is the discount for non-profit organizations?",
    "What encryption standard protects data at rest?", "Which payment gateway processes our invoices?",
    "Summarize the plot of Hamlet.", "What are the Q4 revenue targets?",
]


def q3_facts() -> list[tuple]:
    df = pd.read_csv(settings.customers_csv).drop_duplicates("customer_id")
    from ..ml.features import normalise_category
    df["industry"] = normalise_category(df["industry"])
    top = df.groupby("industry").churned.mean().sort_values(ascending=False).index[0]
    return [("What was the overall churn rate reported in the Q3 review?", Q3, ["churn rate"], f"{df.churned.mean():.1%}", "report"),
            ("Which industries show the highest churn in the Q3 review?", Q3, ["highest churn"], top, "report")]


def customer_record_questions(n: int = 8) -> list[tuple]:
    notes = pd.read_csv(settings.documents_dir / "account_notes.csv").sample(n, random_state=3)
    out = []
    for r in notes.itertuples():
        out.append((f"What do the account notes say about customer {r.customer_id}?", "account_notes.csv", [str(r.customer_id)],
                    str(r.note)[:25], "record"))
    return out


def all_supported() -> list[tuple]:
    return SUPPORTED + q3_facts() + customer_record_questions()
