"""Domain lexicon for query expansion (a classic, transparent way to bridge the vocabulary gap between how
users ask and how policies are written). Keys are *query* words; values are words a policy would use instead.
This is deliberately small and general-purpose (finance / contracts / support vocabulary); the proper fix for
open-ended paraphrase is a neural embedder (EIP_EMBEDDING=sentence-transformers) - see docs/ARCHITECTURE.md."""
from __future__ import annotations

from .text_utils import tokenize

_RAW: dict[str, str] = {
    "maximum": "cap capped exceed above limit upper", "max": "cap capped exceed above limit",
    "highest": "cap capped exceed above", "ceiling": "cap capped limit", "limit": "cap capped exceed",
    "ever": "under any circumstances", "minimum": "least or more at above", "lowest": "least below",
    "allowed": "may permitted require", "permitted": "may allowed", "entitled": "may receive",
    "critical": "P1", "urgent": "P1 critical", "severity": "priority", "severe": "critical P1", "top": "P1 critical",
    "respond": "response", "reply": "response", "time": "hours days",
    "walk": "terminate termination", "leaving": "terminate churn",
    "exit": "terminate", "quit": "terminate", "cancel": "terminate termination", "end": "terminate termination",
    "frozen": "hold", "freeze": "hold", "blocked": "hold", "halt": "hold", "suspended": "hold",
    "nonpayment": "overdue late", "unpaid": "overdue late", "delinquent": "overdue late", "arrears": "overdue",
    "tardy": "late", "behind": "overdue",
    "surveillance": "watch", "monitoring": "watch",
    "compensation": "credit", "reimbursement": "credit", "owed": "credit", "compensate": "credit",
    "interest": "fee charged", "penalty": "fee charged", "surcharge": "fee",
    "availability": "uptime", "downtime": "uptime", "outage": "uptime breach",
    "rival": "competitor", "rivals": "competitor", "competition": "competitor", "undercut": "pricing competitor",
    "factories": "manufacturing", "factory": "manufacturing", "plant": "manufacturing",
    "self-hosting": "on-premise", "selfhosting": "on-premise", "self-hosted": "on-premise", "onprem": "on-premise",
    "cheaper": "discount pricing", "cuts": "discount", "cut": "discount", "rebate": "discount",
    "strategy": "recommended defence", "advised": "recommended", "advice": "recommended", "approach": "recommended",
    "dedicated": "assigned", "signoff": "approval approve require", "sign-off": "approval approve require",
    "sign": "approval approve", "authorise": "approval approve", "authorize": "approval approve",
    "permission": "approval", "likelihood": "probability", "chance": "probability", "odds": "probability",
    "labelled": "classified", "labeled": "classified", "categorised": "classified",
    "reduction": "decline falls below", "decrease": "decline falls below", "drop": "falls below decline", "drops": "falls below",
    "declining": "decline", "decline": "falls below",
    "usage": "utilization", "adoption": "utilization usage", "utilisation": "utilization",
    "lost": "churn churned", "attrition": "churn", "defection": "churn",
    "shortly": "within", "soon": "within", "quickly": "within", "fast": "within", "long": "within", "often": "every",
    "deleted": "deleted", "wiped": "deleted", "erased": "deleted", "purged": "deleted",
    "breach": "breach", "undercuts": "pricing", "pricing": "price fees", "cost": "fees price",
    "contract value": "annual contract value acv",
    "notify": "notify notified notice", "disclose": "notify notice",
    "liable": "liability", "damages": "liability",
}

# phrase-level rewrites applied to the raw query text before tokenisation (number-word -> unit conversions)
_PHRASES = [("two months", "60 days"), ("2 months", "60 days"), ("one month", "30 days"), ("1 month", "30 days"),
            ("three months", "90 days"), ("3 months", "90 days"), ("a quarter", "90 days"), ("a year", "12 months"),
            ("six months", "180 days"), ("past due", "overdue"), ("non-payment", "overdue late"),
            ("non payment", "overdue late"), ("without cause", "convenience"), ("for no reason", "convenience"),
            ("any reason", "convenience"), ("walk away", "terminate"), ("service credit", "credit"),
            ("top-severity", "critical P1"), ("top severity", "critical P1"), ("most critical", "critical P1"),
            ("price cuts", "discount"), ("price cut", "discount"), ("self-hosting", "on-premise"),
            ("self-hosted", "on-premise"), ("sign-off", "approval"), ("sign off", "approval")]

ALIASES: dict[str, list[str]] = {}
for _k, _v in _RAW.items():
    _kt = tokenize(_k)
    if len(_kt) == 1:
        ALIASES[_kt[0]] = sorted({t for w in _v.split() for t in tokenize(w)} - {_kt[0]})


def normalise_query(q: str) -> str:
    ql = q.lower()
    for a, b in _PHRASES:          # replace (not append) so the original wording is not counted as "uncovered"
        ql = ql.replace(a, b)
    return ql


# question-framing words that carry little topical information: they get a reduced IDF weight in coverage/BM25
FILLER = frozenset(tokenize("time happens happen apply applies applied need needs needed get gets give given receive "
                            "find exactly actually ever anyone someone level identifying supports support offered "
                            "mention mentioned state states say says specified specify kind type"))
FILLER_WEIGHT = 0.3


def expand_terms(query: str) -> tuple[list[str], dict[str, list[str]]]:
    """Return (original stemmed tokens, {token: [alias tokens]}) for a query."""
    toks = tokenize(normalise_query(query))
    return toks, {t: ALIASES[t] for t in toks if t in ALIASES}
