"""Account insight generation: ML prediction + explanation  ->  targeted retrieval  ->  grounded LLM briefing.

The recommendation rules decide WHICH policy to look up from the customer's own data and model drivers;
the retrieved policy text then supplies (and cites) the actual rule. Nothing is recommended from memory."""
from __future__ import annotations

from ..ml.predictor import ChurnPredictor
from ..rag.knowledge_base import Hit, KnowledgeBase
from . import prompts
from .client import LLMClient, LLMError
from .generator import Generator, _clean_sentences, verify_citations

POLICY_TYPES = {"pdf", "docx"}


def _rules(r: dict, tier: str) -> list[tuple[str, str, str, str]]:
    """(key, trigger description, action, retrieval query) for rules that fire on this customer."""
    out = []
    util = (r["seats_active"] / max(r["seats_purchased"], 1)) if r.get("seats_purchased") else 1
    if tier == "High":
        out.append(("tier", "predicted High Risk", "Start the High-Risk save plan: executive outreach call, written save plan approved by the Regional VP",
                    "High Risk accounts executive outreach call within business days written save plan approved Regional VP"))
    elif tier == "Medium":
        out.append(("tier", "predicted Medium Risk", "Schedule a customer success manager check-in",
                    "Medium Risk accounts require customer success manager check-in within business days"))
    if (r.get("sla_breaches_90d") or 0) >= 2:
        out.append(("sla", f"{int(r['sla_breaches_90d'])} SLA breaches in 90 days", "Trigger executive escalation and deliver a root cause analysis",
                    "SLA breaches rolling 90-day window executive escalation root cause analysis"))
    if (r.get("late_payments_12m") or 0) >= 3:
        out.append(("credit", f"{int(r['late_payments_12m'])} late payments in 12 months", "Place on Credit Watch and hold a joint Finance + Customer Success review before offering any discount",
                    "three or more late payments Credit Watch joint review Finance Customer Success before retention discount"))
    if util < 0.5 or (r.get("usage_trend_pct") is not None and r["usage_trend_pct"] <= -20):
        out.append(("adoption", f"seat utilization {util:.0%}, usage trend {r.get('usage_trend_pct')}%", "Offer a training and enablement plan and run a customer health review",
                    "seat utilization falls below 50 percent training enablement plan usage decline health review"))
    if (r.get("competitor_mentions_90d") or 0) >= 1:
        out.append(("competitor", f"{int(r['competitor_mentions_90d'])} competitor mentions", "Prepare a competitive defence (value-based, multi-year terms rather than deep discounts)",
                    "Recommended defence at-risk accounts value-based selling multi-year terms competitors"))
    if (r.get("days_to_renewal") or 999) <= 120:
        out.append(("renewal", f"renewal in {int(r['days_to_renewal'])} days", "Open the renewal conversation now (policy requires 120 days notice)",
                    "Renewal conversations must begin 120 days before contract end date Renewal Watch"))
    if (r.get("annual_contract_value") or 0) >= 100_000 and not r.get("exec_sponsor"):
        out.append(("sponsor", f"ACV ${r['annual_contract_value']:,.0f} with no executive sponsor", "Assign an executive sponsor",
                    "accounts annual contract value 100,000 must be assigned executive sponsor quarterly business review"))
    if (r.get("avg_resolution_hours") or 0) > 24:
        out.append(("support", f"avg resolution {r['avg_resolution_hours']:.0f}h", "Add to the Head of Support monthly review (resolution time above the 24h target)",
                    "average ticket resolution time target below 24 hours reviewed monthly Head of Support"))
    return out


def _driver_phrase(d: dict) -> str:
    v = d["value"]
    arrow = "raises" if d["impact"] > 0 else "lowers"
    val = f"{v:,.2f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)
    return f"**{d['label']}** = {val} {arrow} churn risk by {abs(d['impact']) * 100:.1f} pts"


class InsightGenerator:
    def __init__(self, kb: KnowledgeBase, predictor: ChurnPredictor, generator: Generator, llm: LLMClient):
        self.kb, self.pred, self.gen, self.llm = kb, predictor, generator, llm

    def _policy_hit(self, query: str) -> Hit | None:
        for h in self.kb.search(query, k=6):
            if h.chunk.doc_type in POLICY_TYPES and h.coverage >= 0.4:
                return h
        return None

    def briefing(self, customer_id: str) -> dict:
        p = self.pred.predict_customer(customer_id)
        if p is None:
            return {"error": f"Customer {customer_id} not found"}
        r, tier, prob = p["profile"], p["risk_tier"], p["churn_probability"]
        acv = r.get("annual_contract_value") or 0
        # ---- evidence: policy passages for each triggered rule + this customer's own notes/tickets
        hits: list[Hit] = []

        def cite(h: Hit | None) -> int | None:
            if h is None:
                return None
            for i, x in enumerate(hits, 1):
                if x.chunk.chunk_id == h.chunk.chunk_id:
                    return i
            hits.append(h)
            return len(hits)

        tier_hit = self._policy_hit("Risk Tiers High Risk Medium Risk Low Risk predicted churn probability")
        tier_id = cite(tier_hit)
        actions = []
        for key, trig, action, query in _rules(r, tier):
            h = self._policy_hit(query)
            cid = cite(h)
            why = _clean_sentences(h)
            sent = max(why, key=lambda s: self.kb.reranker.coverage(query, s)) if why else ""
            actions.append({"key": key, "trigger": trig, "action": action, "policy_evidence": sent,
                            "citation": cid, "source": h.chunk.source if h else None})
        cust_hits = self.kb.search(f"{customer_id} account notes support tickets", k=4, filters={"customer_id": customer_id})
        cust_ids = [cite(h) for h in cust_hits]

        facts = {"customer_id": customer_id, "industry": r["industry"], "company_size": r["company_size"],
                 "annual_contract_value": acv, "churn_probability": prob, "risk_tier": tier,
                 "model": self.pred.info["model"], "expected_revenue_at_risk": round(prob * acv, 2),
                 "top_drivers": [{"feature": d["label"], "value": d["value"], "impact_pts": round(d["impact"] * 100, 1)}
                                 for d in p["drivers"]]}
        narrative, by = None, "offline-composer"
        if self.llm.available:
            try:
                from ..security import neutralise_context
                from .generator import readable
                ctx = prompts.format_context([(i, h.chunk.source, neutralise_context(readable(h))) for i, h in enumerate(hits, 1)])
                narrative = self.llm.complete(prompts.INSIGHT_SYSTEM, f"CONTEXT:\n{ctx}\n\nML FACTS:\n{facts}\n\n"
                                              f"Rule-based triggers: {[(a['trigger'], a['action']) for a in actions]}", max_tokens=1100)
                by = self.llm.model_name
                problem = self._narrative_problem(narrative, prob, tier, hits)
                if problem:     # never ship an unverifiable briefing from a small model: use the cited deterministic composer
                    narrative, by = None, f"offline-composer (LLM draft rejected: {problem})"
            except LLMError:
                narrative = None
        if narrative is None:
            ts = _clean_sentences(tier_hit) if tier_hit else []
            tier_sent = next((x for x in ts if x.startswith(f"{tier} Risk")), "")
            narrative = self._compose(customer_id, r, p, actions, tier_id, tier_sent, cust_hits, cust_ids)
        ver = verify_citations(narrative, hits)
        from .generator import readable as _rd
        citations = [{"id": i, "source": h.chunk.source, "location": h.chunk.location, "section": h.chunk.section,
                      "snippet": _rd(h)[:350]} for i, h in enumerate(hits, 1)]
        return {"customer_id": p["customer_id"], "prediction": {"churn_probability": prob, "risk_tier": tier,
                "predicted_churn": p["predicted_churn"], "model": self.pred.info["model"],
                "expected_revenue_at_risk": round(prob * acv, 2)},
                "drivers": p["drivers"], "recommendations": actions, "citations": citations,
                "narrative": narrative, "generated_by": by, "verification": ver, "profile": r}

    @staticmethod
    def _narrative_problem(text: str, prob: float, tier: str, hits: list[Hit]) -> str:
        """Return a short reason if the LLM briefing is unsafe to show, else ''."""
        import re
        if not text or len(text.strip()) < 80:
            return "empty/too short"
        if tier.lower() not in text.lower():
            return "risk tier missing or wrong"
        pct = {f"{prob:.0%}", f"{prob * 100:.1f}%", f"{prob * 100:.0f}%", f"{prob:.2f}", f"{prob:.1%}"}
        if not any(x in text for x in pct):
            return "churn probability missing or altered"
        ids = [int(x) for x in re.findall(r"\[(\d+)\]", text)]
        if not ids:
            return "no citations"
        if any(i < 1 or i > len(hits) for i in ids):
            return "invalid citation id"
        return ""

    @staticmethod
    def _compose(cid, r, p, actions, tier_id, tier_sent, cust_hits, cust_ids) -> str:
        prob, tier = p["churn_probability"], p["risk_tier"]
        acv = r.get("annual_contract_value") or 0
        L = ["**Summary**",
             f"{cid} ({r['industry']}, {r['company_size']}, ACV ${acv:,.0f}) has a predicted churn probability of "
             f"{prob:.0%}, which places it in the **{tier} Risk** tier. Expected revenue at risk is about "
             f"${prob * acv:,.0f}."]
        if tier_sent and tier_id:
            L.append(f"Policy definition of the tiers: \"{tier_sent}\" [{tier_id}]")
        L += ["", "**Why this account is at risk**"]
        pos = [d for d in p["drivers"] if d["impact"] > 0.005][:4]
        neg = [d for d in p["drivers"] if d["impact"] < -0.005][:2]
        L += [f"- {_driver_phrase(d)}" for d in pos] or ["- No single factor dominates the risk score."]
        if neg:
            L.append("- Protective factors: " + "; ".join(_driver_phrase(d) for d in neg))
        from .generator import readable
        for h, i in zip(cust_hits[:3], cust_ids):
            L.append(f"- {readable(h)} [{i}]")
        L += ["", "**Recommended actions**"]
        if not actions:
            L.append("1. No policy-triggered actions; continue standard account management.")
        for n, a in enumerate(actions, 1):
            ev = f" Policy: \"{a['policy_evidence']}\" [{a['citation']}]" if a["citation"] else " (no supporting policy passage found)"
            L.append(f"{n}. {a['action']} - triggered by {a['trigger']}.{ev}")
        return "\n".join(L)
