"""Tool-using agent.

Planner -> (tool call -> observation)* -> grounded final answer.
* With an LLM configured, the planner is a ReAct-style loop where the model picks the next tool as JSON.
* Without one, a deterministic intent router selects the tools. Both paths share the same executor,
  guardrails, trace format and final grounded-synthesis step."""
from __future__ import annotations

import json
import re

from ..config import settings
from ..llm import prompts
from ..llm.client import LLMClient, LLMError
from ..llm.generator import NO_EVIDENCE, Generator
from ..rag.knowledge_base import Hit
from ..security import looks_like_injection, redact_pii, validate_query
from .tools import TOOL_SPECS, ToolBox

MAX_STEPS = 6
CUSTOMER_RE = re.compile(r"\bC\d{4}\b", re.I)
COLS = {"industry": "industry", "industries": "industry", "sector": "industry", "region": "region", "regions": "region",
        "size": "company_size", "company size": "company_size", "segment": "company_size",
        "contract": "contract_type", "contract type": "contract_type", "tier": "risk_tier"}
METRICS = {"acv": "annual_contract_value", "contract value": "annual_contract_value", "revenue": "annual_contract_value",
           "tenure": "tenure_months", "nps": "nps_score", "tickets": "support_tickets_90d", "sla breaches": "sla_breaches_90d",
           "late payments": "late_payments_12m", "resolution": "avg_resolution_hours", "discount": "discount_pct",
           "usage trend": "usage_trend_pct", "logins": "logins_30d", "days to renewal": "days_to_renewal"}
POLICY_WORDS = r"polic|contract(?! type)|agreement|sla|clause|terms|notice|renew|discount authority|approv|escalat|credit|late fee|liabilit|uptime|competitor|note|ticket|require|must|allowed|threshold|define|what does|how long|how many days|who (approves|can)|procedure|process|guideline|recommend|what should|what do"


def _num(text: str, default: int) -> int:
    m = re.search(r"\b(?:top|first|worst|highest)\s+(\d{1,2})\b|\b(\d{1,2})\s+(?:accounts|customers|clients)", text, re.I)
    return int(next(g for g in m.groups() if g)) if m else default


_DOC_CLAUSE_RE = re.compile(r"polic|contract|agreement|clause|terms|notice|approv|escalat|credit|late fee|liabilit|uptime|procedure|"
                            r"guideline|require|must|allowed|threshold|define|what does|what do|say about|recommend|what should", re.I)


def _doc_clause(q: str, has_data_part: bool) -> str:
    """For compound questions ("which industry churns most AND what does the policy say ...") search the
    documents with the clause that asks about documents, not the statistics half."""
    if not has_data_part:
        return q
    clauses = [c.strip(" ?.") for c in re.split(r"\band\b|;|,|\bthen\b", q) if c.strip(" ?.")]
    docs = [c for c in clauses if _DOC_CLAUSE_RE.search(c)]
    return " ".join(docs) if docs else q


def rule_based_plan(q: str) -> list[dict]:
    """Deterministic intent router. Returns an ordered list of {tool, args, reason}."""
    ql = q.lower()
    steps: list[dict] = []
    cid = (CUSTOMER_RE.search(q) or [None])[0]
    cid = cid.upper() if cid else None

    if re.search(r"calculat|compute|\d\s*%\s*of|\d\s*[\*/\+\^x]\s*\d|\d\s+(plus|minus|times|divided by)\s+\d", ql) and re.search(r"\d", ql):
        expr = re.sub(r"^(please\s+)?(calculate|compute|what is|what's|work out)\s*", "", q.strip(" ?."), flags=re.I)
        expr = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", expr)
        for w, sym in ((r"\bplus\b", "+"), (r"\bminus\b", "-"), (r"\b(times|multiplied by)\b", "*"), (r"\bdivided by\b", "/")):
            expr = re.sub(w, sym, expr, flags=re.I)
        if re.fullmatch(r"[\d$.,()%\s+\-*/^a-z]+", expr, re.I) and re.search(r"[\d)]\s*(%|\*|/|\+|-|\^|of)", expr):
            steps.append({"tool": "calculate", "args": {"expression": expr}, "reason": "arithmetic requested"})
            return steps

    if re.search(r"\breport\b|\bsummary document\b|write up", ql):
        steps.append({"tool": "generate_report", "args": {"title": "Enterprise Risk Report" if not cid else f"Account Risk Report {cid}",
                                                          "customer_id": cid, "top_n": _num(q, 10)},
                      "reason": "user asked for a report"})
        return steps

    if cid:
        if re.search(r"brief|recommend|what should|action|plan|why|insight|risk|churn|likely", ql) or True:
            steps.append({"tool": "account_briefing", "args": {"customer_id": cid},
                          "reason": f"customer {cid} mentioned: combine ML prediction with policy evidence"})
        return steps

    data_ops = []
    if re.search(r"revenue at risk|expected loss|exposure|money at risk|arr at risk", ql):
        col = next((v for k, v in COLS.items() if re.search(rf"by {k}|per {k}|each {k}", ql)), None)
        data_ops.append(("revenue_at_risk", {"column": col}))
    if re.search(r"\btop\b|most at risk|highest risk|riskiest|at-risk accounts|high[- ]risk (accounts|customers)|worst", ql) and not data_ops:
        filt = {}
        for ind in ["manufacturing", "retail", "healthcare", "finance", "technology", "logistics", "education"]:
            if ind in ql:
                filt["industry"] = ind
        for reg, canon in [("north america", "North America"), ("emea", "EMEA"), ("apac", "APAC"), ("latam", "LATAM")]:
            if reg in ql:
                filt["region"] = canon
        data_ops.append(("top_risk", {"n": _num(q, 10), "filters": filt or None}))
    metric = next((v for k, v in METRICS.items() if k in ql), None)
    stat_words = re.search(r"\b(average|mean|median|total|sum|max|min)\b", ql)
    if metric and stat_words and not data_ops:
        agg = "mean" if re.search(r"average|mean", ql) else "median" if "median" in ql else "sum" if re.search(r"total|sum", ql) else "max" if "max" in ql else "min" if "min" in ql else "mean"
        col = next((v for k, v in COLS.items() if re.search(rf"by {k}|per {k}|each {k}", ql)), None)
        data_ops.append(("segment_stats", {"metric": metric, "agg": agg, "column": col}))
    m = re.search(r"churn(?:ing)? rate|attrition|which (industry|region|segment|contract)|churn\b.*\bby\b", ql)
    if m and not data_ops:
        col = next((v for k, v in COLS.items() if re.search(rf"\b{k}\b", ql)), "industry")
        data_ops.append(("churn_by", {"column": col}))
    if re.search(r"correlat|what drives|driver|factors? (of|for|behind)|why do (customers|accounts) churn", ql):
        data_ops.append(("correlations", {}))
    if re.search(r"how many (customers|accounts)", ql) and not data_ops:
        data_ops.append(("overview", {}))
    if re.search(r"overview|portfolio|kpi|health of|state of", ql) and not data_ops:
        data_ops.append(("overview", {}))
    for op_, args in data_ops:
        steps.append({"tool": "analyze_data", "args": {"operation": op_, **args}, "reason": f"portfolio statistics needed ({op_})"})

    wants_docs = bool(re.search(POLICY_WORDS, ql)) or not data_ops or bool(re.search(r"driver|what drives|why", ql))
    if wants_docs:
        steps.append({"tool": "search_documents", "args": {"query": _doc_clause(q, bool(data_ops)), "k": 5},
                      "reason": "look for supporting evidence in documents"})
    return steps


def _parse_planner_json(text: str) -> dict:
    """Tolerant JSON extraction for small-model output (code fences, leading prose, trailing commentary)."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        d = json.loads(cleaned)
        if isinstance(d, dict):
            return d
    except ValueError:
        pass
    m = re.search(r"\{.*\}", cleaned, re.S)
    if not m:
        raise ValueError("planner returned no JSON")
    d = json.loads(m.group(0))
    if not isinstance(d, dict):
        raise ValueError("planner JSON is not an object")
    return d


class EnterpriseAgent:
    def __init__(self, tools: ToolBox, generator: Generator, llm: LLMClient):
        self.tools, self.gen, self.llm = tools, generator, llm

    # ------------------------------------------------------------------ public
    def run(self, question: str) -> dict:
        q = validate_query(question)
        trace: list[dict] = []
        flags = []
        if looks_like_injection(q):
            flags.append("possible prompt-injection pattern detected in the request; it is treated as plain data")
        plan_by = "rule-based-router"
        obs: list[tuple[dict, dict]] = []

        steps = None
        if self.llm.available:
            try:
                obs, plan_by = self._llm_loop(q, trace)
                steps = True
            except (LLMError, ValueError):
                trace.clear(); obs = []
        if steps is None:
            for st in rule_based_plan(q)[:MAX_STEPS]:
                obs.append((st, self._exec(st, trace, len(trace) + 1)))
        return self._finalise(q, obs, trace, plan_by, flags)

    # ------------------------------------------------------------------ execution
    def _exec(self, step: dict, trace: list, n: int) -> dict:
        res = self.tools.call(step["tool"], **step["args"])
        trace.append({"step": n, "tool": step["tool"], "args": step["args"], "reason": step.get("reason", ""),
                      "ok": res.get("ok", False), "observation": res.get("summary") or res.get("error", ""),
                      "latency_ms": res.get("latency_ms")})
        return res

    def _llm_loop(self, q: str, trace: list) -> tuple[list, str]:
        """ReAct-style loop. Written defensively because small local models (3B) sometimes repeat a call,
        invent tool names or wrap JSON in prose: JSON mode + parsing tolerance + repeat guard + rule-based rescue."""
        obs, history = [], ""
        spec = json.dumps(TOOL_SPECS)
        valid_tools = {t["name"] for t in TOOL_SPECS}
        seen_calls: set[str] = set()
        for i in range(1, MAX_STEPS + 1):
            out = self.llm.complete(prompts.AGENT_PLANNER_SYSTEM,
                                    f"TOOLS: {spec}\n\nUSER QUESTION: {q}\n\nSTEPS SO FAR:\n{history or '(none)'}",
                                    max_tokens=300, json_mode=True)
            d = _parse_planner_json(out)
            tool = d.get("tool") or d.get("name") or d.get("action")
            args = d.get("args") or d.get("parameters") or {}
            if not isinstance(args, dict):
                args = {}
            if not tool:
                if "operation" in args:
                    tool = "analyze_data"
                elif "expression" in args:
                    tool = "calculate"
                elif "query" in args:
                    tool = "search_documents"
                elif "customer_id" in args:
                    tool = "account_briefing"
            if not tool or str(tool).upper() == "FINISH":
                break
            tool = str(tool)
            if tool not in valid_tools:
                raise ValueError(f"planner chose unknown tool '{tool}'")
            key = tool + json.dumps(args, sort_keys=True, default=str)
            if key in seen_calls:        # the model is looping on the same call -> we already have that observation
                break
            seen_calls.add(key)
            step = {"tool": tool, "args": args, "reason": d.get("thought", "")}
            res = self._exec(step, trace, i)
            obs.append((step, res))
            history += f"{i}. {tool}({json.dumps(step['args'], default=str)}) -> {res.get('summary') or res.get('error')}\n"
        if not obs:
            raise ValueError("planner made no calls")
        # Safety net: a 3B planner may stop after one tool even though the question needs documents too.
        # If the question clearly asks about policy/contract text and nothing searched the documents, add that step.
        if not any(s["tool"] in ("search_documents", "account_briefing", "generate_report", "calculate") for s, _ in obs):
            if re.search(POLICY_WORDS, q.lower()):
                st = {"tool": "search_documents", "args": {"query": _doc_clause(q, True), "k": 5},
                      "reason": "rescue: question references policy/contract content"}
                obs.append((st, self._exec(st, trace, len(trace) + 1)))
        return obs, f"{self.llm.model_name} (ReAct planner)"

    # ------------------------------------------------------------------ synthesis
    def _finalise(self, q: str, obs: list, trace: list, plan_by: str, flags: list) -> dict:
        sections, hits, citations_from_briefing, artifacts = [], [], [], {}
        facts: list[str] = []
        abstained_docs = False
        briefing_verification = None
        support, doc_query = None, q
        for step, res in obs:
            tool = step["tool"]
            if not res.get("ok", True):
                sections.append(f"**{tool}** could not complete: {res.get('error')}")
                continue
            if tool == "analyze_data":
                block = f"**Data analysis ({res.get('operation')})**\n{res['summary']}"
                if res.get("markdown"):
                    block += "\n\n" + res["markdown"]
                sections.append(block)
                facts.append(res["summary"])
            elif tool == "predict_churn":
                sections.append(f"**Churn prediction**\n{res['summary']}")
                facts.append(res["summary"])
            elif tool == "account_briefing":
                b = res["briefing"]
                sections.append(b["narrative"])
                citations_from_briefing = b["citations"]
                briefing_verification = b.get("verification")
                facts.append(res["summary"])
            elif tool == "calculate":
                sections.append(f"**Calculation**\n{res['summary']}")
            elif tool == "generate_report":
                artifacts = {"markdown": res["markdown_path"], "docx": res["docx_path"]}
                sections.append(f"**Report generated**\n{res['summary']}")
            elif tool == "search_documents":
                hits.extend(res["hits"])
                support = res["support"]
                doc_query = step["args"].get("query", q)
        citations, generated_by, verification = citations_from_briefing, plan_by, briefing_verification
        if hits:
            # de-duplicate while keeping order
            seen, uniq = set(), []
            for h in hits:
                if h.chunk.chunk_id not in seen:
                    seen.add(h.chunk.chunk_id)
                    uniq.append(h)
            ans = self.gen.answer(doc_query, uniq, support=support, extra_facts="\n".join(facts))
            if ans["abstained"]:
                abstained_docs = True
                if not any(s.startswith("**Data") for s in sections):
                    sections.append(NO_EVIDENCE)
                else:
                    sections.append("**Documents**\n" + NO_EVIDENCE)
            else:
                sections.append(("**Document evidence**\n" if len(sections) else "") + ans["answer"])
                citations, verification = ans["citations"], ans["verification"]
        answer = redact_pii("\n\n".join(sections)) if sections else NO_EVIDENCE
        if self.llm.available and len(obs) > 1 and not abstained_docs:
            answer = self._llm_synthesis(q, sections, hits, citations) or answer
        return {"question": q, "answer": answer, "citations": citations, "trace": trace,
                "tools_used": [s["tool"] for s, _ in obs], "abstained": abstained_docs and not facts,
                "generated_by": generated_by, "artifacts": artifacts, "verification": verification, "warnings": flags}

    def _llm_synthesis(self, q, sections, hits, citations):
        """Let the LLM merge multi-tool results into one answer, but only accept it if it keeps the facts:
        every number in the tool output must survive and citation markers must not be dropped."""
        try:
            draft = self.llm.complete(prompts.GROUNDED_SYSTEM,
                                      "Combine these tool results into one concise answer. Keep citation numbers exactly as given "
                                      "and copy all numbers exactly.\n\n" + "\n\n".join(sections) + f"\n\nQUESTION: {q}", max_tokens=700)
        except LLMError:
            return None
        src = "\n".join(sections)
        src_cites = set(re.findall(r"\[\d+\]", src))
        if src_cites and not (src_cites & set(re.findall(r"\[\d+\]", draft))):
            return None                                     # citations dropped -> keep the deterministic, cited answer
        src_nums = set(re.findall(r"\d[\d,]*\.?\d*%?", src))
        new_nums = set(re.findall(r"\d[\d,]*\.?\d*%?", re.sub(r"\[\d+\]", "", draft)))
        if new_nums - src_nums:                             # numbers that are in no tool output = invented
            return None
        return draft
