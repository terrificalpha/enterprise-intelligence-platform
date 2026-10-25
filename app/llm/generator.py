"""Grounded answer generation. Combines retrieved passages with ML predictions / analytics facts.
Uses a real LLM when configured, otherwise a deterministic extractive composer. Both paths produce
numbered citations that are then *verified* against the retrieved passages."""
from __future__ import annotations

import re

from ..rag.knowledge_base import Hit, KnowledgeBase
from ..security import neutralise_context, redact_pii
from ..text_utils import split_sentences, tokenize
from . import prompts
from .client import LLMClient, LLMError

NO_EVIDENCE = "I don't have enough evidence in the provided documents to answer that."
_CITE_RE = re.compile(r"\[(\d+)\]")


def readable(hit: Hit) -> str:
    """Human-readable rendering of a chunk (turns CSV records into a sentence)."""
    c = hit.chunk
    if c.doc_type == "csv" and c.meta:
        m = c.meta
        if "note" in m:
            return f"Account note for {m.get('customer_id')} ({m.get('date')}, {m.get('author')}): {m['note']}"
        if "description" in m:
            return (f"Support ticket {m.get('ticket_id')} for {m.get('customer_id')} ({m.get('priority')}): "
                    f"{m.get('subject')}. {m['description']}")
        return c.text
    text = c.text
    if c.section and text.startswith(c.section):
        text = text[len(c.section):].lstrip(". ")
    return text


def _clean_sentences(hit: Hit) -> list[str]:
    if hit.chunk.doc_type == "csv":
        return [readable(hit)]          # one record = one atomic statement
    return split_sentences(readable(hit))


def verify_citations(answer: str, hits: list[Hit]) -> dict:
    """Check that (a) every [n] refers to a real passage and (b) each cited sentence is lexically supported
    by the passage(s) it cites (content-token overlap, numbers must match)."""
    norm = re.sub(r"([.!?])\s*((?:\[\d+\])+)", r" \2\1", answer)      # "... higher. [1]" -> "... higher [1]."
    sents = [s for line in norm.splitlines() for s in split_sentences(line) if _CITE_RE.search(s)]
    n_valid = n_total = n_supported = 0
    problems = []
    for s in sents:
        ids = [int(x) for x in _CITE_RE.findall(s)]
        n_total += len(ids)
        good = [i for i in ids if 1 <= i <= len(hits)]
        n_valid += len(good)
        if len(good) != len(ids):
            problems.append(f"invalid citation in: {s[:60]}")
        body = _CITE_RE.sub("", s)
        toks = set(tokenize(body))
        if not toks or not good:
            continue
        src = set()
        for i in good:
            src |= set(tokenize(readable(hits[i - 1]) + " " + hits[i - 1].chunk.text))
        nums = set(re.findall(r"\d+(?:[.,]\d+)?", body))
        src_text = " ".join(hits[i - 1].chunk.text for i in good)
        overlap = len(toks & src) / len(toks)
        nums_ok = all(n in src_text for n in nums) if nums else True
        if overlap >= 0.6 and nums_ok:
            n_supported += 1
        else:
            problems.append(f"weakly supported ({overlap:.0%} overlap): {s[:70]}")
    return {"sentences_with_citations": len(sents), "citations": n_total, "valid_citation_ids": n_valid,
            "supported_sentences": n_supported, "all_valid": n_valid == n_total,
            "support_rate": (n_supported / len(sents)) if sents else 0.0, "problems": problems}


_RANGE_RE = re.compile(r"between (\d+(?:\.\d+)?) and (\d+(?:\.\d+)?)")


def _number_in_range(question: str, sentence: str) -> float:
    """1.0 if a number in the question falls inside a 'between A and B' range in the sentence."""
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", question)]
    for a, b in _RANGE_RE.findall(sentence):
        if any(float(a) < n < float(b) for n in nums):
            return 1.0
    return 0.0


class Generator:
    def __init__(self, kb: KnowledgeBase, llm: LLMClient | None = None):
        self.kb, self.llm = kb, llm or LLMClient()

    # ------------------------------------------------------------------ public API
    def answer(self, question: str, hits: list[Hit], support: dict | None = None,
               extra_facts: str = "", system: str | None = None) -> dict:
        support = support or self.kb.assess_support(question, hits)
        if not support["supported"]:
            return {"answer": NO_EVIDENCE, "abstained": True, "citations": [], "support": support,
                    "generated_by": "guardrail", "verification": verify_citations("", hits)}
        used_hits = hits[: self.kb_top(hits)]
        text, by = None, "offline-extractive"
        if self.llm.available:
            try:
                ctx = prompts.format_context([(i, h.chunk.source, neutralise_context(readable(h)))
                                              for i, h in enumerate(used_hits, 1)])
                user = f"CONTEXT:\n{ctx}\n\n{('FACTS:\n' + extra_facts + chr(10) + chr(10)) if extra_facts else ''}QUESTION: {question}"
                text = self.llm.complete(system or prompts.GROUNDED_SYSTEM, user)
                by = self.llm.model_name
                if not self._citations_ok(text, len(used_hits)):
                    # small local models sometimes forget the [n] markers: one stricter retry, then fall back
                    text = self.llm.complete(system or prompts.GROUNDED_SYSTEM,
                                             user + "\n\nREMINDER: end every sentence with the passage number(s) in square brackets, "
                                                    "e.g. [1]. Use only numbers 1-" + str(len(used_hits)) + ".")
                    if not self._citations_ok(text, len(used_hits)):
                        text, by = None, "offline-extractive (LLM output had no valid citations)"
            except LLMError:
                text = None
        if text is None:
            text = self._extractive(question, used_hits, extra_facts)
        text = redact_pii(text)
        ver = verify_citations(text, used_hits)
        abstained = text.strip().startswith("I don't have enough evidence")
        cited = sorted({int(i) for i in _CITE_RE.findall(text) if 1 <= int(i) <= len(used_hits)})
        cites = [{"id": i, "source": used_hits[i - 1].chunk.source, "location": used_hits[i - 1].chunk.location,
                  "section": used_hits[i - 1].chunk.section, "snippet": readable(used_hits[i - 1])[:400],
                  "score": round(used_hits[i - 1].score, 3)} for i in cited]
        return {"answer": text, "abstained": abstained, "citations": cites, "support": support,
                "generated_by": by, "verification": ver}

    @staticmethod
    def _citations_ok(text: str, n_passages: int) -> bool:
        """An LLM answer is acceptable if it is a clean refusal, or it cites at least one real passage and only real ones."""
        t = (text or "").strip()
        if not t:
            return False
        if t.startswith("I don't have enough evidence"):
            return True
        ids = [int(x) for x in _CITE_RE.findall(t)]
        return bool(ids) and all(1 <= i <= n_passages for i in ids)

    @staticmethod
    def kb_top(hits: list[Hit]) -> int:
        return min(len(hits), 5)

    # ------------------------------------------------------------------ offline composer
    def _extractive(self, question: str, hits: list[Hit], extra_facts: str = "") -> str:
        rr = self.kb.reranker
        cands = []
        for i, h in enumerate(hits, 1):
            for s in _clean_sentences(h):
                if len(s) < 25:
                    continue
                cov = (rr.coverage(question, s) + 0.15 * rr.coverage(question, h.chunk.section or "")
                       + 0.3 * _number_in_range(question, s))
                cands.append((cov - 0.02 * (i - 1), i, s))
        cands.sort(key=lambda t: -t[0])
        if not cands:
            return NO_EVIDENCE
        top = cands[0][0]
        if top < 0.2:                       # nothing in the passages actually answers the question
            return NO_EVIDENCE
        picked = [c for c in cands if c[0] >= 0.85 * top][:2] or cands[:1]
        picked.sort(key=lambda t: t[1])
        lines = [f"- {s} [{i}]" for _, i, s in picked]
        return "\n".join(lines)
