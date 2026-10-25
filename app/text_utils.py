"""Tokenisation, light stemming and sentence splitting shared by BM25, reranker and generator."""
from __future__ import annotations

import re

STOPWORDS = frozenset("""
a an the and or of to in on at by for from with without within into over under is are was were be been being
do does did done has have had having it its this that these those as if then than so such not no nor
what which who whom whose when where why how can could should would will shall may might must
our your their his her we you they i me my us them there here about any all each per also
tell me give show list explain describe please
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_SENT_RE = re.compile(r"(?<![A-Z]\.)(?<=[.!?])\s+(?=[A-Z0-9\[(])")


def stem(tok: str) -> str:
    """Very small suffix stripper: good enough to match discount/discounts, trigger/triggers."""
    if tok.isdigit() or len(tok) <= 3:
        return tok
    for suf in ("ations", "ation", "ings", "ing", "ies", "ied", "es", "ed", "ly", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            base = tok[: -len(suf)]
            if suf in ("ies", "ied"):
                base += "y"
            return base
    return tok


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    toks = _TOKEN_RE.findall(text.lower().replace("_", " "))
    toks = [t.rstrip(".,") for t in toks]
    if not keep_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return [stem(t) for t in toks if t]


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def normalise_ws(text: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\r\n?", "\n", text)).strip()
