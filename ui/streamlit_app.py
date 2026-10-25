"""Streamlit UI for the Autonomous Enterprise Intelligence Platform.
Run:  streamlit run ui/streamlit_app.py     (needs the API on :8000, or set EIP_UI_MODE=local)"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from ui.client import get_client

st.set_page_config(page_title="Enterprise Intelligence Platform", page_icon="📊", layout="wide")
FIG = Path(__file__).resolve().parent.parent / "artifacts" / "figures"
TIER_ICON = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}


@st.cache_resource(show_spinner="Connecting to backend...")
def client():
    return get_client()


def call(fn, *a, **kw):
    """Run a backend call with a spinner and friendly errors."""
    try:
        with st.spinner("Working..."):
            return fn(*a, **kw)
    except Exception as e:
        st.error(str(e))
        return None


def render_citations(cites):
    if not cites:
        return
    with st.expander(f"📎 Sources ({len(cites)})", expanded=True):
        for c in cites:
            st.markdown(f"**[{c['id']}] {c['source']}** · {c['location']}" + (f" · _{c['section']}_" if c.get("section") else ""))
            st.caption(c["snippet"])


def render_verification(v):
    if v and v.get("sentences_with_citations"):
        ok = v["all_valid"] and v["support_rate"] >= 0.99
        st.caption(("✅ " if ok else "⚠️ ") + f"Citation check: {v['valid_citation_ids']}/{v['citations']} ids valid · "
                   f"{v['supported_sentences']}/{v['sentences_with_citations']} cited sentences supported by their source")


api = client()
h = call(api.health) or {}

with st.sidebar:
    st.title("📊 Enterprise Intelligence")
    st.caption("ML + RAG + GenAI + Agents")
    st.markdown(f"**Backend:** `{api.mode}`  \n**LLM:** `{h.get('llm_model', '?')}`  \n**Embeddings:** `{h.get('embedding', '?')}`  \n"
                f"**Docs / chunks:** {h.get('documents', '?')} / {h.get('chunks', '?')}")
    if h.get("model"):
        st.markdown(f"**Churn model:** `{h['model']['model']}`")
    if h.get("llm_available"):
        provider_name = h.get("llm_provider", "local")
        st.success(f"🟢 **LLM Active**: `{h.get('llm_model')}`  \n*Local {provider_name.title()} — 0 cloud inference calls*")
    else:
        st.warning("⚠️ Running on the offline deterministic generator (still cited and grounded).  \n"
                   + (h.get("llm_hint") or "Start Ollama (`ollama serve`) and pull a model: `ollama pull qwen2.5:3b`."))

tabs = st.tabs(["🏠 Dashboard", "🤖 Model", "📄 Documents", "💬 Ask (RAG)", "🧾 Account Insight", "🛠️ Agent", "📈 Evaluation"])

# ------------------------------------------------------------------ Dashboard
with tabs[0]:
    st.header("Portfolio risk dashboard")
    pf = call(api.portfolio, 15)
    if pf:
        ov = pf["overview"]["result"]
        c = st.columns(5)
        c[0].metric("Customers", f"{ov['customers']:,}")
        c[1].metric("Actual churn rate", f"{ov['actual_churn_rate']:.1%}")
        c[2].metric("High-risk accounts", ov["tier_counts"].get("High", 0))
        c[3].metric("Total ACV", f"${ov['total_acv'] / 1e6:,.0f}M")
        c[4].metric("Expected loss", f"${ov['expected_loss'] / 1e6:,.1f}M")
        l, r = st.columns(2)
        with l:
            st.subheader("Churn rate by industry")
            d = pd.DataFrame(pf["by_industry"]["table"]).set_index("industry")
            st.bar_chart(d[["actual_churn_rate", "avg_predicted_churn"]])
        with r:
            st.subheader("Expected revenue loss by industry")
            d = pd.DataFrame(pf["revenue"]["table"]).set_index("industry")
            st.bar_chart(d[["expected_loss"]])
        st.subheader("Top at-risk accounts")
        t = pd.DataFrame(pf["top_risk"]["table"])
        t["churn_probability"] = t["churn_probability"].map("{:.0%}".format)
        st.dataframe(t, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ Model
with tabs[1]:
    st.header("Machine-learning models")
    m = call(api.metrics)
    if m:
        rows = []
        for k in ("baseline_prior", "logistic_regression", "random_forest", "hist_gradient_boosting"):
            v = m[k]
            rows.append({"model": k, "CV ROC-AUC": v.get("cv_roc_auc"), **{f"test {x}": v["test"][x] for x in ("roc_auc", "pr_auc", "f1", "precision", "recall", "brier")}})
        st.dataframe(pd.DataFrame(rows).round(3), use_container_width=True, hide_index=True)
        sel = m["selected"]
        st.success(f"Selected: **{sel['model']}** (best CV ROC-AUC) - calibrated test ROC-AUC {sel['test_calibrated']['roc_auc']:.3f}, "
                   f"F1 {sel['test_calibrated']['f1']:.3f}, threshold {sel['threshold']:.2f}")
        c1, c2 = st.columns(2)
        if (FIG / "roc_pr.png").exists():
            c1.image(str(FIG / "roc_pr.png"), caption="ROC and Precision-Recall curves")
            c2.image(str(FIG / "feature_importance.png"), caption="Permutation importance")
            c1.image(str(FIG / "calibration.png"), caption="Calibration")
        c2.subheader("Confusion matrix (test)")
        c2.dataframe(pd.DataFrame(sel["confusion_matrix"], index=["actual stay", "actual churn"], columns=["pred stay", "pred churn"]))

# ------------------------------------------------------------------ Documents
with tabs[2]:
    st.header("Document intelligence")
    st.write("Upload **PDF, DOCX or CSV** files. Text is extracted, chunked with page/section/row metadata, embedded and indexed immediately.")
    up = st.file_uploader("Add documents", type=["pdf", "docx", "csv", "txt", "md"], accept_multiple_files=True)
    if up and st.button("Ingest files", type="primary"):
        for f in up:
            r = call(api.upload, f.name, f.getvalue())
            if r:
                st.success(f"{r['filename']}: {r['chunks_added']} chunks added (index now {r['total_chunks']})")
    docs = call(api.documents)
    if docs:
        st.dataframe(pd.DataFrame(docs), use_container_width=True, hide_index=True)
    if st.button("Rebuild index from disk"):
        r = call(api.rebuild)
        if r:
            st.success(f"Rebuilt index: {r['chunks']} chunks")
    st.subheader("Raw retrieval explorer")
    q = st.text_input("Query", "credit hold overdue invoices", key="sq")
    mode = st.radio("Retrieval mode", ["bm25", "dense", "hybrid", "hybrid_rerank"], index=3, horizontal=True)
    if st.button("Search") and q:
        r = call(api.search, q, 5, mode)
        if r:
            st.caption(f"{r['latency_ms']} ms · evidence confidence {r['support']['confidence']:.0%} ({'supported' if r['support']['supported'] else 'insufficient'})")
            for x in r["results"]:
                with st.expander(f"{x['source']} · {x['location']}  —  score {x['score']:.3f}"):
                    st.write(x["text"])
                    st.caption(f"dense {x['dense']} · bm25 {x['bm25']} · rerank {x['rerank']} · coverage {x['coverage']}")

# ------------------------------------------------------------------ Ask
with tabs[3]:
    st.header("Ask the documents")
    st.caption("Answers are grounded in retrieved passages with numbered citations. Unsupported questions are refused.")
    mode = st.selectbox("Retrieval mode", ["hybrid_rerank", "hybrid", "dense", "bm25"], key="askmode")
    ex = st.selectbox("Examples", ["", "Who approves a retention discount of 15 percent?", "When is a credit hold applied on overdue invoices?",
                                   "What is the notice period for non-renewal in the MSA?", "Which competitor bundles free support?", "What is our parental leave policy?"])
    q = st.text_input("Your question", ex, key="askq")
    if st.button("Ask", type="primary") and q:
        r = call(api.ask, q, 5, mode)
        if r:
            (st.warning if r["abstained"] else st.success)("Insufficient evidence - refused to answer" if r["abstained"] else f"Answered by {r['generated_by']} · {r['latency_ms']} ms")
            st.markdown(r["answer"])
            render_verification(r.get("verification"))
            render_citations(r["citations"])
            st.caption(f"Evidence confidence {r['support']['confidence']:.0%} - {r['support']['reason']}")

# ------------------------------------------------------------------ Insight
with tabs[4]:
    st.header("Account insight: ML prediction + evidence + recommendations")
    pf = call(api.portfolio, 25)
    options = [x["customer_id"] for x in pf["top_risk"]["table"]] if pf else []
    c1, c2 = st.columns([1, 2])
    pick = c1.selectbox("Top at-risk accounts", options)
    cid = c2.text_input("...or enter a customer id", pick or "C0001")
    if st.button("Generate briefing", type="primary"):
        st.session_state.briefing = call(api.briefing, cid.strip().upper())
    b = st.session_state.get("briefing")      # kept in session_state so later button clicks (report) do not lose it
    if b:
        p = b["prediction"]
        m = st.columns(4)
        m[0].metric("Churn probability", f"{p['churn_probability']:.0%}")
        m[1].metric("Risk tier", f"{TIER_ICON[p['risk_tier']]} {p['risk_tier']}")
        m[2].metric("Expected revenue at risk", f"${p['expected_revenue_at_risk']:,.0f}")
        m[3].metric("Model", p["model"])
        l, r = st.columns([3, 2])
        with l:
            st.markdown(b["narrative"])
            render_verification(b.get("verification"))
        with r:
            st.subheader("What drives this score")
            d = pd.DataFrame(b["drivers"])[["label", "value", "impact"]].set_index("label")
            st.bar_chart(d[["impact"]])
        render_citations(b["citations"])
        if st.button("Create downloadable report for this account"):
            res = call(api.report, f"Account Risk Report {b['customer_id']}", b["customer_id"], 10)
            if res:
                st.session_state.account_report = [(Path(res[k]).name, ext) for ext, k in (("md", "markdown_path"), ("docx", "docx_path"))]
        for name, ext in st.session_state.get("account_report") or []:
            st.download_button(f"Download {ext.upper()}", api.report_bytes(name), file_name=name, key=f"dl_{name}")

# ------------------------------------------------------------------ Agent
with tabs[5]:
    st.header("AI agent")
    st.caption("The agent decides which tools to call: document search, data analysis, churn prediction, calculator, report generation.")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("trace"):
                with st.expander("🔍 Agent trace"):
                    st.dataframe(pd.DataFrame(msg["trace"])[["step", "tool", "reason", "observation", "latency_ms"]], hide_index=True)
            render_citations(msg.get("citations"))
    prompt = st.chat_input("e.g. Which industry has the highest churn and what does policy say about escalation?")
    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})
        res = call(api.agent, prompt)
        if res:
            note = "\n\n> ⚠️ " + "; ".join(res["warnings"]) if res.get("warnings") else ""
            st.session_state.chat.append({"role": "assistant", "content": res["answer"] + note, "trace": res["trace"], "citations": res["citations"]})
            if res.get("artifacts"):
                name = Path(res["artifacts"]["docx"]).name
                st.session_state.last_report = (name, api.report_bytes(name))
        st.rerun()
    if st.session_state.get("last_report"):
        n, data = st.session_state.last_report
        st.download_button(f"⬇️ Download generated report ({n})", data, file_name=n)

# ------------------------------------------------------------------ Evaluation
with tabs[6]:
    st.header("Evaluation results")
    ev = Path(__file__).resolve().parent.parent / "artifacts" / "evaluation" / "results.json"
    if ev.exists():
        import json
        r = json.loads(ev.read_text())
        a, u, g = r["answers"], r["unsupported"], r["agent"]
        c = st.columns(5)
        c[0].metric("Answer accuracy", f"{a['answer_accuracy']:.0%}")
        c[1].metric("Citation id validity", f"{a['citation_id_validity']:.0%}")
        c[2].metric("OOD refusal", f"{u['ood_refusal_rate']:.0%}")
        c[3].metric("Agent tool selection", f"{g['tool_selection_accuracy']:.0%}")
        c[4].metric("Agent task success", f"{g['task_success_rate']:.0%}")
        st.subheader("Retrieval modes compared")
        st.dataframe(pd.DataFrame(r["retrieval"]).T[["hit@1", "hit@3", "hit@5", "mrr", "latency_ms_avg"]].round(3))
        st.subheader("Unsupported-question threshold trade-off")
        st.dataframe(pd.DataFrame(u["threshold_sweep"]).round(3), hide_index=True)
        st.markdown((Path(__file__).resolve().parent.parent / "docs" / "evaluation_results.md").read_text())
    else:
        st.info("Run `python -m app.evaluation.run_all` to generate evaluation results.")
