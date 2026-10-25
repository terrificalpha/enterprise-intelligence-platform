import tempfile
import unittest
from pathlib import Path

from tests.helpers import platform


class TestRAG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.P = platform()
        cls.kb = cls.P.kb

    def test_bm25_ranks_exact_term_first(self):
        from app.rag.bm25 import BM25
        b = BM25().fit(["apples and oranges", "credit hold policy overdue invoices", "weather report"])
        s = b.scores("credit hold")
        self.assertEqual(int(s.argmax()), 1)

    def test_known_facts_retrieved_in_top3_all_modes(self):
        cases = [("Who approves a retention discount of 15 percent?", "Customer_Retention_Policy.pdf", "Regional VP"),
                 ("What triggers Credit Watch?", "Credit_and_Payment_Risk_Policy.pdf", "Three or more late payments"),
                 ("Which competitor bundles free support?", "Competitive_Landscape_Brief.docx", "OrbitDesk")]
        for mode in ("bm25", "dense", "hybrid", "hybrid_rerank"):
            for q, src, kw in cases:
                hits = self.kb.search(q, 3, mode)
                self.assertTrue(any(h.chunk.source == src and kw in h.chunk.text for h in hits), (mode, q))

    def test_hits_carry_citation_metadata(self):
        h = self.kb.search("late fee overdue balances", 1)[0]
        self.assertTrue(h.chunk.source and h.chunk.location and h.chunk.chunk_id)
        self.assertEqual(set(h.to_dict()) >= {"source", "location", "score", "dense", "bm25", "rerank"}, True)

    def test_customer_scoping(self):
        import pandas as pd
        from app.config import settings
        cid = pd.read_csv(settings.documents_dir / "account_notes.csv").customer_id.iloc[0]
        hits = self.kb.search(f"What do the notes say about {cid}?", 5)
        self.assertTrue(hits and all(h.chunk.customer_id == cid for h in hits))

    def test_source_filter(self):
        hits = self.kb.search("discount", 5, filters={"source": "Customer_Retention_Policy.pdf"})
        self.assertTrue(hits and all(h.chunk.source == "Customer_Retention_Policy.pdf" for h in hits))

    def test_policy_outranks_ticket_noise(self):
        hits = self.kb.search("What is the P1 resolution time?", 5, "hybrid_rerank")
        self.assertTrue(hits[0].chunk.doc_type in ("pdf", "docx"))

    def test_support_check_refuses_out_of_domain(self):
        for q in ["Who won the 2022 FIFA World Cup?", "What is the CEO's salary?", "How do I configure Kubernetes autoscaling?"]:
            self.assertFalse(self.kb.assess_support(q, self.kb.search(q, 5))["supported"], q)

    def test_support_check_accepts_answerable(self):
        for q in ["What late fee applies to overdue balances?", "When must an executive sponsor be assigned?"]:
            self.assertTrue(self.kb.assess_support(q, self.kb.search(q, 5))["supported"], q)

    def test_new_document_is_searchable_after_ingest(self):
        from app.config import settings
        from app.rag.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(index_dir=tempfile.mkdtemp())      # isolated: must never overwrite the real index
        kb.build(chunks=self.kb.chunks[:40])
        tmp = Path(tempfile.mkdtemp()) / "zebra_policy.txt"
        tmp.write_text("Zebra Handling Policy. All zebra crossings must be inspected every 14 days by the Stripe Team.")
        n = kb.add_file(tmp)
        self.assertGreaterEqual(n, 1)
        top = kb.search("How often must zebra crossings be inspected?", 1)[0]
        self.assertEqual(top.chunk.source, "zebra_policy.txt")

    def test_index_persists_and_reloads(self):
        from app.rag.knowledge_base import KnowledgeBase
        kb = KnowledgeBase()
        self.assertTrue(kb.load())
        self.assertEqual(len(kb.chunks), len(self.kb.chunks))
        self.assertEqual(kb.search("credit hold", 1)[0].chunk.chunk_id, self.kb.search("credit hold", 1)[0].chunk.chunk_id)

    def test_reranker_prefers_complete_coverage(self):
        rr = self.kb.reranker
        q = "retention discount approval"
        self.assertGreater(rr.score(q, "retention discount approval by the Regional VP")["score"], rr.score(q, "discount")["score"])


if __name__ == "__main__":
    unittest.main()
