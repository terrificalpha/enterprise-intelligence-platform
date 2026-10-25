"""Unit and integration tests for Ollama provider, error handling, index metadata, and agent execution."""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest import mock

import requests

from app.config import settings
from app.llm.client import (
    LLMClient,
    LLMError,
    OllamaConnectionError,
    OllamaError,
    OllamaModelNotFoundError,
)
from app.rag.knowledge_base import KnowledgeBase
from tests.helpers import ROOT, platform


class TestOllamaMocked(unittest.TestCase):
    """Offline unit tests using mocked HTTP responses (no live server needed)."""

    def setUp(self):
        self.client = LLMClient(provider="ollama", base_url="http://localhost:11434", model="qwen2.5:3b")

    @mock.patch("requests.get")
    def test_available_when_ollama_responds_200(self, mock_get):
        mock_get.return_value.status_code = 200
        self.assertTrue(self.client.available)

    @mock.patch("requests.get")
    def test_unavailable_when_ollama_down(self, mock_get):
        mock_get.side_effect = requests.ConnectionError("Connection refused")
        c = LLMClient(provider="ollama", base_url="http://localhost:9999", model="qwen2.5:3b")
        self.assertFalse(c.available)

    @mock.patch("requests.post")
    @mock.patch("requests.get")
    def test_successful_completion(self, mock_get, mock_post):
        mock_get.return_value.status_code = 200
        mock_post.return_value.ok = True
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "message": {"content": "The notice period is 30 days [1]."}
        }

        res = self.client.complete(system="Answer with citations", user="Notice period?")
        self.assertEqual(res, "The notice period is 30 days [1].")

        # Verify payload sent to Ollama
        call_args = mock_post.call_args
        self.assertEqual(call_args[0][0], "http://localhost:11434/api/chat")
        payload = call_args[1]["json"]
        self.assertEqual(payload["model"], "qwen2.5:3b")
        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][1]["role"], "user")

    @mock.patch("requests.post")
    @mock.patch("requests.get")
    def test_missing_model_raises_model_not_found(self, mock_get, mock_post):
        mock_get.return_value.status_code = 200
        mock_post.return_value.ok = False
        mock_post.return_value.status_code = 404
        mock_post.return_value.text = '{"error": "model \'missing:latest\' not found, try pulling it first"}'
        mock_post.return_value.json.return_value = {"error": "model 'missing:latest' not found, try pulling it first"}

        c = LLMClient(provider="ollama", base_url="http://localhost:11434", model="missing:latest")
        with self.assertRaises(OllamaModelNotFoundError) as ctx:
            c._call_ollama("system", "hello", 100, 0.0)
        self.assertIn("not found", str(ctx.exception).lower())

    @mock.patch("requests.post")
    @mock.patch("requests.get")
    def test_connection_error_raises_ollama_connection_error(self, mock_get, mock_post):
        mock_get.return_value.status_code = 200
        mock_post.side_effect = requests.ConnectionError("Connection refused")

        with self.assertRaises(OllamaConnectionError):
            self.client._call_ollama("system", "hello", 100, 0.0)

    @mock.patch("requests.post")
    @mock.patch("requests.get")
    def test_timeout_raises_ollama_error(self, mock_get, mock_post):
        mock_get.return_value.status_code = 200
        mock_post.side_effect = requests.Timeout("Request timed out")

        with self.assertRaises(OllamaError):
            self.client._call_ollama("system", "hello", 100, 0.0)

    @mock.patch("requests.get")
    def test_list_installed_models(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "models": [{"name": "qwen2.5:3b"}, {"name": "llama3.2:3b"}]
        }
        models = self.client.list_installed_models()
        self.assertEqual(models, ["qwen2.5:3b", "llama3.2:3b"])
        self.assertTrue(self.client.check_model_installed("qwen2.5:3b"))
        self.assertFalse(self.client.check_model_installed("gpt-4"))


class TestIndexMetadataAndCompatibility(unittest.TestCase):
    """Tests that knowledge base index records metadata and catches incompatibilities."""

    def setUp(self):
        self.P = platform()

    def test_index_meta_exists_and_records_attributes(self):
        meta_path = self.P.kb.index_path / "index_meta.json"
        self.assertTrue(meta_path.exists(), "index_meta.json must exist in index directory")
        meta = json.loads(meta_path.read_text())
        self.assertIn("embedding_backend", meta)
        self.assertIn("dim", meta)
        self.assertIn("chunk_count", meta)
        self.assertGreater(meta["chunk_count"], 0)
        self.assertGreater(meta["dim"], 0)

    def test_incompatible_backend_detected_on_load(self):
        kb = KnowledgeBase(self.P.kb.index_path)
        # Verify it loads normally
        self.assertTrue(kb.load())

        # Now simulate an incompatible backend in index_meta.json
        meta_path = self.P.kb.index_path / "index_meta.json"
        original = meta_path.read_text()
        try:
            fake_meta = json.loads(original)
            fake_meta["embedding_backend"] = "nonexistent-neural-backend"
            meta_path.write_text(json.dumps(fake_meta))

            kb_test = KnowledgeBase(self.P.kb.index_path)
            # Should detect mismatch and return False so it triggers rebuild
            self.assertFalse(kb_test.load())
        finally:
            meta_path.write_text(original)


class TestAgentToolExecutionVerified(unittest.TestCase):
    """Verifies that the agent tools execute real functions and return valid data."""

    @classmethod
    def setUpClass(cls):
        cls.P = platform()

    def test_calculate_tool_safe_eval(self):
        res = self.P.tools.call("calculate", expression="15% of 250000 + 4000")
        self.assertTrue(res["ok"])
        self.assertEqual(res["value"], 41500.0)

    def test_calculate_tool_blocks_malicious_code(self):
        res = self.P.tools.call("calculate", expression="__import__('os').system('dir')")
        self.assertFalse(res["ok"])

    def test_data_analysis_tool_real_calculations(self):
        res = self.P.tools.call("analyze_data", operation="churn_by", column="industry")
        self.assertTrue(res["ok"])
        self.assertIn("Highest churn by industry", res["summary"])
        self.assertTrue(len(res["table"]) > 0)

    def test_predict_churn_tool_real_predictions(self):
        cid = self.P.tools.pred.portfolio.customer_id.iloc[0]
        res = self.P.tools.call("predict_churn", customer_id=cid)
        self.assertTrue(res["ok"])
        self.assertIn("churn_probability", res["prediction"])
        self.assertTrue(0.0 <= res["prediction"]["churn_probability"] <= 1.0)
        self.assertTrue(len(res["drivers"]) > 0)

    def test_account_briefing_tool_combines_ml_and_rag(self):
        hi_custs = self.P.tools.pred.portfolio[self.P.tools.pred.portfolio.risk_tier == "High"]
        cid = hi_custs.customer_id.iloc[0] if not hi_custs.empty else self.P.tools.pred.portfolio.customer_id.iloc[0]
        res = self.P.tools.call("account_briefing", customer_id=cid)
        self.assertTrue(res["ok"])
        briefing = res["briefing"]
        self.assertIn("narrative", briefing)
        self.assertIn("recommendations", briefing)
        self.assertTrue(len(briefing["recommendations"]) > 0)

    def test_report_generation_tool(self):
        res = self.P.tools.call("generate_report", title="Test Risk Report", top_n=5)
        self.assertTrue(res["ok"])
        self.assertTrue(Path(res["markdown_path"]).exists())
        self.assertTrue(Path(res["docx_path"]).exists())


class TestOllamaLiveSmoke(unittest.TestCase):
    """Optional live smoke test run ONLY if local Ollama server is active."""

    @classmethod
    def setUpClass(cls):
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=1.0)
            if r.status_code != 200:
                raise unittest.SkipTest("Ollama server not active")
            models = [m.get("name", "") for m in r.json().get("models", [])]
            if not any("qwen2.5:3b" in m for m in models):
                raise unittest.SkipTest("qwen2.5:3b model not installed in local Ollama")
        except Exception as e:
            raise unittest.SkipTest(f"Ollama server not available: {e}")

    def test_live_ollama_grounded_answer_and_refusal(self):
        from app.llm import prompts
        from app.llm.client import LLMClient
        c = LLMClient(provider="ollama", model="qwen2.5:3b")

        # 1. Grounded query with citation
        ctx = prompts.format_context([(1, "Policy.pdf", "Notice period for contract termination is 60 calendar days.")])
        user = f"CONTEXT:\n{ctx}\n\nQUESTION: What is the notice period for contract termination?"
        ans = c.complete(prompts.GROUNDED_SYSTEM, user)
        self.assertIn("[1]", ans)
        self.assertIn("60", ans)

        # 2. Unsupported question refusal
        unsupported = f"CONTEXT:\n{ctx}\n\nQUESTION: What is our parental leave policy?"
        refusal = c.complete(prompts.GROUNDED_SYSTEM, unsupported)
        self.assertIn("I don't have enough evidence", refusal)


if __name__ == "__main__":
    unittest.main()
