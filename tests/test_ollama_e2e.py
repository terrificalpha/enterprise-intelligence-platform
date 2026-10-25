"""End-to-end tests of the Ollama integration against a *fake Ollama HTTP server* (real sockets, real `requests`).

They prove the pipeline copes with the failure modes of small local models (3B): missing citations, invented
numbers, malformed planner JSON, looping tool calls, a missing model and a stopped server. No real LLM is needed."""
from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from tests.helpers import platform

from app.agent.agent import EnterpriseAgent, _parse_planner_json
from app.llm.client import LLMClient
from app.llm.generator import Generator
from app.llm.insights import InsightGenerator


class FakeOllama:
    """Tiny stand-in for the Ollama REST API. `behaviour` switches how the 'model' misbehaves."""

    def __init__(self, models=("qwen2.5:3b",)):
        self.models, self.behaviour, self.requests = list(models), "good", []
        self.planner_script: list[str] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def _send(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/api/tags":
                    self._send(200, {"models": [{"name": m} for m in outer.models]})
                else:
                    self._send(404, {"error": "nope"})

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(payload)
                if payload["model"] not in outer.models:
                    return self._send(404, {"error": f"model '{payload['model']}' not found, try pulling it first"})
                self._send(200, {"message": {"role": "assistant", "content": outer.reply(payload)}})

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def reply(self, payload) -> str:
        system, user = payload["messages"][0]["content"], payload["messages"][-1]["content"]
        if "planner" in system:
            if self.behaviour == "garbage_planner":
                return "Sure! I would first look at the data and then think."
            return self.planner_script.pop(0) if self.planner_script else json.dumps({"tool": "FINISH", "args": {}})
        if "account risk briefing" in system:
            if self.behaviour == "bad_briefing":
                return "This customer is probably fine and will renew. " * 4        # no tier, no number, no citation
            return ""
        # grounded answer
        if self.behaviour == "no_cite":
            return "The policy requires an executive escalation within two business days."
        first = user.split("[1]", 1)[1].split("\n", 1)[1].split("\n")[0] if "[1]" in user else "n/a"
        return f"{first.strip()} [1]"

    def close(self):
        self.server.shutdown()


class TestOllamaEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.P = platform()
        cls.fake = FakeOllama()

    @classmethod
    def tearDownClass(cls):
        cls.fake.close()

    def setUp(self):
        self.fake.behaviour, self.fake.planner_script, self.fake.requests = "good", [], []
        self.llm = LLMClient(provider="ollama", base_url=self.fake.url, model="qwen2.5:3b")
        self.gen = Generator(self.P.kb, self.llm)

    # ---------------------------------------------------------------- diagnostics
    def test_status_ready(self):
        st = self.llm.status()
        self.assertTrue(st["ready"])
        self.assertEqual(st["hint"], "")

    def test_status_model_missing_gives_pull_hint(self):
        st = LLMClient(provider="ollama", base_url=self.fake.url, model="llama3:70b").status()
        self.assertTrue(st["server_up"])
        self.assertFalse(st["ready"])
        self.assertIn("ollama pull llama3:70b", st["hint"])

    def test_status_server_down(self):
        st = LLMClient(provider="ollama", base_url="http://127.0.0.1:9", model="qwen2.5:3b").status()
        self.assertFalse(st["server_up"])
        self.assertIn("not running", st["hint"])

    def test_request_carries_context_and_keepalive_options(self):
        self.llm.complete("sys", "hello")
        p = self.fake.requests[-1]
        self.assertEqual(p["options"]["num_ctx"], 4096)
        self.assertIn("keep_alive", p)
        self.assertNotIn("format", p)

    # ---------------------------------------------------------------- grounded answers
    def test_grounded_answer_through_ollama_is_cited_and_verified(self):
        q = "What is the notice period for contract termination?"
        hits = self.P.kb.search(q, k=5)
        res = self.gen.answer(q, hits)
        if res["abstained"]:
            self.skipTest("corpus did not support this question")
        self.assertIn("ollama/qwen2.5:3b", res["generated_by"])
        self.assertTrue(res["citations"])
        self.assertTrue(res["verification"]["all_valid"])

    def test_answer_without_citations_is_rejected_and_falls_back(self):
        self.fake.behaviour = "no_cite"
        q = "What is the notice period for contract termination?"
        hits = self.P.kb.search(q, k=5)
        res = self.gen.answer(q, hits)
        if res["abstained"]:
            self.skipTest("corpus did not support this question")
        self.assertIn("offline-extractive", res["generated_by"])       # never ships an uncited small-model answer
        self.assertTrue(res["citations"])

    def test_unsupported_question_never_reaches_the_llm(self):
        q = "What is our parental leave policy for new employees?"
        res = self.gen.answer(q, self.P.kb.search(q, k=5))
        self.assertTrue(res["abstained"])
        self.assertEqual(self.fake.requests, [])

    # ---------------------------------------------------------------- insights (ML x RAG x LLM)
    def test_briefing_rejects_unverifiable_llm_text(self):
        self.fake.behaviour = "bad_briefing"
        ins = InsightGenerator(self.P.kb, self.P.predictor, self.gen, self.llm)
        cid = self.P.predictor.portfolio.sort_values("churn_probability").customer_id.iloc[-1]
        b = ins.briefing(cid)
        self.assertIn("draft rejected", b["generated_by"])
        self.assertIn("Recommended actions", b["narrative"])            # deterministic cited composer took over
        self.assertTrue(b["verification"]["all_valid"])

    def test_narrative_validator_accepts_good_and_rejects_bad(self):
        h = self.P.kb.search("High Risk accounts executive outreach", k=2)
        good = ("**Summary** Account is High risk with a 72% churn probability [1]. " + "x" * 80)
        self.assertEqual(InsightGenerator._narrative_problem(good, 0.72, "High", h), "")
        self.assertTrue(InsightGenerator._narrative_problem(good.replace("72%", "10%"), 0.72, "High", h))
        self.assertTrue(InsightGenerator._narrative_problem(good.replace("[1]", "[9]"), 0.72, "High", h))

    # ---------------------------------------------------------------- agent
    def test_agent_uses_json_mode_and_tools(self):
        self.fake.planner_script = [json.dumps({"thought": "stats", "tool": "analyze_data", "args": {"operation": "overview"}}),
                                    json.dumps({"tool": "FINISH", "args": {}})]
        res = EnterpriseAgent(self.P.tools, self.gen, self.llm).run("Give me a portfolio overview")
        self.assertEqual(res["tools_used"], ["analyze_data"])
        self.assertIn("ReAct", res["generated_by"])
        planner_calls = [r for r in self.fake.requests if r.get("format") == "json"]
        self.assertTrue(planner_calls, "planner must request JSON mode")

    def test_agent_garbage_planner_falls_back_to_rules(self):
        self.fake.behaviour = "garbage_planner"
        res = EnterpriseAgent(self.P.tools, self.gen, self.llm).run("Calculate 15% of 250000 + 4000")
        self.assertEqual(res["tools_used"], ["calculate"])
        self.assertIn("rule-based", res["generated_by"])
        self.assertIn("41", res["answer"])

    def test_agent_stops_when_model_repeats_the_same_call(self):
        call = json.dumps({"tool": "analyze_data", "args": {"operation": "overview"}})
        self.fake.planner_script = [call, call, call, call]
        res = EnterpriseAgent(self.P.tools, self.gen, self.llm).run("Give me a portfolio overview")
        self.assertEqual(res["tools_used"], ["analyze_data"])             # executed once, not four times

    def test_agent_unknown_tool_falls_back_to_rules(self):
        self.fake.planner_script = [json.dumps({"tool": "hack_the_planet", "args": {}})]
        res = EnterpriseAgent(self.P.tools, self.gen, self.llm).run("Calculate 2 + 2 * 3")
        self.assertEqual(res["tools_used"], ["calculate"])

    def test_agent_survives_ollama_going_down(self):
        dead = LLMClient(provider="ollama", base_url="http://127.0.0.1:9", model="qwen2.5:3b")
        res = EnterpriseAgent(self.P.tools, Generator(self.P.kb, dead), dead).run("Calculate 2 + 2 * 3")
        self.assertEqual(res["tools_used"], ["calculate"])


class TestPlannerJsonParsing(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(_parse_planner_json('{"tool": "calculate"}')["tool"], "calculate")

    def test_fenced_and_chatty(self):
        self.assertEqual(_parse_planner_json('Here you go:\n```json\n{"tool": "calculate", "args": {}}\n```\nDone!')["tool"], "calculate")

    def test_no_json_raises(self):
        with self.assertRaises(ValueError):
            _parse_planner_json("I will think about it")

    def test_non_object_raises(self):
        with self.assertRaises(ValueError):
            _parse_planner_json("[1, 2, 3]")


if __name__ == "__main__":
    unittest.main()
