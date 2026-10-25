"""API tests. With real FastAPI installed they go through fastapi.testclient (full HTTP stack, validation, auth).
Without it, minimal stand-ins let the real route handlers still run (see tests/_fakes.py)."""
import importlib
import sys
import unittest

from tests import _fakes
from tests.helpers import platform

_fakes.install_fake_fastapi()
try:
    from fastapi.testclient import TestClient          # real FastAPI present
    HAVE_REAL = True
except Exception:
    HAVE_REAL = False


def load_app():
    sys.modules.pop("app.api.main", None)
    mod = importlib.import_module("app.api.main")
    mod._platform = platform()                          # reuse the shared, already-built platform
    return mod


@unittest.skipUnless(HAVE_REAL, "real FastAPI not installed - covered by TestHandlers below")
class TestHttp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_app()
        cls.c = TestClient(cls.mod.app)

    def test_health(self):
        r = self.c.get("/health"); self.assertEqual(r.status_code, 200); self.assertEqual(r.json()["status"], "ok")

    def test_ask_with_citations(self):
        r = self.c.post("/ask", json={"question": "What is the notice period for non-renewal in the MSA?"})
        self.assertEqual(r.status_code, 200); self.assertTrue(r.json()["citations"])

    def test_validation_errors(self):
        self.assertEqual(self.c.post("/ask", json={"question": ""}).status_code, 422)
        self.assertEqual(self.c.post("/predict", json={"customer_id": "bad"}).status_code, 422)
        self.assertEqual(self.c.post("/search", json={"query": "x", "mode": "nope"}).status_code, 422)

    def test_auth_when_key_configured(self):
        from app.config import settings
        settings.api_key = "k"
        try:
            self.assertEqual(self.c.post("/ask", json={"question": "late fee?"}).status_code, 401)
            self.assertEqual(self.c.post("/ask", json={"question": "late fee?"}, headers={"X-API-Key": "k"}).status_code, 200)
        finally:
            settings.api_key = ""

    def test_upload_rejects_bad_type(self):
        r = self.c.post("/documents/upload", files={"file": ("evil.exe", b"MZ")}); self.assertEqual(r.status_code, 400)

    def test_report_download_blocks_traversal(self):
        self.assertEqual(self.c.get("/reports/..%2F..%2Fetc%2Fpasswd").status_code, 404)


class TestHandlers(unittest.TestCase):
    """Calls the route functions directly (works with real or stand-in FastAPI)."""
    @classmethod
    def setUpClass(cls):
        cls.mod = load_app()
        cls.P = platform()

    def test_all_expected_routes_exist(self):
        paths = {r.path for r in self.mod.app.routes} if hasattr(self.mod.app.routes, "__iter__") and not isinstance(self.mod.app.routes, dict) \
            else {p for _, p in self.mod.app.routes}
        for p in ("/health", "/search", "/ask", "/predict", "/agent", "/portfolio", "/model/metrics", "/reports",
                  "/documents", "/documents/upload", "/customers/{customer_id}/briefing"):
            self.assertIn(p, paths)

    def test_predict_and_ask_handlers(self):
        from app.api.schemas import AskRequest, PredictRequest
        cid = self.P.tools.pred.portfolio.customer_id.iloc[0]
        r = self.mod.predict(PredictRequest(customer_id=cid))
        self.assertTrue(0 <= r["churn_probability"] <= 1 and r["drivers"])
        a = self.mod.ask(AskRequest(question="Which competitor bundles free support?"))
        self.assertIn("OrbitDesk", a["answer"])

    def test_schema_rejects_bad_ids(self):
        from app.api.schemas import PredictRequest
        with self.assertRaises(Exception):
            PredictRequest(customer_id="1; DROP TABLE")


if __name__ == "__main__":
    unittest.main()
