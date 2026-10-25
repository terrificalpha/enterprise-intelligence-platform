"""Executes the whole Streamlit script body against a mocked `streamlit` + the in-process backend.
Catches NameErrors/KeyErrors in any tab without needing a browser."""
import os
import runpy
import sys
import unittest
from pathlib import Path

from tests import _fakes
from tests.helpers import ROOT, platform


class TestUI(unittest.TestCase):
    def test_streamlit_script_runs_end_to_end(self):
        platform()
        os.environ["EIP_UI_MODE"] = "local"
        real = sys.modules.get("streamlit")
        st = _fakes.install_fake_streamlit(force=True)
        try:
            runpy.run_path(str(ROOT / "ui" / "streamlit_app.py"), run_name="__main__")
            self.assertTrue(st.tabs.called)
            self.assertFalse(st.error.called, st.error.call_args)
        finally:
            if real is not None:
                sys.modules["streamlit"] = real
            else:
                sys.modules.pop("streamlit", None)
            sys.modules.pop("ui.client", None)

    def test_client_fallback_to_local_when_api_down(self):
        os.environ["EIP_UI_MODE"] = "api"
        os.environ["EIP_API_URL"] = "http://127.0.0.1:9"
        from ui import client
        platform()
        self.assertEqual(client.get_client().mode, "local")


if __name__ == "__main__":
    unittest.main()
