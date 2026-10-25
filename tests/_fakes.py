"""Minimal stand-ins for `fastapi` and `streamlit`, installed ONLY when the real packages are missing
(e.g. offline CI). They let the tests execute the real route handlers / the full Streamlit script body.
With the real packages installed, tests use fastapi.testclient.TestClient instead and these are unused."""
from __future__ import annotations

import sys
import types
from unittest import mock


def install_fake_fastapi() -> None:
    try:
        import fastapi  # noqa: F401
        return
    except ImportError:
        pass
    m = types.ModuleType("fastapi")

    class HTTPException(Exception):
        def __init__(self, status_code, detail=None):
            self.status_code, self.detail = status_code, detail

    class FastAPI:
        def __init__(self, **kw):
            self.routes = {}

        def _reg(self, method):
            def deco(path, **kw):
                def wrap(fn):
                    self.routes[(method, path)] = fn
                    return fn
                return wrap
            return deco

        def __getattr__(self, name):
            if name in ("get", "post", "put", "delete"):
                return self._reg(name.upper())
            raise AttributeError(name)

    m.FastAPI, m.HTTPException = FastAPI, HTTPException
    m.Depends = lambda f=None: f
    m.File = lambda *a, **k: None
    m.Header = lambda *a, **k: None
    m.Request = object
    m.UploadFile = object
    resp = types.ModuleType("fastapi.responses")
    resp.FileResponse = lambda p, **k: ("file", str(p))
    sys.modules["fastapi"], sys.modules["fastapi.responses"] = m, resp


def install_fake_streamlit(force: bool = False) -> mock.MagicMock:
    if not force:
        try:
            import streamlit  # noqa: F401
            return None
        except ImportError:
            pass
    st = mock.MagicMock(name="streamlit")

    class State(dict):
        __getattr__ = dict.get

        def __setattr__(self, k, v):
            self[k] = v

    st.session_state = State()
    st.cache_resource = lambda *a, **k: (lambda f: f) if not (a and callable(a[0])) else a[0]
    st.tabs.side_effect = lambda labels: [st for _ in labels]          # containers behave like st itself
    st.columns.side_effect = lambda spec, **k: [st for _ in range(spec if isinstance(spec, int) else len(spec))]
    st.button.return_value = True
    st.file_uploader.return_value = []
    st.chat_input.return_value = None
    st.text_input.side_effect = lambda label, value="", **k: value
    st.selectbox.side_effect = lambda label, options, **k: options[0] if options else ""
    st.radio.side_effect = lambda label, options, **k: options[k.get("index", 0)]
    sys.modules["streamlit"] = st
    return st
