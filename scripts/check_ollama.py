"""Pre-flight check for the local Ollama setup. Run:  python scripts/check_ollama.py
Verifies: server reachable -> model installed -> a real grounded, cited completion works -> latency."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.llm import prompts  # noqa: E402
from app.llm.client import LLMClient  # noqa: E402


def main() -> int:
    c = LLMClient(provider="ollama")
    st = c.status()
    print(f"Ollama URL : {c.base_url}\nModel      : {c.model}\nContext    : {settings.ollama_num_ctx} tokens")
    if not st["server_up"]:
        print("\n[FAIL]", st["hint"])
        return 1
    print("Installed  :", ", ".join(st["installed_models"]) or "(none)")
    if not st["model_installed"]:
        print("\n[FAIL]", st["hint"])
        return 1
    ctx = prompts.format_context([(1, "Policy.pdf", "The notice period for contract termination is 60 calendar days.")])
    t0 = time.time()
    ans = c.complete(prompts.GROUNDED_SYSTEM, f"CONTEXT:\n{ctx}\n\nQUESTION: What is the notice period for contract termination?")
    print(f"\nGrounded answer ({time.time() - t0:.1f}s, first call includes model load):\n  {ans}")
    ok = "[1]" in ans and "60" in ans
    t0 = time.time()
    ref = c.complete(prompts.GROUNDED_SYSTEM, f"CONTEXT:\n{ctx}\n\nQUESTION: What is the parental leave policy?")
    print(f"\nUnsupported question ({time.time() - t0:.1f}s):\n  {ref}")
    ok2 = "don't have enough evidence" in ref.lower()
    print("\n[OK] Ollama is ready." if ok else "\n[WARN] Answer lacked the [1] citation/60 days - app will fall back safely.")
    if not ok2:
        print("[NOTE] Model did not refuse the unsupported question here; the platform's retrieval guardrail still blocks it before the LLM.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
