import functools
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["EIP_LLM_PROVIDER"] = "offline"                # tests must never call a real LLM
os.environ.setdefault("EIP_REPORTS_DIR", tempfile.mkdtemp(prefix="eip_reports_"))   # tests must not litter ./reports


@functools.lru_cache(maxsize=1)
def platform():
    """One shared Platform for the whole test session (index build is the slow part)."""
    ensure_artifacts()
    from app.services import Platform
    return Platform()


def ensure_artifacts():
    from app.config import settings
    if not settings.customers_csv.exists():
        from app.data_gen import generate_customers
        generate_customers.main()
    if not (settings.documents_dir / "Customer_Retention_Policy.pdf").exists():
        from app.data_gen import generate_documents
        generate_documents.main()
    if not (settings.artifacts_dir / "best_model.joblib").exists():
        from app.ml.train import train
        train(n_iter=4, verbose=False)
