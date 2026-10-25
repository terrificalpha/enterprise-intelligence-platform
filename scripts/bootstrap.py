"""One-shot, idempotent setup: generate data -> train models -> build search index -> (optionally) run evaluation.
Usage:  python scripts/bootstrap.py [--force] [--skip-eval]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="regenerate everything")
    ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args()
    settings.ensure_dirs()
    if a.force or not settings.customers_csv.exists():
        print("== generating customers"); from app.data_gen import generate_customers; generate_customers.main()
    if a.force or not (settings.documents_dir / "Customer_Retention_Policy.pdf").exists():
        print("== generating documents"); from app.data_gen import generate_documents; generate_documents.main()
    if a.force or not (settings.artifacts_dir / "best_model.joblib").exists():
        print("== training models"); from app.ml.train import train; train()
    print("== building knowledge-base index")
    from app.rag.knowledge_base import KnowledgeBase
    print(f"   {KnowledgeBase().build()} chunks indexed")
    if not a.skip_eval:
        print("== evaluating"); from app.evaluation import run_all; run_all.main()
    print("Done. Start the stack with:  make api   and   make ui")


if __name__ == "__main__":
    main()
