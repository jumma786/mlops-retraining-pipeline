import os
import sys
import argparse
import logging
import types

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.generator import generate_all_versions
from src.retrain import run_retraining

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_pipeline(min_improvement=0.01, n_samples=5000):
    print("\n" + "="*60)
    print("MLOps RETRAINING PIPELINE - Full Run")
    print("="*60)

    logger.info("Step 1/2: Generating versioned datasets...")
    manifests = generate_all_versions(base_dir="data", n_samples=n_samples)

    print("\n--- Data Versions Generated ---")
    for m in manifests:
        print(f"  v{m['version']}: {m['rows']:,} rows | positive rate: {m['positive_rate']:.1%} | MD5: {m['md5'][:8]}...")

    logger.info("Step 2/2: Running retraining across all versions...")
    results = []

    for version in [1, 2, 3]:
        print(f"\n{'─'*60}")
        print(f"Retraining on v{version}...")
        print(f"{'─'*60}")

        args = types.SimpleNamespace(
            data_version=version,
            random_state=42,
            min_improvement=min_improvement
        )

        metrics, promoted = run_retraining(args)
        results.append({
            "version": version,
            "auc": metrics["roc_auc"],
            "f1": metrics["f1"],
            "promoted": promoted
        })

    print("\n" + "="*60)
    print("PIPELINE COMPLETE - VERSION COMPARISON")
    print("="*60)
    print(f"{'Version':<10} {'AUC':>8} {'F1':>8} {'Promoted':>10}")
    print("-"*60)
    for r in results:
        promoted_str = "YES" if r["promoted"] else "NO"
        print(f"v{r['version']:<9} {r['auc']:>8.4f} {r['f1']:>8.4f} {promoted_str:>10}")
    print("="*60)
    print("\nMLflow UI: mlflow ui --backend-store-uri mlruns")
    print("Audit trail: reports/audit_trail.jsonl")
    return results

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--min-improvement", type=float, default=0.01)
    p.add_argument("--n-samples", type=int, default=5000)
    args = p.parse_args()
    run_pipeline(min_improvement=args.min_improvement, n_samples=args.n_samples)