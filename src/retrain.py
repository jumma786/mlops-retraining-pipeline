import os
import sys
import time
import argparse
import logging
import json

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.preprocessor import load_and_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "mlops-retraining-pipeline"
REGISTRY_NAME   = "BankMarketingRetrained"
CHAMPION_MODEL  = "RandomForest"

def get_model(random_state=42):
    return RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        class_weight="balanced", random_state=random_state, n_jobs=-1
    )

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
    }

def get_champion_auc(experiment_name):
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return 0.0
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.promoted = 'true'",
            order_by=["metrics.roc_auc DESC"],
            max_results=1,
        )
        if runs:
            return runs[0].data.metrics.get("roc_auc", 0.0)
        return 0.0
    except Exception as e:
        logger.warning(f"Could not retrieve champion AUC: {e}")
        return 0.0

def save_audit_trail(info, output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "audit_trail.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(info) + "\n")
    logger.info(f"Audit trail updated: {path}")

def run_retraining(args):
    data_path = f"data/v{args.data_version}/bank_marketing_v{args.data_version}.csv"

    if not os.path.exists(data_path):
        logger.error(f"Data not found: {data_path}")
        logger.error("Run: python src/data/generator.py first")
        sys.exit(1)

    logger.info(f"Loading data version {args.data_version}...")
    X_train, X_test, y_train, y_test, features = load_and_split(data_path)

    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    prev_champion_auc = get_champion_auc(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"retrain-v{args.data_version}") as run:
        mlflow.log_param("data_version", args.data_version)
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("positive_rate", round(float(y_train.mean()), 4))
        mlflow.log_param("prev_champion_auc", prev_champion_auc)
        mlflow.set_tag("pipeline", "scheduled-retraining")

        model = get_model(random_state=args.random_state)
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = round(time.time() - t0, 2)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics["train_time_seconds"] = train_time

        mlflow.log_metrics(metrics)

        improvement = metrics["roc_auc"] - prev_champion_auc
        promoted = improvement >= args.min_improvement

        mlflow.log_metric("auc_improvement", round(improvement, 4))
        mlflow.set_tag("promoted", str(promoted).lower())

        if promoted:
            logger.info(f"AUC improved by {improvement:.4f} - PROMOTING")
            mlflow.sklearn.log_model(model, CHAMPION_MODEL)
            try:
                model_uri = f"runs:/{run.info.run_id}/{CHAMPION_MODEL}"
                reg = mlflow.register_model(model_uri, REGISTRY_NAME)
                logger.info(f"Registered: {REGISTRY_NAME} v{reg.version}")
            except Exception as e:
                logger.warning(f"Registry: {e}")
        else:
            logger.warning(f"Improvement {improvement:.4f} below threshold - NOT promoted")

        audit = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "data_version": args.data_version,
            "run_id": run.info.run_id,
            "model": CHAMPION_MODEL,
            "roc_auc": metrics["roc_auc"],
            "prev_champion_auc": prev_champion_auc,
            "improvement": round(improvement, 4),
            "promoted": promoted,
            "n_train": len(X_train),
        }
        save_audit_trail(audit)

        print("\n" + "="*55)
        print(f"RETRAINING SUMMARY - Data v{args.data_version}")
        print("="*55)
        print(f"  AUC:               {metrics['roc_auc']:.4f}")
        print(f"  Previous champion: {prev_champion_auc:.4f}")
        print(f"  Improvement:       {improvement:+.4f}")
        print(f"  Promoted:          {'YES' if promoted else 'NO'}")
        print("="*55)

        return metrics, promoted

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-version", type=int, default=1)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--min-improvement", type=float, default=0.01)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_retraining(args)