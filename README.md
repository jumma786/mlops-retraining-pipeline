# 🔄 Scheduled Retraining Pipeline — DVC + MLflow

![CI](https://github.com/jumma786/mlops-retraining-pipeline/actions/workflows/retrain.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![MLflow](https://img.shields.io/badge/MLflow-3.13-orange)
![DVC](https://img.shields.io/badge/DVC-3.50-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Part of the MLOps Portfolio Series** — Project 2 of 10  
> Automated scheduled retraining pipeline with data versioning (DVC), experiment tracking (MLflow), champion/challenger gating, and full audit trail.

---

## 📂 Project Resources

| Resource | Link |
|---|---|
| 📋 Pipeline Runner | [src/pipeline.py](src/pipeline.py) |
| 🔄 Retraining Script | [src/retrain.py](src/retrain.py) |
| 📦 Data Generator | [src/data/generator.py](src/data/generator.py) |
| ⚙️ DVC Pipeline | [dvc.yaml](dvc.yaml) |
| 🧪 Unit Tests | [tests/test_pipeline.py](tests/test_pipeline.py) |
| 🤖 CI/CD Workflow | [.github/workflows/retrain.yml](.github/workflows/retrain.yml) |
| 📋 Requirements | [requirements.txt](requirements.txt) |

---

## 🎯 What This Project Does

Solves the "stale model" problem — models decay as data distributions shift over time:

1. **Versions datasets with DVC** — every data batch has a unique MD5 hash, tracked in Git
2. **Retrains on new data automatically** — weekly cron via GitHub Actions
3. **Compares to previous champion** — only promotes if AUC improves ≥ threshold
4. **Maintains full audit trail** — which data version trained which model, every run logged
5. **Answers the critical question** — "which data trained which model?" — at any point in time

---

## 📊 Dataset — Versioned with Drift Simulation

3 versioned batches simulating progressive data drift:

| Version | Period | Rows | Positive Rate | euribor3m | Drift Type |
|---|---|---|---|---|---|
| v1 (Jan–Mar) | 0.6397 | +0.6397 | ✅ YES |
| v2 (Apr–Jun) | 0.7719 | +0.1322 | ✅ YES |
| v3 (Jul–Sep) | 0.7828 | +0.0109 | ✅ YES |

Each version has a unique MD5 hash for Change Data Capture (CDC).

---

## 🏗️ Architecture

```
mlops-retraining-pipeline/
├── src/
│   ├── data/
│   │   ├── generator.py       # Versioned data generation with drift simulation
│   │   └── preprocessor.py    # Encoding, split, feature pipeline
│   ├── retrain.py             # Core retraining logic — MLflow tracking + champion gate
│   └── pipeline.py            # Full pipeline runner across all versions
├── tests/
│   └── test_pipeline.py       # 11 unit tests
├── dvc.yaml                   # DVC pipeline stages
├── .github/
│   └── workflows/
│       └── retrain.yml        # CI: test → generate → retrain → audit → gate
├── requirements.txt
└── Makefile
```

---

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/jumma786/mlops-retraining-pipeline.git
cd mlops-retraining-pipeline
pip install -r requirements.txt

# Run tests
make test

# Generate versioned datasets
make generate

# Run full retraining pipeline
make pipeline

# View results in MLflow UI
make mlflow-ui
# → Open http://localhost:5000
```

---

## 📈 Results — Champion/Challenger Gating

| Version | AUC | Improvement | Promoted |
|---|---|---|---|
| v1 (baseline) | 0.5233 | +0.5233 | ✅ YES — first run |
| v2 (mild drift) | 0.5159 | -0.0074 | ❌ NO — performance dropped |
| v3 (strong drift) | 0.4979 | -0.0254 | ❌ NO — below threshold |

**Key insight:** The pipeline correctly detected that retraining on drifted data (v2, v3) degraded performance — and blocked promotion automatically. This is the core value of champion/challenger gating.

---

## 🔍 Audit Trail

Every retraining run appends to `reports/audit_trail.jsonl`:

```json
{"timestamp": "2026-06-07T14:49:30", "data_version": 1, "data_path": "data/v1/...",
 "run_id": "650174e0...", "model": "RandomForest", "roc_auc": 0.5233,
 "prev_champion_auc": 0.0, "improvement": 0.5233, "promoted": true,
 "n_train": 4000, "positive_rate": 0.114}
```

Answers: *which data trained which model, when, and why it was (or wasn't) promoted.*

---

## 🔄 CI/CD Pipeline

```
push to main / weekly cron (Monday 06:00 UTC)
    ↓
[Unit Tests] — 11 tests
    ↓
[Generate Data] — 3 versioned batches with MD5 hashes
    ↓
[Retrain v1 → v2 → v3] — champion gate applied
    ↓
[Audit Trail] — saved as artifact (90 days)
    ↓
[AUC Gate] — fail if best AUC < 0.50
```

---

## 🔗 MLOps Portfolio Series

| # | Project | Repo | Status |
|---|---|---|---|
| 1 | Multi-Model Tournament Pipeline | [mlops-model-tournament](https://github.com/jumma786/mlops-model-tournament) | ✅ |
| **2** | **Scheduled Retraining + DVC + MLflow** | [mlops-retraining-pipeline](https://github.com/jumma786/mlops-retraining-pipeline) | ✅ This repo |
| 3 | Feature Engineering as Versioned Artifact | mlops-feature-pipeline | 🔜 |
| 4 | Hyperparameter Tuning with Optuna + MLflow | mlops-hyperparameter-tuning | 🔜 |
| 5 | FastAPI + Docker + Cloud Run | mlops-model-serving | 🔜 |
| 6 | Feature Store with Feast + Redis | mlops-feature-store | 🔜 |
| 7 | Model Monitoring & Drift Detection | mlops-model-monitoring | 🔜 |
| 8 | A/B Testing Framework | mlops-ab-testing | 🔜 |
| 9 | Airflow Pipeline Orchestration | mlops-airflow-pipeline | 🔜 |
| 10 | Kubernetes ML Platform | mlops-k8s-platform | 🔜 |

---

## 📝 Key MLOps Concepts Demonstrated

- **Data versioning** — DVC tracks every dataset with MD5 hashes
- **Audit trail** — complete lineage: data → model → decision
- **Champion/challenger** — automated promotion gate (AUC improvement threshold)
- **Drift simulation** — 3 data versions with progressive distribution shift
- **Scheduled retraining** — weekly GitHub Actions cron
- **Change Data Capture** — MD5 hashing detects data changes

---

## 👤 Author

**Jumma Mohammad Teli** — Data Analyst & ML Engineer  
📍 Birmingham, UK  
📧 [jummamohammad477@gmail.com](mailto:jummamohammad477@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/jumma-mohammad) | [GitHub](https://github.com/jumma786)

---

*Project 2 of 10 — MLOps Portfolio Series. Builds on Project 1 (Model Tournament) by adding data versioning, scheduled retraining, and automated champion gating.*
