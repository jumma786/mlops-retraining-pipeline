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
3. **Compares to previous champion** — only promotes if AUC improves >= threshold
4. **Maintains full audit trail** — which data version trained which model, every run logged
5. **Answers the critical question** — "which data trained which model?" — at any point in time

---

## 📊 Dataset — Real UCI Bank Marketing Split by Quarter

3 real data batches split by month from UCI Bank Marketing (41,188 rows):

| Version | Period | Rows | Positive Rate | Data Type |
|---|---|---|---|---|
| v1 | Jan, Feb, Mar | 546 | 50.5% | Real — Q1 |
| v2 | Apr, May, Jun | 21,719 | 9.1% | Real — Q2 |
| v3 | Jul, Aug, Sep | 13,922 | 11.2% | Real — Q3 |

Each version has a unique MD5 hash for Change Data Capture (CDC).

> Note: v1 has high positive rate (50.5%) because Jan-Mar had a targeted campaign in the original dataset.

---

## 🏗️ Architecture

```
mlops-retraining-pipeline/
├── src/
│   ├── data/
│   │   ├── generator.py       # Real data split by month + synthetic fallback
│   │   └── preprocessor.py    # Encoding, split, feature pipeline
│   ├── retrain.py             # Core retraining logic — MLflow tracking + champion gate
│   └── pipeline.py            # Full pipeline runner across all versions
├── tests/
│   └── test_pipeline.py       # 11 unit tests
├── dvc.yaml                   # DVC pipeline stages
├── .github/
│   └── workflows/
│       └── retrain.yml        # CI: test -> generate -> retrain -> audit -> gate
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

# Add real data (required)
# Copy bank-additional-full.csv to data/ folder

# Run full retraining pipeline
python src/pipeline.py

# View results in MLflow UI
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000
```

---

## 📈 Results — Real Data Champion/Challenger Gating

| Version | AUC | Improvement | Promoted |
|---|---|---|---|
| v1 (Jan-Mar, 546 rows) | 0.6397 | +0.6397 | YES — first run |
| v2 (Apr-Jun, 21,719 rows) | 0.7719 | +0.1322 | YES — improved |
| v3 (Jul-Sep, 13,922 rows) | 0.7828 | +0.0109 | YES — improved |

**Key insight:** All 3 versions promoted — model improves with each new real data batch. Champion AUC grew from 0.6397 to 0.7828 across quarters.

---

## 🔍 Audit Trail

Every retraining run appends to `reports/audit_trail.jsonl`:

```json
{"timestamp": "2026-06-07T17:48:46", "data_version": 1,
 "run_id": "4775bb32...", "model": "RandomForest", "roc_auc": 0.6397,
 "prev_champion_auc": 0.5233, "improvement": 0.1164, "promoted": true,
 "n_train": 436, "positive_rate": 0.505}
```

Answers: which data trained which model, when, and why it was promoted.

---

## 🔄 CI/CD Pipeline

```
push to main / weekly cron (Monday 06:00 UTC)
    ↓
[Unit Tests] — 11 tests
    ↓
[Generate Data] — 3 real data batches with MD5 hashes
    ↓
[Retrain v1 -> v2 -> v3] — champion gate applied
    ↓
[Audit Trail] — saved as artifact (90 days)
    ↓
[AUC Gate] — fail if best AUC < 0.50
```

---

## 🔗 MLOps Portfolio Series

| # | Project | Repo | Status |
|---|---|---|---|
| 1 | Multi-Model Tournament | [mlops-model-tournament](https://github.com/jumma786/mlops-model-tournament) | ✅ |
| **2** | **Scheduled Retraining** | [mlops-retraining-pipeline](https://github.com/jumma786/mlops-retraining-pipeline) | ✅ This repo |
| 3 | Feature Engineering | [mlops-feature-pipeline](https://github.com/jumma786/mlops-feature-pipeline) | ✅ |
| 4 | Hyperparameter Tuning | [mlops-hyperparameter-tuning](https://github.com/jumma786/mlops-hyperparameter-tuning) | ✅ |
| 5 | Model Serving | [mlops-model-serving](https://github.com/jumma786/mlops-model-serving) | ✅ |
| 6 | Feature Store | [mlops-feature-store](https://github.com/jumma786/mlops-feature-store) | ✅ |
| 7 | Model Monitoring | [mlops-model-monitoring](https://github.com/jumma786/mlops-model-monitoring) | ✅ |
| 8 | A/B Testing | [mlops-ab-testing](https://github.com/jumma786/mlops-ab-testing) | ✅ |
| 9 | Airflow Pipeline | [mlops-airflow-pipeline](https://github.com/jumma786/mlops-airflow-pipeline) | ✅ |
| 10 | Kubernetes Platform | [mlops-k8s-platform](https://github.com/jumma786/mlops-k8s-platform) | ✅ |

---

## 📝 Key MLOps Concepts Demonstrated

- **Data versioning** — DVC tracks every dataset with MD5 hashes
- **Audit trail** — complete lineage: data -> model -> decision
- **Champion/challenger** — automated promotion gate (AUC improvement threshold)
- **Real data drift** — 3 quarterly batches from UCI Bank Marketing
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
