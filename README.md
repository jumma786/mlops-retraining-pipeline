\# 🔄 Scheduled Retraining Pipeline — DVC + MLflow



!\[Python](https://img.shields.io/badge/Python-3.11%2B-blue)

!\[MLflow](https://img.shields.io/badge/MLflow-3.13-orange)

!\[License](https://img.shields.io/badge/License-MIT-lightgrey)



> \*\*Part of the MLOps Portfolio Series\*\* — Project 2 of 10



\## 🎯 What This Project Does



1\. Versions datasets with DVC — every batch has a unique MD5 hash

2\. Retrains champion model automatically on new data

3\. Compares to previous champion — only promotes if AUC improves

4\. Maintains full audit trail — which data trained which model



\## 📊 Data Versions — Drift Simulation



| Version | Rows | Positive Rate | Drift |

|---|---|---|---|

| v1 | 5,000 | 11.4% | Baseline |

| v2 | 5,000 | 13.0% | Mild drift |

| v3 | 5,000 | 17.3% | Strong drift |



\## 📈 Results



| Version | AUC | Promoted |

|---|---|---|

| v1 | 0.5233 | YES |

| v2 | 0.5159 | NO |

| v3 | 0.4979 | NO |



\## 🚀 Quick Start



```bash

pip install -r requirements.txt

python src/data/generator.py

python src/pipeline.py

mlflow ui --backend-store-uri mlruns

```



\## 🔗 MLOps Portfolio Series



| # | Project | Status |

|---|---|---|

| 1 | \[mlops-model-tournament](https://github.com/jumma786/mlops-model-tournament) | ✅ |

| \*\*2\*\* | \*\*mlops-retraining-pipeline\*\* | ✅ This repo |

| 3–10 | Coming soon | 🔜 |



\## 👤 Author



\*\*Jumma Mohammad Teli\*\* — Data Analyst \& ML Engineer | Birmingham, UK

🔗 \[LinkedIn](https://linkedin.com/in/jumma-mohammad) | \[GitHub](https://github.com/jumma786)

