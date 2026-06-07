"""
Data generator for scheduled retraining pipeline.
Splits real UCI Bank Marketing data into 3 time-based versions by month.

v1 — Jan, Feb, Mar (Q1 baseline)
v2 — Apr, May, Jun (Q2 mild drift)
v3 — Jul, Aug, Sep (Q3 strong drift)

NOTE: Requires real UCI Bank Marketing CSV at data/bank-additional-full.csv
Falls back to synthetic data if real CSV not found.
"""

import numpy as np
import pandas as pd
import hashlib
import os
import logging
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MONTH_VERSIONS = {
    1: ["jan", "feb", "mar"],
    2: ["apr", "may", "jun"],
    3: ["jul", "aug", "sep"],
}

REAL_DATA_PATH = "data/bank-additional-full.csv"


def load_real_data(data_path: str) -> pd.DataFrame:
    """Load real UCI Bank Marketing CSV."""
    df = pd.read_csv(data_path, sep=";")
    logger.info(f"Loaded real data: {df.shape} rows")
    return df


def split_by_month(df: pd.DataFrame, version: int) -> pd.DataFrame:
    """Split dataframe by month for a given version."""
    months = MONTH_VERSIONS[version]
    batch = df[df["month"].isin(months)].copy()
    batch["_version"] = version
    batch["_batch_date"] = f"2022-{'01' if version==1 else '04' if version==2 else '07'}-01"
    logger.info(
        f"v{version} ({', '.join(months)}): {len(batch):,} rows | "
        f"positive rate: {(batch['y']=='yes').mean():.1%}"
    )
    return batch


def generate_synthetic_batch(n_samples: int, version: int, random_state: int = 42) -> pd.DataFrame:
    """Fallback synthetic generator if real data not available."""
    np.random.seed(random_state + version)
    n = n_samples

    if version == 1:
        emp_var_rate = np.random.choice([-1.8, -1.7, 1.1, 1.4], n)
        euribor3m = np.random.uniform(3.0, 5.1, n).round(3)
        nr_employed = np.random.choice([5099.1, 5176.3, 5195.8, 5228.1], n)
        positive_rate = 0.11
        contact = np.random.choice(["cellular", "telephone"], n, p=[0.63, 0.37])
    elif version == 2:
        emp_var_rate = np.random.choice([-2.9, -3.0, -1.8, 1.1], n)
        euribor3m = np.random.uniform(1.5, 4.0, n).round(3)
        nr_employed = np.random.choice([5008.7, 5017.5, 5099.1, 5176.3], n)
        positive_rate = 0.13
        contact = np.random.choice(["cellular", "telephone"], n, p=[0.50, 0.50])
    else:
        emp_var_rate = np.random.choice([-3.4, -3.0, -2.9], n)
        euribor3m = np.random.uniform(0.6, 2.0, n).round(3)
        nr_employed = np.random.choice([4963.6, 5008.7, 5017.5], n)
        positive_rate = 0.17
        contact = np.random.choice(["cellular", "telephone"], n, p=[0.30, 0.70])

    df = pd.DataFrame({
        "age":           np.random.randint(18, 95, n),
        "job":           np.random.choice(["admin.","blue-collar","management","retired","technician","unknown"], n),
        "marital":       np.random.choice(["divorced","married","single"], n),
        "education":     np.random.choice(["basic.4y","high.school","university.degree","unknown"], n),
        "default":       np.random.choice(["no","yes","unknown"], n, p=[0.79,0.01,0.20]),
        "housing":       np.random.choice(["no","yes","unknown"], n, p=[0.45,0.50,0.05]),
        "loan":          np.random.choice(["no","yes","unknown"], n, p=[0.82,0.15,0.03]),
        "contact":       contact,
        "month":         np.random.choice(MONTH_VERSIONS[version], n),
        "day_of_week":   np.random.choice(["mon","tue","wed","thu","fri"], n),
        "campaign":      np.random.randint(1, 15, n),
        "pdays":         np.where(np.random.rand(n) < 0.13, np.random.randint(1,30,n), 999),
        "previous":      np.random.randint(0, 7, n),
        "poutcome":      np.random.choice(["failure","nonexistent","success"], n, p=[0.10,0.86,0.04]),
        "emp.var.rate":  emp_var_rate,
        "cons.price.idx": np.random.uniform(92.2, 94.8, n).round(3),
        "cons.conf.idx": np.random.uniform(-50.8, -26.9, n).round(1),
        "euribor3m":     euribor3m,
        "nr.employed":   nr_employed,
        "y":             np.random.choice(["yes","no"], n, p=[positive_rate, 1-positive_rate]),
        "_version":      version,
        "_batch_date":   f"2022-{'01' if version==1 else '04' if version==2 else '07'}-01"
    })

    logger.info(f"v{version} (synthetic): {n:,} rows | positive rate: {positive_rate:.1%}")
    return df


def compute_md5(filepath: str) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def save_batch(df: pd.DataFrame, version: int, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"bank_marketing_v{version}.csv")
    df.to_csv(path, index=False)
    md5 = compute_md5(path)
    positive_rate = (df["y"] == "yes").mean() if df["y"].dtype == object else df["y"].mean()
    manifest = {
        "version": version,
        "path": path,
        "rows": len(df),
        "md5": md5,
        "positive_rate": round(float(positive_rate), 4),
        "data_type": "real" if os.path.exists(REAL_DATA_PATH) else "synthetic"
    }
    logger.info(f"Saved v{version} -> {path} (MD5: {md5[:8]}...)")
    return manifest


def generate_all_versions(base_dir: str = "data", n_samples: int = 5000):
    """Generate all 3 data versions — real if available, synthetic otherwise."""
    manifests = []
    use_real = os.path.exists(REAL_DATA_PATH)

    if use_real:
        logger.info(f"Using REAL data: {REAL_DATA_PATH}")
        df_full = load_real_data(REAL_DATA_PATH)
    else:
        logger.warning(f"Real data not found at {REAL_DATA_PATH} — using synthetic data")
        logger.warning("Copy real data: data/bank-additional-full.csv to use real batches")

    for v in [1, 2, 3]:
        if use_real:
            df = split_by_month(df_full, version=v)
            if len(df) == 0:
                logger.warning(f"No data for version {v} months — falling back to synthetic")
                df = generate_synthetic_batch(n_samples=n_samples, version=v)
        else:
            df = generate_synthetic_batch(n_samples=n_samples, version=v)

        manifest = save_batch(df, version=v, output_dir=os.path.join(base_dir, f"v{v}"))
        manifests.append(manifest)

    manifest_df = pd.DataFrame(manifests)
    manifest_path = os.path.join(base_dir, "manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    logger.info(f"\nManifest saved: {manifest_path}")
    logger.info(manifest_df.to_string(index=False))
    return manifests


if __name__ == "__main__":
    generate_all_versions()
