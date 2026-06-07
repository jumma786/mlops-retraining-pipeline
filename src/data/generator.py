import numpy as np
import pandas as pd
import hashlib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42

def generate_batch(n_samples, version, random_state=RANDOM_STATE):
    np.random.seed(random_state + version)
    n = n_samples

    age = np.random.randint(18, 95, n)
    job = np.random.choice(["admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown"], n)
    marital = np.random.choice(["divorced","married","single","unknown"], n)
    education = np.random.choice(["basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown"], n)
    default = np.random.choice(["no","yes","unknown"], n, p=[0.79,0.01,0.20])
    housing = np.random.choice(["no","yes","unknown"], n, p=[0.45,0.50,0.05])
    loan = np.random.choice(["no","yes","unknown"], n, p=[0.82,0.15,0.03])

    if version == 1:
        contact = np.random.choice(["cellular","telephone"], n, p=[0.63,0.37])
    elif version == 2:
        contact = np.random.choice(["cellular","telephone"], n, p=[0.50,0.50])
    else:
        contact = np.random.choice(["cellular","telephone"], n, p=[0.30,0.70])

    month = np.random.choice(["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"], n)
    day_of_week = np.random.choice(["mon","tue","wed","thu","fri"], n)
    campaign = np.random.randint(1, 15, n)
    pdays = np.where(np.random.rand(n) < 0.13, np.random.randint(1,30,n), 999)
    previous = np.random.randint(0, 7, n)
    poutcome = np.random.choice(["failure","nonexistent","success"], n, p=[0.10,0.86,0.04])

    if version == 1:
        emp_var_rate = np.random.choice([-1.8,-1.7,1.1,1.4], n)
        euribor3m = np.random.uniform(3.0, 5.1, n).round(3)
        nr_employed = np.random.choice([5099.1,5176.3,5195.8,5228.1], n)
        positive_rate = 0.11
    elif version == 2:
        emp_var_rate = np.random.choice([-2.9,-3.0,-1.8,1.1], n)
        euribor3m = np.random.uniform(1.5, 4.0, n).round(3)
        nr_employed = np.random.choice([5008.7,5017.5,5099.1,5176.3], n)
        positive_rate = 0.13
    else:
        emp_var_rate = np.random.choice([-3.4,-3.0,-2.9], n)
        euribor3m = np.random.uniform(0.6, 2.0, n).round(3)
        nr_employed = np.random.choice([4963.6,5008.7,5017.5], n)
        positive_rate = 0.17

    cons_price_idx = np.random.uniform(92.2, 94.8, n).round(3)
    cons_conf_idx = np.random.uniform(-50.8, -26.9, n).round(1)
    y = (np.random.rand(n) < positive_rate).astype(int)

    df = pd.DataFrame({
        "age": age, "job": job, "marital": marital, "education": education,
        "default": default, "housing": housing, "loan": loan, "contact": contact,
        "month": month, "day_of_week": day_of_week, "campaign": campaign,
        "pdays": pdays, "previous": previous, "poutcome": poutcome,
        "emp.var.rate": emp_var_rate, "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx, "euribor3m": euribor3m,
        "nr.employed": nr_employed, "y": y,
        "_version": version,
        "_batch_date": f"2022-{'01' if version==1 else '04' if version==2 else '07'}-01"
    })

    logger.info(f"v{version}: {df.shape[0]:,} rows | positive rate: {df['y'].mean():.1%}")
    return df

def compute_md5(filepath):
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def save_batch(df, version, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"bank_marketing_v{version}.csv")
    df.to_csv(path, index=False)
    md5 = compute_md5(path)
    manifest = {"version": version, "path": path, "rows": len(df), "md5": md5, "positive_rate": round(df["y"].mean(), 4)}
    logger.info(f"Saved v{version} -> {path} (MD5: {md5[:8]}...)")
    return manifest

def generate_all_versions(base_dir="data", n_samples=5000):
    manifests = []
    for v in [1, 2, 3]:
        df = generate_batch(n_samples=n_samples, version=v)
        manifest = save_batch(df, version=v, output_dir=os.path.join(base_dir, f"v{v}"))
        manifests.append(manifest)
    manifest_df = pd.DataFrame(manifests)
    manifest_df.to_csv(os.path.join(base_dir, "manifest.csv"), index=False)
    logger.info("Manifest saved")
    return manifests

if __name__ == "__main__":
    generate_all_versions()