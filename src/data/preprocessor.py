import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging
import os

logger = logging.getLogger(__name__)

DROP_COLS = ["_version", "_batch_date", "duration"]
TARGET = "y"

def preprocess(df):
    df = df.copy()
    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])
    if df[TARGET].dtype == object:
        df[TARGET] = (df[TARGET] == "yes").astype(int)
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def load_and_split(data_path, test_size=0.2, random_state=42):
    df = pd.read_csv(data_path)
    df = preprocess(df)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    logger.info(f"Positive rate: {y_train.mean():.1%}")
    return X_train, X_test, y_train, y_test, list(X.columns)