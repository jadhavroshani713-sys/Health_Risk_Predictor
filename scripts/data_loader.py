"""
Healthcare Risk Predictor - Data Ingestion and Cleaning Module
Author: Roshani Jadhav
Project: AICTE | IBM SkillsBuild Data Analytics with AI Academic Internship
Organization: In association with BharatCares
"""

import os
import urllib.request
import pandas as pd
import numpy as np

DATA_URL = "https://raw.githubusercontent.com/doguilmak/Heart-Diseaseor-Attack-Classification/main/heart_disease_health_indicators_BRFSS2015.csv"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DATA_FILE = os.path.join(DATA_DIR, "heart_disease_health_indicators_BRFSS2015.csv")


def ensure_dataset(data_path: str = DATA_FILE, download_url: str = DATA_URL) -> str:
    """Ensure that the BRFSS 2015 dataset exists locally; download if absent."""
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    if not os.path.exists(data_path):
        print(f"[*] Dataset not found locally at: {data_path}")
        print(f"[*] Downloading BRFSS 2015 dataset from source: {download_url}...")
        try:
            urllib.request.urlretrieve(download_url, data_path)
            print(f"[+] Download complete. Saved to: {data_path}")
        except Exception as e:
            raise RuntimeError(f"[-] Failed to download dataset: {e}")
    else:
        print(f"[+] Dataset confirmed present at: {data_path}")
    return data_path


def load_and_clean_data(data_path: str = DATA_FILE) -> tuple[pd.DataFrame, dict]:
    """
    Load BRFSS 2015 dataset, perform quality audits (duplicates, nulls, types),
    clean the data, and generate summary descriptive statistics.

    Returns:
        tuple: (cleaned_dataframe, audit_summary_dict)
    """
    ensure_dataset(data_path)
    print(f"[*] Loading raw dataset from {data_path}...")
    df = pd.read_csv(data_path)
    raw_shape = df.shape
    print(f"[+] Raw dataset shape: {raw_shape[0]:,} rows and {raw_shape[1]} columns.")

    # 1. Missing Value Audit & Imputation
    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())
    imputed_columns = []

    if total_nulls > 0:
        print(f"[!] Warning: Found {total_nulls} missing values across features. Imputing...")
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                imputed_columns.append(col)
                if df[col].nunique() <= 5:
                    # Categorical / binary - impute with mode
                    mode_val = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_val)
                else:
                    # Continuous / ordinal - impute with median
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
    else:
        print("[+] Missing Value Audit: 0 missing values detected. Dataset is clean.")

    # 2. Duplicate Records Audit & Removal
    duplicate_count = int(df.duplicated().sum())
    print(f"[*] Duplicate Records Audit: Found {duplicate_count:,} duplicate rows.")
    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"[+] Removed duplicate rows. Updated dataset shape: {df.shape[0]:,} rows.")

    # 3. Data Type Verification & Standardization
    # Convert binary and discrete integer categories from float64 to int64 for computational efficiency
    for col in df.columns:
        if (df[col] % 1 == 0).all():
            df[col] = df[col].astype(int)

    # 4. Target Variable Distribution Audit
    target_col = "HeartDiseaseorAttack"
    if target_col in df.columns:
        target_counts = df[target_col].value_counts().to_dict()
        prevalence = (df[target_col].mean()) * 100
        print(f"[+] Target '{target_col}' Distribution: {target_counts}")
        print(f"[+] Heart Disease Prevalence: {prevalence:.2f}%")
    else:
        target_counts = {}
        prevalence = 0.0

    # 5. Descriptive Summary Statistics
    summary_stats = df.describe().T
    summary_stats["median"] = df.median()
    summary_stats["skewness"] = df.skew()

    audit_summary = {
        "raw_shape": raw_shape,
        "clean_shape": df.shape,
        "duplicate_count": duplicate_count,
        "total_nulls": total_nulls,
        "imputed_columns": imputed_columns,
        "prevalence_percent": prevalence,
        "features": list(df.columns),
        "target_distribution": target_counts
    }

    return df, audit_summary


if __name__ == "__main__":
    print("=" * 70)
    print("HEALTHCARE RISK PREDICTOR: DATA INGESTION & AUDIT")
    print("Student: Roshani Jadhav | AICTE - IBM SkillsBuild & BharatCares")
    print("=" * 70)
    df_clean, audit = load_and_clean_data()
    print("\n--- Summary Audit Report ---")
    for key, val in audit.items():
        if key != "features":
            print(f"  - {key}: {val}")
    print("\nFirst 5 records:")
    print(df_clean.head())
