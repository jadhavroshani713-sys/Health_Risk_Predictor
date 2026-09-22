"""
Healthcare Risk Predictor - Model Training and Evaluation Pipeline
Author: Roshani Jadhav
Project: AICTE | IBM SkillsBuild Data Analytics with AI Academic Internship
Organization: In association with BharatCares
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from data_loader import load_and_clean_data

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def build_and_train_models(save_artifacts: bool = True) -> dict:
    """
    Executes full ML training pipeline:
    - Data ingestion & cleaning
    - Stratified 80/20 train/validation split
    - StandardScaler normalization
    - Logistic Regression & Random Forest model training with balanced weighting
    - Multi-metric evaluation and artifact serialization
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    df, audit = load_and_clean_data()

    target_col = "HeartDiseaseorAttack"
    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = list(X.columns)

    print(f"[*] Splitting dataset (80% Train, 20% Validation) with stratification...", flush=True)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[+] Train set: {X_train.shape[0]:,} rows | Val set: {X_val.shape[0]:,} rows", flush=True)

    # Standardize features
    print("[*] Fitting StandardScaler on training set...", flush=True)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 1. Train Logistic Regression
    print("\n[*] Training Model 1: Logistic Regression (balanced class weights)...", flush=True)
    log_reg = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    log_reg.fit(X_train_scaled, y_train)

    y_pred_lr = log_reg.predict(X_val_scaled)
    y_prob_lr = log_reg.predict_proba(X_val_scaled)[:, 1]

    metrics_lr = {
        "model_name": "Logistic Regression (Class-Weighted)",
        "accuracy": float(accuracy_score(y_val, y_pred_lr)),
        "precision": float(precision_score(y_val, y_pred_lr)),
        "recall": float(recall_score(y_val, y_pred_lr)),
        "f1_score": float(f1_score(y_val, y_pred_lr)),
        "roc_auc": float(roc_auc_score(y_val, y_prob_lr)),
        "confusion_matrix": confusion_matrix(y_val, y_pred_lr).tolist(),
        "classification_report": classification_report(y_val, y_pred_lr, output_dict=True)
    }

    # 2. Train Random Forest Classifier
    print("[*] Training Model 2: Random Forest Classifier (balanced class weights)...", flush=True)
    rf_clf = RandomForestClassifier(
        n_estimators=80,
        max_depth=12,
        max_samples=0.6,
        class_weight="balanced",
        random_state=42,
        n_jobs=2
    )
    rf_clf.fit(X_train_scaled, y_train)

    y_pred_rf = rf_clf.predict(X_val_scaled)
    y_prob_rf = rf_clf.predict_proba(X_val_scaled)[:, 1]

    metrics_rf = {
        "model_name": "Random Forest Classifier (Balanced)",
        "accuracy": float(accuracy_score(y_val, y_pred_rf)),
        "precision": float(precision_score(y_val, y_pred_rf)),
        "recall": float(recall_score(y_val, y_pred_rf)),
        "f1_score": float(f1_score(y_val, y_pred_rf)),
        "roc_auc": float(roc_auc_score(y_val, y_prob_rf)),
        "confusion_matrix": confusion_matrix(y_val, y_pred_rf).tolist(),
        "classification_report": classification_report(y_val, y_pred_rf, output_dict=True)
    }

    print("\n" + "=" * 65, flush=True)
    print("MODEL EVALUATION BENCHMARK SUMMARY", flush=True)
    print("=" * 65, flush=True)
    for res in [metrics_lr, metrics_rf]:
        print(f"Model: {res['model_name']}", flush=True)
        print(f"  - Accuracy:  {res['accuracy'] * 100:.2f}%", flush=True)
        print(f"  - Precision: {res['precision'] * 100:.2f}%", flush=True)
        print(f"  - Recall:    {res['recall'] * 100:.2f}%", flush=True)
        print(f"  - F1-Score:  {res['f1_score'] * 100:.2f}%", flush=True)
        print(f"  - ROC-AUC:   {res['roc_auc']:.4f}", flush=True)
        print("-" * 65, flush=True)

    # Feature Importance / Coefficients Analysis
    rf_importance = dict(zip(feature_names, [float(x) for x in rf_clf.feature_importances_]))
    sorted_rf_importance = dict(sorted(rf_importance.items(), key=lambda item: item[1], reverse=True))

    lr_coefficients = dict(zip(feature_names, [float(x) for x in log_reg.coef_[0]]))
    sorted_lr_coefficients = dict(sorted(lr_coefficients.items(), key=lambda item: abs(item[1]), reverse=True))

    # Determine primary model based on ROC-AUC
    primary_model = rf_clf if metrics_rf["roc_auc"] >= metrics_lr["roc_auc"] else log_reg
    primary_name = "Random Forest Classifier" if primary_model == rf_clf else "Logistic Regression"

    if save_artifacts:
        scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
        rf_path = os.path.join(MODEL_DIR, "heart_risk_model.joblib")
        lr_path = os.path.join(MODEL_DIR, "logistic_regression_model.joblib")
        meta_path = os.path.join(MODEL_DIR, "model_metadata.json")

        joblib.dump(scaler, scaler_path)
        joblib.dump(rf_clf, rf_path)
        joblib.dump(log_reg, lr_path)

        metadata = {
            "project_name": "Healthcare Risk Predictor",
            "student_name": "Roshani Jadhav",
            "internship": "AICTE | IBM SkillsBuild Data Analytics with AI Academic Internship",
            "partner": "BharatCares",
            "primary_model": primary_name,
            "feature_names": feature_names,
            "feature_importances_rf": sorted_rf_importance,
            "coefficients_lr": sorted_lr_coefficients,
            "metrics_rf": metrics_rf,
            "metrics_lr": metrics_lr,
            "train_sample_count": int(X_train.shape[0]),
            "val_sample_count": int(X_val.shape[0]),
            "total_samples": int(df.shape[0])
        }

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"[+] Scaler saved to: {scaler_path}", flush=True)
        print(f"[+] Model saved to:  {rf_path}", flush=True)
        print(f"[+] Metadata saved to: {meta_path}", flush=True)

    return {
        "metrics_lr": metrics_lr,
        "metrics_rf": metrics_rf,
        "feature_names": feature_names,
        "rf_importance": sorted_rf_importance,
        "lr_coefficients": sorted_lr_coefficients
    }


if __name__ == "__main__":
    build_and_train_models(save_artifacts=True)
