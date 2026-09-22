"""
Healthcare Risk Predictor - Inference and Clinical Guidance Engine
Author: Roshani Jadhav
Project: AICTE | IBM SkillsBuild Data Analytics with AI Academic Internship
Organization: In association with BharatCares
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "heart_risk_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
META_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

# Default baseline population values for unselected parameters
POPULATION_DEFAULTS = {
    "HighBP": 0,
    "HighChol": 0,
    "CholCheck": 1,
    "BMI": 25.0,
    "Smoker": 0,
    "Stroke": 0,
    "Diabetes": 0,
    "PhysActivity": 1,
    "Fruits": 1,
    "Veggies": 1,
    "HvyAlcoholConsump": 0,
    "AnyHealthcare": 1,
    "NoDocbcCost": 0,
    "GenHlth": 2,      # 1: Excellent, 2: Very Good, 3: Good, 4: Fair, 5: Poor
    "MentHlth": 0,     # days of poor mental health in past 30 days
    "PhysHlth": 0,     # days of poor physical health in past 30 days
    "DiffWalk": 0,
    "Sex": 0,          # 0: Female, 1: Male
    "Age": 7,          # BRFSS age category (7 corresponds to ~50-54)
    "Education": 5,    # College 1 to 3 years
    "Income": 6        # $35,000 to $50,000
}


def load_artifacts():
    """Load pre-trained model, scaler, and metadata."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            "Model artifacts not found. Please run 'python scripts/train_model.py' first."
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    metadata = {}
    if os.path.exists(META_PATH):
        with open(META_PATH, "r") as f:
            metadata = json.load(f)

    return model, scaler, metadata


def generate_recommendations(inputs: dict, risk_prob: float, is_high_risk: bool) -> list[dict]:
    """Generate tailored, patient-specific community care recommendations."""
    recs = []

    if is_high_risk:
        recs.append({
            "category": "Critical Action",
            "icon": "🚨",
            "title": "Immediate Cardiologist Consultation",
            "detail": "Your clinical profile indicates an elevated cardiovascular risk. Schedule a comprehensive diagnostic cardiac screening (ECG, Lipid Profile, and Echocardiogram) within 14 days."
        })

    # High Blood Pressure
    if inputs.get("HighBP", 0) == 1:
        recs.append({
            "category": "Hypertension Management",
            "icon": "🩺",
            "title": "Blood Pressure Monitoring & Sodium Reduction",
            "detail": "Maintain daily home blood pressure logging. Target BP < 120/80 mmHg. Adopt the DASH diet with sodium intake capped below 1,500 mg daily."
        })

    # High Cholesterol
    if inputs.get("HighChol", 0) == 1:
        recs.append({
            "category": "Lipid Care",
            "icon": "🩸",
            "title": "Lipid Profile Management",
            "detail": "Reduce saturated and trans fat intake. Discuss statin therapy or dietary phytosterols with your primary healthcare provider."
        })

    # Smoking
    if inputs.get("Smoker", 0) == 1:
        recs.append({
            "category": "Lifestyle Cessation",
            "icon": "🚭",
            "title": "Smoking Cessation Support",
            "detail": "Tobacco usage accelerates arterial plaque formation. Enroll in a community tobacco cessation program or utilize nicotine replacement therapy."
        })

    # BMI
    bmi = inputs.get("BMI", 25.0)
    if bmi >= 30.0:
        recs.append({
            "category": "Weight Management",
            "icon": "⚖️",
            "title": "Structured Weight Reduction Program",
            "detail": f"Current BMI is {bmi:.1f} (Classified as Obese). A 5% to 10% sustained weight reduction substantially decreases cardiac workload and vascular resistance."
        })
    elif bmi >= 25.0:
        recs.append({
            "category": "Weight Management",
            "icon": "⚖️",
            "title": "Healthy Weight Maintenance",
            "detail": f"Current BMI is {bmi:.1f} (Overweight range). Focus on portion control and whole-food nutrition."
        })

    # Physical Activity
    if inputs.get("PhysActivity", 1) == 0:
        recs.append({
            "category": "Physical Wellness",
            "icon": "🏃",
            "title": "Cardio Exercise Regimen",
            "detail": "Engage in at least 150 minutes of moderate aerobic activity (e.g. brisk walking, cycling) per week as recommended by the American Heart Association."
        })

    # Stroke / Diabetic History
    if inputs.get("Stroke", 0) == 1:
        recs.append({
            "category": "Neurological & Vascular Care",
            "icon": "🧠",
            "title": "Secondary Stroke Prevention Protocol",
            "detail": "Prior stroke is a high-magnitude cardiovascular indicator. Adhere strictly to antiplatelet regimens and carotid Doppler screening."
        })

    if inputs.get("Diabetes", 0) in [1, 2]:
        recs.append({
            "category": "Endocrine Care",
            "icon": "🔬",
            "title": "Strict Glycemic Control",
            "detail": "Maintain HbA1c levels under 7.0%. High blood glucose damages endothelial vascular linings over time."
        })

    # Community Health Baseline
    if not is_high_risk and len(recs) == 0:
        recs.append({
            "category": "Preventive Care",
            "icon": "✅",
            "title": "Sustain Healthy Habits",
            "detail": "Your risk profile is currently low. Continue balanced Mediterranean-style nutrition, 30 minutes daily activity, and annual preventive health checkups."
        })

    return recs


def predict_patient_risk(patient_traits: dict) -> dict:
    """
    Accepts arbitrary patient input traits, merges with population defaults,
    normalizes features, and predicts risk profile.
    """
    model, scaler, metadata = load_artifacts()
    feature_names = metadata.get("feature_names", list(POPULATION_DEFAULTS.keys()))

    # Build feature record with complete ordered list
    record = {}
    for feat in feature_names:
        record[feat] = patient_traits.get(feat, POPULATION_DEFAULTS.get(feat, 0))

    df_input = pd.DataFrame([record])[feature_names]
    scaled_input = scaler.transform(df_input)

    # Predict probability of Heart Disease
    risk_prob = float(model.predict_proba(scaled_input)[0, 1])
    # Risk threshold: balanced model outputs ~0.50 decision boundary
    is_high_risk = risk_prob >= 0.50

    risk_label = "High Risk Profile" if is_high_risk else "Low Risk Profile"
    recommendations = generate_recommendations(record, risk_prob, is_high_risk)

    return {
        "risk_label": risk_label,
        "is_high_risk": is_high_risk,
        "risk_probability": risk_prob,
        "risk_percentage": round(risk_prob * 100, 2),
        "input_features": record,
        "recommendations": recommendations,
        "confidence_level": "High" if abs(risk_prob - 0.5) > 0.2 else "Moderate"
    }


if __name__ == "__main__":
    test_patient = {
        "HighBP": 1,
        "HighChol": 1,
        "BMI": 32.5,
        "Smoker": 1,
        "Stroke": 0,
        "Age": 10,  # 65-69
        "GenHlth": 4, # Fair
        "PhysActivity": 0
    }
    result = predict_patient_risk(test_patient)
    print("--- Test Patient Prediction Result ---")
    print(f"Risk Label:       {result['risk_label']}")
    print(f"Risk Probability: {result['risk_percentage']}%")
    print(f"Recommendations count: {len(result['recommendations'])}")
    for r in result["recommendations"]:
        print(f"  - [{r['category']}] {r['title']}: {r['detail']}")
