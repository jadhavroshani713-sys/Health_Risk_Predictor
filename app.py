"""
Healthcare Risk Predictor - Production Streamlit Web Application
Title: Data Analytics by Roshani Jadhao
Academic Internship: AICTE | IBM SkillsBuild in association with BharatCares
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add scripts directory to path for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(CURRENT_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from predictor import predict_patient_risk, load_artifacts

# Configure Streamlit page
st.set_page_config(
    page_title="Healthcare Risk Predictor | Roshani Jadhao",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-family: 'Segoe UI', sans-serif;
        background: linear-gradient(135deg, #0d324d 0%, #1e5f74 50%, #133b5c 100%);
        color: #ffffff;
        padding: 24px 30px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .badge-bar {
        margin-top: 10px;
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }
    .tag {
        background-color: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        color: #f1f6f9;
    }
    .metric-card {
        background: #ffffff;
        padding: 18px;
        border-radius: 10px;
        border-left: 5px solid #1e5f74;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 12px;
    }
    .risk-banner-high {
        background-color: #fee2e2;
        border-left: 6px solid #dc2626;
        padding: 20px;
        border-radius: 8px;
        color: #991b1b;
        margin-bottom: 20px;
    }
    .risk-banner-low {
        background-color: #ecfdf5;
        border-left: 6px solid #10b981;
        padding: 20px;
        border-radius: 8px;
        color: #065f46;
        margin-bottom: 20px;
    }
    .rec-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Application Banner Header
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size: 2.2rem; font-weight: 700;">Healthcare Risk Predictor</h1>
    <h3 style="margin: 4px 0 0 0; font-size: 1.15rem; font-weight: 400; color: #d0e8f2;">
        Data Analytics by Roshani Jadhao
    </h3>
    <div class="badge-bar">
        <span class="tag">AICTE Approved</span>
        <span class="tag">IBM SkillsBuild</span>
        <span class="tag">In association with BharatCares</span>
        <span class="tag">BRFSS 2015 Dataset</span>
        <span class="tag">Production AI Model</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Cached Data Loader for Dashboard
@st.cache_data(show_spinner=False)
def load_sample_dashboard_data():
    csv_path = os.path.join(CURRENT_DIR, "data", "heart_disease_health_indicators_BRFSS2015.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if df.duplicated().sum() > 0:
            df = df.drop_duplicates().reset_index(drop=True)
        return df
    return None

@st.cache_resource(show_spinner=False)
def get_cached_artifacts():
    try:
        return load_artifacts()
    except Exception:
        return None, None, {}

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=64)
st.sidebar.title("Navigation Menu")
app_mode = st.sidebar.radio(
    "Choose Section:",
    [
        "📊 Dashboard & Analytics Overview",
        "🩺 Real-Time Patient Risk Assessment",
        "📈 Model Evaluation & Methodology",
        "ℹ️ About BharatCares & Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("👨‍💻 **Author:** Roshani Jadhav (Jadhao)")
st.sidebar.caption("🎓 **Internship:** IBM SkillsBuild & AICTE")
st.sidebar.caption("🏛️ **Partner:** BharatCares Foundation")


# ==============================================================================
# SECTION 1: DASHBOARD OVERVIEW & EXPLORATORY DATA ANALYTICS
# ==============================================================================
if app_mode == "📊 Dashboard & Analytics Overview":
    st.subheader("📊 Executive Healthcare Analytics Dashboard")
    st.markdown(
        "Exploratory data analysis of cardiovascular health indicators from the CDC's "
        "Behavioral Risk Factor Surveillance System (BRFSS 2015)."
    )

    df = load_sample_dashboard_data()

    if df is None:
        st.warning("Dataset not found locally. Please run `python scripts/data_loader.py` to fetch data.")
    else:
        # Top KPI Metric Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        total_records = len(df)
        heart_cases = int(df["HeartDiseaseorAttack"].sum())
        prevalence = (heart_cases / total_records) * 100
        high_bp_pct = (df["HighBP"].sum() / total_records) * 100
        avg_bmi = df["BMI"].mean()

        with kpi1:
            st.metric(label="Total Survey Records", value=f"{total_records:,}")
        with kpi2:
            st.metric(label="Cardiovascular Disease Prevalence", value=f"{prevalence:.2f}%", delta=f"{heart_cases:,} cases", delta_color="inverse")
        with kpi3:
            st.metric(label="High Blood Pressure Rate", value=f"{high_bp_pct:.1f}%")
        with kpi4:
            st.metric(label="Population Mean BMI", value=f"{avg_bmi:.1f} kg/m²")

        st.markdown("---")

        # Row 1: Target Distribution & Grouped Clinical Risk Drivers
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("##### 1. Target Class Distribution (Heart Disease / Attack)")
            target_counts = df["HeartDiseaseorAttack"].value_counts().reset_index()
            target_counts.columns = ["HeartDisease", "Count"]
            target_counts["Label"] = target_counts["HeartDisease"].map({0: "No Heart Disease (89.7%)", 1: "Heart Disease/Attack (10.3%)"})
            
            fig_target = px.pie(
                target_counts,
                values="Count",
                names="Label",
                hole=0.45,
                color="Label",
                color_discrete_map={
                    "No Heart Disease (89.7%)": "#2b6cb0",
                    "Heart Disease/Attack (10.3%)": "#e53e3e"
                }
            )
            fig_target.update_layout(
                margin=dict(t=30, b=10, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_target, use_container_width=True)
            st.caption("Significant class imbalance (1:8.7) confirms the necessity of class-weighted modeling.")

        with col_right:
            st.markdown("##### 2. Heart Disease Occurrence by Hypertension & Cholesterol")
            # Grouped bar chart
            combo = df.groupby(["HighBP", "HighChol"])["HeartDiseaseorAttack"].mean().reset_index()
            combo["Rate"] = combo["HeartDiseaseorAttack"] * 100
            combo["Profile"] = combo.apply(
                lambda r: f"BP: {'High' if r['HighBP']==1 else 'Normal'} | Chol: {'High' if r['HighChol']==1 else 'Normal'}",
                axis=1
            )
            fig_combo = px.bar(
                combo,
                x="Profile",
                y="Rate",
                text=combo["Rate"].apply(lambda v: f"{v:.1f}%"),
                color="Rate",
                color_continuous_scale="Reds",
                labels={"Rate": "Heart Disease Rate (%)", "Profile": "Clinical Combination"}
            )
            fig_combo.update_layout(
                margin=dict(t=20, b=10, l=10, r=10),
                yaxis_title="Prevalence Rate (%)",
                xaxis_tickangle=-25
            )
            st.plotly_chart(fig_combo, use_container_width=True)
            st.caption("Patients presenting with both High BP and High Cholesterol face up to a 4x risk multiplier.")

        st.markdown("---")

        # Row 2: Correlation Heatmap of Top Drivers
        st.markdown("##### 3. Correlation Heatmap of Key Clinical & Lifestyle Drivers")
        key_cols = [
            "HeartDiseaseorAttack", "HighBP", "HighChol", "BMI",
            "Smoker", "Stroke", "Diabetes", "PhysActivity", "GenHlth", "Age"
        ]
        corr_matrix = df[key_cols].corr()

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="Blues",
            title="Pearson Correlation Matrix (BRFSS Key Attributes)"
        )
        fig_corr.update_layout(margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_corr, use_container_width=True)

        # Row 3: Age Breakdown
        st.markdown("##### 4. Heart Disease Prevalence Across Age Brackets")
        age_map = {
            1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
            6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
            11: "70-74", 12: "75-79", 13: "80+"
        }
        age_df = df.groupby("Age")["HeartDiseaseorAttack"].agg(Total="count", Positives="sum").reset_index()
        age_df["Prevalence_Pct"] = (age_df["Positives"] / age_df["Total"]) * 100
        age_df["AgeGroup"] = age_df["Age"].map(age_map)

        fig_age = px.line(
            age_df,
            x="AgeGroup",
            y="Prevalence_Pct",
            markers=True,
            line_shape="spline",
            title="Cardiovascular Disease Prevalence by Age Group (%)"
        )
        fig_age.update_traces(line_color="#e53e3e", line_width=3, marker=dict(size=8, color="#991b1b"))
        fig_age.update_layout(
            margin=dict(t=40, b=10, l=10, r=10),
            yaxis_title="Prevalence (%)",
            xaxis_title="Age Category"
        )
        st.plotly_chart(fig_age, use_container_width=True)


# ==============================================================================
# SECTION 2: REAL-TIME PATIENT RISK ASSESSMENT FORM
# ==============================================================================
elif app_mode == "🩺 Real-Time Patient Risk Assessment":
    st.subheader("🩺 Real-Time Patient Risk Assessment")
    st.markdown(
        "Enter the patient's physiological and lifestyle metrics below. "
        "The machine learning inference engine will evaluate the composite risk score "
        "and produce tailored community care guidance."
    )

    with st.form("risk_assessment_form"):
        st.markdown("#### 1. Patient Demographics & Body Metrics")
        c1, c2, c3 = st.columns(3)

        with c1:
            age_labels = [
                "18-24 (Category 1)", "25-29 (Category 2)", "30-34 (Category 3)",
                "35-39 (Category 4)", "40-44 (Category 5)", "45-49 (Category 6)",
                "50-54 (Category 7)", "55-59 (Category 8)", "60-64 (Category 9)",
                "65-69 (Category 10)", "70-74 (Category 11)", "75-79 (Category 12)",
                "80+ (Category 13)"
            ]
            selected_age = st.selectbox("Patient Age Bracket", age_labels, index=6)
            age_val = age_labels.index(selected_age) + 1

        with c2:
            sex_option = st.radio("Biological Sex", ["Female", "Male"], horizontal=True)
            sex_val = 1 if sex_option == "Male" else 0

        with c3:
            bmi_val = st.slider("Body Mass Index (BMI)", min_value=14.0, max_value=55.0, value=26.5, step=0.5)
            if bmi_val < 18.5:
                bmi_status = "Underweight"
            elif bmi_val < 25.0:
                bmi_status = "Normal weight"
            elif bmi_val < 30.0:
                bmi_status = "Overweight"
            else:
                bmi_status = "Obese"
            st.caption(f"BMI Classification: **{bmi_status}**")

        st.markdown("#### 2. Clinical History & Chronic Indicators")
        c4, c5, c6 = st.columns(3)

        with c4:
            high_bp_sel = st.selectbox("Diagnosed with High Blood Pressure (Hypertension)?", ["No", "Yes"])
            high_bp_val = 1 if high_bp_sel == "Yes" else 0

            chol_check_sel = st.selectbox("Cholesterol Checked in Past 5 Years?", ["Yes", "No"])
            chol_check_val = 1 if chol_check_sel == "Yes" else 0

        with c5:
            high_chol_sel = st.selectbox("Diagnosed with High Blood Cholesterol?", ["No", "Yes"])
            high_chol_val = 1 if high_chol_sel == "Yes" else 0

            stroke_sel = st.selectbox("Ever had a Stroke or Mini-Stroke?", ["No", "Yes"])
            stroke_val = 1 if stroke_sel == "Yes" else 0

        with c6:
            diabetes_sel = st.selectbox(
                "Diabetic Condition",
                ["0: Non-diabetic", "1: Pre-diabetic / Borderline", "2: Diabetic"]
            )
            diabetes_val = int(diabetes_sel[0])

            gen_hlth_sel = st.selectbox(
                "Self-Reported General Health",
                ["1: Excellent", "2: Very Good", "3: Good", "4: Fair", "5: Poor"],
                index=2
            )
            gen_hlth_val = int(gen_hlth_sel[0])

        st.markdown("#### 3. Lifestyle Habits & Functional Mobility")
        c7, c8, c9 = st.columns(3)

        with c7:
            smoker_sel = st.selectbox("Smoked at least 100 cigarettes in entire life?", ["No", "Yes"])
            smoker_val = 1 if smoker_sel == "Yes" else 0

        with c8:
            phys_act_sel = st.selectbox("Physical Activity or Exercise in past 30 days?", ["Yes", "No"])
            phys_act_val = 1 if phys_act_sel == "Yes" else 0

            hvy_alcohol_sel = st.selectbox("Heavy Alcohol Consumption?", ["No", "Yes"])
            hvy_alcohol_val = 1 if hvy_alcohol_sel == "Yes" else 0

        with c9:
            diff_walk_sel = st.selectbox("Serious Difficulty Walking or Climbing Stairs?", ["No", "Yes"])
            diff_walk_val = 1 if diff_walk_sel == "Yes" else 0

            phys_hlth_days = st.slider("Days of physical illness/injury in past 30 days", 0, 30, 2)

        submit_btn = st.form_submit_button("🩺 Predict Risk Profile", use_container_width=True)

    if submit_btn:
        patient_payload = {
            "Age": age_val,
            "Sex": sex_val,
            "BMI": bmi_val,
            "HighBP": high_bp_val,
            "HighChol": high_chol_val,
            "CholCheck": chol_check_val,
            "Stroke": stroke_val,
            "Diabetes": diabetes_val,
            "GenHlth": gen_hlth_val,
            "Smoker": smoker_val,
            "PhysActivity": phys_act_val,
            "HvyAlcoholConsump": hvy_alcohol_val,
            "DiffWalk": diff_walk_val,
            "PhysHlth": phys_hlth_days,
            "MentHlth": 0,
            "Fruits": 1,
            "Veggies": 1,
            "AnyHealthcare": 1,
            "NoDocbcCost": 0,
            "Education": 5,
            "Income": 6
        }

        with st.spinner("Analyzing patient health profile via trained ML pipeline..."):
            try:
                pred_result = predict_patient_risk(patient_payload)
                risk_pct = pred_result["risk_percentage"]
                is_high = pred_result["is_high_risk"]

                st.markdown("### Clinical Risk Evaluation Result")

                if is_high:
                    st.markdown(f"""
                    <div class="risk-banner-high">
                        <h2 style="margin:0; color:#b91c1c;">⚠️ {pred_result['risk_label']} (Elevated Cardiac Vulnerability)</h2>
                        <p style="margin:8px 0 0 0; font-size:1.1rem; color:#7f1d1d;">
                            The diagnostic pipeline calculated a <strong>{risk_pct}%</strong> probability of underlying or imminent cardiovascular complications.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="risk-banner-low">
                        <h2 style="margin:0; color:#047857;">✅ {pred_result['risk_label']} (Favorable Cardiovascular Baseline)</h2>
                        <p style="margin:8px 0 0 0; font-size:1.1rem; color:#064e3b;">
                            The diagnostic pipeline calculated a <strong>{risk_pct}%</strong> probability of heart disease, falling within healthy baseline screening limits.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # Visual Risk Gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=risk_pct,
                    title={'text': "Cardiovascular Risk Score (%)", 'font': {'size': 20}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "#dc2626" if is_high else "#10b981"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 30], 'color': "#dcfce7"},
                            {'range': [30, 50], 'color': "#fef9c3"},
                            {'range': [50, 100], 'color': "#fee2e2"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=280, margin=dict(t=20, b=20, l=30, r=30))
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Tailored Community Care Guidance
                st.markdown("#### 📋 Tailored Community Care & Clinical Action Plan")
                st.markdown(
                    "*Formulated in accordance with BharatCares community healthcare initiatives:*"
                )

                for rec in pred_result["recommendations"]:
                    st.markdown(f"""
                    <div class="rec-card">
                        <h4 style="margin:0 0 4px 0; color:#1e293b;">{rec['icon']} {rec['title']}</h4>
                        <span style="font-size:0.8rem; background:#e2e8f0; padding:2px 8px; border-radius:12px; font-weight:600; color:#475569;">
                            {rec['category']}
                        </span>
                        <p style="margin:8px 0 0 0; color:#334155; font-size:0.95rem;">{rec['detail']}</p>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as ex:
                st.error(f"Error during risk inference: {ex}")


# ==============================================================================
# SECTION 3: MODEL EVALUATION & METHODOLOGY
# ==============================================================================
elif app_mode == "📈 Model Evaluation & Methodology":
    st.subheader("📈 Machine Learning Modeling Architecture & Benchmarks")
    st.markdown(
        "Detailed performance comparison between candidate classification models trained on "
        "stratified 80/20 train/validation splits."
    )

    model, scaler, metadata = get_cached_artifacts()

    if not metadata:
        st.info("Model metadata not loaded. Please ensure `scripts/train_model.py` has been executed.")
    else:
        m_lr = metadata.get("metrics_lr", {})
        m_rf = metadata.get("metrics_rf", {})

        # Benchmark Comparison Table
        st.markdown("##### 1. Comparative Metrics Matrix")
        comp_data = {
            "Evaluation Metric": [
                "Overall Accuracy",
                "Positive Class Recall (Sensitivity)",
                "Positive Class Precision",
                "F1-Score",
                "ROC-AUC Score"
            ],
            "Logistic Regression (Class-Weighted)": [
                f"{m_lr.get('accuracy', 0)*100:.2f}%",
                f"{m_lr.get('recall', 0)*100:.2f}%",
                f"{m_lr.get('precision', 0)*100:.2f}%",
                f"{m_lr.get('f1_score', 0)*100:.2f}%",
                f"{m_lr.get('roc_auc', 0):.4f}"
            ],
            "Random Forest Classifier (Balanced)": [
                f"{m_rf.get('accuracy', 0)*100:.2f}%",
                f"{m_rf.get('recall', 0)*100:.2f}%",
                f"{m_rf.get('precision', 0)*100:.2f}%",
                f"{m_rf.get('f1_score', 0)*100:.2f}%",
                f"{m_rf.get('roc_auc', 0):.4f}"
            ]
        }
        st.table(pd.DataFrame(comp_data))

        st.info(
            "💡 **Clinical Rationale:** In healthcare screening, maximizing **Recall (~78%)** "
            "is prioritized over raw accuracy because failing to identify a patient at risk "
            "(False Negative) carries severe clinical repercussions."
        )

        st.markdown("---")

        # Top 10 Feature Importances
        st.markdown("##### 2. Top 10 Random Forest Predictive Risk Factors")
        rf_imp = metadata.get("feature_importances_rf", {})
        if rf_imp:
            top_features = list(rf_imp.keys())[:10]
            top_weights = [rf_imp[k] for k in top_features]

            df_imp = pd.DataFrame({"Feature": top_features, "Importance": top_weights}).sort_values("Importance", ascending=True)
            fig_imp = px.bar(
                df_imp,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Viridis",
                title="Gini Feature Importance Ranking"
            )
            fig_imp.update_layout(margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig_imp, use_container_width=True)


# ==============================================================================
# SECTION 4: ABOUT BHARATCARES & PROJECT
# ==============================================================================
elif app_mode == "ℹ️ About BharatCares & Project":
    st.subheader("ℹ️ About the Academic Internship & BharatCares Partnership")
    st.markdown("""
    ### Project Overview
    The **Healthcare Risk Predictor** was developed by **Roshani Jadhav** as part of the 
    **AICTE | IBM SkillsBuild Data Analytics with AI Academic Internship** conducted in association with **BharatCares**.

    #### Mission & Community Impact
    - **BharatCares** works actively across community healthcare empowerment, preventive screening, and digital interventions.
    - This project demonstrates how modern predictive AI and interactive data analytics can identify high-risk cardiovascular individuals early, enabling targeted primary healthcare interventions before acute cardiac events occur.

    #### Technical Architecture
    - **Data Source:** CDC Behavioral Risk Factor Surveillance System (BRFSS 2015).
    - **Preprocessing:** Duplicate removal, robust missing data imputation, standardized normalization.
    - **Modeling:** Class-weighted Logistic Regression and Random Forest Classification pipelines.
    - **Deployment:** Real-time interactive Streamlit web dashboard with instant inference and personalized health guidance.

    ---
    *Disclaimer: This predictive tool is developed for educational and clinical decision-support research. It does not replace formal clinical diagnosis by a licensed medical practitioner.*
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
    "Healthcare Risk Predictor | Data Analytics by Roshani Jadhao | AICTE • IBM SkillsBuild • BharatCares"
    "</div>",
    unsafe_allow_html=True
)
