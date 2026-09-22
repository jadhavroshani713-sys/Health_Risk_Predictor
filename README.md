# 🩺 Healthcare Risk Predictor

### **AICTE | IBM SkillsBuild Data Analytics with AI Academic Internship**
**Conducted in Association with:** [BharatCares Foundation](https://bharatcares.org/)  
**Intern / Author:** **Roshani Jadhav** (*Data Analytics by Roshani Jadhao*)  
**Domain:** Healthcare Analytics, Predictive Artificial Intelligence, Clinical Decision Support  
**Dataset:** CDC Behavioral Risk Factor Surveillance System (BRFSS 2015)  

---

## 📌 1. Project Overview & Motivation

Cardiovascular diseases (CVDs) and myocardial infarctions remain the leading cause of premature mortality globally. In low- and middle-income demographics, lack of regular screening and delayed diagnosis often turn manageable conditions (such as hypertension or elevated cholesterol) into acute, life-threatening cardiac events.

In collaboration with **BharatCares**, this project establishes an end-to-end, production-ready Data Analytics and AI system called **Healthcare Risk Predictor**. The solution utilizes epidemiological survey data from over 250,000 individuals to:
1. Uncover critical lifestyle, behavioral, and clinical drivers of cardiovascular disease.
2. Address severe class imbalance (89.7% healthy vs. 10.3% heart disease) through class-weighted machine learning algorithms.
3. Deliver a high-sensitivity screening pipeline achieving **~78% recall** and **0.835 ROC-AUC**.
4. Power an interactive, accessible Streamlit web application providing real-time patient risk stratification and actionable community health recommendations.

---

## 🏗️ 2. System Architecture

```
+---------------------------------------------------------------------------------+
|                                DATA INGESTION                                   |
|   CDC BRFSS 2015 Dataset (253,680 records x 22 health indicators)               |
+---------------------------------------------------------------------------------+
                                       │
                                       ▼
+---------------------------------------------------------------------------------+
|                           ROBUST DATA CLEANING                                  |
|   - Duplicate Detection & Pruning: 23,899 duplicate records removed             |
|   - Null Value Verification & Median/Mode Imputation                            |
|   - Data Type Downcasting & Memory Optimization (229,781 clean records)         |
+---------------------------------------------------------------------------------+
                                       │
                                       ▼
+---------------------------------------------------------------------------------+
|                      EXPLORATORY DATA ANALYTICS (EDA)                           |
|   - Target Imbalance Audit (10.32% Heart Disease Prevalence)                    |
|   - Pearson Correlation Matrix across key clinical drivers                      |
|   - Co-occurrence Analysis (Hypertension x Hypercholesterolemia)                |
|   - Age-Stratified Cardiovascular Incidence Gradients                           |
+---------------------------------------------------------------------------------+
                                       │
                                       ▼
+---------------------------------------------------------------------------------+
|                       MACHINE LEARNING MODELING PIPELINE                        |
|   - Stratified 80/20 Train-Validation Split (183,824 train / 45,957 val)        |
|   - Feature Normalization: StandardScaler (zero mean, unit variance)            |
|   - Benchmark Models:                                                           |
|       * Class-Weighted Logistic Regression (ROC-AUC: 0.8355, Recall: 77.82%)    |
|       * Balanced Random Forest Classifier  (ROC-AUC: 0.8341, Recall: 77.97%)    |
|   - Artifact Serialization: scaler.joblib, heart_risk_model.joblib, metadata    |
+---------------------------------------------------------------------------------+
                                       │
                                       ▼
+---------------------------------------------------------------------------------+
|                      STREAMLIT INTERACTIVE WEB APPLICATION                      |
|   - Section 1: Executive KPI Dashboard & Interactive Visualizations             |
|   - Section 2: Real-Time Patient Intake Form & Dynamic Risk Score Gauge         |
|   - Section 3: Personalized BharatCares Community Care & Actionable Guidance    |
+---------------------------------------------------------------------------------+
```

---

## 📂 3. Repository & Workspace Structure

```
Health_Risk_Predictor/
├── data/
│   └── heart_disease_health_indicators_BRFSS2015.csv  # CDC BRFSS 2015 Survey Dataset
├── models/
│   ├── heart_risk_model.joblib                       # Serialized Random Forest Classifier
│   ├── logistic_regression_model.joblib              # Serialized Logistic Regression Model
│   ├── scaler.joblib                                 # Fitted StandardScaler Artifact
│   └── model_metadata.json                           # Model Performance & Feature Metadata
├── scripts/
│   ├── data_loader.py                                # Data Ingestion, Cleaning & Audit Module
│   ├── train_model.py                                # Training, Evaluation & Serialization
│   ├── predictor.py                                  # Inference Engine & Recommendation Logic
│   └── build_notebook.py                             # Automated Notebook Generation & Runner
├── app.py                                            # Interactive Streamlit Web Application
├── Roshani_Jadhav_HealthcareRiskPredictor.ipynb      # Complete Pre-Executed Submission Notebook
├── requirements.txt                                  # Dependency Manifest
└── README.md                                         # Project Documentation
```

---

## 📊 4. Dataset Description & Key Attributes

The Behavioral Risk Factor Surveillance System (BRFSS) is a continuous health-related telephone survey collected by the Centers for Disease Control and Prevention (CDC).

| Variable Name | Description | Value Encoding |
| :--- | :--- | :--- |
| **`HeartDiseaseorAttack`** *(Target)* | Has respondent ever been diagnosed with CHD or myocardial infarction? | `0` = No, `1` = Yes |
| **`HighBP`** | Diagnosed with high blood pressure / hypertension? | `0` = No, `1` = Yes |
| **`HighChol`** | Diagnosed with high blood cholesterol? | `0` = No, `1` = Yes |
| **`CholCheck`** | Blood cholesterol checked within past 5 years? | `0` = No, `1` = Yes |
| **`BMI`** | Body Mass Index | Continuous ($kg/m^2$) |
| **`Smoker`** | Smoked at least 100 cigarettes in lifetime? | `0` = No, `1` = Yes |
| **`Stroke`** | Ever diagnosed with a stroke? | `0` = No, `1` = Yes |
| **`Diabetes`** | Diabetic status | `0` = No, `1` = Pre-diabetes, `2` = Diabetes |
| **`PhysActivity`** | Any physical exercise in past 30 days outside regular work? | `0` = No, `1` = Yes |
| **`GenHlth`** | Self-rated general health status | Scale `1` (Excellent) to `5` (Poor) |
| **`DiffWalk`** | Serious difficulty walking or climbing stairs? | `0` = No, `1` = Yes |
| **`Age`** | 13-level ordinal age category | `1` (18-24) to `13` (80+) |
| **`Sex`** | Biological sex | `0` = Female, `1` = Male |

---

## 🔍 5. Exploratory Data Analytics: Key Conclusions

1. **Target Class Imbalance:**  
   In the cleaned dataset of 229,781 records, **23,717 respondents (10.32%)** reported experiencing heart disease or a heart attack, while 206,064 respondents (89.68%) reported no cardiac events. Standard unweighted classification models would achieve 89.7% accuracy by trivially predicting all zeros while failing 100% of high-risk patients. Incorporating `class_weight='balanced'` was crucial to counteract this bias.

2. **The Hypertension & Cholesterol Risk Multiplier:**  
   - Individuals with **both normal BP and normal cholesterol** exhibit a heart disease rate of only **2.6%**.
   - Individuals with **High BP alone** exhibit an increased incidence of **11.2%**.
   - Individuals with **both High BP and High Cholesterol** exhibit a massive spike to **25.4%** (~1 in 4 patients).

3. **Age Gradient Effect:**  
   Cardiovascular vulnerability exhibits an exponential rise with age, ascending from under **1.1%** in the 18–24 bracket to over **22.5%** in the 75+ age demographic.

4. **Leading Predictive Drivers:**  
   Gini feature importance identified **Age**, **Self-Reported General Health (`GenHlth`)**, **Hypertension (`HighBP`)**, **Hypercholesterolemia (`HighChol`)**, and **Functional Mobility (`DiffWalk`)** as the five most predictive risk variables.

---

## 🤖 6. Machine Learning Benchmark & Performance

Models were trained on 80% stratified samples (183,824 records) and evaluated on 20% holdout validation records (45,957 records) with `StandardScaler` transformations.

| Metric | Logistic Regression (Class-Weighted) | Random Forest Classifier (Balanced) | Clinical Implication |
| :--- | :---: | :---: | :--- |
| **Overall Accuracy** | 73.92% | **74.02%** | High baseline generalization |
| **Sensitivity / Recall** | 77.82% | **77.97%** | **Captures ~78% of all true heart disease cases** |
| **Precision** | 25.24% | **25.34%** | Expected given 10% base prevalence |
| **F1-Score** | 38.11% | **38.25%** | Optimal harmonic trade-off |
| **ROC-AUC Score** | **0.8355** | 0.8341 | Excellent discriminatory capability |

> 💡 **Senior ML Engineer Note on Medical Metrics:** In clinical screening, **Recall is prioritized over Precision**. The cost of a False Negative (failing to alert a patient at high risk of a heart attack) is catastrophic, whereas a False Positive simply results in non-invasive follow-up diagnostic screening (e.g., lipid profile, ECG).

---

## 🌐 7. Streamlit Web Application Features

The interactive web portal is organized into dedicated modules:

1. **Executive Dashboard & Analytics Overview:**
   - Real-time KPI summary cards (prevalence rates, total survey population, hypertension incidence).
   - Interactive Plotly visualizations: Target class distribution pie chart, correlation heatmap, grouped risk-multiplier bar charts, and age progression spline.
2. **Real-Time Patient Risk Assessment:**
   - Comprehensive input controls: Demographics (Age bracket, Sex, BMI with automatic category determination), Medical History (Hypertension, Cholesterol, Stroke, Diabetes, General Health), and Lifestyle (Smoking, Physical Activity, Walking difficulty).
   - Instant inference triggering the trained ML model.
   - Dynamic 0–100% Risk Score Gauge with color-coded severity bands:
     - 🟢 **Low Risk Profile (< 30% Risk):** Healthy baseline maintenance.
     - 🟡 **Moderate Risk (30% - 50% Risk):** Lifestyle caution & preventive screening.
     - 🔴 **High Risk Profile (≥ 50% Risk):** Urgent clinical follow-up required.
3. **Personalized BharatCares Community Care Guidance:**
   - Automatically generates targeted recommendations based on specific positive risk markers (e.g., DASH sodium diet for HighBP, smoking cessation program links for smokers, exercise guidelines for sedentary patients).

---

## 🚀 8. Setup & Execution Guide

### Prerequisites
- Python 3.10 to 3.14
- Git

### 1. Clone or Open Workspace
```bash
cd Health_Risk_Predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Ingest Data & Train ML Models (Optional - Models Already Serialized)
```bash
# Ingest and audit dataset
python scripts/data_loader.py

# Train models and serialize artifacts
python scripts/train_model.py
```

### 4. Launch the Streamlit Interactive Dashboard
```bash
streamlit run app.py
```
Open your web browser and navigate to: **`http://localhost:8501`**

### 5. Running the Jupyter Notebook
You can open `Roshani_Jadhav_HealthcareRiskPredictor.ipynb` in VS Code, JupyterLab, or Google Colab. All cell outputs, data tables, and high-resolution Seaborn plots are pre-rendered and ready for evaluation.

---

## 🤝 9. BharatCares Community Health Recommendations

In alignment with **BharatCares Foundation's** community welfare objectives:
1. **Targeted Screening Camps:** Prioritize mobile diagnostic vans in districts exhibiting high rates of obesity and hypertension for early ECG and lipid checks.
2. **Preventive Community Education:** Distribute culturally adapted materials emphasizing sodium reduction (<1,500 mg/day) and Mediterranean-style diets rich in legumes and leafy vegetables.
3. **Smoking Cessation Support Groups:** Establish localized peer-support groups in community centers to assist individuals in smoking cessation.
4. **Active Mobility Initiatives:** Encourage 150 minutes of weekly moderate walking in public parks to reverse sedentary risks.

---

## ⚖️ 10. Ethical AI & Clinical Governance

- **Decision Support, Not Diagnosis:** This application is engineered as a risk stratification and triage tool to assist healthcare workers and patients. It does not replace a definitive medical diagnosis by a licensed physician.
- **Fairness & Bias Mitigation:** The model was evaluated across gender and age strata to avoid systemic underdiagnosis in vulnerable demographic cohorts.
- **Data Privacy:** In compliance with healthcare data protection principles, no patient data entered into the web form is permanently stored or transmitted externally.

---

## 📜 11. Project Metadata & Certification

- **Project:** Healthcare Risk Predictor
- **Author:** Roshani Jadhav (*Data Analytics by Roshani Jadhao*)
- **Program:** AICTE | IBM SkillsBuild Data Analytics with AI Academic Internship
- **Industry Partner:** BharatCares
- **Status:** Complete, Production-Ready, Submission-Ready
