import streamlit as st
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Churn Prediction App", layout="centered")
st.title("📊 Telecom Customer Churn Prediction")
st.write("Fill in the details below to estimate churn probability.")

# -----------------------------
# Load Metadata
# -----------------------------
@st.cache_resource
def load_metadata():
    with open("metadata.pkl", "rb") as f:
        return pickle.load(f)

metadata = load_metadata()
cat_features = metadata["cat_features"]
feature_names = metadata["feature_names"]
best_threshold = metadata["best_threshold"]

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("catboost_churn_model.cbm")
    return model

model = load_model()

# -----------------------------
# Helper Functions
# -----------------------------
def yes_no_to_int(x):
    return 1 if x == "Yes" else 0

def gender_to_int(x):
    return 1 if x == "Male" else 0

# -----------------------------
# Streamlit Form
# -----------------------------
with st.form("customer_form"):
    st.subheader("Customer Information")

    # Binary/Numeric Inputs
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])

    # Categorical Inputs (strings including "No phone service")
    multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])

    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

    # Numeric Inputs
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0)

    # Submit button
    submitted = st.form_submit_button("Predict Churn")

# -----------------------------
# Prediction Logic
# -----------------------------
if submitted:
    # Engineered features
    services_count = (np.array([
        phone_service, multiple_lines, online_security, online_backup,
        device_protection, tech_support, streaming_tv, streaming_movies
    ]) == "Yes").sum()

    payment_ratio = monthly_charges / (total_charges + 1)
    avg_charges = total_charges / (tenure + 1)
    is_new_client = int(tenure <= 6)

    tenure_group = pd.cut(
        [tenure],
        bins=[0, 6, 12, 24, 48, 72],
        labels=["0–6", "6–12", "12–24", "24–48", "48–72"],
        include_lowest=True
    )[0]

    contract_payment = contract + "_" + payment_method

    # Build input row
    row = {
        "gender": gender_to_int(gender),
        "SeniorCitizen": yes_no_to_int(senior),
        "Partner": yes_no_to_int(partner),
        "Dependents": yes_no_to_int(dependents),
        "tenure": tenure,
        "PhoneService": yes_no_to_int(phone_service),

        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,

        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,

        "services_count": services_count,
        "payment_ratio": payment_ratio,
        "avg_charges": avg_charges,
        "is_new_client": is_new_client,
        "tenure_group": tenure_group,
        "contract_payment": contract_payment
    }

    df = pd.DataFrame([row])

    # -----------------------------
    # Safe Type Enforcement
    # -----------------------------
    # Ensure correct column order
    df = df[feature_names]

    # Cast categorical features to string
    for c in cat_features:
        df[c] = df[c].astype(str)

    # Cast numeric features to float
    for c in df.columns:
        if c not in cat_features:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # -----------------------------
    # Prediction
    # -----------------------------
    proba = model.predict_proba(df)[0][1]
    prediction = int(proba >= best_threshold)

    # -----------------------------
    # Display Results
    # -----------------------------
    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ The customer is **likely to churn**.\n\n**Probability: {proba:.2f}**")
    else:
        st.success(f"✅ The customer is **not likely to churn**.\n\n**Probability: {proba:.2f}**")

    st.progress(float(proba))

    st.write("---")
    st.write("### 📌 Interpretation")
    if prediction == 1:
        st.write("This customer shows churn-risk signals. Consider retention strategies.")
    else:
        st.write("Customer appears stable based on current behavioral patterns.")
