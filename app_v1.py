import streamlit as st
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier
import plotly.express as px

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Churn Prediction App",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("🛠 Telecom Customer Churn Prediction - Development Mode")

# -----------------------------
# Colors for Churn Visualizations
# -----------------------------
COLORS = {
    "Yes": "#d62728",   # churn
    "No":  "#1f77b4"    # stay
}

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned_telecom_churn_data.csv")
    df["Churn"] = df["Churn"].map({0: "No", 1: "Yes"})
    return df

df_data = load_data()

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
# Objective / Context
# -----------------------------
st.header("🎯 Objective")
st.write("""
The goal of this application is to predict **whether a telecom customer will churn or not**.  
It is designed in **development mode** for exploring features, understanding model behavior, and testing predictions interactively.
""")

# -----------------------------
# Data Visualizations
# -----------------------------
st.header("📊 Data Insights")

def plot_histogram(x_col, title):
    df_grouped = df_data.groupby([x_col, "Churn"]).size().reset_index(name="count")
    df_totals = df_data.groupby([x_col]).size().reset_index(name="total")
    df_grouped = df_grouped.merge(df_totals, on=x_col)
    df_grouped["percent"] = df_grouped["count"] / df_grouped["total"] * 100

    fig = px.bar(
        df_grouped,
        x=x_col,
        y="count",
        color="Churn",
        text=df_grouped["percent"].apply(lambda x: f"{x:.1f}%"),
        barmode="group",
        color_discrete_map=COLORS,
        title=title
    )
    fig.update_layout(xaxis_tickangle=-30)
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

plot_histogram("PaymentMethod", "Payment Method vs Churn")
plot_histogram("Contract", "Contract Type vs Churn")

# Overall Churn Distribution
fig_pie = px.pie(
    df_data,
    names="Churn",
    hole=0.6,
    color="Churn",
    color_discrete_map=COLORS,
    title="Overall Churn Distribution",
)
fig_pie.update_traces(textinfo="percent+label", pull=[0, 0.06])
st.plotly_chart(fig_pie, use_container_width=True)

# -----------------------------
# Model Metrics
# -----------------------------
st.header("📈 Model Performance Metrics")
metrics = {
    "Accuracy": 0.784414,
    "F1 Score": 0.642790,
    "ROC AUC": 0.843333,
    "Precision": 0.574074,
    "Recall": 0.730193,
    "Threshold": 0.584878,
    "Test Size": 0.25
}

cols = st.columns(len(metrics))
for i, (k, v) in enumerate(metrics.items()):
    with cols[i]:
        st.metric(label=k, value=f"{v:.2f}")

# -----------------------------
# Prediction Interface
# -----------------------------
st.header("🤖 Predict Churn for a Customer")

def yes_no_to_int(x):
    return 1 if x == "Yes" else 0

def gender_to_int(x):
    return 1 if x == "Male" else 0

with st.form("customer_form"):
    st.subheader("Customer Information")

    # Binary/Numeric Inputs
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])

    # Categorical Inputs
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
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

    # Numeric Inputs
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0)

    submitted = st.form_submit_button("Predict Churn")

# -----------------------------
# Prediction Logic
# -----------------------------
if submitted:
    # Feature engineering
    services_count = (
        np.array([
            phone_service, multiple_lines, online_security, online_backup,
            device_protection, tech_support, streaming_tv, streaming_movies
        ]) == "Yes"
    ).sum()

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
    df = df[feature_names]

    for c in cat_features:
        df[c] = df[c].astype(str)
    for c in df.columns:
        if c not in cat_features:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    proba = model.predict_proba(df)[0][1]
    prediction = int(proba >= best_threshold)

    # Display Result
    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ Likely to churn — Probability: {proba:.2f}")
    else:
        st.success(f"✅ Not likely to churn — Probability: {proba:.2f}")
    st.progress(float(proba))

# -----------------------------
# Conclusion / Feature Improvements
# -----------------------------
st.header("📝 Conclusion & Feature Improvements")
st.write("""
- This app allows testing customer churn interactively.  
- Observing features like **contract type, payment method, tenure, and services used** helps understand churn drivers.  
- Future improvements:
    - Include **SHAP or feature importance** visualization for better explainability.
    - Extend dataset with more behavioral features for improved prediction accuracy.
""")
