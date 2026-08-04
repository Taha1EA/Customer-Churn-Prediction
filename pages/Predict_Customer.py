import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Predict Customer",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

model = joblib.load(open("log_reg.pkl", "rb"))
transformer = joblib.load(open("transformer.pkl", "rb"))

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title(" Customer Churn Prediction")

st.write(
    "Enter the customer's information then click Predict."
)

st.markdown("---")

# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        ['0','1']
    )

    partner = st.selectbox(
        "Partner",
        ["Yes","No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes","No"]
    )

    tenure = st.slider(
        "Tenure (Months)",
        0,
        72,
        12
    )

    phone = st.selectbox(
        "Phone Service",
        ["Yes","No"]
    )

    multiple = st.selectbox(
        "Multiple Lines",
        ["Yes","No","No phone service"]
    )

with col2:

    internet = st.selectbox(
        "Internet Service",
        ["DSL","Fiber optic","No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes","No","No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes","No","No internet service"]
    )

    tech = st.selectbox(
        "Tech Support",
        ["Yes","No","No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        ["Yes","No"]
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=150.0,
    value=70.0
)

total = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)

# ---------------------------------------------------
# CREATE DATAFRAME
# ---------------------------------------------------

customer = pd.DataFrame({

    "gender":[gender],
    "seniorcitizen":[senior],
    "partner":[partner],
    "dependents":[dependents],
    "tenure":[tenure],
    "phoneservice":[phone],
    "multipleLines":[multiple],
    "internetservice":[internet],
    "onlinesecurity":[online_security],
    "onlinebackup":[online_backup],
    "techsupport":[tech],
    "contract":[contract],
    "paperlessbilling":[paperless],
    "paymentmethod":[payment],
    "monthlycharges":[monthly],
    "totalcharges":[total]

},index=[0])

# ---------------------------------------------------
# BUTTON
# ---------------------------------------------------

if st.button("Predict"):

    X = transformer.transform(customer)

    prediction = model.predict(X)[0]

    probability = model.predict_proba(X)[0][1]

    st.markdown("---")

    if prediction == 1:

        st.error("⚠ Customer will Churn")

    else:

        st.success("✅ Customer will Stay")

    st.metric(
        "Churn Probability",
        f"{probability*100:.2f}%"
    )

    st.progress(float(probability))