import streamlit as st
import pandas as pd
from pathlib import Path

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="About",
    layout="wide"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("About Customer Churn Prediction")

st.write(
    "This project predicts whether a telecom customer is likely to stop using the service. "
    "It uses customer demographic, account, service, and billing information to estimate churn risk."
)

st.markdown("---")

# ---------------------------------------------------
# PROJECT PURPOSE
# ---------------------------------------------------

st.subheader("Project Purpose")

st.write(
    """
    Customer churn is the percentage of customers who leave a company during a given time period.

    This application helps identify customers with a high probability of churn so that
    retention actions can be prioritized.

    The system provides:
    - Single customer prediction
    - Batch prediction from a CSV file
    - Model performance evaluation
    - Project and dataset explanation
    """
)

st.markdown("---")

# ---------------------------------------------------
# APPLICATION FILES
# ---------------------------------------------------

st.subheader("Application Files")

files_info = pd.DataFrame(
    {
        "File": [
            "predict_customer.py",
            "predict_customers.py",
            "model_performance.py",
            "about.py",
            "log_reg.pkl",
            "transformer.pkl",
            "Telco_Cus_Churn.csv",
        ],
        "Purpose": [
            "Predict churn for one customer using a form.",
            "Predict churn for multiple customers by uploading a CSV file.",
            "Evaluate model performance using metrics and charts.",
            "Explain the project, dataset, and model.",
            "Saved logistic regression model.",
            "Saved preprocessing transformer.",
            "Original telecom customer churn dataset.",
        ],
    }
)

st.table(files_info)

st.markdown("---")

# ---------------------------------------------------
# DATASET OVERVIEW
# ---------------------------------------------------

st.subheader("Dataset Overview")

DATA_FILE = "data/Telco_Cus_Churn.csv"

df = None

if Path(DATA_FILE).exists():
    try:
        df = pd.read_csv(DATA_FILE)
    except Exception as e:
        st.warning(f"Could not read {DATA_FILE}: {e}")

if df is not None and not df.empty:

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        len(df)
    )

    col2.metric(
        "Total Columns",
        df.shape[1]
    )

    if "Churn" in df.columns:
        churn_count = int((df["Churn"].astype(str).str.strip() == "Yes").sum())
        stay_count = len(df) - churn_count
        churn_rate = (churn_count / len(df)) * 100

        col3.metric(
            "Churn Customers",
            churn_count
        )

        col4.metric(
            "Churn Rate",
            f"{churn_rate:.2f}%"
        )

        st.markdown("### Target Distribution")

        target_df = pd.DataFrame(
            {
                "Class": ["Stay", "Churn"],
                "Count": [stay_count, churn_count],
            }
        )

        st.bar_chart(
            target_df.set_index("Class")
        )

    st.markdown("### Sample Data")
    st.dataframe(df.head(10))

else:
    st.info(
        "Telco_Cus_Churn.csv is not available in the current folder. "
        "Add the dataset file to see automatic dataset statistics."
    )

st.markdown("---")

# ---------------------------------------------------
# COLUMN DESCRIPTIONS
# ---------------------------------------------------

st.subheader("Column Descriptions")

column_info = pd.DataFrame(
    {
        "Column": [
            "customerID",
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "tenure",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "MonthlyCharges",
            "TotalCharges",
            "Churn",
        ],
        "Type": [
            "Identifier",
            "Categorical",
            "Numeric",
            "Categorical",
            "Categorical",
            "Numeric",
            "Categorical",
            "Categorical",
            "Categorical",
            "Categorical",
            "Categorical",
            "Categorical",
            "Categorical",
            "Categorical",
            "Categorical",
            "Categorical",
            "Categorical",
            "Categorical",
            "Numeric",
            "Numeric",
            "Target",
        ],
        "Description": [
            "Unique identifier for each customer.",
            "Customer gender.",
            "Whether the customer is a senior citizen. 0 means No, 1 means Yes.",
            "Whether the customer has a partner.",
            "Whether the customer has dependents.",
            "Number of months the customer has stayed with the company.",
            "Whether the customer has phone service.",
            "Whether the customer has multiple phone lines.",
            "Type of internet service.",
            "Whether the customer has online security.",
            "Whether the customer has online backup.",
            "Whether the customer has device protection.",
            "Whether the customer has tech support.",
            "Whether the customer has streaming TV.",
            "Whether the customer has streaming movies.",
            "Customer contract type.",
            "Whether the customer uses paperless billing.",
            "Customer payment method.",
            "Amount charged monthly.",
            "Total amount charged over the customer lifetime.",
            "Whether the customer churned. Yes means churn, No means stayed.",
        ],
    }
)

st.table(column_info)

st.markdown("---")

# ---------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------

