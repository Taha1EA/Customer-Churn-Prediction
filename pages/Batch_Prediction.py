import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Predict Customers",
    layout="wide"
)

# ---------------------------------------------------
# FILES
# ---------------------------------------------------

MODEL_FILE = "log_reg.pkl"
TRANSFORMER_FILE = "transformer.pkl"

EXPECTED_COLUMNS = [
    "gender",
    "seniorcitizen",
    "partner",
    "dependents",
    "tenure",
    "phoneservice",
    "multipleLines",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "techsupport",
    "contract",
    "paperlessbilling",
    "paymentmethod",
    "monthlycharges",
    "totalcharges",
]

# ---------------------------------------------------
# LOAD MODEL AND TRANSFORMER
# ---------------------------------------------------


def load_pickle(path):
    if not Path(path).exists():
        st.error(f"Missing file: {path}")
        st.stop()

    with open(path, "rb") as f:
        return pickle.load(f)


model = load_pickle(MODEL_FILE)
transformer = load_pickle(TRANSFORMER_FILE)

# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------


def normalize_columns(df):
    """
    Strip whitespace from column names and attempt to match
    case-insensitive column names to the expected column names.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    lower_to_original = {col.lower(): col for col in df.columns}
    rename_map = {}

    for expected in EXPECTED_COLUMNS:
        if expected not in df.columns and expected.lower() in lower_to_original:
            rename_map[lower_to_original[expected.lower()]] = expected

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def churn_label(value):
    """
    Convert model prediction to Churn / Stay.
    Works for 1/0, Yes/No, True/False style outputs.
    """
    if str(value).strip().lower() in {"1", "yes", "true", "churn"}:
        return "Churn"

    return "Stay"


# ---------------------------------------------------
# PAGE HEADER
# ---------------------------------------------------

st.title("Customer Churn Prediction - CSV Batch")

st.write(
    "Upload a CSV file containing multiple customers. "
    "One row must represent one customer."
)

st.markdown("---")

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload customer CSV",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    raw_df = normalize_columns(raw_df)

    if raw_df.empty:
        st.error("CSV is empty.")
        st.stop()

    st.subheader("Uploaded Data")
    st.dataframe(raw_df)

    missing_columns = [
        col for col in EXPECTED_COLUMNS
        if col not in raw_df.columns
    ]

    if missing_columns:
        st.error(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )
        st.stop()

    # ---------------------------------------------------
    # PREDICT BUTTON
    # ---------------------------------------------------

    if st.button("Predict All Customers"):

        try:
            df = raw_df[EXPECTED_COLUMNS].copy()

            # ---------------------------------------------------
            # BASIC CLEANING
            # ---------------------------------------------------

            df["seniorcitizen"] = (
                df["seniorcitizen"]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace(
                    {
                        "true": "1",
                        "false": "0",
                        "yes": "1",
                        "no": "0",
                    }
                )
            )

            numeric_columns = [
                "tenure",
                "monthlycharges",
                "totalcharges",
            ]

            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df["tenure"] = df["tenure"].fillna(0).astype(int)
            df["monthlycharges"] = df["monthlycharges"].fillna(0.0)

            missing_total = df["totalcharges"].isna()

            df.loc[missing_total, "totalcharges"] = (
                df.loc[missing_total, "tenure"]
                * df.loc[missing_total, "monthlycharges"]
            )

            df["totalcharges"] = df["totalcharges"].fillna(0.0)

            categorical_columns = [
                col for col in EXPECTED_COLUMNS
                if col not in numeric_columns
            ]

            for col in categorical_columns:
                df[col] = df[col].astype(str).str.strip()

            # ---------------------------------------------------
            # TRANSFORM AND PREDICT
            # ---------------------------------------------------

            X = transformer.transform(df)

            predictions = model.predict(X)
            probabilities = model.predict_proba(X)

            # ---------------------------------------------------
            # GET PROBABILITY FOR CHURN CLASS
            # ---------------------------------------------------

            prob_index = None

            if hasattr(model, "classes_"):
                classes = list(model.classes_)

                for target in [1, "1", "Yes", "yes", True]:
                    if target in classes:
                        prob_index = classes.index(target)
                        break

            if prob_index is None:
                if probabilities.shape[1] > 1:
                    prob_index = 1
                else:
                    prob_index = 0

            churn_probability = probabilities[:, prob_index]

            # ---------------------------------------------------
            # CREATE RESULT TABLE
            # ---------------------------------------------------

            result = raw_df.copy()

            result["churn_prediction"] = predictions
            result["churn_label"] = [
                churn_label(value)
                for value in predictions
            ]
            result["churn_probability_percent"] = (
                churn_probability * 100
            ).round(2)

            # ---------------------------------------------------
            # DISPLAY RESULTS
            # ---------------------------------------------------

            st.subheader("Prediction Results")
            st.dataframe(result)

            churn_count = int(
                (result["churn_label"] == "Churn").sum()
            )

            stay_count = int(
                (result["churn_label"] == "Stay").sum()
            )

            average_probability = float(
                result["churn_probability_percent"].mean()
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Customers",
                len(result)
            )

            col2.metric(
                "Predicted Churn",
                churn_count
            )

            col3.metric(
                "Predicted Stay",
                stay_count
            )

            col4.metric(
                "Average Churn Probability",
                f"{average_probability:.2f}%"
            )

            # ---------------------------------------------------
            # DOWNLOAD RESULTS
            # ---------------------------------------------------

            output_csv = result.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download predictions CSV",
                data=output_csv,
                file_name="customer_churn_predictions.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")