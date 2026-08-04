import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    ConfusionMatrixDisplay,
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Model Performance",
    layout="wide"
)

# ---------------------------------------------------
# FILES
# ---------------------------------------------------

MODEL_FILE = "log_reg.pkl"
TRANSFORMER_FILE = "transformer.pkl"
DATA_FILE = "data/Telco_Cus_Churn.csv"

for file_name in [MODEL_FILE, TRANSFORMER_FILE, DATA_FILE]:
    if not Path(file_name).exists():
        st.error(f"Missing file: {file_name}")
        st.stop()

# ---------------------------------------------------
# LOAD MODEL AND TRANSFORMER
# ---------------------------------------------------


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


model = load_pickle(MODEL_FILE)
transformer = load_pickle(TRANSFORMER_FILE)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

try:
    df = pd.read_csv(DATA_FILE)
except Exception as e:
    st.error(f"Could not read {DATA_FILE}: {e}")
    st.stop()

df.columns = df.columns.str.strip()

st.title("Customer Churn Model Performance")

st.write(
    "This page evaluates the saved churn model using Telco_Cus_Churn.csv."
)

st.markdown("---")

# ---------------------------------------------------
# BASIC DATA CHECKS
# ---------------------------------------------------

if df.empty:
    st.error("The dataset is empty.")
    st.stop()

if "Churn" not in df.columns:
    st.error("The dataset does not contain a Churn column.")
    st.stop()

df["Churn"] = df["Churn"].astype(str).str.strip()
df["Churn_Binary"] = df["Churn"].map({"Yes": 1, "No": 0})

invalid_rows = int(df["Churn_Binary"].isna().sum())

if invalid_rows > 0:
    st.warning(
        f"Removed {invalid_rows} rows with invalid Churn values."
    )
    df = df.dropna(subset=["Churn_Binary"])

if df.empty:
    st.error("No valid rows remain after cleaning Churn values.")
    st.stop()

df["Churn_Binary"] = df["Churn_Binary"].astype(int)

# ---------------------------------------------------
# CLEAN NUMERIC COLUMNS
# ---------------------------------------------------

numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "TotalCharges" in df.columns:
    missing_total = df["TotalCharges"].isna()

    if "tenure" in df.columns and "MonthlyCharges" in df.columns:
        df.loc[missing_total, "TotalCharges"] = (
            df.loc[missing_total, "tenure"]
            * df.loc[missing_total, "MonthlyCharges"]
        )

    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

if "tenure" in df.columns:
    df["tenure"] = df["tenure"].fillna(0)

if "MonthlyCharges" in df.columns:
    df["MonthlyCharges"] = df["MonthlyCharges"].fillna(0.0)

# ---------------------------------------------------
# RESOLVE FEATURE COLUMNS
# ---------------------------------------------------


def resolve_features(dataframe, transformer_object):
    """
    Try to use the exact feature names stored in the transformer.
    If that fails, use all columns except ID and target.
    """

    excluded_columns = {
        "customerid",
        "churn",
        "churn_binary",
    }

    if hasattr(transformer_object, "feature_names_in_"):
        required_columns = [
            col
            for col in list(transformer_object.feature_names_in_)
            if col.lower() not in excluded_columns
        ]

        if required_columns:
            lower_to_actual = {
                col.lower(): col
                for col in dataframe.columns
            }

            rename_map = {}
            resolved_columns = []
            missing_columns = []

            for required_col in required_columns:
                if required_col in dataframe.columns:
                    resolved_columns.append(required_col)

                elif required_col.lower() in lower_to_actual:
                    actual_col = lower_to_actual[required_col.lower()]
                    rename_map[actual_col] = required_col
                    resolved_columns.append(required_col)

                else:
                    missing_columns.append(required_col)

            if not missing_columns:
                resolved_df = dataframe.rename(columns=rename_map)
                return resolved_df[resolved_columns].copy(), resolved_columns

    fallback_columns = [
        col
        for col in dataframe.columns
        if col.lower() not in excluded_columns
    ]

    return dataframe[fallback_columns].copy(), fallback_columns


X, feature_columns = resolve_features(df, transformer)
y = df["Churn_Binary"]

for col in X.select_dtypes(include=["object"]).columns:
    X[col] = X[col].fillna("Missing").astype(str).str.strip()

if len(feature_columns) == 0:
    st.error("No feature columns could be resolved.")
    st.stop()

# ---------------------------------------------------
# SIDEBAR OPTIONS
# ---------------------------------------------------

st.sidebar.header("Evaluation Settings")

test_size = st.sidebar.slider(
    "Test split size",
    min_value=0.10,
    max_value=0.50,
    value=0.20,
    step=0.05,
)

random_state = st.sidebar.number_input(
    "Random state",
    min_value=0,
    max_value=10000,
    value=42,
    step=1,
)

threshold = st.sidebar.slider(
    "Churn probability threshold",
    min_value=0.00,
    max_value=1.00,
    value=0.50,
    step=0.01,
)

run_evaluation = st.sidebar.button("Run Evaluation")

# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------


def get_positive_class_index(model_object):
    """
    Return the probability column index for the positive class.
    Positive class is assumed to be 1, Yes, True, or Churn.
    """

    if hasattr(model_object, "classes_"):
        classes = list(model_object.classes_)
        normalized_classes = [str(c).strip().lower() for c in classes]

        for target in ["1", "yes", "true", "churn"]:
            if target in normalized_classes:
                return normalized_classes.index(target)

        if len(classes) > 1:
            return 1

        return 0

    return 1


def convert_predictions_to_binary(values):
    """
    Convert model predictions to 0/1 if they are returned as strings.
    """

    binary_values = []

    for value in values:
        normalized_value = str(value).strip().lower()

        if normalized_value in {"1", "yes", "true", "churn"}:
            binary_values.append(1)
        else:
            binary_values.append(0)

    return np.array(binary_values)


def evaluate_split(X_transformed, y_true):
    """
    Evaluate one data split.
    """

    y_true = np.asarray(y_true).astype(int)

    y_prob = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_transformed)

        if probabilities.ndim == 1:
            y_prob = probabilities
        else:
            positive_index = get_positive_class_index(model)

            if probabilities.shape[1] > positive_index:
                y_prob = probabilities[:, positive_index]
            else:
                y_prob = probabilities[:, 0]

        y_pred = (y_prob >= threshold).astype(int)

    else:
        raw_predictions = model.predict(X_transformed)
        y_pred = convert_predictions_to_binary(raw_predictions)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    auc = None

    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = None

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=["Stay", "Churn"],
        zero_division=0
    )

    return {
        "true": y_true,
        "pred": y_pred,
        "prob": y_prob,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion": cm,
        "report": report,
    }


def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(5, 4))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Stay", "Churn"]
    )

    display.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title(title)

    return fig


def plot_roc_curve(y_true, y_prob, title):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(
        fpr,
        tpr,
        label=f"AUC = {auc_value:.3f}"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray"
    )

    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

    return fig


def render_metrics(metrics, title):
    st.subheader(title)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{metrics['accuracy']:.3f}"
    )

    col2.metric(
        "Precision",
        f"{metrics['precision']:.3f}"
    )

    col3.metric(
        "Recall",
        f"{metrics['recall']:.3f}"
    )

    col4.metric(
        "F1 Score",
        f"{metrics['f1']:.3f}"
    )

    if metrics["auc"] is not None:
        st.metric(
            "ROC AUC",
            f"{metrics['auc']:.3f}"
        )
    else:
        st.warning("ROC AUC is not available for this model.")

    st.markdown("#### Classification Report")
    st.code(metrics["report"], language=None)

    st.markdown("#### Confusion Matrix")
    cm_fig = plot_confusion_matrix(
        metrics["confusion"],
        f"{title} Confusion Matrix"
    )
    st.pyplot(cm_fig)
    plt.close(cm_fig)

    if metrics["prob"] is not None:
        try:
            roc_fig = plot_roc_curve(
                metrics["true"],
                metrics["prob"],
                f"{title} ROC Curve"
            )
            st.pyplot(roc_fig)
            plt.close(roc_fig)
        except Exception:
            st.warning("Could not draw ROC curve.")


# ---------------------------------------------------
# DATASET SUMMARY
# ---------------------------------------------------

st.subheader("Dataset Summary")

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

churn_count = int(y.sum())
stay_count = int(len(y) - churn_count)
churn_rate = float(y.mean() * 100)

summary_col1.metric(
    "Total Customers",
    len(df)
)

summary_col2.metric(
    "Churn Customers",
    churn_count
)

summary_col3.metric(
    "Stay Customers",
    stay_count
)

summary_col4.metric(
    "Churn Rate",
    f"{churn_rate:.2f}%"
)

with st.expander("View resolved feature columns"):
    st.write(feature_columns)

st.markdown("---")

# ---------------------------------------------------
# EVALUATION
# ---------------------------------------------------

if run_evaluation:

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=int(random_state),
            stratify=y
        )

    except Exception as e:
        st.error(f"Could not split the data: {e}")
        st.stop()

    try:
        X_train_transformed = transformer.transform(X_train)
        X_test_transformed = transformer.transform(X_test)

    except Exception as e:
        st.error(f"Transformer failed: {e}")
        st.stop()

    try:
        train_metrics = evaluate_split(
            X_train_transformed,
            y_train
        )

        test_metrics = evaluate_split(
            X_test_transformed,
            y_test
        )

    except Exception as e:
        st.error(f"Model evaluation failed: {e}")
        st.stop()

    st.markdown("---")

    tab_test, tab_train = st.tabs(["Test Performance", "Train Performance"])

    with tab_test:
        render_metrics(
            test_metrics,
            "Test Set"
        )

    with tab_train:
        render_metrics(
            train_metrics,
            "Train Set"
        )

    # ---------------------------------------------------
    # DOWNLOAD REPORT
    # ---------------------------------------------------

    report_lines = []

    report_lines.append("Customer Churn Model Performance Report")
    report_lines.append("=" * 50)
    report_lines.append(f"Dataset file: {DATA_FILE}")
    report_lines.append(f"Model file: {MODEL_FILE}")
    report_lines.append(f"Transformer file: {TRANSFORMER_FILE}")
    report_lines.append(f"Total rows: {len(df)}")
    report_lines.append(f"Churn rows: {churn_count}")
    report_lines.append(f"Stay rows: {stay_count}")
    report_lines.append(f"Churn rate: {churn_rate:.2f}%")
    report_lines.append(f"Test split size: {test_size}")
    report_lines.append(f"Random state: {int(random_state)}")
    report_lines.append(f"Churn probability threshold: {threshold}")
    report_lines.append("")

    for split_name, metrics in [
        ("TEST", test_metrics),
        ("TRAIN", train_metrics),
    ]:
        report_lines.append(f"{split_name} METRICS")
        report_lines.append("-" * 30)
        report_lines.append(f"Accuracy: {metrics['accuracy']:.4f}")
        report_lines.append(f"Precision: {metrics['precision']:.4f}")
        report_lines.append(f"Recall: {metrics['recall']:.4f}")
        report_lines.append(f"F1 Score: {metrics['f1']:.4f}")

        if metrics["auc"] is not None:
            report_lines.append(f"ROC AUC: {metrics['auc']:.4f}")
        else:
            report_lines.append("ROC AUC: Not available")

        report_lines.append("")
        report_lines.append("Classification Report:")
        report_lines.append(metrics["report"])
        report_lines.append("")

    report_text = "\n".join(report_lines)

    st.download_button(
        "Download Evaluation Report",
        data=report_text.encode("utf-8"),
        file_name="model_performance_report.txt",
        mime="text/plain"
    )