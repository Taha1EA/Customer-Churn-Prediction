import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📱",
    layout="wide"
)

# Title
st.title("Telecom Customer Churn Prediction")

st.markdown("---")

# Description
st.markdown("""
## Welcome

This application predicts whether a telecom customer is likely to **churn** (leave the company)
using a **Machine Learning** model trained on the **IBM Telco Customer Churn Dataset**.

The application allows you to:

-  Explore the dataset
-  Predict churn for a single customer
-  Predict churn for multiple customers (CSV upload)
-  View model performance
-  Understand business insights
""")

st.markdown("---")

# Project Information
st.subheader(" Project Goal")

st.write("""
Predict whether a telecom customer is likely to leave the company
based on demographic information, subscribed services,
contract type, and billing information.
""")



st.markdown("---")

# Dataset Information
st.subheader(" Dataset")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Customers", "7,043")

with c2:
    st.metric("Features", "20")

with c3:
    st.metric("Target", "Churn")

st.info("""
Dataset: IBM Telco Customer Churn Dataset

The dataset contains customer demographics, services,
account information, and whether the customer churned.
""")


