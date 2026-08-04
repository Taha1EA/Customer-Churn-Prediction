import streamlit as st
import pandas as pd
import plotly.express as px

# ----------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------

st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide"
)

# ----------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("data/Telco_Cus_Churn.csv")
    return df

df = load_data()

# ----------------------------------------------------
# PAGE TITLE
# ----------------------------------------------------

st.title("📊 Telecom Customer Dashboard")

st.markdown("""
Explore the IBM Telecom Customer Churn dataset.
""")

st.markdown("---")

# ----------------------------------------------------
# KPIs
# ----------------------------------------------------

customers = len(df)
churn_rate =  round(df["Churn"].map({"Yes": 1, "No": 0}).mean() * 100, 2)
avg_monthly = round(df["MonthlyCharges"].mean(),2)
avg_tenure = round(df["tenure"].mean(),2)

col1,col2,col3,col4 = st.columns(4)

col1.metric("Customers",customers)
col2.metric("Churn Rate",f"{churn_rate}%")
col3.metric("Avg Monthly Charges",f"${avg_monthly}")
col4.metric("Avg Tenure",avg_tenure)

st.markdown("---")

# ----------------------------------------------------
# DATASET PREVIEW
# ----------------------------------------------------

st.subheader("Dataset Preview")

st.dataframe(df)

st.markdown("---")

# ----------------------------------------------------
# CHURN DISTRIBUTION
# ----------------------------------------------------

st.subheader("Churn Distribution")

fig = px.pie(
    df,
    names="Churn",
    title="Customer Churn Distribution"
)

st.plotly_chart(fig,use_container_width=True)

# ----------------------------------------------------
# CONTRACT TYPE
# ----------------------------------------------------

st.subheader("Contract Types")

fig = px.histogram(
    df,
    x="Contract",
    color="Churn",
    barmode="group"
)

st.plotly_chart(fig,use_container_width=True)

# ----------------------------------------------------
# INTERNET SERVICE
# ----------------------------------------------------

st.subheader("Internet Service")

fig = px.histogram(
    df,
    x="InternetService",
    color="Churn"
)

st.plotly_chart(fig,use_container_width=True)

# ----------------------------------------------------
# MONTHLY CHARGES
# ----------------------------------------------------

st.subheader("Monthly Charges")

fig = px.histogram(
    df,
    x="MonthlyCharges",
    color="Churn",
    nbins=40
)

st.plotly_chart(fig,use_container_width=True)

# ----------------------------------------------------
# TENURE
# ----------------------------------------------------

st.subheader("Customer Tenure")

fig = px.histogram(
    df,
    x="tenure",
    color="Churn",
    nbins=40
)

st.plotly_chart(fig,use_container_width=True)

# ----------------------------------------------------
# DATASET STATISTICS
# ----------------------------------------------------

st.subheader("Dataset Statistics")

st.write(df.describe())

# ----------------------------------------------------
# CORRELATION MATRIX
# ----------------------------------------------------

st.subheader("Correlation")

corr = df.corr(numeric_only=True)

fig = px.imshow(
    corr,
    text_auto=True,
    aspect="auto"
)

st.plotly_chart(fig,use_container_width=True)