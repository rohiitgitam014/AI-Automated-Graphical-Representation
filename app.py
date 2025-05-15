import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import re

# ────────────────────────────────────────────────────────────────────────────────
# Page configuration & title
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🤖 AI Auto-Visualizer (CSV-Powered)")

# ────────────────────────────────────────────────────────────────────────────────
# File upload
# ────────────────────────────────────────────────────────────────────────────────
file = st.file_uploader("📤 Upload CSV File", type=["csv"])

# ────────────────────────────────────────────────────────────────────────────────
# Helper: convert size / percent strings → float (GB or %)
# ────────────────────────────────────────────────────────────────────────────────
def convert_size(val):
    try:
        val = str(val).strip().upper()
        if val.endswith("G"):   return float(val[:-1])                     # GB
        if val.endswith("M"):   return float(val[:-1]) / 1024              # MB → GB
        if val.endswith("K"):   return float(val[:-1]) / (1024 * 1024)     # KB → GB
        if val.endswith("T"):   return float(val[:-1]) * 1024              # TB → GB
        if val.endswith("%"):   return float(val[:-1])                     # %
        return float(val)                                                  # plain number
    except Exception:
        return np.nan

# ────────────────────────────────────────────────────────────────────────────────
# Main app logic
# ────────────────────────────────────────────────────────────────────────────────
if file:
    # 1️⃣ RAW PREVIEW -----------------------------------------------------------
    df = pd.read_csv(file)
    st.subheader("🔍 Raw Data Preview")
    st.dataframe(df.head())

    # 2️⃣ CLEAN DATA -----------------------------------------------------------
    df_cleaned = df.copy()
    size_pat = r"\d+(\.\d+)?[GMKT%]"

    for col in df.columns:
        if df[col].dtype == "object" and df[col].str.contains(size_pat, na=False).any():
            df_cleaned[col] = df[col].apply(convert_size)

    df_cleaned = df_cleaned.loc[:, ~df_cleaned.columns.str.contains("^Unnamed")]

    st.subheader("🧹 Cleaned Dataset (auto-converted)")
    st.dataframe(df_cleaned.head())

    # 3️⃣ COLUMN TYPES ---------------------------------------------------------
    numeric_cols     = df_cleaned.select_dtypes(include="number").columns.tolist()
    categorical_cols = df_cleaned.select_dtypes(include="object").columns.tolist()

    # 4️⃣ UNIVARIATE HISTOGRAMS ------------------------------------------------
    if numeric_cols:
        st.subheader("📊 Univariate Analysis")
        for col in numeric_cols:
            if df_cleaned[col].dropna().nunique() > 1:
                st.markdown(f"**Histogram of `{col}`**")
                fig = px.histogram(df_cleaned, x=col)
                st.plotly_chart(fig, use_container_width=True)

    # 5️⃣ BIVARIATE SCATTER PLOTS ---------------------------------------------
    if len(numeric_cols) >= 2:
        st.subheader("📈 Bivariate Analysis")
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                x_col, y_col = numeric_cols[i], numeric_cols[j]
                if (df_cleaned[x_col].dropna().nunique() > 1
                        and df_cleaned[y_col].dropna().nunique() > 1):
                    st.markdown(f"**Scatter Plot: `{x_col}` vs `{y_col}`**")
                    fig = px.scatter(df_cleaned, x=x_col, y=y_col)
                    st.plotly_chart(fig, use_container_width=True)

    # 6️⃣ BASIC BAR CHARTS (NO AGGREGATION) ------------------------------------
    if categorical_cols and numeric_cols:
        st.subheader("📊 Bar Charts: Categorical vs Numeric (Raw Values)")
        for cat_col in categorical_cols:
            for num_col in numeric_cols:
                tmp = df_cleaned[[cat_col, num_col]].dropna()
                if tmp.empty or tmp[cat_col].nunique() > 50:
                    continue        # skip empty or high-cardinality
                st.markdown(f"**Bar Chart: `{num_col}` by `{cat_col}`**")
                fig = px.bar(tmp, x=cat_col, y=num_col,
                             title=f"{num_col} per {cat_col} (Raw Values)")
                st.plotly_chart(fig, use_container_width=True)

    # 7️⃣ CORRELATION HEATMAP --------------------------------------------------
    if len(numeric_cols) >= 2:
        valid = [c for c in numeric_cols if df_cleaned[c].dropna().nunique() > 1]
        if len(valid) >= 2:
            st.subheader("🔥 Correlation Heatmap")
            corr = df_cleaned[valid].corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    # 8️⃣ DOWNLOAD CLEANED CSV -------------------------------------------------
    st.subheader("📤 Download Cleaned CSV")
    csv_data = df_cleaned.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Cleaned Data",
        data=csv_data,
        file_name="cleaned_data.csv",
        mime="text/csv",
    )

