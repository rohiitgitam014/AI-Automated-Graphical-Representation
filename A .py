import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import os
import google.generativeai as genai

# ────────────────────────────────────────────────────────────────────────────────
# Configure Gemini (set your API key as environment variable GENAI_API_KEY)
# ────────────────────────────────────────────────────────────────────────────────
GENAI_API_KEY = "AIzaSyCpu_OmvM5ElHNsT89SwJ1umKEUUj5j_h4"  # Replace with your actual API key
USE_GEMINI = GENAI_API_KEY not in (None, "")
if USE_GEMINI:
    genai.configure(api_key=  GENAI_API_KEY )
    gemini_model = genai.GenerativeModel("gemini-2.0-flash")



def generate_summary(prompt: str) -> str:
    """Return Gemini summary or placeholder if disabled."""
    if not USE_GEMINI:
        return "*(Provide your free Gemini API key in the sidebar to enable AI summaries)*"
    try:
        resp = gemini_model.generate_content(prompt, generation_config={"temperature": 0.2})
        return resp.text
    except Exception as exc:
        return f"*Summary unavailable: {exc}*"

# ────────────────────────────────────────────────────────────────────────────────
# Page configuration & title
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🤖 AI Auto‑Visualizer (CSV‑Powered) + Gemini Insights")

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

    st.subheader("🧹 Cleaned Dataset (auto‑converted)")
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

                # AI Summary
                stats = df_cleaned[col].describe()
                prompt = f"You are a data analyst. Provide a concise (max 3 sentences) insight summary for a histogram of the column '{col}'. Here are descriptive stats:\n{stats.to_string()}"
                summary = generate_summary(prompt)
                st.markdown(f"> **AI Summary:** {summary}")

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

                    corr_val = df_cleaned[[x_col, y_col]].corr().iloc[0,1]
                    prompt = (
                        f"You are a data analyst. Write a short insight (max 3 sentences) " 
                        f"about a scatter plot between '{x_col}' (x‑axis) and '{y_col}' (y‑axis). " 
                        f"The Pearson correlation is {corr_val:.2f}. Mention strength and direction.")
                    summary = generate_summary(prompt)
                    st.markdown(f"> **AI Summary:** {summary}")

    # 6️⃣ BASIC BAR CHARTS (NO AGGREGATION) ------------------------------------
    if categorical_cols and numeric_cols:
        st.subheader("📊 Bar Charts: Categorical vs Numeric (Raw Values)")
        for cat_col in categorical_cols:
            for num_col in numeric_cols:
                tmp = df_cleaned[[cat_col, num_col]].dropna()
                if tmp.empty or tmp[cat_col].nunique() > 50:
                    continue        # skip empty or high‑cardinality
                st.markdown(f"**Bar Chart: `{num_col}` by `{cat_col}`**")
                fig = px.bar(tmp, x=cat_col, y=num_col,
                             title=f"{num_col} per {cat_col} (Raw Values)")
                st.plotly_chart(fig, use_container_width=True)

                top_vals = tmp.groupby(cat_col)[num_col].mean().sort_values(ascending=False).head(5)
                prompt = (
                    f"You are a data analyst. Summarize insights (max 3 sentences) from a bar chart of '{num_col}' by '{cat_col}'. " 
                    f"Here are the top 5 category means:\n{top_vals.to_string()}")
                summary = generate_summary(prompt)
                st.markdown(f"> **AI Summary:** {summary}")

    # 7️⃣ CORRELATION HEATMAP --------------------------------------------------
    if len(numeric_cols) >= 2:
        valid = [c for c in numeric_cols if df_cleaned[c].dropna().nunique() > 1]
        if len(valid) >= 2:
            st.subheader("🔥 Correlation Heatmap")
            corr = df_cleaned[valid].corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

            # Summarize strongest correlations
            corr_pairs = (
                corr.where(~np.tril(np.ones(corr.shape)).astype(bool))
                .stack()
                .sort_values(ascending=False)
            )
            top_corr = corr_pairs.head(5)
            prompt = ("You are a data analyst. Provide a concise narrative (max 3 sentences) highlighting the strongest correlations " 
                      "observed in the heatmap. Here are the top correlation pairs:\n" 
                      f"{top_corr.to_string()}")
            summary = generate_summary(prompt)
            st.markdown(f"> **AI Summary:** {summary}")

    # 8️⃣ DOWNLOAD CLEANED CSV -------------------------------------------------
    st.subheader("📤 Download Cleaned CSV")
    csv_data = df_cleaned.to_csv(index=False).encode("utf‑8")
    st.download_button(
        "Download Cleaned Data",
        data=csv_data,
        file_name="cleaned_data.csv",
        mime="text/csv",
    )
