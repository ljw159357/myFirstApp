import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ---------------------------------------------
# Page configuration
# ---------------------------------------------
st.set_page_config(page_title="Thrombosis Prediction", layout="centered")

# ---------------------------------------------
# Feature definitions & mappings
# ---------------------------------------------
feature_defs = {
    "Age": ("numerical", 0.0),
    "Postoperative platelet count (x10⁹/L)": ("numerical", 0.0),
    "Postoperative BUN (μmol/L)": ("numerical", 0.0),
    "Day 1 postoperative antithrombin III activity (%)": ("numerical", 0.0),
    "NYHA": ("categorical", ["＞2", "≤2"]),
    "HBP": ("categorical", ["Yes", "No"]),
    "Postoperative CRRT (Continuous Renal Replacement Therapy)": ("categorical", ["Yes", "No"]),
    "Postoperative Anticoagulation": ("categorical", ["Yes", "No"]),
}

categorical_mapping = {
    "NYHA": {"＞2": 1, "≤2": 0},
    "HBP": {"Yes": 1, "No": 0},
    "Postoperative CRRT (Continuous Renal Replacement Therapy)": {"Yes": 1, "No": 0},
    "Postoperative Anticoagulation": {"Yes": 1, "No": 0},
}

numerical_cols = [k for k, v in feature_defs.items() if v[0] == "numerical"]
categorical_cols = [k for k, v in feature_defs.items() if v[0] == "categorical"]

# ---------------------------------------------
# Load model & scaler
# ---------------------------------------------
@st.cache_resource(show_spinner=False)
def load_assets():
    model_ = joblib.load("rf.pkl")
    scaler_ = joblib.load("minmax_scaler.pkl")
    return model_, scaler_

model, scaler = load_assets()

# ---------------------------------------------
# UI
# ---------------------------------------------
st.title("Prediction Model for Thrombosis After Lung Transplantation")

user_inputs = {}
for feat, (ftype, default) in feature_defs.items():
    if ftype == "numerical":
        user_inputs[feat] = st.number_input(feat, value=float(default))
    else:
        user_inputs[feat] = st.selectbox(feat, default, index=0)

user_df_raw = pd.DataFrame([user_inputs])

# ---------------------------------------------
# Pre‑processing (use *training‑time* scaler!)
# ---------------------------------------------
user_df = user_df_raw.copy()
user_df[categorical_cols] = user_df[categorical_cols].replace(categorical_mapping)
user_df[numerical_cols] = scaler.transform(user_df[numerical_cols])

# ---------------------------------------------
# Prediction
# ---------------------------------------------
if st.button("Predict"):
    proba = model.predict_proba(user_df)[:, 1][0]
    st.success(f"Predicted risk of postoperative thrombosis: {proba * 100:.2f}%")

    # -------- SHAP explanation --------
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(user_df)

    base_value = explainer.expected_value[1] if isinstance(shap_values, list) else explainer.expected_value
    shap_vec   = shap_values[1][0]          if isinstance(shap_values, list) else shap_values[0]

    # 优先用 HTML 版本（交互效果更好）；若环境不支持再退回 matplotlib
    try:
        shap_html = shap.plots.force(
            base_value,
            shap_vec,
            features=user_df_raw,
        ).html()
        st.components.v1.html(shap_html, height=300, scrolling=True)
    except Exception:
        force_fig = shap.plots.force(
            base_value,
            shap_vec,
            features=user_df_raw,
            matplotlib=True,
            show=False,
        )
        st.pyplot(force_fig)
