import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

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
# Load model & (optional) external scaler
# ---------------------------------------------
@st.cache_resource(show_spinner=False)
def load_assets():
    model_ = joblib.load("rf.pkl")
    scaler_ = None if isinstance(model_, Pipeline) else joblib.load("minmax_scaler.pkl")
    return model_, scaler_

model, external_scaler = load_assets()
uses_pipeline = isinstance(model, Pipeline)

# ---------------------------------------------
# UI
# ---------------------------------------------
st.title("Prediction Model for Thrombosis After Lung Transplantation")

user_inputs = {}
for feat, (ftype, default) in feature_defs.items():
    if ftype == "numerical":
        user_inputs[feat] = st.number_input(feat, value=float(default))
    else:
        user_inputs[feat] = st.selectbox(feat, feature_defs[feat][1], index=0)

user_df_raw = pd.DataFrame([user_inputs])

# ---------------------------------------------
# Pre‑processing
# ---------------------------------------------
user_df_proc = user_df_raw.copy()
user_df_proc[categorical_cols] = user_df_proc[categorical_cols].replace(categorical_mapping)
if not uses_pipeline:
    user_df_proc[numerical_cols] = external_scaler.transform(user_df_proc[numerical_cols])

# ---------------------------------------------
# Prediction & SHAP
# ---------------------------------------------
if st.button("Predict"):
    proba = model.predict_proba(user_df_proc)[:, 1][0]
    st.success(f"Predicted risk of postoperative thrombosis: {proba * 100:.2f}%")

    # -------- SHAP TreeExplainer --------
    tree_model = model.steps[-1][1] if uses_pipeline else model
    explainer  = shap.TreeExplainer(tree_model)
    shap_vals  = explainer.shap_values(user_df_proc)

    # 针对二分类：取正类 (=1)
    if isinstance(shap_vals, list):
        shap_vec = shap_vals[1][0]
        base_val = explainer.expected_value[1]
    else:  # ndarray
        shap_vec = shap_vals[0]
        base_val = explainer.expected_value[1] if hasattr(explainer, "expected_value") and isinstance(explainer.expected_value, (list, tuple)) else explainer.expected_value

    # -------- 绘制 SHAP 力图 (matplotlib) --------
    if isinstance(shap_vals, list):
        shap_array = shap_vals[1]              # (n_samples, n_features)
        base_val   = explainer.expected_value[1]
    else:
        shap_array = shap_vals                 # ndarray (n_samples, n_features)
        base_val   = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, tuple)) else explainer.expected_value

    try:
        plt.clf()
        force_fig = shap.force_plot(
            base_val,
            shap_array[0],                     # 取首样本 SHAP 向量
            user_df_raw.iloc[0],               # 特征原值 (Series)
            matplotlib=True,
            show=False,
        )
        st.pyplot(force_fig)
    except Exception as e:
        st.error(f"SHAP force plot failed: {e}")
