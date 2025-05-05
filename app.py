import streamlit as st
import joblib
import pandas as pd
import shap
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
    # 外部 scaler 只有在模型不是 Pipeline 时才需要
    scaler_ = None
    if not isinstance(model_, Pipeline):
        scaler_ = joblib.load("minmax_scaler.pkl")
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
        user_inputs[feat] = st.selectbox(feat, default, index=0)

user_df_raw = pd.DataFrame([user_inputs])

# ---------------------------------------------
# ---------------------------------------------
# Pre‑processing -------------------------------------------------------------
# 1. 将所有分类特征先映射为数字，保证 DataFrame 无字符串；
# 2. 如果模型不是 Pipeline，则对数值列做外部 scaler.transform；
# ---------------------------------------------------------------------------
user_df_proc = user_df_raw.copy()
user_df_proc[categorical_cols] = user_df_proc[categorical_cols].replace(categorical_mapping)

if not uses_pipeline:
    user_df_proc[numerical_cols] = external_scaler.transform(user_df_proc[numerical_cols])
# ---------------------------------------------------------------------------
if uses_pipeline:
    user_df_proc = user_df_raw.copy()
else:
    user_df_proc = user_df_raw.copy()
    user_df_proc[categorical_cols] = user_df_proc[categorical_cols].replace(categorical_mapping)
    user_df_proc[numerical_cols] = external_scaler.transform(user_df_proc[numerical_cols])

# ---------------------------------------------
# Prediction
# ---------------------------------------------
if st.button("Predict"):
    proba = model.predict_proba(user_df_proc)[:, 1][0]
    st.success(f"Predicted risk of postoperative thrombosis: {proba * 100:.2f}%")

    # -------- SHAP explanation --------
    # 1) 取出可被 TreeExplainer 处理的树模型；
    # 2) 计算 shap_values，兼容 ndarray / list 两种返回格式。
    if uses_pipeline:
        # Pipeline: 取最后一步作为树模型
        tree_model = model.steps[-1][1]
    else:
        tree_model = model

    # 对于 Pipeline，需要输入已经经过前处理的矩阵；否则直接用 user_df_proc
    X_explain = user_df_proc

    try:
        explainer = shap.TreeExplainer(tree_model)
        shap_vals = explainer.shap_values(X_explain)
    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")
        st.stop()

    # --- 兼容不同返回格式 ---
    if isinstance(shap_vals, list):
        shap_vec = shap_vals[1][0]  # 正类
        base_val = explainer.expected_value[1]
    else:  # ndarray
        shap_vec = shap_vals[0]
        # expected_value 可能是标量或长度为2的数组
        if isinstance(explainer.expected_value, (list, tuple)):
            base_val = explainer.expected_value[1]
        else:
            base_val = explainer.expected_value

    # -------- Force plot：HTML 优先，matplotlib 兜底 --------
    try:
        shap_html = shap.plots.force(
            base_val,
            shap_vec,
            features=user_df_raw,
        ).html()
        st.components.v1.html(shap_html, height=300, scrolling=True)
    except Exception:
        force_fig = shap.plots.force(
            base_val,
            shap_vec,
            features=user_df_raw,
            matplotlib=True,
            show=False,
        )
        st.pyplot(force_fig)
