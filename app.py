import io
import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

# ---------------------------------------------
# Page configuration
# ---------------------------------------------
st.set_page_config(page_title="Thrombosis Risk Prediction", layout="centered")

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

# ------------ 新增：显示名 → 训练名映射 ------------ #
internal_name_map = {
    "Age": "age",
    "Postoperative platelet count (x10⁹/L)": "post_PLT",
    "Postoperative BUN (μmol/L)": "post_BUN",
    "Day 1 postoperative antithrombin III activity (%)": "post_antithrombase_III_1",
    "NYHA": "NYHA",
    "HBP": "HBP",
    "Postoperative CRRT (Continuous Renal Replacement Therapy)": "post_CRRT",
    "Postoperative Anticoagulation": "anticoagulant_therapy",
}
# -------------------------------------------------

# Utility
def _fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    buf.seek(0)
    return buf.read()

# Load model & (optional) external scaler
@st.cache_resource(show_spinner=False)
def load_assets():
    model_ = joblib.load("rf.pkl")
    scaler_ = None
    if not isinstance(model_, Pipeline):
        try:
            scaler_ = joblib.load("minmax_scaler.pkl")
        except FileNotFoundError:
            pass
    return model_, scaler_

model, external_scaler = load_assets()
uses_pipeline = isinstance(model, Pipeline)

# UI – feature input
st.title("Prediction Model for Thrombosis After Lung Transplantation")

user_inputs = {}
for feat, (ftype, default) in feature_defs.items():
    if ftype == "numerical":
        user_inputs[feat] = st.number_input(feat, value=float(default))
    else:
        user_inputs[feat] = st.selectbox(feat, feature_defs[feat][1], index=0)

user_df_raw = pd.DataFrame([user_inputs])

# Pre‑processing
user_df_proc = user_df_raw.copy()
user_df_proc[categorical_cols] = user_df_proc[categorical_cols].replace(categorical_mapping)
if (external_scaler is not None) and (not uses_pipeline):
    user_df_proc[numerical_cols] = external_scaler.transform(user_df_proc[numerical_cols])

# Inference & SHAP explanation
if st.button("Predict"):
    proba = model.predict_proba(user_df_proc)[:, 1][0]
    st.success(f"Predicted risk of postoperative thrombosis: {proba * 100:.2f}%")

    @st.cache_resource(show_spinner=False)
    def build_explainer(_m):
        try:
            return shap.Explainer(_m)
        except Exception:
            if isinstance(_m, Pipeline):
                return shap.TreeExplainer(_m.steps[-1][1])
            raise

    explainer = build_explainer(model)
    shap_exp  = explainer(user_df_proc)
    instance_exp = shap_exp[0]
    if instance_exp.values.ndim == 2:
        instance_exp = instance_exp[:, 1]

    # ----------- 关键：替换特征名 -----------
    mapped_names = [internal_name_map.get(n, n) for n in instance_exp.feature_names]
    instance_exp.feature_names = mapped_names
    # ----------------------------------------

    # WATERFALL
    st.subheader("Model Explanation – SHAP Waterfall Plot")
    shap.plots.waterfall(instance_exp, max_display=15, show=False)
    fig_water = plt.gcf()
    st.pyplot(fig_water)
    with st.expander("Download SHAP waterfall plot"):
        st.download_button("Download PNG", _fig_to_png_bytes(fig_water),
                           "shap_waterfall_plot.png", "image/png")

    # FORCE
    st.subheader("Model Explanation – SHAP Force Plot")
    shap.plots.force(
        float(instance_exp.base_values if hasattr(instance_exp.base_values, "__len__") else instance_exp.base_values),
        instance_exp.values,
        features=instance_exp.data,
        feature_names=instance_exp.feature_names,  # 已替换
        matplotlib=True,
        show=False,
    )
    fig_force = plt.gcf()
    st.pyplot(fig_force)
    with st.expander("Download SHAP force plot"):
        st.download_button("Download PNG", _fig_to_png_bytes(fig_force),
                           "shap_force_plot.png", "image/png")
