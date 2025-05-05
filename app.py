import io
import streamlit as st
import joblib
import pandas as pd
import shap
import numpy as np
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
# Utility
# ---------------------------------------------

def _fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    buf.seek(0)
    return buf.read()

# ---------------------------------------------
# Load model & (optional) external scaler
# ---------------------------------------------
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

# ---------------------------------------------
# UI – feature input
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
if (external_scaler is not None) and (not uses_pipeline):
    user_df_proc[numerical_cols] = external_scaler.transform(user_df_proc[numerical_cols])

# ---------------------------------------------
# Inference & SHAP explanation (probability scale)
# ---------------------------------------------
if st.button("Predict"):
    # ---------------- Prediction ----------------
    proba = model.predict_proba(user_df_proc)[:, 1][0]
    st.success(f"Predicted risk of postoperative thrombosis: {proba * 100:.2f}%")

    # ---------------- Build SHAP explainer ----------------
    @st.cache_resource(show_spinner=False)
    def build_explainer(_m):
        base_est = _m.steps[-1][1] if isinstance(_m, Pipeline) else _m
        background = user_df_proc  # minimal background for interventional mode
        return shap.TreeExplainer(
            base_est,
            data=background,
            model_output="probability",
            feature_perturbation="interventional",
        )

    explainer = build_explainer(model)

    # ---------------- Compute SHAP values safely ----------------
    shap_values = explainer.shap_values(user_df_proc)

        # ---- Robustly pick positive‑class vector & expected value ----
    if isinstance(shap_values, list):
        # TreeExplainer returns list in raw‑output mode — we requested probability, but stay safe
        pos_index = 1 if len(shap_values) > 1 else 0
        shap_vec = shap_values[pos_index][0]  # (n_features,)
        base_val = explainer.expected_value[pos_index] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    else:
        # ndarray shape can be (n_samples, n_features) OR (n_samples, n_features, n_outputs)
        arr = shap_values[0]
        if arr.ndim == 2 and arr.shape[1] == 2:
            # shape (n_features, 2) → choose positive class (index 1)
            shap_vec = arr[:, 1]
        elif arr.ndim == 3:
            # shape (n_features, n_outputs, something) uncommon, pick [:,1]
            shap_vec = arr[:, 1, ...].squeeze()
        else:
            shap_vec = arr  # already 1‑D

        base_val_raw = explainer.expected_value
        if isinstance(base_val_raw, (list, np.ndarray)) and len(base_val_raw) > 1:
            base_val = base_val_raw[1]  # positive class
        else:
            base_val = base_val_raw

    # Ensure we ended with a 1‑D contribution vector
    shap_vec = np.asarray(shap_vec).flatten()

    # Wrap into Explanation
    instance_exp = shap.Explanation(
        values=shap_vec,
        base_values=base_val,
        data=user_df_proc.iloc[0].values,
        feature_names=user_df_proc.columns,
    )(
        values=shap_vec,
        base_values=base_val,
        data=user_df_proc.iloc[0].values,
        feature_names=user_df_proc.columns,
    )

    # ====================================================
    # WATERFALL PLOT (probability scale)
    # ====================================================
    st.subheader("Model Explanation – SHAP Waterfall (Probability)")
    shap.plots.waterfall(instance_exp, max_display=15, show=False)
    fig_water = plt.gcf()
    st.pyplot(fig_water)

    with st.expander("Download SHAP waterfall (probability)"):
        st.download_button(
            label="Download PNG",
            data=_fig_to_png_bytes(fig_water),
            file_name="shap_waterfall_probability.png",
            mime="image/png",
        )

    # ====================================================
    # FORCE PLOT (probability scale)
    # ====================================================
    st.subheader("Model Explanation – SHAP Force (Probability)")

    shap.plots.force(
        base_val,
        shap_vec,
        features=user_df_proc.iloc[0],
        feature_names=user_df_proc.columns,
        matplotlib=True,
        show=False,
    )
    fig_force = plt.gcf()
    st.pyplot(fig_force)

    with st.expander("Download SHAP force (probability)"):
        st.download_button(
            label="Download PNG",
            data=_fig_to_png_bytes(fig_force),
            file_name="shap_force_probability.png",
            mime="image/png",
        )
