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
    """Serialize a Matplotlib figure as PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    buf.seek(0)
    return buf.read()

# ---------------------------------------------
# Load model & (optional) external scaler
# ---------------------------------------------
@st.cache_resource(show_spinner=False)
def load_assets():
    """Load model and optional scaler from disk."""
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
# Build SHAP explainer (probability scale, interventional perturbation)
# ---------------------------------------------
@st.cache_resource(show_spinner=False)
def build_explainer(_m):
    """Return a TreeExplainer that outputs *probabilities* (0‑1)."""
    base_est = _m.steps[-1][1] if isinstance(_m, Pipeline) else _m
    return shap.TreeExplainer(
        base_est,
        model_output="probability",
        feature_perturbation="interventional",
    )

explainer = build_explainer(model)

# ---------------------------------------------
# Streamlit UI – feature inputs
# ---------------------------------------------
st.title("Prediction Model for Thrombosis After Lung Transplantation")

user_inputs: dict = {}
for feat, (ftype, default) in feature_defs.items():
    if ftype == "numerical":
        user_inputs[feat] = st.number_input(feat, value=float(default))
    else:
        user_inputs[feat] = st.selectbox(feat, feature_defs[feat][1], index=0)

user_df_raw = pd.DataFrame([user_inputs])

# ---------------------------------------------
# Pre‑processing (mirror training pipeline)
# ---------------------------------------------
user_df_proc = user_df_raw.copy()
user_df_proc[categorical_cols] = user_df_proc[categorical_cols].replace(categorical_mapping)
if (external_scaler is not None) and (not uses_pipeline):
    user_df_proc[numerical_cols] = external_scaler.transform(user_df_proc[numerical_cols])

# ---------------------------------------------
# Prediction & SHAP explanation
# ---------------------------------------------
if st.button("Predict"):
    # ----- Prediction -----
    proba = model.predict_proba(user_df_proc)[:, 1][0]
    st.success(f"Predicted risk of postoperative thrombosis: {proba * 100:.2f}%")

    # ----- SHAP values -----
    shap_exp = explainer(user_df_proc)

    # Select explanation for single sample & positive class
    instance_exp = shap_exp[0]
    if instance_exp.values.ndim == 2:
        instance_exp = instance_exp[:, 1]

    # ====================================================
    # Waterfall plot (probability)
    # ====================================================
    st.subheader("SHAP Waterfall Plot (Probability scale)")
    shap.plots.waterfall(instance_exp, max_display=15, show=False)
    fig_water = plt.gcf()
    st.pyplot(fig_water)

    with st.expander("Download waterfall plot"):
        st.download_button(
            "Download PNG",
            _fig_to_png_bytes(fig_water),
            "shap_waterfall_plot.png",
            "image/png",
        )

    # ====================================================
    # Force plot (probability)
    # ====================================================
    st.subheader("SHAP Force Plot (Probability scale)")

    shap.plots.force(
        float(instance_exp.base_values),
        instance_exp.values,
        features=instance_exp.data,
        feature_names=instance_exp.feature_names,
        matplotlib=True,
        show=False,
    )
    fig_force = plt.gcf()
    st.pyplot(fig_force)

    with st.expander("Download force plot"):
        st.download_button(
            "Download PNG",
            _fig_to_png_bytes(fig_force),
            "shap_force_plot.png",
            "image/png",
        )
