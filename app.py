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
    # Model prediction
    proba = model.predict_proba(user_df_proc)[:, 1][0]
    st.success(f"Predicted risk of postoperative thrombosis: {proba * 100:.2f}%")

    # -------------------------------------------------
    # SHAP explanation (force plot)
    # -------------------------------------------------
    st.subheader("Model Explanation (SHAP)")

    # Build SHAP explainer – works for both pipelines and raw estimators
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(user_df_proc)
    except Exception:
        # Fallback in case the model is a Pipeline and TreeExplainer fails
        base_model = model.steps[-1][1] if uses_pipeline else model
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(user_df_proc if not uses_pipeline else model[:-1].transform(user_df_raw))

    # For binary classification, shap_values is a list [class0, class1]
    class_index = 1  # positive (thrombosis) class
    expected_value = explainer.expected_value[class_index] if isinstance(explainer.expected_value, (list, tuple)) else explainer.expected_value
    shap_sample_values = shap_values[class_index][0] if isinstance(shap_values, list) else shap_values[0]

    # Generate force plot (static matplotlib figure)
    shap.force_plot(expected_value, shap_sample_values, user_df_raw.iloc[0], matplotlib=True, show=False)
    fig = plt.gcf()
    st.pyplot(fig)

    # Optional: offer download of the figure
    with st.expander("Download SHAP force plot"):
        st.download_button(
            label="Download PNG",
            data=fig_to_png_bytes(fig),
            file_name="shap_force_plot.png",
            mime="image/png",
        )

def fig_to_png_bytes(fig):
    """Convert a Matplotlib figure to PNG bytes."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    buf.seek(0)
    return buf.read()
