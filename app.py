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
    """Load the trained model and optional scaler from disk."""
    model_ = joblib.load("rf.pkl")

    # If the pickled object is **not** a Pipeline we also need a scaler
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
# Pre‑processing (mirror training pipeline)
# ---------------------------------------------
user_df_proc = user_df_raw.copy()
user_df_proc[categorical_cols] = user_df_proc[categorical_cols].replace(categorical_mapping)
if (external_scaler is not None) and (not uses_pipeline):
    user_df_proc[numerical_cols] = external_scaler.transform(user_df_proc[numerical_cols])

# ---------------------------------------------
# Inference & SHAP explanation
# ---------------------------------------------
if st.button("Predict"):
    # ---------------- Prediction ----------------
    proba = model.predict_proba(user_df_proc)[:, 1][0]
    st.success(f"Predicted risk of postoperative thrombosis: {proba * 100:.2f}%")

    # ---------------- Build SHAP explainer ----------------
    @st.cache_resource(show_spinner=False)
    def build_explainer(m):
        """Return a SHAP explainer that works for both pipelines and bare models."""
        try:
            return shap.Explainer(m)
        except Exception:
            # If the generic path fails (rare), fall back to the last estimator in Pipeline
            if isinstance(m, Pipeline):
                return shap.TreeExplainer(m.steps[-1][1])
            raise  # Re‑raise if it's some other unexpected error

    explainer = build_explainer(model)

    # ---------------- Compute SHAP values ----------------
    shap_result = explainer(user_df_proc)

    # For single‑sample prediction shap_result.base_values is scalar or array‑like length 1
    base_val = float(shap_result.base_values[0] if hasattr(shap_result.base_values, "__len__") else shap_result.base_values)
    shap_vec = shap_result.values[0]  # 1‑D array of per‑feature contributions

    # ---------------- Force plot ----------------
    st.subheader("Model Explanation (SHAP Force Plot)")

    # Provide raw feature values for annotation
    feature_vals = user_df_raw.iloc[0]

    # Draw static matplotlib force plot
    shap.plots.force(
        base_val,
        shap_vec,
        features=feature_vals,
        feature_names=feature_vals.index,
        matplotlib=True,
        show=False,
    )
    fig = plt.gcf()
    st.pyplot(fig)

    # ---------------- Download figure ----------------
    with st.expander("Download SHAP force plot"):
        st.download_button(
            label="Download PNG",
            data=_fig_to_png_bytes(fig),
            file_name="shap_force_plot.png",
            mime="image/png",
        )
