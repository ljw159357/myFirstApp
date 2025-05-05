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
    def build_explainer(_m):
        """Return a SHAP *TreeExplainer* forced to probability scale.

        * Why not `shap.Explainer`?
          In SHAP ≤0.47 当传入 `Pipeline` + `model_output="probability"` 时，
          `Explainer` 会内部调用 `TreeExplainer`，但 **忽略我们指定的
          `feature_perturbation`**，退回默认 `tree_path_dependent` → 与概率
          输出冲突，从而报 *Only model_output="raw" is supported...*。

        * 解决策略：
          - 拆出 Pipeline 的末级树模型（RandomForestClassifier）。
          - 直接构造 `TreeExplainer`，同时显式设置
            `feature_perturbation="interventional"` 与
            `model_output="probability"`，二者兼容。
        """
        # 拆出底层树模型（假设在步骤末尾）
        base_est = _m.steps[-1][1] if isinstance(_m, Pipeline) else _m

        return shap.TreeExplainer(
            base_est,
            model_output="probability",
            feature_perturbation="interventional",
        )

    explainer = build_explainer(model)

    # ---------------- Compute SHAP values ----------------
    shap_exp = explainer(user_df_proc)

    # ---------------- Select single‑output explanation ----------------
    instance_exp = shap_exp[0]
    if instance_exp.values.ndim == 2:  # (n_features, n_outputs)
        instance_exp = instance_exp[:, 1]  # positive class

    # ====================================================
    # WATERFALL PLOT – probability scale
    # ====================================================
    st.subheader("Model Explanation – SHAP Waterfall (Probability)")
    shap.plots.waterfall(instance_exp, max_display=15, show=False)
    fig_water = plt.gcf()
    st.pyplot(fig_water)

    with st.expander("Download SHAP waterfall plot"):
        st.download_button(
            label="Download PNG",
            data=_fig_to_png_bytes(fig_water),
            file_name="shap_waterfall_plot.png",
            mime="image/png",
        )

    # ====================================================
    # FORCE PLOT – probability scale
    # ====================================================
    st.subheader("Model Explanation – SHAP Force (Probability)")
    base_val = float(instance_exp.base_values)
    shap_vec = instance_exp.values
    feature_vals = instance_exp.data
    feature_names = instance_exp.feature_names

    shap.plots.force(
        base_val,
        shap_vec,
        features=feature_vals,
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )
    fig_force = plt.gcf()
    st.pyplot(fig_force)

    with st.expander("Download SHAP force plot"):
        st.download_button(
            label="Download PNG",
            data=_fig_to_png_bytes(fig_force),
            file_name="shap_force_plot.png",
            mime="image/png",
        )
