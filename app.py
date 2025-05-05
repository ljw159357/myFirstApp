import io, streamlit as st, joblib, pandas as pd, shap, matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

# ---------------------------------------------
# Page configuration
# ---------------------------------------------
st.set_page_config(page_title="Thrombosis Prediction", layout="centered")

# ---------------------------------------------
# ❶ UI Feature definitions (展示名)
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

# ---------------------------------------------
# ❷ 展示名 → 训练名 映射（务必与训练脚本保持一致）
#    这里示例用简写；请按实际训练列名修改！
# ---------------------------------------------
rename_cols = {
    "Age": "age",
    "Postoperative platelet count (x10⁹/L)": "post_PLT",
    "Postoperative BUN (μmol/L)": "post_BUN",
    "Day 1 postoperative antithrombin III activity (%)": "post_AT_1",
    "NYHA": "NYHA",
    "HBP": "HBP",
    "Postoperative CRRT (Continuous Renal Replacement Therapy)": "post_CRRT",
    "Postoperative Anticoagulation": "anticoagulant_therapy",
}

# ---------------------------------------------
# ❸ 类别映射（键用训练名）
# ---------------------------------------------
categorical_mapping_internal = {
    "NYHA": {"＞2": 1, "≤2": 0},
    "HBP": {"Yes": 1, "No": 0},
    "post_CRRT": {"Yes": 1, "No": 0},
    "anticoagulant_therapy": {"Yes": 1, "No": 0},
}

# 训练名列列表
numerical_cols_internal   = [rename_cols[k] for k, v in feature_defs.items() if v[0] == "numerical"]
categorical_cols_internal = [rename_cols[k] for k, v in feature_defs.items() if v[0] == "categorical"]

# ---------------------------------------------
# Utility
# ---------------------------------------------
def _fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    buf.seek(0)
    return buf.read()

# ---------------------------------------------
# Load model & scaler
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
# UI – gather inputs
# ---------------------------------------------
st.title("Prediction Model for Thrombosis After Lung Transplantation")

user_inputs = {}
for feat, (ftype, default) in feature_defs.items():
    if ftype == "numerical":
        user_inputs[feat] = st.number_input(feat, value=float(default))
    else:
        user_inputs[feat] = st.selectbox(feat, feature_defs[feat][1], index=0)

user_df_raw = pd.DataFrame([user_inputs])           # 列名 = 展示名

# ---------------------------------------------
# Pre‑processing
# ---------------------------------------------
user_df_proc = user_df_raw.rename(columns=rename_cols)                 # 1) 改列名
user_df_proc[categorical_cols_internal] = (
    user_df_proc[categorical_cols_internal]
        .replace(categorical_mapping_internal)                         # 2) 类别映射
)

if (external_scaler is not None) and (not uses_pipeline):              # 3) 数值标准化
    user_df_proc[numerical_cols_internal] = external_scaler.transform(
        user_df_proc[numerical_cols_internal]
    )

# 若模型保存了 feature_names_in_，按其顺序排列
if hasattr(model, "feature_names_in_"):
    user_df_proc = user_df_proc[model.feature_names_in_]

# ---------------------------------------------
# Inference & SHAP
# ---------------------------------------------
if st.button("Predict"):
    proba = model.predict_proba(user_df_proc)[:, 1][0]
    st.success(f"Predicted risk of postoperative thrombosis: {proba * 100:.2f}%")

    @st.cache_resource(show_spinner=False)
    def build_explainer(_m):
        try: return shap.Explainer(_m)
        except Exception:
            if isinstance(_m, Pipeline):
                return shap.TreeExplainer(_m.steps[-1][1])
            raise

    explainer   = build_explainer(model)
    shap_values = explainer(user_df_proc)
    instance    = shap_values[0]
    if instance.values.ndim == 2:
        instance = instance[:, 1]

    # ---------------- Waterfall ----------------
    st.subheader("SHAP Waterfall Plot (训练列名)")
    shap.plots.waterfall(instance, max_display=15, show=False)
    fig_water = plt.gcf()
    st.pyplot(fig_water)
    with st.expander("Download waterfall PNG"):
        st.download_button("Download PNG", _fig_to_png_bytes(fig_water),
                           "shap_waterfall_plot.png", "image/png")

    # ---------------- Force ----------------
    st.subheader("SHAP Force Plot (训练列名)")
    shap.plots.force(
        instance.base_values, instance.values,
        features=instance.data, feature_names=instance.feature_names,
        matplotlib=True, show=False
    )
    fig_force = plt.gcf()
    st.pyplot(fig_force)
    with st.expander("Download force PNG"):
        st.download_button("Download PNG", _fig_to_png_bytes(fig_force),
                           "shap_force_plot.png", "image/png")
