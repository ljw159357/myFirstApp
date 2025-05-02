import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 加载模型
try:
    model = joblib.load('rf.pkl')  # 确保文件名正确
except FileNotFoundError:
    st.error("模型文件 'rf.pkl' 没有找到，请检查文件路径。")
    st.stop()

scaler = StandardScaler()

# 特征定义
feature_ranges = {
    "Age": {"type": "numerical"},
    "Postoperative platelet count (x10⁹/L)": {"type": "numerical"},
    "Postoperative BUN (μmol/L)": {"type": "numerical"},
    "Day 1 postoperative antithrombin III activity (%)": {"type": "numerical"},
    "NYHA": {"type": "categorical", "options": ["＞2", "≤2"]},
    "HBP": {"type": "categorical", "options": ["Yes", "No"]},
    "Postoperative CRRT (Continuous Renal Replacement Therapy)": {"type": "categorical", "options": ["Yes", "No"]},
    "Postoperative Anticoagulation": {"type": "categorical", "options": ["Yes", "No"]}
}

category_to_numeric_mapping = {
    "NYHA": {"＞2": 1, "≤2": 0},
    "HBP": {"Yes": 1, "No": 0},
    "Postoperative CRRT (Continuous Renal Replacement Therapy)": {"Yes": 1, "No": 0},
    "Postoperative Anticoagulation": {"Yes": 1, "No": 0}
}

# UI
st.title("Prediction Model for Thrombosis After Lung Transplantation")
st.header("Enter the following feature values:")

feature_values = []
feature_keys = list(feature_ranges.keys())

# 输入
for feature in feature_keys:
    prop = feature_ranges[feature]
    if prop["type"] == "numerical":
        value = st.number_input(label=f"{feature}", value=0.0)
        feature_values.append(value)
    elif prop["type"] == "categorical":
        value = st.selectbox(label=f"{feature} (Select a value)", options=prop["options"], index=0)
        numeric_value = category_to_numeric_mapping[feature][value]
        feature_values.append(numeric_value)

# 数值特征标准化
numerical_features = [f for f, p in feature_ranges.items() if p["type"] == "numerical"]
numerical_values = [feature_values[feature_keys.index(f)] for f in numerical_features]

if numerical_values:
    numerical_values_scaled = scaler.fit_transform([numerical_values])
    for idx, f in enumerate(numerical_features):
        feature_values[feature_keys.index(f)] = numerical_values_scaled[0][idx]

features = np.array([feature_values])

if st.button("Predict"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果
    text = f"Based on feature values, predicted possibility of thrombosis after lung transplantation is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(0.5, 0.5, text, fontsize=16, ha='center', va='center', fontname='Arial', transform=ax.transAxes)  # 改为已安装的字体
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # SHAP 解释
    # 提取底层模型（支持 pipeline 或直接模型）
    def get_tree_model(model):
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            return model.named_steps['clf']
        return model

    tree_model = get_tree_model(model)
    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_keys))

    shap.initjs()
    shap_fig = shap.plots.force(
        explainer.expected_value[1],  # 类别 1 的基准值
        shap_values[0, :, 1],  # 类别 1 的 SHAP 值
        pd.DataFrame([feature_values], columns=feature_keys),
        matplotlib=True,
        show=False  # 不自动显示图形
    )
    st.pyplot(shap_fig)
