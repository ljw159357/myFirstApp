import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 特征的缩写字典
feature_abbr = {
    "Age": "age",
    "Postoperative platelet count (x10⁹/L)": "post_plt",
    "Postoperative BUN (μmol/L)": "post_BUN",
    "Day 1 postoperative antithrombin III activity (%)": "post_antithrombin_III_1",
    "NYHA": "NYHA",
    "HBP": "HBP",
    "Postoperative CRRT (Continuous Renal Replacement Therapy)": "post_CRRT",
    "Postoperative Anticoagulation": "post_anticoagulation"
}

# 加载模型
model = joblib.load('rf.pkl')
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

# 特征名缩写
feature_keys_abbr = [feature_abbr.get(f, f) for f in feature_keys]  # 将特征名替换为缩写

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果，使用 Matplotlib 渲染指定字体
    text = f"Based on feature values, predicted possibility of AKI is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    # 生成 SHAP 力图
    class_index = predicted_class  # 当前预测类别
    shap_fig = shap.force_plot(
        explainer.expected_value[class_index],
        shap_values[:,:,class_index],
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
    )
    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
