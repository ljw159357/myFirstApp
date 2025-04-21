#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 特征的缩写字典
feature_abbr = {
    "Postoperative Platelet Count (x10⁹/L)": "post_plt",
    "Urgent Postoperative APTT (s)": "post_APTT_u",
    "Day 1 Postoperative APTT (s)": "post_APTT_1",
    "Day 1 Postoperative Antithrombin III Activity (%)": "post_antithrombin_III_1",
    "Postoperative CRRT (Continuous Renal Replacement Therapy)": "post_CRRT",
    "Postoperative Anticoagulation": "post_anticoagulation",
    "Transplant Side": "trans_side",
    "Primary Graft Dysfunction (PGD, Level)": "PGD",
    "Height": "height",  # 其他特征也可以添加缩写
    "HBP": "hbp"
}

# 加载模型
model = joblib.load('xgb.pkl')
scaler = StandardScaler()

# 特征定义
feature_ranges = {
    "Height": {"type": "numerical"},
    "HBP": {"type": "categorical", "options": ["Yes", "No"]},
    "Postoperative Platelet Count (x10⁹/L)": {"type": "numerical"},
    "Urgent Postoperative APTT (s)": {"type": "numerical"},
    "Day 1 Postoperative APTT (s)": {"type": "numerical"},
    "Day 1 Postoperative Antithrombin III Activity (%)": {"type": "numerical"},
    "Postoperative CRRT (Continuous Renal Replacement Therapy)": {"type": "categorical", "options": ["Yes", "No"]},
    "Postoperative Anticoagulation": {"type": "categorical", "options": ["Yes", "No"]},
    "Transplant Side": {"type": "categorical", "options": ["Left", "Right", "Both"]},
    "Primary Graft Dysfunction (PGD, Level)": {"type": "categorical", "options": ["3", "2", "1", "0"]},
}

category_to_numeric_mapping = {
    "Transplant Side": {"Left": 1, "Right": 2, "Both": 0},
    "HBP": {"Yes": 1, "No": 0},
    "Postoperative CRRT (Continuous Renal Replacement Therapy)": {"Yes": 1, "No": 0},
    "Postoperative Anticoagulation": {"Yes": 1, "No": 0},
    "Primary Graft Dysfunction (PGD, Level)": {"3": 3, "2": 2, "1": 1, "0": 0}
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

# 预测
if st.button("Predict"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果
    text = f"Based on feature values, predicted possibility of hemorrhage after lung transplantation is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(0.5, 0.5, text, fontsize=16, ha='center', va='center', fontname='Times New Roman', transform=ax.transAxes)
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
    
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values_for_display = shap_values[1]  
        base_value = explainer.expected_value[1]
    elif isinstance(shap_values, list) and len(shap_values) == 1:
        shap_values_for_display = shap_values[0]
        base_value = explainer.expected_value[0]
    else:
        shap_values_for_display = shap_values
        base_value = explainer.expected_value

    # 将特征名替换为缩写
    feature_keys_abbr = [feature_abbr.get(f, f) for f in feature_keys]  # 将特征名替换为缩写

    # 生成 SHAP 力图
    shap.initjs()
    shap_fig = shap.plots.force(
        base_value,  # 基准值
        shap_values_for_display,  # SHAP 值
        pd.DataFrame([feature_values], columns=feature_keys_abbr),  # 使用缩写作为列名
        matplotlib=True,
        show=False  # 不自动显示图形
    )
    
    st.pyplot(shap_fig)
