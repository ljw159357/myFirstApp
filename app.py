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

if st.button("Predict"):
    prediction = model.predict(features)[0]
    Predict_proba = model.predict_proba(features)[:, 1][0]
    # 输出概率
    st.subheader(f"Based on feature values, predicted possibility of thrombosis after lung transplantation is :  {'%.2f' % float(Predict_proba * 100) + '%'}")
