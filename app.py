import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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
st.header("Prediction Model for Thrombosis After Lung Transplantation")
st.write("Enter the following feature values:")

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
    st.write(f"Based on feature values, predicted possibility of thrombosis after lung transplantation is :  {'%.2f' % float(Predict_proba * 100) + '%'}")
    
    # 构造 DataFrame 供 SHAP 使用
    X = pd.DataFrame([feature_values], columns=feature_keys)

    # 如果 model 是 Pipeline，就取出实际的树模型
    from sklearn.pipeline import Pipeline
    if isinstance(model, Pipeline):
        tree_model = model.named_steps.get('clf', model.steps[-1][1])
    else:
        tree_model = model

    # 构造解释器并计算 SHAP 值
    explainer = shap.TreeExplainer(tree_model)
    shap_vals = explainer.shap_values(X)

    # 根据返回类型取正类的 SHAP 向量和基准值
    if isinstance(shap_vals, list):
        # 旧接口：list 长度=类别数，二分类时用索引1
        vals = shap_vals[1][0]
        base = explainer.expected_value[1]
    else:
        # 新接口：直接返回 (n_samples, n_features)
        vals = shap_vals[0]
        base = explainer.expected_value

    # 绘制 force plot
    force_fig = shap.plots.force(
        base,     # 基准值
        vals,     # 单样本的 SHAP 值向量
        X,        # 特征 DataFrame
        matplotlib=True,
        show=False
    )

    # 用 Streamlit 展示
    st.pyplot(force_fig)
    # ---------- 在这里结束追加 SHAP 力图 ----------
