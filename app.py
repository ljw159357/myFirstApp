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

    # 如果 model 是 Pipeline，就取出最终的树模型
    if isinstance(model, Pipeline):
        # 尝试通过 named_steps 找 clf，否则取最后一步
        tree_model = model.named_steps.get('clf', model.steps[-1][1])
    else:
        tree_model = model

    # 用提取出的树模型创建解释器
    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(X)

    # 绘制正类（索引 1）的 force plot
    force_fig = shap.plots.force(
        explainer.expected_value[1],  # 正类的基准值
        shap_values[1][0],            # 样本的 SHAP 值
        X,                            # 原始特征 DataFrame
        matplotlib=True,
        show=False
    )

    # 通过 Streamlit 显示 force plot
    st.pyplot(force_fig)
