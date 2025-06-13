import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 页面标题
st.title("Random Forest 预测模型演示")

# 加载模型与 scaler
model = joblib.load("RF_model.pkl")
scaler = joblib.load("scaler.pkl")

# 手动定义特征列名，防止泄露原始数据
categorical_cols = ['Vertebral intraosseous abscess', 'Endplate inflammatory reaction line', 'Paravertebral abscess']
quantitative_cols = ['Number of intervertebral discs destroyed', 'CRP', 'N%', 'L', 'Age', 'Height(m)']

# Streamlit 侧边栏输入真实值
st.sidebar.header("请输入真实特征值：")
input_data = {}

for col in quantitative_cols:
    input_data[col] = st.sidebar.number_input(col, value=0.0)

for col in categorical_cols:
    input_data[col] = st.sidebar.selectbox(col, options=[0, 1, 2] if col == 'Paravertebral abscess' else [0, 1])

# 创建 DataFrame 并标准化 + 预测
if st.button("开始预测"):
    input_df = pd.DataFrame([input_data])
    input_quant_scaled = scaler.transform(input_df[quantitative_cols])
    input_combined = pd.DataFrame(input_quant_scaled, columns=quantitative_cols)
    input_combined = pd.concat([input_combined, input_df[categorical_cols].reset_index(drop=True)], axis=1)

    prediction = model.predict(input_combined)[0]
    prediction_proba = model.predict_proba(input_combined)[0]

    label_mapping = {0: "Group 1", 1: "Group 2", 2: "Group 3", 3: "Group 4"}
    st.success(f"模型预测结果：{label_mapping.get(prediction, prediction)}")

    st.subheader("各组别预测概率：")
    for i, prob in enumerate(prediction_proba):
        st.write(f"{label_mapping.get(i)}: {prob:.3f}")
