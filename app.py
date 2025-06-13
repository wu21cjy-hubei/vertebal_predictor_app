import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# 页面标题
st.title("Random Forest 预测模型演示")

# 加载模型和数据示例
model = joblib.load("RF_model.pkl")
data_example = pd.read_excel("X_scaled_RF.xlsx")

# 获取特征列（定量+定性）
feature_columns = data_example.columns.tolist()

# 用户上传数据或手动输入
st.sidebar.header("输入特征")

input_data = {}
for col in feature_columns:
    dtype = data_example[col].dtype
    if np.issubdtype(dtype, np.number):
        min_val = float(data_example[col].min())
        max_val = float(data_example[col].max())
        default_val = float(data_example[col].mean())
        input_data[col] = st.sidebar.slider(col, min_val, max_val, default_val)
    else:
        input_data[col] = st.sidebar.text_input(col, "")

# 创建 DataFrame 并预测
if st.button("开始预测"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    label_mapping = {0: "Group 1", 1: "Group 2", 2: "Group 3", 3: "Group 4"}
    st.success(f"模型预测结果：{label_mapping.get(prediction, prediction)}")

    st.subheader("各组别预测概率：")
    for i, prob in enumerate(prediction_proba):
        st.write(f"{label_mapping.get(i)}: {prob:.3f}")
