import streamlit as st
import pandas as pd
import numpy as np
import joblib

try:
    import matplotlib.pyplot as plt
    matplotlib_available = True
except ModuleNotFoundError:
    matplotlib_available = False

# 页面标题
st.set_page_config(page_title="脊柱感染预测模型演示", layout="wide")
st.title("🌟 Random Forest 脊柱感染预测模型演示")
st.markdown("""
<style>
    html, body, [class*="css"]  {
        font-size: 16px !important;
        color: #333333 !important;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    label, .stSelectbox label, .stNumberInput label {
        color: #333333 !important;
        font-weight: 500;
    }
    .stSelectbox div[role="combobox"], .stNumberInput input {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# 加载模型与 scaler
model = joblib.load("RF_model.pkl")
scaler = joblib.load("scaler.pkl")

# 打印 scaler 期望的特征列（调试用）
st.sidebar.markdown("### 🛠️ 调试信息")
st.sidebar.text("Scaler 特征列：")
st.sidebar.write(scaler.feature_names_in_)

# 定性与定量特征列名
categorical_cols = ['Thoracic', 'Lumbar and Sacrum', 'Number of vertebrae involved',
                    'Extent of vertebral destruction', 'Vertebral intraosseous abscess',
                    'Degree of disk destruction', 'Subligamentous spread', 'Skip lesion',
                    'Endplate inflammatory reaction line', 'Paravertebral abscess',
                    'Neurological symptom', 'Fever']

quantitative_cols = ['involved/normal', 'ESR', 'CRP', 'A/G', 'WBC', 'L%',
                     'Time elapsed to diagnosis of spondylodiscitis (m)', 'Height(m)']

# 输入模块卡片风格 + 提示说明
st.subheader("📝 请输入真实特征值：")
st.info("请填写患者的基本临床特征和实验室指标，部分项目默认值为 0，可根据实际情况修改。")
with st.container():
    col1, col2 = st.columns(2)
    input_data = {}

    with col1:
        st.markdown("#### 📊 定量特征")
        with st.expander("📥 展开/折叠 定量输入", expanded=True):
            for col in quantitative_cols:
                input_data[col] = st.number_input(col, value=0.0, format="%.2f", help="请输入连续变量，如化验值")

    with col2:
        st.markdown("#### 🧬 定性特征")
        with st.expander("📥 展开/折叠 定性选择", expanded=True):
            for col in categorical_cols:
                options = [0, 1, 2] if col in ['Extent of vertebral destruction', 'Degree of disk destruction', 'Paravertebral abscess'] else [0, 1]
                input_data[col] = st.selectbox(col, options=options, help="选择分类变量（0/1 或 0/1/2）")

# 创建 DataFrame 并标准化 + 预测
if st.button("🚀 开始预测"):
    input_df = pd.DataFrame([input_data])

    # 展示当前输入列名
    st.sidebar.text("输入数据列：")
    st.sidebar.write(input_df.columns.tolist())

    # 检查列是否匹配 scaler 要求
    missing_cols = [col for col in scaler.feature_names_in_ if col not in input_df.columns]
    if missing_cols:
        st.error(f"❌ 缺少特征列：{missing_cols}，请检查列名是否与 scaler 拟合时一致。")
    else:
        input_df_scaled = scaler.transform(input_df[scaler.feature_names_in_])
        input_combined = pd.DataFrame(input_df_scaled, columns=scaler.feature_names_in_)
        input_combined = pd.concat([input_combined, input_df[categorical_cols].reset_index(drop=True)], axis=1)

        prediction = model.predict(input_combined)[0]
        prediction_proba = model.predict_proba(input_combined)[0]

        label_mapping = {0: "Group 1", 1: "Group 2", 2: "Group 3", 3: "Group 4"}
        st.balloons()
        st.success(f"✅ 模型预测结果：{label_mapping.get(prediction, prediction)}")

        st.subheader("📈 各组别预测概率（条形图）：")
        proba_df = pd.DataFrame({"Group": [label_mapping[i] for i in range(len(prediction_proba))],
                                 "Probability": prediction_proba})
        st.bar_chart(proba_df.set_index("Group"))

        if matplotlib_available:
            st.subheader("🧭 各组别预测概率（雷达图）：")
            categories = list(label_mapping.values())
            values = list(prediction_proba) + [prediction_proba[0]]

            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
            ax.plot(angles, values, linewidth=2, linestyle='solid')
            ax.fill(angles, values, alpha=0.3)
            ax.set_thetagrids(np.degrees(angles[:-1]), categories)
            ax.set_title("Prediction Radar")
            st.pyplot(fig)
        else:
            st.warning("⚠️ 当前环境缺少 matplotlib，无法显示雷达图。请联系管理员安装 matplotlib 库。")