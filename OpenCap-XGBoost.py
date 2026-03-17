import streamlit as st
import os
import requests
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import io
import zipfile
import traceback
from utilsAuthentication import get_token

# ===================== 1. Streamlit 页面配置 =====================
st.set_page_config(page_title="ACL 损伤风险预测", layout="wide")

st.title("🏃‍♂️ OpenCap ACL 载荷预测分析系统")

# 侧边栏配置
st.sidebar.header("配置参数")
session_id = st.sidebar.text_input("Session ID", value="995d44f9-022a-449b-9dd2-2424318c3f54")
trial_keyword = st.sidebar.text_input("试次筛选关键词", value="single-jumpGR_6_1")
model_file = st.sidebar.file_uploader("上传模型文件 (.pkl)", type=["pkl"])

# ===================== 2. 核心分析函数 =====================
def run_analysis(sid, keyword, model_obj):
    try:
        st.info(f"[*] 正在获取 Token 并下载 Session 数据...")
        token = get_token() 
        headers = {"Authorization": f"Token {token}"}
        url = f"https://api.opencap.ai/sessions/{sid}/download/"
        
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            st.error(f"❌ 下载失败，状态码: {resp.status_code}")
            return

        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            all_files = z.namelist()
            all_mots = [f for f in all_files if f.endswith('.mot') and 'Kinematics' in f and 'static' not in f.lower()]
            
            # 试次选择
            selected_mot = next((m for m in all_mots if keyword.lower() in m.lower()), all_mots[0])
            trial_id = os.path.basename(selected_mot).replace('.mot', '')
            selected_trc = [f for f in all_files if f.endswith('.trc') and trial_id in f and 'MarkerData' in f][0]

            # 解析数据
            with z.open(selected_trc) as f:
                df_trc = pd.read_csv(f, sep='\t', skiprows=6, header=None)
            with z.open(selected_mot) as f:
                content = f.read().decode('utf-8').splitlines()
                header_idx = next(i for i, line in enumerate(content) if 'time' in line.lower() and '\t' in line)
                df_mot = pd.read_csv(io.StringIO('\n'.join(content[header_idx:])), sep='\t')

        # --- IC 瞬间定位 ---
        y_values = df_trc.iloc[:, 54].values 
        apex_idx = np.argmax(y_values)
        ic_idx = apex_idx
        for i in range(apex_idx + 1, len(y_values)):
            if y_values[i] < 0.15 and y_values[i] > y_values[i-1]:
                ic_idx = i
                break
        
        # --- 特征提取 ---
        row = df_mot.iloc[ic_idx].copy()
        feature_names = ["HFA", "HAA", "KFA", "ADF", "FPA", "TFA"]
        feature_values = [
            row.get('hip_flexion_r', 0),
            row.get('hip_adduction_r', 0),
            row.get('knee_angle_r', 0),
            row.get('ankle_angle_r', 0),
            row.get('subtalar_angle_r', 0) * -1,
            row.get('lumbar_extension', 0) * -1
        ]
        features_array = np.array([feature_values])
        
        # --- 模型预测 ---
        model = joblib.load(model_obj)
        score = float(np.asarray(model.predict(features_array)).ravel()[0])
        
        # --- 结果展示 ---
        is_high_risk = score >= 2.45
        risk_text = "High Risk" if is_high_risk else "Low Risk"
        risk_color = "red" if is_high_risk else "green"

        col1, col2 = st.columns(2)
        col1.metric("Predicted ACL load (×BW)", f"{score:.2f}")
        col2.markdown(f"### Risk Level: :{risk_color}[{risk_text}]")

        st.markdown("---")
        st.subheader("💡 Analysis & Recommendations")
        st.write(f"分析试次: **{trial_id}** (IC Frame: {ic_idx+1})")
        
        st.markdown("""
        * **Strength Training:** Focus on hamstrings and core stability.
        * **Landing Technique:** Avoid excessive knee valgus and maintain hip control.
        """)

        # --- SHAP 可视化 ---
        st.subheader("📊 SHAP 解释性分析 (Feature Contribution)")
        explainer = shap.TreeExplainer(model)
        input_df = pd.DataFrame(features_array, columns=feature_names).round(1)
        shap_values = explainer.shap_values(input_df)

        # 在 Streamlit 中绘制 Matplotlib 图形
        fig, ax = plt.subplots(figsize=(12, 3))
        shap.force_plot(
            explainer.expected_value, 
            shap_values[0,:], 
            input_df.iloc[0,:], 
            matplotlib=True, 
            show=False,
            plot_cmap=["#ff0051", "#008bfb"]
        )
        st.pyplot(plt.gcf())

    except Exception as e:
        st.error(f"分析过程中出错: {e}")
        st.code(traceback.format_exc())

# ===================== 3. 运行逻辑 =====================
if st.button("开始分析"):
    if model_file is not None:
        run_analysis(session_id, trial_keyword, model_file)
    else:
        st.warning("请先在左侧上传模型文件 (.pkl)")