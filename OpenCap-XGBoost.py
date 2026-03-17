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
import streamlit as st

# ===================== 0. 全局配置 =====================
DEFAULT_MODEL_NAME = "final_XGJ_model.pkl"  # <--- 新增：定义仓库里的模型文件名
# ===================== 0. 全局配置 (新加) =====================
ADVICE_MAP = {
    "HFA": "您的**髋关节屈曲角度(HFA)**贡献了较多风险。建议：加强臀大肌离心训练（如深蹲、硬拉），增加触地时的缓冲行程。",
    "HAA": "您的**髋关节内收角度(HAA)**偏大，这可能导致膝外翻。建议：强化臀中肌力量（如蚌式开合、侧抬腿），提高额状面稳定性。",
    "KFA": "**膝关节屈曲(KFA)**不足（着地过直）。建议：练习软着地技术，着地时主动增加膝关节屈曲程度以吸收冲击力。",
    "ADF": "**踝关节背屈(ADF)**不足。建议：改善踝关节活动度，加强小腿后群肌肉拉伸，避免硬性着地。",
    "FPA": "**足偏角(FPA)**异常。建议：注意脚尖指向，避免过度内八或外八字着地，保持下肢力线一致。",
    "TFA": "**躯干倾斜(TFA)**不稳定。建议：加强核心稳定性训练（如平板支撑、侧桥），减少触地时重心的剧烈波动。"
}

# ===================== 1. 身份验证逻辑 (集成版) =====================
def get_opencap_token():
    """从 Secrets 获取凭据并向 OpenCap 请求 Token"""
    try:
        # 从 Streamlit Cloud Secrets 读取
        username = st.secrets["OPENCAP_USER"]
        password = st.secrets["OPENCAP_PASS"]
    except Exception:
        st.error("❌ 错误：未在 Streamlit Secrets 中找到 OPENCAP_USER 或 OPENCAP_PASS")
        st.stop()

    login_url = "https://api.opencap.ai/login/"
    try:
        resp = requests.post(login_url, data={'username': username, 'password': password})
        if resp.status_code == 200:
            return resp.json().get('token')
        else:
            st.error(f"❌ 登录 OpenCap 失败 (状态码: {resp.status_code}): {resp.text}")
            st.stop()
    except Exception as e:
        st.error(f"❌ 网络请求异常: {e}")
        st.stop()

# ===================== 2. Streamlit 页面配置 =====================
st.set_page_config(page_title="ACL 损伤风险预测", layout="wide")

st.title("🏃‍♂️ ACL损伤风险快速筛查系统")

# 侧边栏配置
st.sidebar.header("⚙️ 配置参数")
session_id = st.sidebar.text_input("Session ID", value="995d44f9-022a-449b-9dd2-2424318c3f54")
trial_keyword = st.sidebar.text_input("动作试次名称", value="single-jumpGR_6_1")
# 修改点：将上传组件设为可选，不再是必填项
model_file = st.sidebar.file_uploader("上传自定义模型 (不上传则使用内置默认模型)", type=["pkl"])

# ===================== 3. 核心分析逻辑 =====================
def run_analysis(sid, keyword, model_obj):
    try:
        with st.status("🔍 正在执行分析流程...", expanded=True) as status:
            st.write("正在获取身份令牌...")
            token = get_opencap_token()
            
            st.write("正在从 OpenCap 下载 Session 数据...")
            headers = {"Authorization": f"Token {token}"}
            url = f"https://api.opencap.ai/sessions/{sid}/download/"
            
            resp = requests.get(url, headers=headers)
            if resp.status_code != 200:
                st.error(f"❌ 下载失败，状态码: {resp.status_code}")
                return

            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                all_files = z.namelist()
                # 筛选动态试次的 MOT 文件
                all_mots = [f for f in all_files if f.endswith('.mot') and 'Kinematics' in f and 'static' not in f.lower()]
                
                if not all_mots:
                    st.error("❌ 该 Session 中未发现有效的 MOT 运动数据文件。")
                    return

                # 根据关键词选择试次
                selected_mot = next((m for m in all_mots if keyword.lower() in m.lower()), all_mots[0])
                trial_id = os.path.basename(selected_mot).replace('.mot', '')
                
                # 寻找对应的 TRC 文件
                try:
                    selected_trc = [f for f in all_files if f.endswith('.trc') and trial_id in f and 'MarkerData' in f][0]
                except IndexError:
                    st.error(f"❌ 未找到与试次 {trial_id} 对应的 TRC 标记点数据。")
                    return

                st.write(f"正在解析试次数据: {trial_id}...")
                with z.open(selected_trc) as f:
                    df_trc = pd.read_csv(f, sep='\t', skiprows=6, header=None)
                with z.open(selected_mot) as f:
                    content = f.read().decode('utf-8').splitlines()
                    header_idx = next(i for i, line in enumerate(content) if 'time' in line.lower() and '\t' in line)
                    df_mot = pd.read_csv(io.StringIO('\n'.join(content[header_idx:])), sep='\t')

            # --- IC 瞬间定位 (最高点后首个回弹点) ---
            # 索引 54 对应 RBigToe 的垂直轴
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
            
            st.write("正在加载模型并进行风险预测...")
            model = joblib.load(model_obj)
            score = float(np.asarray(model.predict(features_array)).ravel()[0])
            status.update(label="✅ 分析完成！", state="complete")

        # --- 结果展示面板 ---
        st.divider()
        is_high_risk = score >= 2.45
        risk_text = "高风险" if is_high_risk else "低风险"
        risk_color = "#d63031" if is_high_risk else "#27ae60"

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("ACL 应力值 (×BW)", f"{score:.2f}")
        with m_col2:
            st.markdown(f"### 风险判定: <span style='color:{risk_color};'>{risk_text}</span>", unsafe_allow_html=True)

        # --- SHAP 可视化 ---
        st.subheader("📊 关键动作特征贡献分析 (SHAP)")
        explainer = shap.TreeExplainer(model)
        input_df = pd.DataFrame(features_array, columns=feature_names).round(1)
        
        # 计算 SHAP 值
        shap_values_all = explainer(input_df)
        # 提取当前这一行数据的 Explanation 对象（用于瀑布图）
        exp = shap_values_all[0]

        # 创建两个标签页，用户可以切换查看
        tab1, tab2 = st.tabs(["瀑布图 (Waterfall)", "力图 (Force Plot)"])

        with tab1:
            st.write("瀑布图展示了各特征对基准值的累加贡献：")
            fig_wf, ax_wf = plt.subplots(figsize=(10, 6))
            # 这里的 max_display 控制显示的特征数量
            shap.plots.waterfall(exp, max_display=10, show=False)
            plt.tight_layout()
            st.pyplot(plt.gcf())

        with tab2:
            st.write("力图展示了特征之间相互“推拉”的过程：")
            # Force plot 需要特定的显示逻辑，我们将其包装在 matplotlib 模式下
            shap.force_plot(
                explainer.expected_value, 
                exp.values, 
                input_df.iloc[0,:], 
                matplotlib=True, 
                show=False,
                plot_cmap=["#ff0051", "#008bfb"]
            )
            st.pyplot(plt.gcf(), clear_figure=True)

       # --- 动态建议生成逻辑 ---
        # 获取所有特征名和对应的 SHAP 值
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'contribution': exp.values
        })
        
        # 筛选出贡献值为正（即增加风险）的特征，并按贡献度从大到小排序
        risk_factors = shap_df[shap_df['contribution'] > 0].sort_values(by='contribution', ascending=False)

        with st.expander("📋 针对性动作改善建议", expanded=True):
            st.markdown(f"**分析对象**: `{trial_id}`  |  **触地瞬间 (IC)**: 第 {ic_idx+1} 帧")
            
            if not risk_factors.empty:
                st.markdown("#### ⚠️ 需重点关注的风险项：")
                for _, row in risk_factors.iterrows():
                    f_name = row['feature']
                    advice = ADVICE_MAP.get(f_name, "保持良好姿势。")
                    # 使用 st.info 或 markdown 展示
                    st.write(f"👉 **{f_name}**: {advice}")
            else:
                st.success("✨ 您的动作表现非常平衡，未发现明显的力学风险项！")
            
            st.markdown("---")
            st.markdown("**通用基础建议：**\n1. 强化后群肌肉（Hamstrings）和核心稳定性训练。\n2. 避免在疲劳状态下进行高强度的单腿着地练习。")

    except Exception as e:
        st.error(f"🚨 分析执行出错: {e}")
        st.code(traceback.format_exc())

# ===================== 4. 运行逻辑 =====================
if st.button("🚀 开始自动化分析", use_container_width=True):
    # 逻辑点：判断使用哪个模型源
    if model_file is not None:
        # 如果用户上传了，用上传的
        run_analysis(session_id, trial_keyword, model_file)
    elif os.path.exists(DEFAULT_MODEL_NAME):
        # 如果没上传，但在仓库里找到了默认模型，用默认的
        run_analysis(session_id, trial_keyword, DEFAULT_MODEL_NAME)
    else:
        # 如果两个都没有，再报错
        st.error(f"⚠️ 找不到模型文件！请上传 .pkl 文件或确保仓库中存在 {DEFAULT_MODEL_NAME}")

st.sidebar.markdown("---")
st.sidebar.caption("Powered by OpenCap & XGBoost Model")