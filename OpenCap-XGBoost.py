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
import plotly.graph_objects as go

# ===================== 0. 全局配置 =====================
DEFAULT_MODEL_NAME = "final_MLP_multioutput_model.pkl"  # <--- 新增：定义仓库里的模型文件名
# ===================== 0. 全局配置 (新加) =====================
ADVICE_MAP = {
    "HFA": (
        " 🏋️‍♂️ 髋关节屈曲 (HFA) 优化方案\n\n"
        "**⚠️ 风险机制：** 髋屈不足导致下肢刚性着地，冲击力无法通过大肌群有效吸收。\n\n"
        "**💡 训练建议：**\n"
        "* **基础 (髋铰链)：** 罗马尼亚硬拉 (RDL)。强调臀部向后推，膝盖微屈但不前推，保持背部平直。*(3组 x 8-12次)*\n"
        "* **进阶 (爆发力)：** 壶铃摆动 (Kettlebell Swing)。利用臀部爆发式伸展“弹”起壶铃，在动作顶点收紧臀肌。*(3组 x 15次)*\n"
        "* **动态 (落地控制)：** 高台深落地 (Drop Landing)。从 30cm 高台迈下，触地瞬间迅速“向后坐”，像坐椅子一样。*(3组 x 5-8次)*\n\n"
        "**🎯 技术要点：** 触地瞬间增加“向后坐”的缓冲动作，延长力传导时间。"
    ),
    "HAA": (
        " 🛡️ 髋关节内收 (HAA) 优化方案\n\n"
        "**⚠️ 风险机制：** 髋内收过大会诱发膝外翻（动态内扣），是 ACL (前交叉韧带) 损伤的核心风险因素。\n\n"
        "**💡 训练建议：**\n"
        "* **激活 (臀中肌)：** 弹力带蚌式开合。保持骨盆绝对稳定不向后翻转，体会臀部侧上方的酸胀感。*(3组 x 15-20次/侧)*\n"
        "* **力量 (抗内扣)：** 怪兽步 (Monster Walk)。保持 1/4 蹲姿，侧向行走时主动对抗弹力带拉力，绝不允许膝盖内扣。*(3组 x 15米/方向)*\n"
        "* **功能性 (单腿稳定)：** 单腿下蹲。对着镜子练习，全程盯住膝盖，确保髌骨中心对准第2-3足趾。*(3组 x 8-10次/侧)*\n\n"
        "**🎯 技术要点：** 确保跳跃/落地过程中，膝盖中心始终对准脚尖方向。"
    ),
    "KFA": (
        " 🦵 膝关节屈曲 (KFA) 优化方案\n\n"
        "**⚠️ 风险机制：** 着地时膝关节过直（伸膝位）会产生巨大的前向剪切力，容易导致关节损伤。\n\n"
        "**💡 训练建议：**\n"
        "* **基础 (等长收缩)：** 靠墙静蹲。重心压在足中和脚后跟，下背部贴紧墙壁，感受大腿前侧发力。*(3-4组 x 45-60秒)*\n"
        "* **离心 (制动力)：** 诺迪克后屈腿 (Nordic Curls)。固定双脚，身体挺直缓慢前倾，用大腿后侧控制下降速度。*(3组 x 5-8次)*\n"
        "* **专项 (吸震反应)：** 连续跳深缓冲。从高处跳下，触地瞬间主动屈膝屈髋“吸收”冲击，落地声越轻越好。*(3组 x 6-8次)*\n\n"
        "**🎯 技术要点：** 触地时主动通过屈膝动作来“吸震”，做到落地无声（像猫一样）。"
    ),
    "ADF": (
        " 🦶 踝关节背屈 (ADF) 优化方案\n\n"
        "**⚠️ 风险机制：** 踝关节活动度受限会限制整体下肢缓冲行程，迫使膝关节进行代偿。\n\n"
        "**💡 训练建议：**\n"
        "* **松解 (软组织)：** 泡沫轴深度松解。重点滚压小腿肚和靠近跟腱的区域，痛点停留 20 秒并转动脚踝。*(双侧各 1-2 分钟)*\n"
        "* **拉伸 (活动度)：** 靠墙弓箭步拉伸 (Knee-to-Wall)。足跟死死踩住地面，膝盖尽量向前顶触碰墙壁。*(3组 x 每侧 30秒)*\n"
        "* **动态 (受压灵活性)：** 负重高脚杯深蹲停留。在最深蹲姿下，将重物压在单侧大腿前侧，迫使膝盖超过脚尖。*(3组 x 每侧压迫 20秒)*\n\n"
        "**🎯 技术要点：** 提高踝关节灵活性，确保足跟在落地支撑期能平稳过渡。"
    ),
    "FPA": (
        " 🧭 足偏角 (FPA) 优化方案\n\n"
        "**⚠️ 风险机制：** 过度内八或外八会改变下肢动力链力线，增加膝关节的扭转应力。\n\n"
        "**💡 训练建议：**\n"
        "* **平衡 (足底感知)：** 赤足单腿闭眼站立。感受前脚掌内外侧和足跟构成的“足底三角”均匀受力。*(3组 x 30-45秒/侧)*\n"
        "* **整合 (力线纠正)：** 弓箭步走 (Lunge Walk)。强制盯住前后脚，必须 100% 朝向正前方，拒绝脚尖外撇。*(3组 x 每侧 10-12步)*\n"
        "* **敏捷 (动态对齐)：** 十字象限跳 (Dot Drill)。在四个象限间快速跳跃，关注落地时双脚的平行度，避免疲劳时外八。*(3组 x 20-30秒)*\n\n"
        "**🎯 技术要点：** 保持足部纵轴与行进方向基本一致，消除多余的旋转负荷。"
    ),
    "TFA": (
        " ⚖️ 躯干倾斜 (TFA) 优化方案\n\n"
        "**⚠️ 风险机制：** 躯干不稳会导致重心偏移，产生巨大的膝关节冠状面力矩，破坏身体平衡。\n\n"
        "**💡 训练建议：**\n"
        "* **静态 (侧向抗力)：** 侧桥 (Side Plank)。从头到脚跟呈一条直线，主动将骨盆顶起，核心绷紧不塌腰。*(3组 x 30-45秒/侧)*\n"
        "* **动态 (抗侧屈)：** 单臂农夫行走 (Suitcase Carry)。单手提重物行走，刻意对抗重力，保持双肩水平不向一侧倾斜。*(3组 x 每侧 20米)*\n"
        "* **专项 (动态中立)：** 滑冰跳 (Skater Jumps)。左右单腿横向跳跃，落地停顿 2 秒，控制上半身不要过度侧倾。*(3组 x 每侧 8-10次)*\n\n"
        "**🎯 技术要点：** 维持脊柱中立位稳定，减少触地瞬间身体重心的左右摇晃。"
    )
}
# 各关节在触地瞬间(IC)的正常参考角度（度），需根据你的实验标准微调
NORMAL_VALUES = {
    "HFA": 28.0,  # 髋关节屈曲正常值
    "HAA": 5,   # 髋关节内收正常值
    "KFA": 26.0,  # 膝关节屈曲正常值
    "ADF": 10.0,  # 踝关节背屈正常值
    "FPA": 10.0,   # 足偏角正常值
    "TFA": 25.0   # 躯干侧倾正常值
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

st.title("🏃‍♂️ 韧带损伤风险快速筛查系统")

# 侧边栏配置
st.sidebar.header("⚙️ 配置参数")
session_id = st.sidebar.text_input("Session ID", value="995d44f9-022a-449b-9dd2-2424318c3f54")
trial_keyword = st.sidebar.text_input("动作试次名称", value="single-jumpGR_6_1")
# 修改点：将上传组件设为可选，不再是必填项
model_file = st.sidebar.file_uploader("上传自定义模型 (不上传则使用内置默认模型)", type=["pkl"])

# ===================== 3. 核心分析逻辑 =====================
# 💡 新增：定义 3D 骨架绘制函数
def create_3d_skeleton_plot(df_trc, df_mot, ic_idx):
    marker_map = {
        "Neck": 2, "RShoulder": 5, "RElbow": 8, "RWrist": 11,
        "LShoulder": 14, "LElbow": 17, "LWrist": 20,
        "midHip": 23, "RHip": 26, "RKnee": 29, "RAnkle": 32,
        "LHip": 35, "LKnee": 38, "LAnkle": 41,
        "LBigToe": 44, "LHeel": 50, 
        "RBigToe": 53, "RHeel": 59
    }

    segments = [
        ["Neck", "midHip"],  
        ["LHip", "midHip", "RHip"],  
        ["Neck", "RShoulder", "RElbow", "RWrist"],  
        ["Neck", "LShoulder", "LElbow", "LWrist"],  
        ["RHip", "RKnee", "RAnkle", "RHeel", "RBigToe", "RAnkle"],  
        ["LHip", "LKnee", "LAnkle", "LHeel", "LBigToe", "LAnkle"]   
    ]

    # --- 1. 动态范围计算 (修复上半身消失的关键) ---
    # 提取所有标记点在所有时刻的垂直高度 (TRC数据的第2列是Y轴，即Plotly的Z轴)
    z_cols = [marker_map[m] + 1 for m in marker_map]
    all_z_data = pd.to_numeric(df_trc.iloc[:, z_cols].values.flatten(), errors='coerce')
    max_height_reached = np.nanmax(all_z_data) 
    
    # 自动中心点定位 (基于骨盆水平轨迹)
    pelvis_x = pd.to_numeric(df_trc.iloc[:, marker_map["midHip"]], errors='coerce').dropna().values
    pelvis_y = pd.to_numeric(df_trc.iloc[:, marker_map["midHip"] + 2], errors='coerce').dropna().values
    cx = (np.percentile(pelvis_x, 95) + np.percentile(pelvis_x, 5)) / 2
    cy = (np.percentile(pelvis_y, 95) + np.percentile(pelvis_y, 5)) / 2

    # 动态设定盒子大小：取 (最高点+0.3米缓冲) 或 (保底1.8米)
    box_size = max(1.8, max_height_reached + 0.3) 
    
    range_x = [cx - box_size/2, cx + box_size/2]
    range_y = [cy - box_size/2, cy + box_size/2]
    range_z = [0, box_size] 

    # --- 2. 帧数据提取函数 ---
    def get_frame_data(frame_idx):
        x_vals, y_vals, z_vals = [], [], []
        for seg in segments:
            for point in seg:
                col = marker_map[point]
                x_vals.append(df_trc.iloc[frame_idx, col])
                y_vals.append(df_trc.iloc[frame_idx, col + 2])
                z_vals.append(df_trc.iloc[frame_idx, col + 1])
            x_vals.append(None); y_vals.append(None); z_vals.append(None)
        return x_vals, y_vals, z_vals

    def get_label_data(frame_idx):
        label_joints = ["RKnee", "RHip", "RAnkle"]
        mot_columns = ["knee_angle_r", "hip_flexion_r", "ankle_angle_r"]
        lx, ly, lz, texts = [], [], [], []
        for joint, col_name in zip(label_joints, mot_columns):
            col = marker_map[joint]
            lx.append(df_trc.iloc[frame_idx, col])
            ly.append(df_trc.iloc[frame_idx, col + 2])
            lz.append(df_trc.iloc[frame_idx, col + 1])
            val = df_mot.iloc[frame_idx].get(col_name, 0)
            texts.append(f"{val:.1f}°")
        return lx, ly, lz, texts

    # --- 3. 初始化图表 ---
    x_init, y_init, z_init = get_frame_data(ic_idx)
    lx_init, ly_init, lz_init, lt_init = get_label_data(ic_idx)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=x_init, y=y_init, z=z_init, mode='lines+markers',
        line=dict(color='#ff0051', width=5), marker=dict(size=2, color='#2d3436'), name="Skeleton"
    ))

    fig.add_trace(go.Scatter3d(
        x=lx_init, y=ly_init, z=lz_init, mode='text+markers',
        text=lt_init, textposition="top center",
        textfont=dict(family="Arial Black", size=13, color="#ff0051"),
        marker=dict(size=2, color="#ff0051"), name="Joint Angles"
    ))

    # --- 4. 构建动画帧 ---
    frames = []
    step = 2
    total_frames = len(df_trc)
    for i in range(0, total_frames, step):
        xf, yf, zf = get_frame_data(i)
        lxf, lyf, lzf, ltf = get_label_data(i)
        is_ic = abs(i - ic_idx) <= step
        color = '#ff0051' if is_ic else '#008bfb'
        
        frames.append(go.Frame(
            data=[
                go.Scatter3d(x=xf, y=yf, z=zf, line=dict(color=color)),
                go.Scatter3d(x=lxf, y=lyf, z=lzf, text=ltf)
            ],
            name=str(i)
        ))
    fig.frames = frames

    # --- 5. 交互控件：播放、暂停、进度条 ---
    fig.update_layout(
        paper_bgcolor='white',
        scene=dict(
            xaxis=dict(range=range_x, showticklabels=False, title='', showgrid=True, gridcolor='#f0f0f0'),
            yaxis=dict(range=range_y, showticklabels=False, title='', showgrid=True, gridcolor='#f0f0f0'),
            zaxis=dict(range=range_z, showticklabels=False, title='', showgrid=True, gridcolor='#f0f0f0'),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.2, y=1.2, z=0.6))
        ),
        margin=dict(r=0, l=0, b=0, t=0),
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.05, y=0, xanchor="right", yanchor="top",
            pad={"t": 60},
            buttons=[
                dict(label="▶ 播放", method="animate", args=[None, dict(frame=dict(duration=30, redraw=True), fromcurrent=True)]),
                dict(label="⏸ 暂停", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
            ]
        )],
        sliders=[dict(
            active=0, x=0.07, y=0, len=0.9, xanchor="left", yanchor="top",
            pad={"t": 50},
            currentvalue=dict(font=dict(size=12), prefix="当前帧: ", visible=True, xanchor="right"),
            steps=[dict(method="animate", args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True))], label=str(i)) 
                   for i in range(0, total_frames, step)]
        )]
    )
    
    return fig

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
            
            # 1. 加载模型及核心变量
            loaded_package = joblib.load(model_obj)
            model = loaded_package['model']
            scaler_X = loaded_package['scaler_X']
            scaler_y = loaded_package['scaler_y']
            background_data = loaded_package['background_data']
            important_features = loaded_package['important_features']

            # 2. 💡 定义唯一的预测出口：输入原始特征 -> 输出真实物理力
            def get_real_prediction(raw_features_df):
                # 第一步：标准化特征
                scaled_x = scaler_X.transform(raw_features_df)
                scaled_x_df = pd.DataFrame(scaled_x, columns=scaler_X.feature_names_in_)
                # 第二步：筛选核心特征
                sel_x = scaled_x_df[important_features]
                # 第三步：模型预测
                pred_s = model.predict(sel_x)
                # 第四步：逆向还原力值
                return scaler_y.inverse_transform(pred_s)

            # 3. 执行面板预测
            # 这里的 input_df_raw 必须包含所有原始特征 [HFA, HAA, KFA, ADF, FPA, TFA]
            input_df_raw = pd.DataFrame(features_array, columns=feature_names)
            pred_real = get_real_prediction(input_df_raw)
            
            score_acl = float(pred_real[0][0])      # 严格对应 targets[0]
            score_kneeload = float(pred_real[0][1]) # 严格对应 targets[1]
            
            status.update(label="✅ 分析完成！", state="complete")
        # --- 结果展示面板 ---
        st.divider()
        
        # 💡 1. 设定两个指标的高风险阈值 (请根据你的实验标准修改下面的数字)
        ACL_THRESHOLD = 2.45
        KNEE_LOAD_THRESHOLD = 4.5  # <--- ⚠️ 请把 3.0 改为你判定 knee-load 高风险的实际标准
        
        # 💡 2. 分别判断两个指标是否超标
        is_acl_risk = score_acl >= ACL_THRESHOLD
        is_knee_risk = score_kneeload >= KNEE_LOAD_THRESHOLD
        
        # 💡 3. 总体风险判定：只要有一个指标超标，总体动作就判定为“高风险”
        is_overall_risk = is_acl_risk or is_knee_risk
        overall_text = "🚨 高风险" if is_overall_risk else "✅ 低风险"
        overall_color = "#d63031" if is_overall_risk else "#27ae60"

        # 调整为 3 列，展示指标的同时加上超标警告小标签
        m_col1, m_col2 = st.columns(2)
        
        with m_col1:
            acl_warning = "<span style='color:#d63031; font-size:22px; font-weight:bold;'> ▲</span>" if is_acl_risk else ""
            st.markdown(f"### ACL 应力值（×BW）: <span style='color:#2d3436;'>{score_acl:.2f}</span>{acl_warning}", unsafe_allow_html=True)
            
        with m_col2:
            knee_warning = "<span style='color:#d63031; font-size:22px; font-weight:bold;'> ▲</span>" if is_knee_risk else ""
            st.markdown(f"### 膝关节总接触力（×BW）: <span style='color:#2d3436;'>{score_kneeload:.2f}</span>{knee_warning}", unsafe_allow_html=True)

        # 换行：展示总体风险判定 (占据整行宽度)
        st.write("") # 加一点微小的间距
        
        # 使用 st.markdown 配合简单的背景色，让风险判定更醒目
        bg_color = "rgba(214, 48, 49, 0.1)" if is_overall_risk else "rgba(39, 174, 96, 0.1)"
        st.markdown(
            f"""
            <div style="background-color:{bg_color}; padding:15px; border-radius:10px; border-left: 5px solid {overall_color};">
                <h2 style="color:{overall_color}; margin:0; padding:0;">风险判定：{overall_text}</h2>
            </div>
            """, 
            unsafe_allow_html=True
        )
        # =========================================================
        # 💡 新增：在这里插入 3D 动作姿态重构视图
        # =========================================================
        st.subheader("🟢 动作捕捉数字孪生回放")
        col_3d, col_info = st.columns([2, 1])
        
        with col_3d:
            # 调用我们在上面定义的 3D 绘制函数
            fig_3d = create_3d_skeleton_plot(df_trc, df_mot, ic_idx)
            st.plotly_chart(fig_3d, use_container_width=True)
            
        with col_info:
            st.info(f"""
            **📍 触地瞬间 (IC) 定位**: 第 {ic_idx + 1} 帧
            
            **🔬 动态观察指南**:
            1. 点击下方 **▶ 播放** 按钮，观看全身跳跃力学传导全过程。
            2. 动画变为 **红色** 的瞬间，即为系统抓取的触地高危时刻。
            3. 你可以随时暂停，用鼠标旋转视角，观察躯干是否偏移，以及双膝是否在落地时产生内扣。
            """)
        st.divider()
        # =========================================================

        # --- SHAP 可视化 ---
        st.subheader("📊 关键动作特征贡献分析 (SHAP)")
        
        # 💡 SHAP 专用的预测包装 (因为 SHAP 内部会传标准化的 sel 数据，我们只需做 预测+还原)
        def shap_predict_wrapper(scaled_sel_data):
            p = model.predict(scaled_sel_data)
            return scaler_y.inverse_transform(p)

        # 准备 SHAP 输入
        input_scaled_full = pd.DataFrame(scaler_X.transform(input_df_raw), columns=scaler_X.feature_names_in_)
        input_sel_for_shap = input_scaled_full[important_features]

        explainer = shap.KernelExplainer(shap_predict_wrapper, background_data)
        shap_values_raw = explainer.shap_values(input_sel_for_shap)
        
        # 5. 提取并对齐索引
        if isinstance(shap_values_raw, list):
            val_acl = shap_values_raw[0][0]      
            val_kneeload = shap_values_raw[1][0] 
        else:
            # 3D Array 模式 [样本, 特征, 输出]
            val_acl = shap_values_raw[0, :, 0]
            val_kneeload = shap_values_raw[0, :, 1]

        # 6. 基准值对齐
        if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1:
            expected_val_acl = explainer.expected_value[0]
            expected_val_kneeload = explainer.expected_value[1]
        else:
            expected_val_acl = explainer.expected_value
            expected_val_kneeload = explainer.expected_value
            
       # 7. 组装数据 (对显示的特征数值进行四舍五入，保留一位小数)
        input_raw_sel = np.round(input_df_raw[important_features].iloc[0].values.astype(float), 1)

        exp_acl = shap.Explanation(
            values=val_acl, 
            base_values=expected_val_acl, 
            data=input_raw_sel,  # 此时这里的数值已经是 28.4 这种格式了
            feature_names=important_features
        )
        exp_kneeload = shap.Explanation(
            values=val_kneeload, 
            base_values=expected_val_kneeload, 
            data=input_raw_sel, 
            feature_names=important_features
        )

        # --- 开始渲染 UI 标签页 ---
        tab_acl, tab_kneeload = st.tabs(["🦵 ACL SHAP 解释图", "🏋️‍♂️ 膝关节总接触力 SHAP 解释图"])

        def draw_shap_plots(exp_obj, exp_val, target_name):
            st.markdown(f"**{target_name} - 瀑布图 (Waterfall):** 展示各特征对基准值的累加贡献 (单位: 真实量纲)")
            fig_wf, ax_wf = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(exp_obj, max_display=10, show=False)
            plt.tight_layout()
            st.pyplot(fig_wf)

            st.markdown("---")
            
            st.markdown(f"**{target_name} - 力图 (Force Plot):** 展示特征之间相互“推拉”的过程")
            shap.force_plot(
                exp_val, 
                exp_obj.values, 
                pd.Series(exp_obj.data, index=exp_obj.feature_names), 
                matplotlib=True, 
                show=False,
                plot_cmap=["#ff0051", "#008bfb"]
            )
            st.pyplot(plt.gcf(), clear_figure=True)

        with tab_acl:
            draw_shap_plots(exp_acl, expected_val_acl, "ACL")

        with tab_kneeload:
            draw_shap_plots(exp_kneeload, expected_val_kneeload, "knee-load")

        # --- 动态建议生成逻辑 ---
        # 基于 ACL 的贡献值生成建议
        shap_df = pd.DataFrame({
            'feature': important_features,
            'contribution': exp_acl.values,  
            'actual_value': input_raw_sel   
        })
        
        risk_factors = shap_df[shap_df['contribution'] > 0].sort_values(by='contribution', ascending=False)

        with st.expander("📋 针对性动作改善建议", expanded=True):
            st.markdown(f"**分析对象**: `{trial_id}`  |  **触地瞬间 (IC)**: 第 {ic_idx+1} 帧")
            
            if not risk_factors.empty:
                st.markdown("#### ⚠️ 高风险动作特征偏差分析 (实测值 vs 正常值)")
                
                 # --- 开始绘制哑铃图 ---
                # 根据高风险特征数量动态调整图表高度
                fig_db, ax_db = plt.subplots(figsize=(3.0, len(risk_factors) * 0.25 + 0.8), dpi=200)
                
                y_labels = []
                y_ticks = []
                
                for idx, (index, row) in enumerate(risk_factors[::-1].iterrows()):
                    f_name = row['feature']
                    actual_val = row['actual_value']
                    normal_val = NORMAL_VALUES.get(f_name, 0)
                    y = idx
                    
                    # 连线
                    ax_db.plot([actual_val, normal_val], [y, y], color='#dcdde1', zorder=1, lw=1)
                    
                    # 极小圆点
                    ax_db.scatter(normal_val, y, color='#008bfb', s=15, zorder=2, label='Normal Value' if idx==len(risk_factors)-1 else "")
                    ax_db.scatter(actual_val, y, color='#ff0051', s=15, zorder=2, label='Actual Risk' if idx==len(risk_factors)-1 else "")
                    
                    left_val, right_val = min(actual_val, normal_val), max(actual_val, normal_val)
                    offset = max(abs(actual_val - normal_val) * 0.1, 0.5) 
                    
                    # 极限小字体
                    if actual_val < normal_val:
                        ax_db.text(actual_val - offset, y, f"{actual_val:.1f}", va='center', ha='right', fontsize=6, color='#ff0051', fontweight='bold')
                        ax_db.text(normal_val + offset, y, f"{normal_val:.1f}", va='center', ha='left', fontsize=6, color='#008bfb')
                    else:
                        ax_db.text(normal_val - offset, y, f"{normal_val:.1f}", va='center', ha='right', fontsize=6, color='#008bfb')
                        ax_db.text(actual_val + offset, y, f"{actual_val:.1f}", va='center', ha='left', fontsize=6, color='#ff0051', fontweight='bold')
                    
                    y_labels.append(f_name)
                    y_ticks.append(y)
                
                # 设置Y轴和X轴
                ax_db.set_yticks(y_ticks)
                ax_db.set_yticklabels(y_labels, fontsize=7, fontweight='bold', color='#2d3436')
                ax_db.set_xlabel("Angle (Degree)", fontsize=7, color='#636e72')
                ax_db.tick_params(axis='x', labelsize=6)
                
                # 美化图表
                ax_db.spines['top'].set_visible(False)
                ax_db.spines['right'].set_visible(False)
                ax_db.spines['left'].set_visible(False)
                ax_db.spines['bottom'].set_color('#b2bec3')
                ax_db.grid(axis='x', linestyle='--', alpha=0.5)
                
                # 两侧留白
                x_min, x_max = ax_db.get_xlim()
                ax_db.set_xlim(x_min - (x_max-x_min)*0.25, x_max + (x_max-x_min)*0.25)
                
                # 【修改点 1】：把上限从 len(risk_factors) 改成 len(risk_factors) + 0.5
                # 这样会把图表内部的“天花板”再往上抬高一截，给最上面的 ADF 留出更多呼吸空间
                ax_db.set_ylim(-1.0, len(risk_factors) + 0.7)
                
                # 【修改点 2】：把图例的高度位置从 1.15 调高到 1.25 或 1.3
                ax_db.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False, fontsize=6)
                
                plt.tight_layout()
                st.pyplot(fig_db, clear_figure=True, use_container_width=False)
                # --- 哑铃图绘制结束 ---

                st.markdown("#### 🎯 动作处方：")
                for _, row in risk_factors.iterrows():
                    f_name = row['feature']
                    advice = ADVICE_MAP.get(f_name, "保持良好姿势。")
                    st.info(f"👉 **{f_name}**: {advice}")
            else:
                st.success("✨ 您的动作表现非常平衡，未发现明显的力学风险项！")
            
            st.markdown("---")

    except Exception as e:
        st.error(f"🚨 分析执行出错: {e}")
        st.code(traceback.format_exc())

# ===================== 4. 运行逻辑 =====================
if st.button("🚀 开始自动化分析", use_container_width=True):
    if model_file is not None:
        run_analysis(session_id, trial_keyword, model_file)
    elif os.path.exists(DEFAULT_MODEL_NAME):
        run_analysis(session_id, trial_keyword, DEFAULT_MODEL_NAME)
    else:
        st.error(f"⚠️ 找不到模型文件！请上传 .pkl 文件或确保仓库中存在 {DEFAULT_MODEL_NAME}")

st.sidebar.markdown("---")
st.sidebar.caption("Powered by OpenCap & MLP Model")