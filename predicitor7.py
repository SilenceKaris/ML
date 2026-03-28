import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
STAGE_MAPPING = {1: "临床前期", 2: "无症状期", 3: "症状期", 4: "失代偿期"}

# 关键：与训练时一致的顺序（ALBI由程序自动计算填入）
MODEL_FEATURE_ORDER = ['ALP', 'Stages', 'ALBI', 'GGT', 'C3', 'D_D', 'DBIL']
SHAP_LABELS = ['ALP', 'Stages', 'ALBI', 'GGT', 'C3', 'D_D', 'DBIL']

# ==================== 页面配置 ====================
st.set_page_config(page_title="UDCA应答不佳预测", layout="wide")
st.title("UDCA应答不佳风险预测")
# ==================== 加载模型 ====================
@st.cache_resource
def load_model():
    try:
        with open('lgb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model, None
    except Exception as e:
        return None, str(e)

model, error = load_model()
if model is None:
    st.error(f"模型加载失败: {error}")
    st.stop()

# ==================== 输入界面 ====================
st.subheader("📋 临床指标输入")
col1, col2 = st.columns(2)

with col1:
    # 总胆红素和白蛋白输入（用于自动计算ALBI）
    tbil = st.number_input("总胆红素 TBIL (μmol/L)", 
                          min_value=1.0, max_value=1000.0, 
                          value=15.0, step=0.1, key='tbil')
    
    alb = st.number_input("白蛋白 ALB (g/L)", 
                         min_value=1.0, max_value=100.0, 
                         value=40.0, step=0.1, key='alb')
    
    # 自动计算ALBI评分并显示（公式：ALBI = 0.66×log10(TBIL) - 0.085×ALB）
    albi_score = 0.66 * np.log10(tbil) - 0.085 * alb
    st.metric("ALBI评分 (自动计算)", f"{albi_score:.3f}", 
              help="计算公式：ALBI = 0.66×log₁₀(TBIL) - 0.085×ALB")
    
    # 其他指标
    ggt = st.number_input("GGT (U/L)", 
                         min_value=0.0, max_value=2000.0, 
                         value=120.0, step=1.0, key='ggt')
    
    dbil = st.number_input("DBIL (μmol/L)", 
                          min_value=0.0, max_value=500.0, 
                          value=15.0, step=0.1, key='dbil')

with col2:
    alp = st.number_input("ALP (U/L)", 
                         min_value=0.0, max_value=3000.0, 
                         value=200.0, step=1.0, key='alp')
    
    c3 = st.number_input("补体C3 (g/L)", 
                        min_value=0.0, max_value=5.0, 
                        value=0.9, step=0.01, key='c3')
    
    d_d = st.number_input("D二聚体 (mg/L)", 
                         min_value=0.0, max_value=100.0, 
                         value=0.8, step=0.01, key='d_d')
    
    stages = st.selectbox("自然史分期", 
                         options=list(STAGE_MAPPING.keys()),
                         format_func=lambda x: f"{x}期 - {STAGE_MAPPING[x]}",
                         key='stages')

# 构建输入字典（按模型训练时的顺序）
inputs = {
    'ALP': alp,
    'Stages': stages,
    'ALBI': albi_score,  # 使用自动计算的值
    'GGT': ggt,
    'C3': c3,
    'D_D': d_d,
    'DBIL': dbil
}

with st.expander("ℹ️ 隐私与使用说明"):
    st.markdown("""
- 输入数据仅用于实时预测，不存储，关闭页面后清除
- 演示数据为模拟值，不涉及真实患者信息  
- 本工具仅供论文演示，不作为诊断依据
""")
# ==================== 预测 ====================
st.divider()
if st.button("🔍 预测", type="primary"):
    feature_values = [inputs[col] for col in MODEL_FEATURE_ORDER]
    X = pd.DataFrame([feature_values], columns=MODEL_FEATURE_ORDER)
    
    try:
        proba = model.predict_proba(X)[0][1]
        risk_pct = proba * 100
        
        col_res, col_shap = st.columns([1, 2])
        
        with col_res:
            st.subheader("预测结果")
            st.metric("UDCA应答不佳概率", f"{risk_pct:.1f}%")
            st.progress(min(float(proba), 1.0))
            
            # 阈值保持0.5（50%）
            if proba >= 0.5:
                st.error("高风险")
            else:
                st.success("低风险")
            
            # 特征对照表
            with st.expander("📊 特征名称对照"):
                st.markdown("""
                | 图中标签 | 中文含义 |
                |---------|---------|
                | ALP | 碱性磷酸酶 |
                | Stages | 自然史分期 |
                | ALBI | ALBI评分（由TBIL和ALB自动计算）|
                | GGT | γ-谷氨酰转肽酶 |
                | C3 | 补体C3 |
                | D_D | D二聚体 |
                | DBIL | 直接胆红素 |
                """)
        
        with col_shap:
            st.subheader("SHAP特征贡献")
            with st.spinner("计算中..."):
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X)
                
                if isinstance(shap_vals, list):
                    sv = shap_vals[1][0]
                    base = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
                else:
                    sv = shap_vals[0]
                    base = explainer.expected_value
                
                # SHAP仍显示ALBI（作为一个整体指标）
                fig, ax = plt.subplots(figsize=(10, 6))
                exp = shap.Explanation(sv, base, X.iloc[0].values, SHAP_LABELS)
                shap.plots.waterfall(exp, max_display=8, show=False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
    except Exception as e:
        st.error(f"预测错误: {e}")
