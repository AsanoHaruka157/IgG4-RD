"""
HC稳态初值、参数、RHS方程定义

包含：
1. HC_init() - 返回HC稳态下的初始条件
2. HC_param() - 返回HC稳态下的参数字典
3. rhs_hc() - 强制 Antigen=0 的RHS方程
4. STATE_NAMES, IDX - 状态变量名和索引
"""

import numpy as np
from typing import Dict

# Central list of all parameter names. Keep this as the single source of truth.
ALL_PARAMS = [
    "k_act_CD4_CTL_antigen_f",
    "k_act_CD4_CTL_basal_f",
    "k_act_CD4_IFN1_f",
    "k_act_CD4_IL_15_d",
    "k_act_CD4_IL_15_f",
    "k_act_CD4_IL_15_m",
    "k_act_CD4_IL_2_d",
    "k_act_CD4_IL_2_f",
    "k_act_CD4_IL_2_m",
    "k_act_CD4_IL_33_d",
    "k_act_CD4_IL_4_d",
    "k_act_CD4_IL_7_d",
    "k_act_CD4_IL_7_f",
    "k_act_CD4_IL_7_m",
    "k_act_CD4_IFN1_d",
    "k_act_CD4_d",
    "k_act_CD4_f",
    "k_act_CD4_m",
    "k_act_CD4_mDC_f",
    "k_act_CD4_mDC_m",
    "k_act_NK_IFN1_d",
    "k_act_NK_IFN1_f",
    "k_act_NK_IFN1_m",
    "k_act_NK_IFN_g_d",
    "k_act_NK_IFN_g_f",
    "k_act_NK_IFN_g_m",
    "k_act_NK_IL_12_d",
    "k_act_NK_IL_12_f",
    "k_act_NK_IL_12_m",
    "k_act_NK_IL_2_d",
    "k_act_NK_IL_2_f",
    "k_act_NK_IL_2_m",
    "k_act_NK_base_f",
    "k_act_NK_d",
    "k_act_NK_f",
    "k_act_NK_m",
    "k_Act_B_Antigen_f",
    "k_Act_B_Antigen_pro_f",
    "k_Act_B_basal_f",
    "k_Act_B_d",
    "k_Act_B_f",
    "k_Act_B_m",
    "k_CD4_CTL_d",
    "k_CD4_CTL_f",
    "k_CD4_CTL_m",
    "k_CD4_f",
    "k_CD4_m",
    "k_GMCSF_Th2_Antigen_f",
    "k_GMCSF_Th2_f",
    "k_GMCSF_act_NK_f",
    "k_GMCSF_d",
    "k_IFN1_CD4_CTL_m",
    "k_IFN1_d",
    "k_IFN1_pDC_f",
    "k_IFN_g_CD4_CTL_f",
    "k_IFN_g_act_NK_f",
    "k_IFN_g_d",
    "k_IL_10_d",
    "k_IL_10_iTreg_f",
    "k_IL_10_nTreg_f",
    "k_IL_10_nTreg_mDC_m",
    "k_IL_12_d",
    "k_IL_12_mDC_f",
    "k_IL_15_Antigen_f",
    "k_IL_15_d",
    "k_IL_15_f",
    "k_IL_1_d",
    "k_IL_1_mDC_f",
    "k_IL_2_act_CD4_Antigen_f",
    "k_IL_2_act_CD4_f",
    "k_IL_2_d",
    "k_IL_33_d",
    "k_IL_33_pDC_f",
    "k_IL_4_Th2_Antigen_f",
    "k_IL_4_Th2_f",
    "k_IL_4_d",
    "k_IL_6_TFH_f",
    "k_IL_6_d",
    "k_IL_6_mDC_f",
    "k_IL_6_pDC_f",
    "k_IL_7_d",
    "k_IL_7_f",
    "k_IgG4_TI_f",
    "k_IgG4_TD_f",
    "k_IgG4_d",
    "k_NK_d",
    "k_NK_f",
    "k_NK_m",
    "k_Naive_B_Antigen_f",
    "k_Naive_B_d",
    "k_Naive_B_f",
    "k_Naive_B_m",
    "k_TD_IL_4_f",
    "k_TD_base_f",
    "k_TD_d",
    "k_TD_f",
    "k_TD_m",
    "k_TFH_IFN1_f",
    "k_TFH_IFN1_m",
    "k_TFH_IL_6_d",
    "k_TFH_IL_6_f",
    "k_TFH_IL_6_m",
    "k_TFH_mDC_Antigen_f",
    "k_TFH_mDC_f",
    "k_TFH_d",
    "k_TFH_f",
    "k_TFH_m",
    "k_TFH_nTreg_m",
    "k_TI_IFN_g_f",
    "k_TI_IL_10_f",
    "k_TI_base_f",
    "k_TI_d",
    "k_TI_f",
    "k_TI_m",
    "k_Th2_IL_10_m",
    "k_Th2_IL_12_m",
    "k_Th2_IL_33_f",
    "k_Th2_IL_33_m",
    "k_Th2_IL_4_f",
    "k_Th2_IL_4_m",
    "k_Th2_TGFbeta_m",
    "k_Th2_d",
    "k_Th2_f",
    "k_Th2_m",
    "k_TGFbeta_CD4_CTL_f",
    "k_TGFbeta_d",
    "k_TGFbeta_iTreg_f",
    "k_TGFbeta_nTreg_f",
    "k_TGFbeta_nTreg_mDC_m",
    "k_act_CD4_IL_33_f",
    "k_iTreg_IL_10_f",
    "k_iTreg_IL_10_m",
    "k_iTreg_IL_1_m",
    "k_iTreg_TGFbeta_f",
    "k_iTreg_TGFbeta_m",
    "k_iTreg_d",
    "k_iTreg_f",
    "k_iTreg_m",
    "k_iTreg_mDC_d",
    "k_iTreg_mDC_f",
    "k_mDC_Antigen_f",
    "k_mDC_GMCSF_f",
    "k_mDC_GMCSF_d",
    "k_mDC_GMCSF_m",
    "k_mDC_IL_10_m",
    "k_mDC_d",
    "k_mDC_f",
    "k_mDC_m",
    "k_naive_CD4_IL_15_d",
    "k_naive_CD4_IL_15_f",
    "k_naive_CD4_IL_15_m",
    "k_naive_CD4_IL_7_d",
    "k_naive_CD4_IL_7_f",
    "k_naive_CD4_IL_7_m",
    "k_naive_CD4_d",
    "k_nDC_d",
    "k_nDC_f",
    "k_nDC_m",
    "k_nTreg_d",
    "k_nTreg_m",
    "k_nTreg_mDC_f",
    "k_nTreg_mDC_m",
    "k_pDC_Antigen_f",
    "k_pDC_d",
    "k_pDC_f",
    "k_pDC_m",
]

# ============================================================================
# 状态变量定义（与main.py/sci.py一致）
# ============================================================================
STATE_NAMES = [
    "Antigen", "nDC", "mDC", "GMCSF", "pDC",
    "IL_33", "IL_6", "IL_12", "IL_15", "IL_7", "IFN1", "IL_1", "IL_2", "IL_4", "IL_10", "TGFbeta", "IFN_g",
    "naive_CD4", "act_CD4", "Th2", "iTreg", "CD4_CTL", "nTreg", "TFH",
    "CD56_NK", "CD16_NK",
    "Naive_B", "Act_B", "TD_Plasma", "TI_Plasma",
    "IgG4",
]

N_STATE = len(STATE_NAMES)
IDX = {n: i for i, n in enumerate(STATE_NAMES)}

def HC_init():
    """
    返回HC（健康对照）稳态下的初始条件字典
    
    Returns:
    --------
    y0_dict : dict，变量名 → 初值
    """
    hc_init = {
        "Antigen": 0.0,
        "nDC": 33.0,
        "mDC": 20.0,
        "pDC": 2.0,
        "naive_CD4": 170.0,
        "act_CD4": 400.0,
        "Th2": 0.30,
        "iTreg": 250.0,
        "CD4_CTL": 1.0,
        "nTreg": 20.0,
        "TFH": 18.0,
        "NK": 50.0,
        "act_NK": 200.0,
        "Naive_B": 88.0,
        "Act_B": 95.0,
        "TD_IS_B": 2.0,
        "TI_IS_B": 0.0,
        "IgG4": 0.0,
        "GMCSF": 3.0e3,
        "IL_33": 0.0,
        "IL_6": 3.5e4,
        "IL_12": 1.5e4,
        "IL_15": 1.0e5,
        "IL_7": 3.5e5,
        "IFN1": 0.0,
        "IL_1": 2.5e3,
        "IL_2": 1.2e4,
        "IL_4": 1.0e2,
        "IL_10": 8.0e3,
        "TGFbeta": 1.0e2,
        "IFN_g": 2.0e5,
    }
    return hc_init


def HC_param():
    """
    返回HC（健康对照）稳态下的参数字典
    
    关键问题修复：避免"所有东西都衰减到0"
    - 对于"只有死亡、无生成"的项，必须加入basal生成项
    - 例如 dX = -k_d * X，若无生成项则X→0，需要 dX = production - k_d * X
    - basal生成项通常很小（如1e-3到1e-2量级），足以维持稳态但不导致无限增长
    
    Returns:
    --------
    params : dict，参数名 → 参数值
    """
    params = {}
    
    # ========== 规则1️⃣：抗原依赖激活参数 → 0.0 ==========
    antigen_dependent = [
        "k_mDC_Antigen_f", "k_mDC_GMCSF_f", "k_pDC_Antigen_f",
        "k_IL_15_Antigen_f",
        "k_GMCSF_Th2_Antigen_f", "k_IL_4_Th2_Antigen_f",
        "k_Act_B_Antigen_f", "k_Naive_B_Antigen_f", "k_Act_B_Antigen_pro_f",
        "k_act_CD4_CTL_antigen_f", "k_TFH_mDC_Antigen_f", "k_IL_2_act_CD4_Antigen_f",
    ]
    for p in antigen_dependent:
        params[p] = 0.0
    
    # ========== 规则2️⃣：IgG4 产生参数 → 0.0 ==========
    params["k_IgG4_TI_f"] = 0.0
    params["k_IgG4_TD_f"] = 0.0
    
    # ========== 规则3️⃣：细胞自我增殖参数 → 0.0 ==========
    proliferation = {
        "k_nDC_f": 0.0, "k_mDC_f": 0.0, "k_pDC_f": 0.0,
        "k_CD4_f": 0.0, "k_act_CD4_f": 0.0,
        "k_Th2_f": 0.0, "k_iTreg_f": 0.0, "k_nTreg_f": 0.0, "k_CD4_CTL_f": 0.0,
        "k_NK_f": 0.0, "k_act_NK_f": 0.0,
        "k_Naive_B_f": 0.0, "k_Act_B_f": 0.0, "k_TD_f": 0.0, "k_TI_f": 0.0, "k_TFH_f": 0.0,
    }
    params.update(proliferation)
    
    # ========== 规则4️⃣：Basal/Homeostatic 参数 - 大幅增加！==========
    # 关键：这些是维持稳态的"生命线"
    
    # ---- 细胞的basal生成参数（非抗原驱动的增殖/激活/分化） ----
    # 这些参数直接决定稳态时各细胞类型的数量
    params["k_nDC_f"] = 0.5  # nDC基础维持（抵消死亡）
    params["k_mDC_f"] = 0.1  # mDC基础维持
    params["k_pDC_f"] = 0.05  # pDC基础维持
    
    params["k_CD4_f"] = 0.8  # naive_CD4基础增殖
    params["k_act_CD4_f"] = 0.6  # act_CD4基础增殖
    params["k_act_CD4_mDC_f"] = 0.2  # mDC诱导naive→act CD4激活
    params["k_act_CD4_IL_2_f"] = 0.1  # IL-2诱导激活
    params["k_naive_CD4_IL_15_f"] = 0.05  # IL-15增强
    params["k_naive_CD4_IL_7_f"] = 0.03  # IL-7增强
    params["k_act_CD4_IL_15_f"] = 0.05  # act_CD4 IL-15
    params["k_act_CD4_IL_7_f"] = 0.03  # act_CD4 IL-7
    
    params["k_Th2_f"] = 0.08  # Th2基础维持
    params["k_Th2_IL_4_f"] = 0.05  # IL-4诱导分化
    params["k_Th2_IL_33_f"] = 0.08  # IL-33诱导分化
    
    params["k_iTreg_f"] = 0.15  # iTreg基础增殖
    params["k_iTreg_mDC_f"] = 0.1  # mDC诱导分化
    params["k_iTreg_TGFbeta_f"] = 0.12  # TGF-β诱导分化
    params["k_iTreg_IL_10_f"] = 0.05  # IL-10增强
    
    params["k_nTreg_f"] = 0.06  # nTreg基础维持
    params["k_nTreg_mDC_f"] = 0.05  # mDC维持nTreg
    
    params["k_CD4_CTL_f"] = 0.08  # CD4_CTL基础增殖
    params["k_act_CD4_CTL_basal_f"] = 0.03  # act_CD4→CD4_CTL基础分化
    
    params["k_TFH_f"] = 0.06  # TFH基础维持
    params["k_TFH_mDC_f"] = 0.08  # mDC诱导分化
    params["k_TFH_IL_6_f"] = 0.05  # IL-6诱导分化
    params["k_TFH_IFN1_f"] = 0.02  # IFN-I诱导
    
    params["k_NK_f"] = 0.6  # NK基础维持
    params["k_act_NK_f"] = 0.4  # act_NK基础维持
    params["k_act_NK_base_f"] = 0.15  # NK→act_NK基础激活
    params["k_act_NK_IL_12_f"] = 0.08  # IL-12诱导
    params["k_act_NK_IL_2_f"] = 0.05  # IL-2诱导
    params["k_act_NK_IFN1_f"] = 0.02  # IFN-I诱导
    params["k_act_NK_IFN_g_f"] = 0.01  # IFN-γ自增强
    
    params["k_Naive_B_f"] = 0.5  # Naive_B基础增殖
    params["k_Act_B_f"] = 0.3  # Act_B基础增殖
    params["k_Act_B_basal_f"] = 0.08  # Naive_B→Act_B基础激活
    
    params["k_TD_f"] = 0.15  # TD_IS_B基础增殖
    params["k_TI_f"] = 0.08  # TI_IS_B基础增殖
    params["k_TD_base_f"] = 0.06  # Act_B→TD_IS_B基础分化
    params["k_TI_base_f"] = 0.02  # Act_B→TI_IS_B基础分化
    params["k_TD_IL_4_f"] = 0.04  # IL-4增强TD分化
    params["k_TI_IFN_g_f"] = 0.03  # IFN-γ增强TI分化
    params["k_TI_IL_10_f"] = 0.02  # IL-10增强TI分化
    
    # ---- 细胞因子生成参数（最关键！）----
    # 这些决定了cytokine的稳态浓度
    # 在稳态时：production_rate = k_d * [cytokine]_ss
    # 所以需要足够大的生成速率来抵消清除
    
    # IL_7: 目标稳态约 3.5e5，k_d通常≈0.05-0.1，则需要生成速率≈1e4-3.5e4
    params["k_IL_7_f"] = 1.0e4  # IL_7 basal生成，非常重要！
    
    # IL_15: 目标稳态约 1.0e5，类似计算
    params["k_IL_15_f"] = 5e3  # IL_15 basal生成
    
    # IL_6: 目标稳态约 3.5e4，由mDC、pDC、TFH贡献
    params["k_IL_6_pDC_f"] = 1e3  # pDC生成IL-6
    params["k_IL_6_mDC_f"] = 2e3  # mDC主要生成
    params["k_IL_6_TFH_f"] = 1e3  # TFH生成
    
    # IL_12: 目标稳态约 1.5e4
    params["k_IL_12_mDC_f"] = 3e3  # mDC生成IL-12
    params["k_IL_12_d"] = 0.05  # 减小清除速率以便更容易达到稳态
    
    # IL_2: 目标稳态约 1.2e4
    params["k_IL_2_act_CD4_f"] = 2e3  # act_CD4生成IL-2
    
    # IL_10: 目标稳态约 8.0e3
    params["k_IL_10_iTreg_f"] = 1e3  # iTreg生成IL-10
    params["k_IL_10_nTreg_f"] = 2e2  # nTreg生成
    
    # IL_4: 目标稳态约 1.0e2
    params["k_IL_4_Th2_f"] = 3e1  # Th2生成IL-4
    
    # IL_1: 目标稳态约 2.5e3
    params["k_IL_1_mDC_f"] = 5e2  # mDC生成IL-1
    
    # GMCSF: 目标稳态约 3.0e3
    params["k_GMCSF_Th2_f"] = 1e2  # Th2生成
    params["k_GMCSF_act_NK_f"] = 5e1  # act_NK生成
    
    # TGFbeta: 目标稳态约 1.0e2
    params["k_TGFbeta_iTreg_f"] = 2e1  # iTreg生成
    params["k_TGFbeta_CD4_CTL_f"] = 1e1  # CD4_CTL生成
    params["k_TGFbeta_nTreg_f"] = 5e0  # nTreg生成
    
    # IFN_g: 目标稳态约 2.0e5
    params["k_IFN_g_CD4_CTL_f"] = 2e3  # CD4_CTL生成
    params["k_IFN_g_act_NK_f"] = 5e3  # act_NK主要生成
    
    # IFN1: 极小值（防止过度激活）
    params["k_IFN1_pDC_f"] = 1e0  # pDC极小生成
    
    # IL_33: 极小值
    params["k_IL_33_pDC_f"] = 1e0  # pDC极小生成
    
    params["k_IgG4_d"] = 0.1  # IgG4清除
    
    # ========== 规则5️⃣：死亡/清除参数 → 0.1 ==========
    death_params = [
        "k_nDC_d", "k_mDC_d", "k_pDC_d", "k_mDC_GMCSF_d",
        "k_IL_33_d", "k_IL_6_d", "k_IL_12_d", "k_IL_15_d", "k_IL_7_d",
        "k_IFN1_d", "k_IL_1_d", "k_IL_2_d", "k_IL_4_d", "k_IL_10_d",
        "k_TGFbeta_d", "k_IFN_g_d", "k_GMCSF_d",
        "k_naive_CD4_d", "k_act_CD4_d", "k_CD4_CTL_d",
        "k_Th2_d", "k_iTreg_d", "k_nTreg_d", "k_TFH_d",
        "k_NK_d", "k_Naive_B_d", "k_Act_B_d", "k_TD_d", "k_TI_d",
        "k_act_NK_d", "k_iTreg_mDC_d",
        "k_act_CD4_IL_33_d", "k_act_CD4_IFN1_d",
        "k_act_NK_IFN1_d", "k_act_NK_IFN_g_d", "k_act_NK_IL_12_d", "k_act_NK_IL_2_d",
        "k_act_CD4_IL_15_d", "k_act_CD4_IL_7_d", "k_act_CD4_IL_4_d", "k_act_CD4_IL_2_d",
        "k_naive_CD4_IL_15_d", "k_naive_CD4_IL_7_d", "k_TFH_IL_6_d",
    ]
    for p in death_params:
        if p not in params:
            params[p] = 0.1
    
    # ========== 规则6️⃣：Hill常数 (_m) ==========
    m_params = [
        # 细胞数相关
        "k_nDC_m", "k_mDC_m", "k_pDC_m",
        "k_CD4_m", "k_act_CD4_m", "k_act_CD4_mDC_m",
        "k_Th2_m", "k_iTreg_m", "k_nTreg_m", "k_CD4_CTL_m", "k_TFH_m",
        "k_NK_m", "k_act_NK_m",
        "k_Naive_B_m", "k_Act_B_m", "k_TD_m", "k_TI_m",
        "k_act_CD4_IL_2_m", "k_act_CD4_IL_15_m", "k_act_CD4_IL_7_m",
        "k_naive_CD4_IL_15_m", "k_naive_CD4_IL_7_m",
        "k_act_NK_IL_12_m", "k_act_NK_IL_2_m", "k_act_NK_IFN_g_m", "k_act_NK_IFN1_m",
        "k_nTreg_mDC_m", "k_TFH_nTreg_m",
    ]
    for p in m_params:
        params[p] = 100
    
    cytokine_m_params = [
        "k_IL_33_m", "k_IL_1_m",
        "k_Th2_IL_33_m", "k_Th2_IL_4_m", "k_Th2_IL_10_m", "k_Th2_IL_12_m", "k_Th2_TGFbeta_m",
        "k_iTreg_IL_1_m", "k_iTreg_IL_10_m", "k_iTreg_TGFbeta_m",
        "k_IL_6_m", "k_IL_12_m", "k_IL_2_m", "k_IL_4_m", "k_IL_10_m", "k_IL_15_m", "k_IL_7_m",
        "k_TFH_IL_6_m", "k_TGFbeta_nTreg_mDC_m", "k_IL_10_nTreg_mDC_m",
        "k_IFN1_CD4_CTL_m", "k_TFH_IFN1_m", "k_mDC_GMCSF_m", "k_mDC_IL_10_m",
    ]
    for p in cytokine_m_params:
        params[p] = 1e4
    
    # ========== 第七步：填充所有尚未设置的参数 ==========
    for param_name in ALL_PARAMS:
        if param_name not in params:
            if param_name.endswith('_f'):
                params[param_name] = 1e-2  # basal参数默认值升高
            elif param_name.endswith('_d'):
                params[param_name] = 0.1
            elif param_name.endswith('_m'):
                params[param_name] = 100
            else:
                params[param_name] = 1e-2
    
    return params


def IgG_param():
    """Variant of HC parameters with all Antigen-related rates reduced."""
    params = HC_param().copy()
    for name in params:
        if "Antigen" in name:
            params[name] = 1e-1
    return params

def rhs_hc(t: float, y: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    """
    HC稳态下的RHS方程（Antigen 取自状态向量 y[Antigen]）

    与 sci.py 的 rhs() 完全相同，但这里不再强制 Antigen=0，
    而是从状态向量读取 Antigen。这允许外部调用者在刺激模拟时
    通过修改 y[0]（Antigen）驱动系统。

    Parameters:
    -----------
    t : 时间（未使用，为了兼容 solve_ivp）
    y : 状态向量 (31维)
    p : 参数字典

    Returns:
    --------
    dy : 导数向量
    """
    # ========== Antigen 直接来自状态向量 ==========
    Antigen = y[IDX["Antigen"]]
    # 默认不自发变化；如需外部刺激，可在包装函数里改写 dAntigen 或 y[0]
    dAntigen = 0.0
    
    # 解包所有状态变量
    nDC       = y[IDX["nDC"]]
    mDC       = y[IDX["mDC"]]
    GMCSF     = y[IDX["GMCSF"]]
    pDC       = y[IDX["pDC"]]
    IL_33     = y[IDX["IL_33"]]
    IL_6      = y[IDX["IL_6"]]
    IL_12     = y[IDX["IL_12"]]
    IL_15     = y[IDX["IL_15"]]
    IL_7      = y[IDX["IL_7"]]
    IFN1      = y[IDX["IFN1"]]
    IL_1      = y[IDX["IL_1"]]
    IL_2      = y[IDX["IL_2"]]
    IL_4      = y[IDX["IL_4"]]
    IL_10     = y[IDX["IL_10"]]
    TGFbeta   = y[IDX["TGFbeta"]]
    IFN_g     = y[IDX["IFN_g"]]
    naive_CD4 = y[IDX["naive_CD4"]]
    act_CD4   = y[IDX["act_CD4"]]
    Th2       = y[IDX["Th2"]]
    iTreg     = y[IDX["iTreg"]]
    CD4_CTL   = y[IDX["CD4_CTL"]]
    nTreg     = y[IDX["nTreg"]]
    TFH       = y[IDX["TFH"]]
    CD56_NK   = y[IDX["CD56_NK"]]
    CD16_NK   = y[IDX["CD16_NK"]]
    Naive_B   = y[IDX["Naive_B"]]
    Act_B     = y[IDX["Act_B"]]
    TD_Plasma = y[IDX["TD_Plasma"]]
    TI_Plasma = y[IDX["TI_Plasma"]]
    IgG4      = y[IDX["IgG4"]]

    # ========== nDC 方程 ==========
    dnDC = (
        p["k_nDC_f"] * nDC * (1 - nDC / p["k_nDC_m"])
        - p["k_mDC_Antigen_f"] * Antigen * nDC * (p["k_mDC_IL_10_m"] / (p["k_mDC_IL_10_m"] + IL_10))
        - p["k_mDC_GMCSF_f"] * Antigen * nDC * (GMCSF / (GMCSF + p["k_mDC_GMCSF_m"])) * (p["k_mDC_IL_10_m"] / (p["k_mDC_IL_10_m"] + IL_10))
        - p["k_pDC_Antigen_f"] * nDC * Antigen
        - p["k_nDC_d"] * nDC
    )

    # ========== mDC 方程 ==========
    dmDC = (
        p["k_mDC_Antigen_f"] * Antigen * nDC * (p["k_mDC_IL_10_m"] / (p["k_mDC_IL_10_m"] + IL_10))
        + p["k_mDC_GMCSF_f"] * Antigen * nDC * (GMCSF / (GMCSF + p["k_mDC_GMCSF_m"])) * (p["k_mDC_IL_10_m"] / (p["k_mDC_IL_10_m"] + IL_10))
        + p["k_mDC_f"] * mDC * (1 - mDC / p["k_mDC_m"])
        - p["k_mDC_d"] * mDC
    )

    # ========== GMCSF 方程 ==========
    dGMCSF = (
        - p["k_mDC_GMCSF_d"] * Antigen * nDC * (GMCSF / (GMCSF + p["k_mDC_GMCSF_m"])) * (p["k_mDC_IL_10_m"] / (p["k_mDC_IL_10_m"] + IL_10))
        + p["k_GMCSF_Th2_f"] * Th2
        + p["k_GMCSF_Th2_Antigen_f"] * Th2 * Antigen
        + p["k_GMCSF_act_NK_f"] * CD16_NK
        - p["k_GMCSF_d"] * GMCSF
    )

    # ========== pDC 方程 ==========
    dpDC = (
        p["k_pDC_Antigen_f"] * nDC * Antigen
        + p["k_pDC_f"] * pDC * (1 - pDC / p["k_pDC_m"])
        - p["k_pDC_d"] * pDC
    )

    # ========== IL_33 方程 ==========
    dIL_33 = (
        p["k_IL_33_pDC_f"] * pDC
        - p["k_act_CD4_IL_33_d"] * act_CD4 * IL_33 / (p["k_Th2_IL_33_m"] + IL_33)
        - p["k_IL_33_d"] * IL_33
    )

    # ========== IL_6 方程 ==========
    dIL_6 = (
        p["k_IL_6_pDC_f"] * pDC
        + p["k_IL_6_mDC_f"] * mDC
        + p["k_IL_6_TFH_f"] * TFH * (p["k_TFH_nTreg_m"] / (nTreg + p["k_TFH_nTreg_m"]))
        - p["k_TFH_IL_6_d"] * act_CD4 * IL_6 / (p["k_TFH_IL_6_m"] + IL_6)
        - p["k_IL_6_d"] * IL_6
    )

    # ========== IL_12 方程 ==========
    dIL_12 = (
        p["k_IL_12_mDC_f"] * mDC
        - p["k_act_NK_IL_12_d"] * CD56_NK * IL_12 / (IL_12 + p["k_act_NK_IL_12_m"])
        - p["k_IL_12_d"] * IL_12
    )

    # ========== IL_15 方程 ==========
    dIL_15 = (
        p["k_IL_15_f"]
        + p["k_IL_15_Antigen_f"] * Antigen
        - p["k_naive_CD4_IL_15_d"] * naive_CD4 * IL_15 / (p["k_naive_CD4_IL_15_m"] + IL_15)
        - p["k_act_CD4_IL_15_d"] * act_CD4 * IL_15 / (p["k_act_CD4_IL_15_m"] + IL_15)
        - p["k_IL_15_d"] * IL_15
    )

    # ========== IL_7 方程 ==========
    dIL_7 = (
        p["k_IL_7_f"]
        - p["k_naive_CD4_IL_7_d"] * naive_CD4 * IL_7 / (p["k_naive_CD4_IL_7_m"] + IL_7)
        - p["k_act_CD4_IL_7_d"] * act_CD4 * IL_7 / (p["k_act_CD4_IL_7_m"] + IL_7)
        - p["k_IL_7_d"] * IL_7
    )

    # ========== IFN1 方程 ==========
    dIFN1 = (
        p["k_IFN1_pDC_f"] * pDC
        - p["k_act_CD4_IFN1_d"] * act_CD4 * IFN1 / (p["k_IFN1_CD4_CTL_m"] + IFN1)
        - p["k_act_NK_IFN1_d"] * CD56_NK * IFN1 / (IFN1 + p["k_act_NK_IFN1_m"])
        - p["k_IFN1_d"] * IFN1
    )

    # ========== IL_1 方程 ==========
    dIL_1 = (
        p["k_IL_1_mDC_f"] * mDC
        - p["k_IL_1_d"] * IL_1
    )

    # ========== IL_2 方程 ==========
    dIL_2 = (
        p["k_IL_2_act_CD4_f"] * act_CD4
        + p["k_IL_2_act_CD4_Antigen_f"] * act_CD4 * Antigen
        - p["k_act_CD4_IL_2_d"] * naive_CD4 * IL_2 / (p["k_act_CD4_IL_2_m"] + IL_2)
        - p["k_act_NK_IL_2_d"] * CD56_NK * IL_2 / (IL_2 + p["k_act_NK_IL_2_m"])
        - p["k_IL_2_d"] * IL_2
    )

    # ========== IL_4 方程 ==========
    dIL_4 = (
        p["k_IL_4_Th2_f"] * Th2
        + p["k_IL_4_Th2_Antigen_f"] * Th2 * Antigen
        - p["k_act_CD4_IL_4_d"] * act_CD4 * IL_4 / (p["k_Th2_IL_4_m"] + IL_4)
        - p["k_IL_4_d"] * IL_4
    )

    # ========== IL_10 方程 ==========
    dIL_10 = (
        p["k_IL_10_iTreg_f"] * iTreg
        + p["k_IL_10_nTreg_f"] * nTreg * mDC / (p["k_IL_10_nTreg_mDC_m"] + mDC)
        - p["k_iTreg_mDC_d"] * act_CD4 * IL_10 / (p["k_iTreg_IL_10_m"] + IL_10)
        - p["k_IL_10_d"] * IL_10
    )

    # ========== TGFbeta 方程 ==========
    dTGFbeta = (
        p["k_TGFbeta_iTreg_f"] * iTreg
        + p["k_TGFbeta_CD4_CTL_f"] * CD4_CTL
        + p["k_TGFbeta_nTreg_f"] * nTreg * mDC / (p["k_TGFbeta_nTreg_mDC_m"] + mDC)
        - p["k_iTreg_mDC_d"] * act_CD4 * TGFbeta / (p["k_iTreg_TGFbeta_m"] + TGFbeta)
        - p["k_TGFbeta_d"] * TGFbeta
    )

    # ========== IFN_g 方程 ==========
    dIFN_g = (
        p["k_IFN_g_CD4_CTL_f"] * CD4_CTL
        + p["k_IFN_g_act_NK_f"] * CD16_NK
        - p["k_act_NK_IFN_g_d"] * CD56_NK * IFN_g / (IFN_g + p["k_act_NK_IFN_g_m"])
        - p["k_IFN_g_d"] * IFN_g
    )

    # ========== naive_CD4 方程 ==========
    dnaive_CD4 = (
        p["k_CD4_f"] * naive_CD4 * (1 - naive_CD4 / p["k_CD4_m"])
        + p["k_naive_CD4_IL_15_f"] * naive_CD4 * (1 - naive_CD4 / p["k_CD4_m"]) * IL_15 / (p["k_naive_CD4_IL_15_m"] + IL_15)
        + p["k_naive_CD4_IL_7_f"] * naive_CD4 * (1 - naive_CD4 / p["k_CD4_m"]) * IL_7 / (p["k_naive_CD4_IL_7_m"] + IL_7)
        - p["k_act_CD4_mDC_f"] * naive_CD4 * mDC / (p["k_act_CD4_mDC_m"] + mDC)
        - p["k_act_CD4_IL_2_f"] * naive_CD4 * IL_2 / (p["k_act_CD4_IL_2_m"] + IL_2)
        - p["k_naive_CD4_d"] * naive_CD4
    )

    # ========== act_CD4 方程 ==========
    dact_CD4 = (
        p["k_act_CD4_mDC_f"] * naive_CD4 * mDC / (p["k_act_CD4_mDC_m"] + mDC)
        + p["k_act_CD4_IL_2_f"] * naive_CD4 * IL_2 / (p["k_act_CD4_IL_2_m"] + IL_2)
        + p["k_act_CD4_f"] * act_CD4 * (1 - act_CD4 / p["k_act_CD4_m"])
        + p["k_act_CD4_IL_15_f"] * act_CD4 * (1 - act_CD4 / p["k_act_CD4_m"]) * IL_15 / (p["k_act_CD4_IL_15_m"] + IL_15)
        + p["k_act_CD4_IL_7_f"] * act_CD4 * (1 - act_CD4 / p["k_act_CD4_m"]) * IL_7 / (p["k_act_CD4_IL_7_m"] + IL_7)
        - act_CD4 * p["k_Th2_f"] * (p["k_Th2_TGFbeta_m"] / (p["k_Th2_TGFbeta_m"] + TGFbeta)) * (p["k_Th2_IL_10_m"] / (p["k_Th2_IL_10_m"] + IL_10)) * (p["k_Th2_IL_12_m"] / (p["k_Th2_IL_12_m"] + IL_12))
        - act_CD4 * p["k_Th2_IL_4_f"] * IL_4 / (p["k_Th2_IL_4_m"] + IL_4) * (p["k_Th2_TGFbeta_m"] / (p["k_Th2_TGFbeta_m"] + TGFbeta)) * (p["k_Th2_IL_10_m"] / (p["k_Th2_IL_10_m"] + IL_10)) * (p["k_Th2_IL_12_m"] / (p["k_Th2_IL_12_m"] + IL_12))
        - act_CD4 * p["k_Th2_IL_33_f"] * IL_33 / (p["k_Th2_IL_33_m"] + IL_33) * (p["k_Th2_TGFbeta_m"] / (p["k_Th2_TGFbeta_m"] + TGFbeta)) * (p["k_Th2_IL_10_m"] / (p["k_Th2_IL_10_m"] + IL_10)) * (p["k_Th2_IL_12_m"] / (p["k_Th2_IL_12_m"] + IL_12))
        - act_CD4 * p["k_iTreg_mDC_f"] * p["k_iTreg_TGFbeta_f"] * TGFbeta / (p["k_iTreg_TGFbeta_m"] + TGFbeta) * (p["k_iTreg_IL_1_m"] / (p["k_iTreg_IL_1_m"] + IL_1))
        - act_CD4 * p["k_iTreg_mDC_f"] * p["k_iTreg_IL_10_f"] * IL_10 / (p["k_iTreg_IL_10_m"] + IL_10) * (p["k_iTreg_IL_1_m"] / (p["k_iTreg_IL_1_m"] + IL_1))
        - p["k_act_CD4_CTL_basal_f"] * act_CD4
        - p["k_act_CD4_CTL_antigen_f"] * act_CD4 * Antigen
        - p["k_act_CD4_IFN1_f"] * act_CD4 * IFN1 / (p["k_IFN1_CD4_CTL_m"] + IFN1)
        - p["k_TFH_mDC_f"] * act_CD4
        - p["k_TFH_mDC_Antigen_f"] * act_CD4 * Antigen
        - p["k_TFH_IFN1_f"] * act_CD4 * IFN1 / (p["k_TFH_IFN1_m"] + IFN1)
        - p["k_TFH_IL_6_f"] * act_CD4 * IL_6 / (p["k_TFH_IL_6_m"] + IL_6)
        - p["k_act_CD4_d"] * act_CD4
    )

    # ========== Th2 方程 ==========
    dTh2 = (
        p["k_Th2_f"] * Th2 * (1 - Th2 / p["k_Th2_m"])
        + act_CD4 * p["k_Th2_f"] * (p["k_Th2_TGFbeta_m"] / (p["k_Th2_TGFbeta_m"] + TGFbeta)) * (p["k_Th2_IL_10_m"] / (p["k_Th2_IL_10_m"] + IL_10)) * (p["k_Th2_IL_12_m"] / (p["k_Th2_IL_12_m"] + IL_12))
        + act_CD4 * p["k_Th2_IL_4_f"] * IL_4 / (p["k_Th2_IL_4_m"] + IL_4) * (p["k_Th2_TGFbeta_m"] / (p["k_Th2_TGFbeta_m"] + TGFbeta)) * (p["k_Th2_IL_10_m"] / (p["k_Th2_IL_10_m"] + IL_10)) * (p["k_Th2_IL_12_m"] / (p["k_Th2_IL_12_m"] + IL_12))
        + act_CD4 * p["k_Th2_IL_33_f"] * IL_33 / (p["k_Th2_IL_33_m"] + IL_33) * (p["k_Th2_TGFbeta_m"] / (p["k_Th2_TGFbeta_m"] + TGFbeta)) * (p["k_Th2_IL_10_m"] / (p["k_Th2_IL_10_m"] + IL_10)) * (p["k_Th2_IL_12_m"] / (p["k_Th2_IL_12_m"] + IL_12))
        - p["k_Th2_d"] * Th2
    )

    # ========== iTreg 方程 ==========
    diTreg = (
        act_CD4 * p["k_iTreg_mDC_f"] * p["k_iTreg_TGFbeta_f"] * TGFbeta / (p["k_iTreg_TGFbeta_m"] + TGFbeta) * (p["k_iTreg_IL_1_m"] / (p["k_iTreg_IL_1_m"] + IL_1))
        + act_CD4 * p["k_iTreg_mDC_f"] * p["k_iTreg_IL_10_f"] * IL_10 / (p["k_iTreg_IL_10_m"] + IL_10) * (p["k_iTreg_IL_1_m"] / (p["k_iTreg_IL_1_m"] + IL_1))
        + p["k_iTreg_f"] * iTreg * (1 - iTreg / p["k_iTreg_m"])
        - p["k_iTreg_d"] * iTreg
    )

    # ========== CD4_CTL 方程 ==========
    dCD4_CTL = (
        p["k_act_CD4_CTL_basal_f"] * act_CD4
        + p["k_act_CD4_CTL_antigen_f"] * act_CD4 * Antigen
        + p["k_act_CD4_IFN1_f"] * act_CD4 * IFN1 / (p["k_IFN1_CD4_CTL_m"] + IFN1)
        + p["k_CD4_CTL_f"] * CD4_CTL * (1 - CD4_CTL / p["k_CD4_CTL_m"])
        - p["k_CD4_CTL_d"] * CD4_CTL
    )

    # ========== nTreg 方程 ==========
    dnTreg = (
        p["k_nTreg_mDC_f"] * nTreg * (1 - nTreg / p["k_nTreg_m"]) * mDC / (p["k_nTreg_mDC_m"] + mDC)
        - p["k_nTreg_d"] * nTreg
    )

    # ========== TFH 方程 ==========
    dTFH = (
        p["k_TFH_mDC_f"] * act_CD4
        + p["k_TFH_mDC_Antigen_f"] * act_CD4 * Antigen
        + p["k_TFH_IFN1_f"] * act_CD4 * IFN1 / (p["k_TFH_IFN1_m"] + IFN1)
        + p["k_TFH_IL_6_f"] * act_CD4 * IL_6 / (p["k_TFH_IL_6_m"] + IL_6)
        + p["k_TFH_f"] * TFH * (1 - TFH / p["k_TFH_m"])
        - p["k_TFH_d"] * TFH
    )

    # ========== CD56_NK 方程 ==========
    dCD56_NK = (
        p["k_NK_f"] * CD56_NK * (1 - CD56_NK / p["k_NK_m"])
        - p["k_act_NK_base_f"] * CD56_NK
        - p["k_act_NK_IL_12_f"] * CD56_NK * IL_12 / (IL_12 + p["k_act_NK_IL_12_m"])
        - p["k_act_NK_IL_2_f"] * CD56_NK * IL_2 / (IL_2 + p["k_act_NK_IL_2_m"])
        - p["k_act_NK_IFN1_f"] * CD56_NK * IFN1 / (IFN1 + p["k_act_NK_IFN1_m"])
        - p["k_act_NK_IFN_g_f"] * CD56_NK * IFN_g / (IFN_g + p["k_act_NK_IFN_g_m"])
        - p["k_NK_d"] * CD56_NK
    )

    # ========== CD16_NK 方程 ==========
    dCD16_NK = (
        p["k_act_NK_base_f"] * CD56_NK
        + p["k_act_NK_IL_12_f"] * CD56_NK * IL_12 / (IL_12 + p["k_act_NK_IL_12_m"])
        + p["k_act_NK_IL_2_f"] * CD56_NK * IL_2 / (IL_2 + p["k_act_NK_IL_2_m"])
        + p["k_act_NK_IFN1_f"] * CD56_NK * IFN1 / (IFN1 + p["k_act_NK_IFN1_m"])
        + p["k_act_NK_IFN_g_f"] * CD56_NK * IFN_g / (IFN_g + p["k_act_NK_IFN_g_m"])
        + p["k_act_NK_f"] * CD16_NK * (1 - CD16_NK / p["k_act_NK_m"])
        - p["k_act_NK_d"] * CD16_NK
    )

    # ========== Naive_B 方程 ==========
    dNaive_B = (
        p["k_Naive_B_f"] * Naive_B * (1 - Naive_B / p["k_Naive_B_m"])
        + p["k_Naive_B_Antigen_f"] * Naive_B * Antigen * (1 - Naive_B / p["k_Naive_B_m"])
        - p["k_Act_B_basal_f"] * Naive_B
        - p["k_Act_B_Antigen_f"] * Naive_B * Antigen
        - p["k_Naive_B_d"] * Naive_B
    )

    # ========== Act_B 方程 ==========
    dAct_B = (
        p["k_Act_B_basal_f"] * Naive_B
        + p["k_Act_B_Antigen_f"] * Naive_B * Antigen
        + p["k_Act_B_f"] * Act_B * (1 - Act_B / p["k_Act_B_m"])
        + p["k_Act_B_Antigen_pro_f"] * Act_B * Antigen * (1 - Act_B / p["k_Act_B_m"])
        - p["k_Act_B_d"] * Act_B
    )

    # ========== TD_Plasma 方程 ==========
    dTD_Plasma = (
        p["k_TD_base_f"] * Act_B
        + p["k_TD_IL_4_f"] * Act_B * IL_4
        + p["k_TD_f"] * TD_Plasma * (1 - TD_Plasma / p["k_TD_m"])
        - p["k_TD_d"] * TD_Plasma
    )

    # ========== TI_Plasma 方程 ==========
    dTI_Plasma = (
        p["k_TI_base_f"] * Act_B
        + p["k_TI_IFN_g_f"] * Act_B * IFN_g
        + p["k_TI_IL_10_f"] * Act_B * IL_10
        + p["k_TI_f"] * TI_Plasma * (1 - TI_Plasma / p["k_TI_m"])
        - p["k_TI_d"] * TI_Plasma
    )

    # ========== IgG4 方程 ==========
    dIgG4 = (
        p["k_IgG4_TI_f"] * 1e8 * TI_Plasma
        + p["k_IgG4_TD_f"] * 1e8 * TD_Plasma
        - p["k_IgG4_d"] * IgG4
    )

    # 返回完整导数向量
    return np.array([
        dAntigen, dnDC, dmDC, dGMCSF, dpDC,
        dIL_33, dIL_6, dIL_12, dIL_15, dIL_7, dIFN1, dIL_1, dIL_2, dIL_4, dIL_10, dTGFbeta, dIFN_g,
        dnaive_CD4, dact_CD4, dTh2, diTreg, dCD4_CTL, dnTreg, dTFH,
        dCD56_NK, dCD16_NK,
        dNaive_B, dAct_B, dTD_Plasma, dTI_Plasma,
        dIgG4
    ], dtype=float)

def HC0_params():
    """
    优化后的不动点参数
    
    使得HC_init()中的初始值y0是系统的不动点：
    dy/dt = f(y0; p) ≈ 0
    
    Returns:
    --------
    params : dict，参数名 → 参数值
    """
    params = {}
    params["k_Act_B_Antigen_f"] = 0.000000000000000e+00
    params["k_Act_B_Antigen_pro_f"] = 0.000000000000000e+00
    params["k_Act_B_basal_f"] = 7.657112440621612e-02
    params["k_Act_B_d"] = 9.893069039561722e-02
    params["k_Act_B_f"] = 2.983366229551116e-01
    params["k_Act_B_m"] = 1.048397539980038e+02
    params["k_CD4_CTL_d"] = 1.770152861116173e-01
    params["k_CD4_CTL_f"] = 4.935559604278415e-02
    params["k_CD4_CTL_m"] = 9.995283918075899e+01
    params["k_CD4_f"] = 2.455414623938323e-01
    params["k_CD4_m"] = 5.304174986636976e+02
    params["k_GMCSF_Th2_Antigen_f"] = 0.000000000000000e+00
    params["k_GMCSF_Th2_f"] = 9.526356498462960e+01
    params["k_GMCSF_act_NK_f"] = 2.494127498663141e+00
    params["k_GMCSF_d"] = 1.758015694598238e-01
    params["k_IFN1_CD4_CTL_m"] = 1.000000000000001e+04
    params["k_IFN1_d"] = 1.000000000000000e-01
    params["k_IFN1_pDC_f"] = 1.209510617784153e-04
    params["k_IFN_g_CD4_CTL_f"] = 1.004823983803905e+02
    params["k_IFN_g_act_NK_f"] = 1.003551895584697e+02
    params["k_IFN_g_d"] = 1.008322461984089e-01
    params["k_IL_10_d"] = 1.714154388122759e-01
    params["k_IL_10_iTreg_f"] = 5.541689728652778e+00
    params["k_IL_10_m"] = 1.000000000000000e+04
    params["k_IL_10_nTreg_f"] = 1.004222701073421e+02
    params["k_IL_10_nTreg_mDC_m"] = 1.004381895282544e+04
    params["k_IL_12_d"] = 1.037850141429764e-02
    params["k_IL_12_m"] = 1.000000000000000e+04
    params["k_IL_12_mDC_f"] = 8.030600757516053e+00
    params["k_IL_15_Antigen_f"] = 0.000000000000000e+00
    params["k_IL_15_d"] = 3.273232800552565e-04
    params["k_IL_15_f"] = 8.484236357445444e+01
    params["k_IL_15_m"] = 1.000000000000000e+04
    params["k_IL_1_d"] = 2.111603966399026e-01
    params["k_IL_1_m"] = 1.000000000000000e+04
    params["k_IL_1_mDC_f"] = 2.639507212535932e+01
    params["k_IL_2_act_CD4_Antigen_f"] = 0.000000000000000e+00
    params["k_IL_2_act_CD4_f"] = 5.407573873064636e+00
    params["k_IL_2_d"] = 1.784192253001379e-01
    params["k_IL_2_m"] = 1.000000000000000e+04
    params["k_IL_33_d"] = 1.000000000000000e-01
    params["k_IL_33_m"] = 1.000000000000000e+04
    params["k_IL_33_pDC_f"] = 1.209510617784153e-04
    params["k_IL_4_Th2_Antigen_f"] = 0.000000000000000e+00
    params["k_IL_4_Th2_f"] = 2.718323202449611e+01
    params["k_IL_4_d"] = 7.780415919441316e-02
    params["k_IL_4_m"] = 1.000000000000000e+04
    params["k_IL_6_TFH_f"] = 9.857685020865901e+01
    params["k_IL_6_d"] = 1.029351310961767e-01
    params["k_IL_6_m"] = 1.000000000000000e+04
    params["k_IL_6_mDC_f"] = 9.742080030760668e+01
    params["k_IL_6_pDC_f"] = 1.003557217611631e+02
    params["k_IL_7_d"] = 1.001639682749864e-04
    params["k_IL_7_f"] = 1.001227609972763e+02
    params["k_IL_7_m"] = 1.000000000000000e+04
    params["k_IgG4_TD_f"] = 0.000000000000000e+00
    params["k_IgG4_TI_f"] = 0.000000000000000e+00
    params["k_IgG4_d"] = 1.000000000000000e-01
    params["k_NK_d"] = 9.410492966120493e-02
    params["k_NK_f"] = 6.409999358765121e-01
    params["k_NK_m"] = 1.185358957940208e+02
    params["k_Naive_B_Antigen_f"] = 0.000000000000000e+00
    params["k_Naive_B_d"] = 9.369080090771800e-02
    params["k_Naive_B_f"] = 5.058830434042246e-01
    params["k_Naive_B_m"] = 1.326424618349616e+02
    params["k_TD_IL_4_f"] = 1.075454716738192e-06
    params["k_TD_base_f"] = 1.997609768897674e-03
    params["k_TD_d"] = 1.783231780711757e-01
    params["k_TD_f"] = 7.994251341162460e-02
    params["k_TD_m"] = 9.904910753374645e+01
    params["k_TFH_IFN1_f"] = 2.000000000000000e-02
    params["k_TFH_IFN1_m"] = 1.000000000000001e+04
    params["k_TFH_IL_6_d"] = 1.003353374476153e-01
    params["k_TFH_IL_6_f"] = 3.565344081658315e-03
    params["k_TFH_IL_6_m"] = 2.022374897827762e+04
    params["k_TFH_d"] = 1.212297336684900e-01
    params["k_TFH_f"] = 3.957014003566890e-02
    params["k_TFH_m"] = 9.118588290584248e+01
    params["k_TFH_mDC_Antigen_f"] = 0.000000000000000e+00
    params["k_TFH_mDC_f"] = 1.766571215912776e-03
    params["k_TFH_nTreg_m"] = 1.002111678826213e+02
    params["k_TGFbeta_CD4_CTL_f"] = 1.005918387369668e+01
    params["k_TGFbeta_d"] = 9.886355774929642e-02
    params["k_TGFbeta_iTreg_f"] = 1.136950637933731e-04
    params["k_TGFbeta_nTreg_f"] = 5.029933319575649e+00
    params["k_TGFbeta_nTreg_mDC_m"] = 1.003494138114710e+04
    params["k_TI_IFN_g_f"] = 9.955206204452296e-07
    params["k_TI_IL_10_f"] = 9.955206204452296e-07
    params["k_TI_base_f"] = 9.955206204452296e-07
    params["k_TI_d"] = 1.000000000000000e-01
    params["k_TI_f"] = 7.999999999999999e-02
    params["k_TI_m"] = 1.000000000000000e+02
    params["k_Th2_IL_10_m"] = 3.045523847857607e+03
    params["k_Th2_IL_12_m"] = 1.247508554676655e+03
    params["k_Th2_IL_33_f"] = 7.999999999999999e-02
    params["k_Th2_IL_33_m"] = 1.000000000000001e+04
    params["k_Th2_IL_4_f"] = 4.783214742387871e-02
    params["k_Th2_IL_4_m"] = 1.058877690623945e+04
    params["k_Th2_TGFbeta_m"] = 9.971578656504977e+03
    params["k_Th2_d"] = 1.451215396957014e-01
    params["k_Th2_f"] = 4.581769296331513e-03
    params["k_Th2_m"] = 9.999716760296194e+01
    params["k_act_CD4_CTL_antigen_f"] = 0.000000000000000e+00
    params["k_act_CD4_CTL_basal_f"] = 3.203835086594402e-04
    params["k_act_CD4_IFN1_d"] = 1.000000000000000e-01
    params["k_act_CD4_IFN1_f"] = 1.000000000000000e-02
    params["k_act_CD4_IL_15_d"] = 8.853926681292469e-02
    params["k_act_CD4_IL_15_f"] = 4.665480218328792e-02
    params["k_act_CD4_IL_15_m"] = 1.000130061562039e+02
    params["k_act_CD4_IL_2_d"] = 1.009671095924272e-01
    params["k_act_CD4_IL_2_f"] = 9.545658759296374e-02
    params["k_act_CD4_IL_2_m"] = 1.003204889520155e+02
    params["k_act_CD4_IL_33_d"] = 1.000000000000000e-01
    params["k_act_CD4_IL_33_f"] = 1.000000000000000e-02
    params["k_act_CD4_IL_4_d"] = 1.000862144160391e-01
    params["k_act_CD4_IL_7_d"] = 9.495052460560600e-02
    params["k_act_CD4_IL_7_f"] = 2.867088626333135e-02
    params["k_act_CD4_IL_7_m"] = 1.000001258587967e+02
    params["k_act_CD4_d"] = 8.849222733448388e-02
    params["k_act_CD4_f"] = 3.104696084787493e-01
    params["k_act_CD4_m"] = 4.472796476772964e+02
    params["k_act_CD4_mDC_f"] = 1.980712114504834e-01
    params["k_act_CD4_mDC_m"] = 1.015957706594708e+02
    params["k_act_NK_IFN1_d"] = 1.000000000000000e-01
    params["k_act_NK_IFN1_f"] = 2.000000000000000e-02
    params["k_act_NK_IFN1_m"] = 1.000000000000000e+02
    params["k_act_NK_IFN_g_d"] = 9.999779357904054e-02
    params["k_act_NK_IFN_g_f"] = 1.001101765577010e-02
    params["k_act_NK_IFN_g_m"] = 1.000348619056858e+02
    params["k_act_NK_IL_12_d"] = 1.004062902609396e-01
    params["k_act_NK_IL_12_f"] = 7.785485603384419e-02
    params["k_act_NK_IL_12_m"] = 1.002837310745972e+02
    params["k_act_NK_IL_2_d"] = 1.003005369735191e-01
    params["k_act_NK_IL_2_f"] = 4.928210490525303e-02
    params["k_act_NK_IL_2_m"] = 1.003996058991740e+02
    params["k_act_NK_base_f"] = 1.402921332313577e-01
    params["k_act_NK_d"] = 8.715801465823918e-02
    params["k_act_NK_f"] = 2.970163041978084e-01
    params["k_act_NK_m"] = 2.129254852859514e+02
    params["k_iTreg_IL_10_f"] = 5.034518831684288e-02
    params["k_iTreg_IL_10_m"] = 9.962070957676277e+03
    params["k_iTreg_IL_1_m"] = 1.005050468683719e+04
    params["k_iTreg_TGFbeta_f"] = 1.208306815791386e-01
    params["k_iTreg_TGFbeta_m"] = 1.001764845502115e+04
    params["k_iTreg_d"] = 5.880775729629839e-03
    params["k_iTreg_f"] = 2.881886048008269e-03
    params["k_iTreg_m"] = 1.460714893533799e+04
    params["k_iTreg_mDC_d"] = 1.015444627165897e-01
    params["k_iTreg_mDC_f"] = 1.007406336187880e-01
    params["k_mDC_Antigen_f"] = 0.000000000000000e+00
    params["k_mDC_GMCSF_d"] = 1.000000000000000e-01
    params["k_mDC_GMCSF_f"] = 0.000000000000000e+00
    params["k_mDC_GMCSF_m"] = 1.000000000000001e+04
    params["k_mDC_IL_10_m"] = 1.000000000000001e+04
    params["k_mDC_d"] = 6.628884072298968e-02
    params["k_mDC_f"] = 8.330783001300870e-02
    params["k_mDC_m"] = 9.791896792412297e+01
    params["k_nDC_d"] = 9.021781376817076e-02
    params["k_nDC_f"] = 1.656884891944422e-01
    params["k_nDC_m"] = 7.244628011569414e+01
    params["k_nTreg_d"] = 1.000340429909296e-04
    params["k_nTreg_f"] = 6.000000000000000e-02
    params["k_nTreg_m"] = 5.114113029052412e+01
    params["k_nTreg_mDC_f"] = 2.602037587816170e-03
    params["k_nTreg_mDC_m"] = 1.038793680814494e+03
    params["k_naive_CD4_IL_15_d"] = 9.522479724642381e-02
    params["k_naive_CD4_IL_15_f"] = 4.675446539043470e-02
    params["k_naive_CD4_IL_15_m"] = 1.001673193565809e+02
    params["k_naive_CD4_IL_7_d"] = 9.798848797651291e-02
    params["k_naive_CD4_IL_7_f"] = 2.878965588053295e-02
    params["k_naive_CD4_IL_7_m"] = 1.000348674522370e+02
    params["k_naive_CD4_d"] = 9.089686348308211e-02
    params["k_pDC_Antigen_f"] = 0.000000000000000e+00
    params["k_pDC_d"] = 2.854375591133581e-02
    params["k_pDC_f"] = 2.912661546330699e-02
    params["k_pDC_m"] = 9.927109143546703e+01
    return params



# ============================================================================
# Baseline steady-state (from HC_init with HC_param at long-run t=100000s)
# ============================================================================
def HC_bl():
    """
    稳态值（直接解 dy/dt=0 得到）
    """
    y0 = {}
    y0["Antigen"] = 0.000000000000000e+00
    y0["nDC"] = 7.999999999999997e+01
    y0["mDC"] = 5.038097723718902e-04
    y0["GMCSF"] = 5.285975659478403e+04
    y0["pDC"] = 2.510833534565723e-13
    y0["IL_33"] = 1.201145518380320e-12
    y0["IL_6"] = 9.828066798626479e+05
    y0["IL_12"] = 1.684600277155585e+01
    y0["IL_15"] = 4.984791564430515e+04
    y0["IL_7"] = 9.984776301668264e+04
    y0["IFN1"] = 1.233772066907432e-12
    y0["IL_1"] = 2.519048862333498e+00
    y0["IL_2"] = 1.502265210572392e+06
    y0["IL_4"] = 7.684696146341679e+02
    y0["IL_10"] = 4.609801790883978e+05
    y0["TGFbeta"] = 1.338081578111910e+04
    y0["IFN_g"] = 5.868532293118685e+06
    y0["naive_CD4"] = 7.727000821654539e+01
    y0["act_CD4"] = 7.511944419842132e+01
    y0["Th2"] = 2.579434529231575e+00
    y0["iTreg"] = 4.610537035727966e+01
    y0["CD4_CTL"] = 4.202732486965451e+01
    y0["nTreg"] = 9.413414364985554e-13
    y0["TFH"] = 9.828709665184974e+01
    y0["CD56_NK"] = 4.641162008954428e+01
    y0["CD16_NK"] = 1.005606441314536e+02
    y0["Naive_B"] = 6.399999999999998e+01
    y0["Act_B"] = 8.641607796638505e+01
    y0["TD_Plasma"] = 1.348814515484752e+03
    y0["TI_Plasma"] = 1.414565344251858e+05
    y0["IgG4"] = 9.413409156823363e-13
    return y0

# ============================================================================
# IgG4 target steady-state (estimated from Figure A and Figure B)
# ============================================================================
def IgG_target():
    """
    IgG4诱导后的稳态值（从实验图估计）
    """
    y0 = {}
    y0["Antigen"] = 1.000000000000000e+00
    y0["nDC"] = 5.000000000000000e+00
    y0["mDC"] = 1.200000000000000e+01
    y0["GMCSF"] = 5.000000000000000e+03
    y0["pDC"] = 5.000000000000000e+00
    y0["IL_33"] = 4.000000000000000e+04
    y0["IL_6"] = 6.000000000000000e+04
    y0["IL_12"] = 2.000000000000000e+04
    y0["IL_15"] = 1.300000000000000e+05
    y0["IL_7"] = 8.000000000000000e+04
    y0["IFN1"] = 1.500000000000000e+00
    y0["IL_1"] = 6.500000000000000e+03
    y0["IL_2"] = 1.300000000000000e+04
    y0["IL_4"] = 1.800000000000000e+03
    y0["IL_10"] = 1.000000000000000e+04
    y0["TGFbeta"] = 5.000000000000000e+01
    y0["IFN_g"] = 1.000000000000000e+06
    y0["naive_CD4"] = 8.000000000000000e+01
    y0["act_CD4"] = 5.000000000000000e+02
    y0["Th2"] = 4.000000000000000e-01
    y0["iTreg"] = 4.500000000000000e+01
    y0["CD4_CTL"] = 4.000000000000000e+00
    y0["nTreg"] = 8.000000000000000e+00
    y0["TFH"] = 3.000000000000000e+01
    y0["CD56_NK"] = 2.000000000000000e+01
    y0["CD16_NK"] = 3.200000000000000e+02
    y0["Naive_B"] = 9.400000000000000e+01
    y0["Act_B"] = 8.000000000000000e+01
    y0["TD_Plasma"] = 1.000000000000000e+01
    y0["TI_Plasma"] = 1.000000000000000e+00
    y0["IgG4"] = 1.400000000000000e+02
    return y0