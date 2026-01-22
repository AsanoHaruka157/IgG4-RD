"""
HC稳态初值、参数、RHS方程定义

包含：
1. HC_state() - 返回HC稳态下的初始条件
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
    "CD56 NK", "CD16 NK",
    "Naive_B", "Act_B", "TD Plasma", "TI Plasma",
    "IgG4",
]

N_STATE = len(STATE_NAMES)
IDX = {n: i for i, n in enumerate(STATE_NAMES)}

# Backward-compatible aliases for previous state names
ALIASES = {
    "NK": "CD56 NK",
    "act_NK": "CD16 NK",
    "TD_IS_B": "TD Plasma",
    "TI_IS_B": "TI Plasma",
}
for old, new in ALIASES.items():
    IDX[old] = IDX[new]

def HC_state():
    """
    返回HC（健康对照）稳态下的初始条件字典
    
    注意：这些值是用户提供的HC稳态参考值
    健康人的IgG4应该接近0
    
    Returns:
    --------
    y0_dict : dict，变量名 → 初值
    """
    hc_state = {
        "Antigen": 0.0,
        # 细胞（16个）
        "nDC": 30.0,
        "mDC": 0.0,
        "pDC": 0.0,
        "naive_CD4": 180.0,
        "act_CD4": 300.0,
        "Th2": 0.3,
        "iTreg": 50.0,
        "CD4_CTL": 1.0,
        "nTreg": 10.0,
        "TFH": 20.0,
        "CD56 NK": 50.0,
        "CD16 NK": 200.0,
        "Naive_B": 86.0,
        "Act_B": 80.0,
        "TD Plasma": 1.5,
        "TI Plasma": 1,

        # 细胞因子和抗体（14个）
        "GMCSF": 3e3,
        "IL_33": 1e0,
        "IL_6": 1.0e4,
        "IL_12": 1e0,
        "IL_15": 1.0e4,
        "IL_7": 3.5e5,
        "IFN1": 1e0,
        "IL_1": 3.0e3,
        "IL_2": 5.0e3,
        "IL_4": 2.5e2,
        "IL_10": 4.0e3,
        "TGFbeta": 5.0e1,
        "IFN_g": 2.0e5,
        "IgG4": 0.0,
    }
    hc_state["NK"] = hc_state["CD56 NK"]
    hc_state["act_NK"] = hc_state["CD16 NK"]
    hc_state["TD_IS_B"] = hc_state["TD Plasma"]
    hc_state["TI_IS_B"] = hc_state["TI Plasma"]
    return hc_state


def IgG_state():
    """
    IgG4诱导图中的目标值（按图中顺序：左到右、上到下）
    包含所有31个状态变量的目标值
    """
    igg = {
        # ========== 抗原（稳态后为1.0） ==========
        "Antigen": 1.0,  # 抗原刺激后稳态为1
        
        # ========== 细胞（16个） ==========
        "nDC": 0.0,
        "mDC": 20.0,
        "pDC": 4.0,
        "naive_CD4": 80.0,
        "act_CD4": 500.0,
        "Th2": 0.3,
        "iTreg": 40.0,
        "CD4_CTL": 3.0,
        "nTreg": 10.0,
        "TFH": 20.0,
        "CD56 NK": 25.0,
        "CD16 NK": 310.0,
        "Naive_B": 94.0,
        "Act_B": 85.0,
        "TD Plasma": 10.0,
        "TI Plasma": 0.1,
        
        # ========== 细胞因子和抗体（14个） ==========
        "GMCSF": 5.2e3,
        "IL_33": 4.0e4,
        "IL_6": 6.0e4,
        "IL_12": 2.0e4,
        "IL_15": 1.4e5,
        "IL_7": 1.0e5,
        "IFN1": 1.0,
        "IL_1": 7.0e3,
        "IL_2": 1.3e4,
        "IL_4": 1.8e3,
        "IL_10": 9.0e3,
        "TGFbeta": 5.0e1,
        "IFN_g": 1.0e6,
        "IgG4": 140.0,
    }
    # aliases for consistency
    igg["NK"] = igg["CD56 NK"]
    igg["act_NK"] = igg["CD16 NK"]
    igg["TD_IS_B"] = igg["TD Plasma"]
    igg["TI_IS_B"] = igg["TI Plasma"]
    return igg


def param():
    """
    模型参数（基于Supplementary Table 7参考值）
    
    这些参数值来自文献中对多组学数据的拟合结果。
    所有参数默认为1.0，然后用文献值覆盖。
    """
    # 初始化所有参数为1.0
    params = {k: 1.0 for k in ALL_PARAMS}
    
    # ========== 基于 Supplementary Table 7 的参数参考值 ==========
    # 高影响参数（mean > 100）
    params["k_IFN_g_CD4_CTL_f"] = 22978.0  # IFN-γ 由 CD4_CTL 产生
    params["k_GMCSF_Th2_Antigen_f"] = 3053.6
    params["k_IL_10_nTreg_f"] = 2774.5
    params["k_IL_4_Th2_Antigen_f"] = 1721.1
    params["k_TI_d"] = 860.8  # TI Plasma 降解 (k_TI_IS_B_cells_d)
    params["k_Naive_B_Antigen_f"] = 814.1
    params["k_IL_6_mDC_f"] = 654.9
    params["k_IL_6_TFH_f"] = 490.5
    params["k_IL_12_mDC_f"] = 478.1
    params["k_IL_2_act_CD4_Antigen_f"] = 443.4
    params["k_IL_33_pDC_f"] = 416.9
    params["k_IL_6_pDC_f"] = 270.6
    params["k_TI_f"] = 266.0  # TI Plasma 自增殖 (k_TI_IS_B_cells_TI_IS_B_cells_f)
    params["k_Act_B_Antigen_f"] = 149.6
    params["k_mDC_GMCSF_f"] = 94.6
    params["k_nDC_f"] = 86.9
    
    # 中等影响参数（10 < mean < 100）
    params["k_act_CD4_IL_7_d"] = 55.6
    params["k_act_CD4_mDC_f"] = 39.5
    params["k_IFN_g_act_NK_f"] = 34.3
    params["k_naive_CD4_IL_7_f"] = 22.4
    params["k_IL_2_d"] = 16.7
    params["k_TD_d"] = 15.5  # TD Plasma 降解 (k_TD_IS_B_cells_d)
    params["k_IL_10_iTreg_f"] = 14.1
    params["k_act_NK_IL_12_f"] = 12.0  # k_pro_act_NK_IL_12_f
    params["k_GMCSF_act_NK_f"] = 11.4
    params["k_mDC_Antigen_f"] = 8.5
    params["k_pDC_Antigen_f"] = 7.9
    params["k_mDC_d"] = 3.2
    params["k_IL_1_mDC_f"] = 3.0
    params["k_CD4_f"] = 2.75
    params["k_naive_CD4_IL_15_f"] = 2.68
    params["k_pDC_d"] = 2.0
    params["k_Act_B_Antigen_pro_f"] = 1.69  # k_Act_B_cells_Antigen_f
    params["k_iTreg_mDC_f"] = 1.21
    params["k_NK_f"] = 1.2
    
    # IgG4 生成参数（关键！）
    params["k_IgG4_TD_f"] = 1.0  # k_IgG4_TD_IS_B_cells_f
    params["k_IgG4_TI_f"] = 1.0  # k_IgG4_TI_IS_B_cells_f
    params["k_TFH_f"] = 1.0
    
    # 较小参数（mean < 1）
    params["k_act_NK_IL_12_f"] = 0.96
    params["k_GMCSF_d"] = 0.83
    params["k_iTreg_d"] = 0.57
    params["k_nTreg_d"] = 0.51
    params["k_TFH_mDC_Antigen_f"] = 0.3
    params["k_nTreg_mDC_f"] = 0.3
    params["k_IL_6_d"] = 0.28
    params["k_act_CD4_CTL_antigen_f"] = 0.25
    params["k_IL_10_d"] = 0.23
    params["k_act_NK_d"] = 0.23
    params["k_IL_4_d"] = 0.19
    params["k_CD4_CTL_f"] = 0.13
    params["k_act_NK_IFN1_f"] = 0.13
    params["k_IFN1_pDC_f"] = 0.12
    params["k_Th2_f"] = 0.12
    params["k_act_CD4_IL_4_d"] = 0.12
    params["k_IL_12_d"] = 0.11
    params["k_iTreg_TGFbeta_f"] = 0.11
    params["k_TGFbeta_iTreg_f"] = 0.09
    params["k_IFN_g_d"] = 0.07
    params["k_IFN1_d"] = 0.06
    params["k_TFH_IFN1_f"] = 0.06
    params["k_TFH_IL_6_f"] = 0.06
    params["k_TGFbeta_CD4_CTL_f"] = 0.06
    params["k_TGFbeta_d"] = 0.06
    params["k_Th2_IL_4_f"] = 0.06
    params["k_act_CD4_IFN1_d"] = 0.06
    params["k_act_CD4_IL_15_d"] = 0.06
    params["k_act_NK_IFN1_d"] = 0.06
    params["k_act_NK_IFN_g_d"] = 0.06
    params["k_iTreg_mDC_d"] = 0.06
    params["k_mDC_GMCSF_d"] = 0.06
    params["k_naive_CD4_IL_15_d"] = 0.06
    params["k_naive_CD4_IL_7_d"] = 0.06
    params["k_naive_CD4_d"] = 0.06
    params["k_Act_B_d"] = 0.05
    params["k_IL_15_d"] = 0.05
    params["k_IL_33_d"] = 0.05
    params["k_TGFbeta_nTreg_f"] = 0.05
    params["k_Th2_IL_33_f"] = 0.05
    params["k_act_CD4_IL_7_f"] = 0.05
    params["k_act_NK_IL_12_d"] = 0.05
    params["k_act_NK_IL_2_d"] = 0.05
    params["k_TI_base_f"] = 0.04  # k_TI_IS_B_cells_base_f
    params["k_act_CD4_d"] = 0.04
    params["k_act_NK_f"] = 0.04
    params["k_iTreg_IL_10_f"] = 0.04
    params["k_IL_7_d"] = 0.03
    params["k_Naive_B_d"] = 0.03
    params["k_TFH_d"] = 0.03
    params["k_Th2_d"] = 0.03
    params["k_act_CD4_IL_2_f"] = 0.03
    params["k_act_CD4_IL_15_f"] = 0.02
    params["k_act_CD4_f"] = 0.02
    params["k_IL_1_d"] = 0.01
    params["k_NK_d"] = 0.01
    params["k_TD_base_f"] = 0.01  # k_TD_IS_B_cells_TD_IS_B_cells_f
    params["k_act_NK_IFN_g_f"] = 0.01
    params["k_act_NK_IL_2_f"] = 0.01
    params["k_act_NK_base_f"] = 0.01
    params["k_nDC_d"] = 0.01
    
    # 极小/零参数
    params["k_TD_IL_4_f"] = 0.0  # k_TD_IS_B_cells_IL_4_f
    
    # IgG4 降解
    params["k_IgG4_d"] = 0.01
    
    return params


def HC_param() -> Dict[str, float]:
    """返回HC稳态使用的参数字典（为后续修改提供副本）。"""
    params = param()  # 生成一份基础参数副本
    return params  # 返回副本


def IgG_param() -> Dict[str, float]:
    """返回IgG诱导阶段的参数初值（当前与HC一致，可在此基础上调整）。"""
    params = param()  # 先获取基础参数
    return params  # 直接返回，可在外部按需修改


def rhs_hc(t: float, y: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    """在健康对照情境下计算RHS（固定抗原为0）。"""
    y_mod = y.copy()  # 复制原始状态以避免原地修改
    y_mod[IDX["Antigen"]] = 0.0  # 强制抗原浓度为0保持稳态背景
    return rhs(t, y_mod, p)  # 调用通用RHS求导

def rhs(t: float, y: np.ndarray, p: Dict[str, float]) -> np.ndarray:
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
    NK   = y[IDX["NK"]]
    act_NK   = y[IDX["act_NK"]]
    Naive_B   = y[IDX["Naive_B"]]
    Act_B     = y[IDX["Act_B"]]
    TD_IS_B = y[IDX["TD_IS_B"]]
    TI_IS_B = y[IDX["TI_IS_B"]]
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
        + p["k_GMCSF_act_NK_f"] * act_NK
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
        - p["k_act_NK_IL_12_d"] * NK * IL_12 / (IL_12 + p["k_act_NK_IL_12_m"])
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
        - p["k_act_NK_IFN1_d"] * NK * IFN1 / (IFN1 + p["k_act_NK_IFN1_m"])
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
        - p["k_act_NK_IL_2_d"] * NK * IL_2 / (IL_2 + p["k_act_NK_IL_2_m"])
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
        + p["k_IFN_g_act_NK_f"] * act_NK
        - p["k_act_NK_IFN_g_d"] * NK * IFN_g / (IFN_g + p["k_act_NK_IFN_g_m"])
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

    # ========== NK 方程 ==========
    dNK = (
        p["k_NK_f"] * NK * (1 - NK / p["k_NK_m"])
        - p["k_act_NK_base_f"] * NK
        - p["k_act_NK_IL_12_f"] * NK * IL_12 / (IL_12 + p["k_act_NK_IL_12_m"])
        - p["k_act_NK_IL_2_f"] * NK * IL_2 / (IL_2 + p["k_act_NK_IL_2_m"])
        - p["k_act_NK_IFN1_f"] * NK * IFN1 / (IFN1 + p["k_act_NK_IFN1_m"])
        - p["k_act_NK_IFN_g_f"] * NK * IFN_g / (IFN_g + p["k_act_NK_IFN_g_m"])
        - p["k_NK_d"] * NK
    )

    # ========== act_NK 方程 ==========
    dact_NK = (
        p["k_act_NK_base_f"] * NK
        + p["k_act_NK_IL_12_f"] * NK * IL_12 / (IL_12 + p["k_act_NK_IL_12_m"])
        + p["k_act_NK_IL_2_f"] * NK * IL_2 / (IL_2 + p["k_act_NK_IL_2_m"])
        + p["k_act_NK_IFN1_f"] * NK * IFN1 / (IFN1 + p["k_act_NK_IFN1_m"])
        + p["k_act_NK_IFN_g_f"] * NK * IFN_g / (IFN_g + p["k_act_NK_IFN_g_m"])
        + p["k_act_NK_f"] * act_NK * (1 - act_NK / p["k_act_NK_m"])
        - p["k_act_NK_d"] * act_NK
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

    # ========== TD_IS_B 方程 ==========
    dTD_IS_B = (
        p["k_TD_base_f"] * Act_B
        + p["k_TD_IL_4_f"] * Act_B * IL_4
        + p["k_TD_f"] * TD_IS_B * (1 - TD_IS_B / p["k_TD_m"])
        - p["k_TD_d"] * TD_IS_B
    )

    # ========== TI_IS_B 方程 ==========
    dTI_IS_B = (
        p["k_TI_base_f"] * Act_B
        + p["k_TI_IFN_g_f"] * Act_B * IFN_g
        + p["k_TI_IL_10_f"] * Act_B * IL_10
        + p["k_TI_f"] * TI_IS_B * (1 - TI_IS_B / p["k_TI_m"])
        - p["k_TI_d"] * TI_IS_B
    )

    # ========== IgG4 方程 ==========
    # 将 1e8 量纲吸收到 k_IgG4_* 参数定义里，避免对参数的极端敏感放大
    dIgG4 = (
        p["k_IgG4_TI_f"] * TI_IS_B
        + p["k_IgG4_TD_f"] * TD_IS_B
        - p["k_IgG4_d"] * IgG4
    )

    # 返回完整导数向量
    return np.array([
        dAntigen, dnDC, dmDC, dGMCSF, dpDC,
        dIL_33, dIL_6, dIL_12, dIL_15, dIL_7, dIFN1, dIL_1, dIL_2, dIL_4, dIL_10, dTGFbeta, dIFN_g,
        dnaive_CD4, dact_CD4, dTh2, diTreg, dCD4_CTL, dnTreg, dTFH,
        dNK, dact_NK,
        dNaive_B, dAct_B, dTD_IS_B, dTI_IS_B,
        dIgG4
    ], dtype=float)

def param_optimized_candidates():
    """返回Stage A找到的候选参数列表"""
    candidates = []
    params = {}
    params["k_Act_B_Antigen_f"] = 2.0924090242e-02
    params["k_Act_B_Antigen_pro_f"] = 2.1482623678e-02
    params["k_Act_B_basal_f"] = 2.6824138668e-01
    params["k_Act_B_d"] = 7.7585462390e-01
    params["k_Act_B_f"] = 3.4889876477e-02
    params["k_Act_B_m"] = 1.0591926356e+02
    params["k_CD4_CTL_d"] = 5.0000000000e+00
    params["k_CD4_CTL_f"] = 3.8613443623e-01
    params["k_CD4_CTL_m"] = 6.3355391665e+00
    params["k_CD4_f"] = 4.0729063652e-02
    params["k_CD4_m"] = 4.7218208665e+01
    params["k_GMCSF_Th2_Antigen_f"] = 1.5894652437e-01
    params["k_GMCSF_Th2_f"] = 6.5532387033e-01
    params["k_GMCSF_act_NK_f"] = 2.4697826467e+00
    params["k_GMCSF_d"] = 2.5850477173e+00
    params["k_IFN1_CD4_CTL_m"] = 5.3233700019e+05
    params["k_IFN1_d"] = 4.9033346646e-01
    params["k_IFN1_pDC_f"] = 4.7443497345e-02
    params["k_IFN_g_CD4_CTL_f"] = 3.0411280404e-02
    params["k_IFN_g_act_NK_f"] = 1.3564310118e+01
    params["k_IFN_g_d"] = 1.6091510125e+00
    params["k_IL_10_d"] = 2.1724514623e+00
    params["k_IL_10_iTreg_f"] = 3.6202125193e-01
    params["k_IL_10_nTreg_f"] = 1.1964324541e+01
    params["k_IL_10_nTreg_mDC_m"] = 2.9881344476e+05
    params["k_IL_12_d"] = 6.5453472805e-02
    params["k_IL_12_mDC_f"] = 3.0488999368e-01
    params["k_IL_15_Antigen_f"] = 2.4850683260e-02
    params["k_IL_15_d"] = 2.2741200218e-01
    params["k_IL_15_f"] = 7.3315140446e-01
    params["k_IL_1_d"] = 3.6069563136e+00
    params["k_IL_1_mDC_f"] = 7.3896326698e-01
    params["k_IL_2_act_CD4_Antigen_f"] = 3.1164101926e-01
    params["k_IL_2_act_CD4_f"] = 1.5736909957e-02
    params["k_IL_2_d"] = 1.1883558498e+00
    params["k_IL_33_d"] = 1.4910603514e-01
    params["k_IL_33_pDC_f"] = 1.2844848683e+01
    params["k_IL_4_Th2_Antigen_f"] = 1.1708045048e-01
    params["k_IL_4_Th2_f"] = 1.6054510281e+00
    params["k_IL_4_d"] = 3.0519339338e+00
    params["k_IL_6_TFH_f"] = 1.1737947803e+00
    params["k_IL_6_d"] = 1.1566469991e-01
    params["k_IL_6_mDC_f"] = 6.2792477319e+00
    params["k_IL_6_pDC_f"] = 3.4174953895e+00
    params["k_IL_7_d"] = 2.5604073180e+00
    params["k_IL_7_f"] = 2.1432891117e-02
    params["k_IgG4_TD_f"] = 1.7687246861e+01
    params["k_IgG4_TI_f"] = 6.7046470826e-01
    params["k_IgG4_d"] = 1.1977497633e+00
    params["k_NK_d"] = 3.4147258450e-01
    params["k_NK_f"] = 1.4498876048e-02
    params["k_NK_m"] = 2.6392583219e+01
    params["k_Naive_B_Antigen_f"] = 9.9378713223e+00
    params["k_Naive_B_d"] = 1.8606774865e-01
    params["k_Naive_B_f"] = 2.3792252675e-01
    params["k_Naive_B_m"] = 1.5892448739e+02
    params["k_TD_IL_4_f"] = 1.0000000000e-02
    params["k_TD_base_f"] = 1.0000000000e-02
    params["k_TD_d"] = 5.0000000000e+00
    params["k_TD_f"] = 3.5753372580e-01
    params["k_TD_m"] = 2.3506102902e+01
    params["k_TFH_IFN1_f"] = 1.6227686725e-02
    params["k_TFH_IFN1_m"] = 3.3568897564e+05
    params["k_TFH_IL_6_d"] = 4.6159384653e+00
    params["k_TFH_IL_6_f"] = 2.1034530885e-01
    params["k_TFH_IL_6_m"] = 3.8763582310e+05
    params["k_TFH_d"] = 4.2699133816e-01
    params["k_TFH_f"] = 2.9238672155e-02
    params["k_TFH_m"] = 4.3374911086e+01
    params["k_TFH_mDC_Antigen_f"] = 1.8926920096e+00
    params["k_TFH_mDC_f"] = 3.4417821908e-01
    params["k_TFH_nTreg_m"] = 1.4968238388e+04
    params["k_TGFbeta_CD4_CTL_f"] = 1.9942459728e-01
    params["k_TGFbeta_d"] = 7.9100278929e-01
    params["k_TGFbeta_iTreg_f"] = 1.3500020513e+00
    params["k_TGFbeta_nTreg_f"] = 2.0585048734e-02
    params["k_TGFbeta_nTreg_mDC_m"] = 1.3492513554e+04
    params["k_TI_IFN_g_f"] = 1.0000000000e-02
    params["k_TI_IL_10_f"] = 1.0000000000e-02
    params["k_TI_base_f"] = 1.0000000000e-02
    params["k_TI_d"] = 5.0000000000e+00
    params["k_TI_f"] = 1.0000000000e-02
    params["k_TI_m"] = 5.0000000000e-01
    params["k_Th2_IL_10_m"] = 1.0149499756e+04
    params["k_Th2_IL_12_m"] = 1.0058275360e+04
    params["k_Th2_IL_33_f"] = 4.8063028271e+00
    params["k_Th2_IL_33_m"] = 1.0000000000e+06
    params["k_Th2_IL_4_f"] = 5.8454488948e-02
    params["k_Th2_IL_4_m"] = 2.0838911827e+05
    params["k_Th2_TGFbeta_m"] = 9.2088827181e+03
    params["k_Th2_d"] = 4.4680327353e-01
    params["k_Th2_f"] = 1.2551725630e-02
    params["k_Th2_m"] = 2.2373879849e+00
    params["k_act_CD4_CTL_antigen_f"] = 1.6591260632e-02
    params["k_act_CD4_CTL_basal_f"] = 1.0000000000e-02
    params["k_act_CD4_IFN1_d"] = 7.3145202277e-02
    params["k_act_CD4_IFN1_f"] = 6.6379740267e-02
    params["k_act_CD4_IL_15_d"] = 6.1584310369e-02
    params["k_act_CD4_IL_15_f"] = 9.6373680906e-02
    params["k_act_CD4_IL_15_m"] = 1.9497927629e+04
    params["k_act_CD4_IL_2_d"] = 1.8742297237e+00
    params["k_act_CD4_IL_2_f"] = 1.7356438199e+01
    params["k_act_CD4_IL_2_m"] = 1.6173152569e+04
    params["k_act_CD4_IL_33_d"] = 9.1008907000e-01
    params["k_act_CD4_IL_33_f"] = 1.3996014958e+00
    params["k_act_CD4_IL_4_d"] = 1.7352481354e+00
    params["k_act_CD4_IL_7_d"] = 1.2632253792e-01
    params["k_act_CD4_IL_7_f"] = 3.3758999699e-02
    params["k_act_CD4_IL_7_m"] = 5.0695237377e+05
    params["k_act_CD4_d"] = 4.2099820843e-01
    params["k_act_CD4_f"] = 2.5446825675e-02
    params["k_act_CD4_m"] = 5.8595963095e+02
    params["k_act_CD4_mDC_f"] = 3.9261565995e-02
    params["k_act_CD4_mDC_m"] = 1.5601863947e+04
    params["k_act_NK_IFN1_d"] = 3.3616664572e+00
    params["k_act_NK_IFN1_f"] = 1.7402257010e-02
    params["k_act_NK_IFN1_m"] = 2.8438641315e+03
    params["k_act_NK_IFN_g_d"] = 8.2119536958e-01
    params["k_act_NK_IFN_g_f"] = 8.4062080594e-01
    params["k_act_NK_IFN_g_m"] = 2.0069806084e+03
    params["k_act_NK_IL_12_d"] = 3.0057716704e+00
    params["k_act_NK_IL_12_f"] = 1.3785657464e+00
    params["k_act_NK_IL_12_m"] = 3.1662596178e+04
    params["k_act_NK_IL_2_d"] = 4.3052877245e+00
    params["k_act_NK_IL_2_f"] = 2.6048281470e+00
    params["k_act_NK_IL_2_m"] = 2.0506727288e+04
    params["k_act_NK_base_f"] = 1.3848625586e-01
    params["k_act_NK_d"] = 4.0481537246e-01
    params["k_act_NK_f"] = 2.0509636386e-02
    params["k_act_NK_m"] = 1.1225536460e+03
    params["k_iTreg_IL_10_f"] = 3.7572223192e-01
    params["k_iTreg_IL_10_m"] = 3.7930724139e+05
    params["k_iTreg_IL_1_m"] = 2.1689112469e+04
    params["k_iTreg_TGFbeta_f"] = 1.1913538260e+01
    params["k_iTreg_TGFbeta_m"] = 4.0363527241e+05
    params["k_iTreg_d"] = 2.4148039148e+00
    params["k_iTreg_f"] = 2.6661567999e+00
    params["k_iTreg_m"] = 4.9331668328e+01
    params["k_iTreg_mDC_d"] = 1.0572025463e+00
    params["k_iTreg_mDC_f"] = 2.7469165138e-02
    params["k_mDC_Antigen_f"] = 4.4140914271e-02
    params["k_mDC_GMCSF_d"] = 1.3945753929e+00
    params["k_mDC_GMCSF_f"] = 1.7265801546e+00
    params["k_mDC_GMCSF_m"] = 7.2361007289e+05
    params["k_mDC_IL_10_m"] = 1.1988137728e+05
    params["k_mDC_d"] = 2.2395216716e-01
    params["k_mDC_f"] = 6.3563710587e-01
    params["k_mDC_m"] = 5.2371132281e+01
    params["k_nDC_d"] = 3.0566225513e-01
    params["k_nDC_f"] = 1.0759263285e-01
    params["k_nDC_m"] = 1.4844504491e+00
    params["k_nTreg_d"] = 7.0447499091e-02
    params["k_nTreg_m"] = 7.1375423781e+00
    params["k_nTreg_mDC_f"] = 2.9135749007e-01
    params["k_nTreg_mDC_m"] = 9.1224966666e+05
    params["k_naive_CD4_IL_15_d"] = 1.1154764168e+00
    params["k_naive_CD4_IL_15_f"] = 9.2398440122e-01
    params["k_naive_CD4_IL_15_m"] = 3.2294380371e+03
    params["k_naive_CD4_IL_7_d"] = 1.6614678511e-01
    params["k_naive_CD4_IL_7_f"] = 1.2426446811e-01
    params["k_naive_CD4_IL_7_m"] = 4.3854577123e+04
    params["k_naive_CD4_d"] = 2.7741986150e-01
    params["k_pDC_Antigen_f"] = 1.8627642329e+01
    params["k_pDC_d"] = 1.4238701693e+00
    params["k_pDC_f"] = 4.8519067219e+00
    params["k_pDC_m"] = 5.4338686962e+00
    candidates.append(params)
    return candidates
