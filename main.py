from __future__ import annotations
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# ============================================================================
# 超参数配置区域 - 方便调整
# ============================================================================
OPTIMIZATION_CONFIG = {
    'max_iter': 100,              # 优化最大迭代次数
    'log_param_lb': -10,          # log参数空间下界（对应约1e-4）
    'log_param_ub': 10,           # log参数空间上界（对应约1e4）
    'progress_bar_total': 100,    # 进度条显示的总迭代次数
    # SciPy优化器选项
    'scipy_method': 'L-BFGS-B',     # 优化器类型: 'L-BFGS-B', 'SLSQP', 'Nelder-Mead'
    'scipy_ftol': 1e-6,             # 函数值容差
    'scipy_gtol': 1e-6,             # 梯度容差
}

SIMULATION_CONFIG = {
    't_end': 300.0,               # 模拟结束时间（秒）
    'time_step': 1.0,             # 时间步长（秒）
}

# ============================================================================
try:
    from tqdm import tqdm
except ImportError:
    # 如果没有tqdm，创建一个简单的占位符
    def tqdm(iterable=None, total=None, desc=None, unit=None, ncols=None):
        class TqdmPlaceholder:
            def __init__(self, *args, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def update(self, n=1):
                pass
            def set_postfix(self, **kwargs):
                pass
            def close(self):
                pass
        return TqdmPlaceholder()
import matplotlib.pyplot as plt
import matplotlib
from utils import (
    ALL_PARAMS,
    STATE_NAMES,
    IDX,
    N_STATE,
    HC_init,
    HC_param,
    IgG_param,
    HC_bl,
    rhs_hc,
)

# 设置matplotlib支持中文显示
try:
    # 尝试使用系统中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    # 如果失败，使用英文标签
    pass


# -------------------------
# 1) 参数向量化：用 log 参数化保证正数
# -------------------------
@dataclass
class ParamSpec:
    names: List[str]
    lb: np.ndarray  # 下界（log空间）
    ub: np.ndarray  # 上界（log空间）

def pack_params(theta_dict: Dict[str, float], spec: ParamSpec) -> np.ndarray:
    """dict -> log(theta) 向量"""
    v = np.array([theta_dict[n] for n in spec.names], dtype=float)
    if np.any(v <= 0):
        raise ValueError("All parameters must be > 0 for log-parameterization.")
    return np.log(v)

def unpack_params(logtheta: np.ndarray, spec: ParamSpec) -> Dict[str, float]:
    """log(theta) 向量 -> dict"""
    v = np.exp(np.asarray(logtheta, dtype=float))
    p = {n: float(v[i]) for i, n in enumerate(spec.names)}
    for k in ALL_PARAMS:
        if k not in p:
            p[k] = 1.0
    return p

# -------------------------
# 2) 31-state RHS：你在这里逐条写“完整版”
#    不要省略。保持变量名与论文一致。
# -------------------------
def rhs(t: float, y: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    # 解包 states（写全比较清楚，也方便你对照论文）
    # Antigen根据时间变化：0-100s为0，100-150s线性增长到1，150s后保持1
    # 计算Antigen的当前值（用于计算其他变量的导数）
    if t < 100.0:
        Antigen = 0.0
    elif t < 150.0:
        Antigen = (t - 100.0) / 50.0  # 100-150s线性增长到1
    else:
        Antigen = 1.0
    # 注意：在rhs函数中使用计算出的Antigen值，而不是y[IDX["Antigen"]]
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
    CD56_NK        = y[IDX["CD56_NK"]]
    CD16_NK    = y[IDX["CD16_NK"]]
    Naive_B   = y[IDX["Naive_B"]]
    Act_B     = y[IDX["Act_B"]]
    TD_Plasma   = y[IDX["TD_Plasma"]]
    TI_Plasma   = y[IDX["TI_Plasma"]]
    IgG4      = y[IDX["IgG4"]]

    # ---- Antigen根据时间变化：0-100s为0，100-150s线性增长到1，150s后保持1 ----
    if t < 100.0:
        dAntigen = 0.0  # 保持为0
    elif t < 150.0:
        dAntigen = 1.0 / 50.0  # 100-150s线性增长，斜率为1/50
    else:
        dAntigen = 0.0  # 150s后保持为1

    dnDC = (
        p["k_nDC_f"] * nDC * (1 - nDC / p["k_nDC_m"])
        - p["k_mDC_Antigen_f"] * Antigen * nDC * (p["k_mDC_IL_10_m"] / (p["k_mDC_IL_10_m"] + IL_10))
        - p["k_mDC_GMCSF_f"] * Antigen * nDC * (GMCSF / (GMCSF + p["k_mDC_GMCSF_m"])) * (p["k_mDC_IL_10_m"] / (p["k_mDC_IL_10_m"] + IL_10))
        - p["k_pDC_Antigen_f"] * nDC * Antigen
        - p["k_nDC_d"] * nDC
    )

    dmDC = (
        p["k_mDC_Antigen_f"] * Antigen * nDC * (p["k_mDC_IL_10_m"] / (p["k_mDC_IL_10_m"] + IL_10))
        + p["k_mDC_GMCSF_f"] * Antigen * nDC * (GMCSF / (GMCSF + p["k_mDC_GMCSF_m"])) * (p["k_mDC_IL_10_m"] / (p["k_mDC_IL_10_m"] + IL_10))
        + p["k_mDC_f"] * mDC * (1 - mDC / p["k_mDC_m"])
        - p["k_mDC_d"] * mDC
    )

    dGMCSF = (
        - p["k_mDC_GMCSF_d"] * Antigen * nDC * (GMCSF / (GMCSF + p["k_mDC_GMCSF_m"])) * (p["k_mDC_IL_10_m"] / (p["k_mDC_IL_10_m"] + IL_10))
        + p["k_GMCSF_Th2_f"] * Th2
        + p["k_GMCSF_Th2_Antigen_f"] * Th2 * Antigen
        + p["k_GMCSF_act_NK_f"] * CD16_NK
        - p["k_GMCSF_d"] * GMCSF
    )

    dpDC = (
        p["k_pDC_Antigen_f"] * nDC * Antigen
        + p["k_pDC_f"] * pDC * (1 - pDC / p["k_pDC_m"])
        - p["k_pDC_d"] * pDC
    )

    # ---- 从这里开始：你把剩下 27 条全部按论文补齐（不要省略）----
    dIL_33 = (
        p["k_IL_33_pDC_f"] * pDC
        - p["k_act_CD4_IL_33_d"] * act_CD4 * IL_33 / (p["k_Th2_IL_33_m"] + IL_33)
        - p["k_IL_33_d"] * IL_33
    )

    dIL_6 = (
        p["k_IL_6_pDC_f"] * pDC
        + p["k_IL_6_mDC_f"] * mDC
        + p["k_IL_6_TFH_f"] * TFH * (p["k_TFH_nTreg_m"] / (nTreg + p["k_TFH_nTreg_m"]))
        - p["k_TFH_IL_6_d"] * act_CD4 * IL_6 / (p["k_TFH_IL_6_m"] + IL_6)
        - p["k_IL_6_d"] * IL_6
    )

    dIL_12 = (
        p["k_IL_12_mDC_f"] * mDC
        - p["k_act_NK_IL_12_d"] * CD56_NK * IL_12 / (IL_12 + p["k_act_NK_IL_12_m"])
        - p["k_IL_12_d"] * IL_12
    )

    dIL_15 = (
        p["k_IL_15_f"]
        + p["k_IL_15_Antigen_f"] * Antigen
        - p["k_naive_CD4_IL_15_d"] * naive_CD4 * IL_15 / (p["k_naive_CD4_IL_15_m"] + IL_15)
        - p["k_act_CD4_IL_15_d"] * act_CD4 * IL_15 / (p["k_act_CD4_IL_15_m"] + IL_15)
        - p["k_IL_15_d"] * IL_15
    )

    dIL_7 = (
        p["k_IL_7_f"]
        - p["k_naive_CD4_IL_7_d"] * naive_CD4 * IL_7 / (p["k_naive_CD4_IL_7_m"] + IL_7)
        - p["k_act_CD4_IL_7_d"] * act_CD4 * IL_7 / (p["k_act_CD4_IL_7_m"] + IL_7)
        - p["k_IL_7_d"] * IL_7
    )

    dIFN1 = (
        p["k_IFN1_pDC_f"] * pDC
        - p["k_act_CD4_IFN1_d"] * act_CD4 * IFN1 / (p["k_IFN1_CD4_CTL_m"] + IFN1)
        - p["k_act_NK_IFN1_d"] * CD56_NK * IFN1 / (IFN1 + p["k_act_NK_IFN1_m"])
        - p["k_IFN1_d"] * IFN1
    )

    dIL_1 = (
        p["k_IL_1_mDC_f"] * mDC
        - p["k_IL_1_d"] * IL_1
    )

    dIL_2 = (
        p["k_IL_2_act_CD4_f"] * act_CD4
        + p["k_IL_2_act_CD4_Antigen_f"] * act_CD4 * Antigen
        - p["k_act_CD4_IL_2_d"] * naive_CD4 * IL_2 / (p["k_act_CD4_IL_2_m"] + IL_2)
        - p["k_act_NK_IL_2_d"] * CD56_NK * IL_2 / (IL_2 + p["k_act_NK_IL_2_m"])
        - p["k_IL_2_d"] * IL_2
    )

    dIL_4 = (
        p["k_IL_4_Th2_f"] * Th2
        + p["k_IL_4_Th2_Antigen_f"] * Th2 * Antigen
        - p["k_act_CD4_IL_4_d"] * act_CD4 * IL_4 / (p["k_Th2_IL_4_m"] + IL_4)
        - p["k_IL_4_d"] * IL_4
    )

    dIL_10 = (
        p["k_IL_10_iTreg_f"] * iTreg
        + p["k_IL_10_nTreg_f"] * nTreg * mDC / (p["k_IL_10_nTreg_mDC_m"] + mDC)
        - p["k_iTreg_mDC_d"] * act_CD4 * IL_10 / (p["k_iTreg_IL_10_m"] + IL_10)
        - p["k_IL_10_d"] * IL_10
    )

    dTGFbeta = (
        p["k_TGFbeta_iTreg_f"] * iTreg
        + p["k_TGFbeta_CD4_CTL_f"] * CD4_CTL
        + p["k_TGFbeta_nTreg_f"] * nTreg * mDC / (p["k_TGFbeta_nTreg_mDC_m"] + mDC)
        - p["k_iTreg_mDC_d"] * act_CD4 * TGFbeta / (p["k_iTreg_TGFbeta_m"] + TGFbeta)
        - p["k_TGFbeta_d"] * TGFbeta
    )

    dIFN_g = (
        p["k_IFN_g_CD4_CTL_f"] * CD4_CTL
        + p["k_IFN_g_act_NK_f"] * CD16_NK
        - p["k_act_NK_IFN_g_d"] * CD56_NK * IFN_g / (IFN_g + p["k_act_NK_IFN_g_m"])
        - p["k_IFN_g_d"] * IFN_g
    )

    dnaive_CD4 = (
        p["k_CD4_f"] * naive_CD4 * (1 - naive_CD4 / p["k_CD4_m"])
        + p["k_naive_CD4_IL_15_f"] * naive_CD4 * (1 - naive_CD4 / p["k_CD4_m"]) * IL_15 / (p["k_naive_CD4_IL_15_m"] + IL_15)
        + p["k_naive_CD4_IL_7_f"] * naive_CD4 * (1 - naive_CD4 / p["k_CD4_m"]) * IL_7 / (p["k_naive_CD4_IL_7_m"] + IL_7)
        - p["k_act_CD4_mDC_f"] * naive_CD4 * mDC / (p["k_act_CD4_mDC_m"] + mDC)
        - p["k_act_CD4_IL_2_f"] * naive_CD4 * IL_2 / (p["k_act_CD4_IL_2_m"] + IL_2)
        - p["k_naive_CD4_d"] * naive_CD4
    )

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

    dTh2 = (
        p["k_Th2_f"] * Th2 * (1 - Th2 / p["k_Th2_m"])
        + act_CD4 * p["k_Th2_f"] * (p["k_Th2_TGFbeta_m"] / (p["k_Th2_TGFbeta_m"] + TGFbeta)) * (p["k_Th2_IL_10_m"] / (p["k_Th2_IL_10_m"] + IL_10)) * (p["k_Th2_IL_12_m"] / (p["k_Th2_IL_12_m"] + IL_12))
        + act_CD4 * p["k_Th2_IL_4_f"] * IL_4 / (p["k_Th2_IL_4_m"] + IL_4) * (p["k_Th2_TGFbeta_m"] / (p["k_Th2_TGFbeta_m"] + TGFbeta)) * (p["k_Th2_IL_10_m"] / (p["k_Th2_IL_10_m"] + IL_10)) * (p["k_Th2_IL_12_m"] / (p["k_Th2_IL_12_m"] + IL_12))
        + act_CD4 * p["k_Th2_IL_33_f"] * IL_33 / (p["k_Th2_IL_33_m"] + IL_33) * (p["k_Th2_TGFbeta_m"] / (p["k_Th2_TGFbeta_m"] + TGFbeta)) * (p["k_Th2_IL_10_m"] / (p["k_Th2_IL_10_m"] + IL_10)) * (p["k_Th2_IL_12_m"] / (p["k_Th2_IL_12_m"] + IL_12))
        - p["k_Th2_d"] * Th2
    )

    diTreg = (
        act_CD4 * p["k_iTreg_mDC_f"] * p["k_iTreg_TGFbeta_f"] * TGFbeta / (p["k_iTreg_TGFbeta_m"] + TGFbeta) * (p["k_iTreg_IL_1_m"] / (p["k_iTreg_IL_1_m"] + IL_1))
        + act_CD4 * p["k_iTreg_mDC_f"] * p["k_iTreg_IL_10_f"] * IL_10 / (p["k_iTreg_IL_10_m"] + IL_10) * (p["k_iTreg_IL_1_m"] / (p["k_iTreg_IL_1_m"] + IL_1))
        + p["k_iTreg_f"] * iTreg * (1 - iTreg / p["k_iTreg_m"])
        - p["k_iTreg_d"] * iTreg
    )

    dCD4_CTL = (
        p["k_act_CD4_CTL_basal_f"] * act_CD4
        + p["k_act_CD4_CTL_antigen_f"] * act_CD4 * Antigen
        + p["k_act_CD4_IFN1_f"] * act_CD4 * IFN1 / (p["k_IFN1_CD4_CTL_m"] + IFN1)
        + p["k_CD4_CTL_f"] * CD4_CTL * (1 - CD4_CTL / p["k_CD4_CTL_m"])
        - p["k_CD4_CTL_d"] * CD4_CTL
    )

    dnTreg = (
        p["k_nTreg_mDC_f"] * nTreg * (1 - nTreg / p["k_nTreg_m"]) * mDC / (p["k_nTreg_mDC_m"] + mDC)
        - p["k_nTreg_d"] * nTreg
    )

    dTFH = (
        p["k_TFH_mDC_f"] * act_CD4
        + p["k_TFH_mDC_Antigen_f"] * act_CD4 * Antigen
        + p["k_TFH_IFN1_f"] * act_CD4 * IFN1 / (p["k_TFH_IFN1_m"] + IFN1)
        + p["k_TFH_IL_6_f"] * act_CD4 * IL_6 / (p["k_TFH_IL_6_m"] + IL_6)
        + p["k_TFH_f"] * TFH * (1 - TFH / p["k_TFH_m"])
        - p["k_TFH_d"] * TFH
    )

    dNK = (
        p["k_NK_f"] * CD56_NK * (1 - CD56_NK / p["k_NK_m"])
        - p["k_act_NK_base_f"] * CD56_NK
        - p["k_act_NK_IL_12_f"] * CD56_NK * IL_12 / (IL_12 + p["k_act_NK_IL_12_m"])
        - p["k_act_NK_IL_2_f"] * CD56_NK * IL_2 / (IL_2 + p["k_act_NK_IL_2_m"])
        - p["k_act_NK_IFN1_f"] * CD56_NK * IFN1 / (IFN1 + p["k_act_NK_IFN1_m"])
        - p["k_act_NK_IFN_g_f"] * CD56_NK * IFN_g / (IFN_g + p["k_act_NK_IFN_g_m"])
        - p["k_NK_d"] * CD56_NK
    )

    dact_NK = (
        p["k_act_NK_base_f"] * CD56_NK
        + p["k_act_NK_IL_12_f"] * CD56_NK * IL_12 / (IL_12 + p["k_act_NK_IL_12_m"])
        + p["k_act_NK_IL_2_f"] * CD56_NK * IL_2 / (IL_2 + p["k_act_NK_IL_2_m"])
        + p["k_act_NK_IFN1_f"] * CD56_NK * IFN1 / (IFN1 + p["k_act_NK_IFN1_m"])
        + p["k_act_NK_IFN_g_f"] * CD56_NK * IFN_g / (IFN_g + p["k_act_NK_IFN_g_m"])
        + p["k_act_NK_f"] * CD16_NK * (1 - CD16_NK / p["k_act_NK_m"])
        - p["k_act_NK_d"] * CD16_NK
    )

    dNaive_B = (
        p["k_Naive_B_f"] * Naive_B * (1 - Naive_B / p["k_Naive_B_m"])
        + p["k_Naive_B_Antigen_f"] * Naive_B * Antigen * (1 - Naive_B / p["k_Naive_B_m"])
        - p["k_Act_B_basal_f"] * Naive_B
        - p["k_Act_B_Antigen_f"] * Naive_B * Antigen
        - p["k_Naive_B_d"] * Naive_B
    )

    dAct_B = (
        p["k_Act_B_basal_f"] * Naive_B
        + p["k_Act_B_Antigen_f"] * Naive_B * Antigen
        + p["k_Act_B_f"] * Act_B * (1 - Act_B / p["k_Act_B_m"])
        + p["k_Act_B_Antigen_pro_f"] * Act_B * Antigen * (1 - Act_B / p["k_Act_B_m"])
        - p["k_Act_B_d"] * Act_B
    )

    dTD_IS_B = (
        p["k_TD_base_f"] * Act_B
        + p["k_TD_IL_4_f"] * Act_B * IL_4
        + p["k_TD_f"] * TD_Plasma * (1 - TD_Plasma / p["k_TD_m"])
        - p["k_TD_d"] * TD_Plasma
    )

    dTI_IS_B = (
        p["k_TI_base_f"] * Act_B
        + p["k_TI_IFN_g_f"] * Act_B * IFN_g
        + p["k_TI_IL_10_f"] * Act_B * IL_10
        + p["k_TI_f"] * TI_Plasma * (1 - TI_Plasma / p["k_TI_m"])
        - p["k_TI_d"] * TI_Plasma
    )

    dIgG4 = (
        p["k_IgG4_TI_f"] * 1e8 * TI_Plasma
        + p["k_IgG4_TD_f"] * 1e8 * TD_Plasma
        - p["k_IgG4_d"] * IgG4
    )

    return np.array([
        dAntigen, dnDC, dmDC, dGMCSF, dpDC,
        dIL_33, dIL_6, dIL_12, dIL_15, dIL_7, dIFN1, dIL_1, dIL_2, dIL_4, dIL_10, dTGFbeta, dIFN_g,
        dnaive_CD4, dact_CD4, dTh2, diTreg, dCD4_CTL, dnTreg, dTFH,
        dNK, dact_NK,
        dNaive_B, dAct_B, dTD_IS_B, dTI_IS_B,
        dIgG4
    ], dtype=float)

# -------------------------
# 3) 仿真：支持多受试者（不同 y0）+ 指定输出时间点
# -------------------------
def simulate(ts: np.ndarray, y0: np.ndarray, p: Dict[str, float], verbose: bool = False, try_bdf_on_fail: bool = True) -> np.ndarray:
    """
    求解ODE系统
    
    Parameters:
    -----------
    ts : 时间点数组
    y0 : 初始条件
    p : 参数字典
    verbose : 是否打印详细信息
    try_bdf_on_fail : 如果LSODA失败，是否尝试BDF方法（对刚性问题更有效）
    """
    ts = np.asarray(ts, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    assert y0.shape == (N_STATE,)

    if verbose:
        print(f"  开始求解ODE: t_span=[{ts[0]:.2f}, {ts[-1]:.2f}], 时间点数={len(ts)}")

    methods_to_try = ["LSODA"]
    if try_bdf_on_fail:
        methods_to_try.append("BDF")  # BDF方法对刚性问题更稳健
    
    for method_idx, method in enumerate(methods_to_try):
        if verbose and method_idx > 0:
            print(f"  尝试备用方法: {method}")
        
        try:
            sol = solve_ivp(
                fun=lambda t, y: rhs(t, y, p),
                t_span=(float(ts[0]), float(ts[-1])),
                y0=y0,
                t_eval=ts,
                method=method,
                rtol=1e-6,
                atol=1e-9,
            )
            
            if verbose:
                if sol.success:
                    print(f"  ODE求解完成 ({method}): 成功, 内部计算了 {len(sol.t)} 个时间步")
                    # 检查是否有警告信息
                    if hasattr(sol, 'message') and sol.message:
                        print(f"  求解器消息: {sol.message}")
                    # 检查是否可能遇到刚性问题（步数异常多）
                    if len(sol.t) > len(ts) * 100:
                        print(f"  警告: 内部时间步数 ({len(sol.t)}) 远大于输出点数 ({len(ts)})，可能遇到刚性问题")
                else:
                    print(f"  ODE求解失败 ({method}): {sol.message if hasattr(sol, 'message') else '未知错误'}")
            
            if sol.success:
                return sol.y.T  # (T, N_STATE)
            elif method_idx == len(methods_to_try) - 1:
                # 所有方法都失败了
                if verbose:
                    print(f"  警告: 所有求解方法都失败，返回NaN")
                return np.full((len(ts), N_STATE), np.nan)
            # 否则继续尝试下一个方法
            
        except Exception as e:
            if verbose:
                print(f"  ODE求解异常 ({method}): {type(e).__name__}: {e}")
            if method_idx == len(methods_to_try) - 1:
                # 最后一个方法也失败了
                return np.full((len(ts), N_STATE), np.nan)
            # 否则继续尝试下一个方法
    
    # 如果所有方法都失败
    return np.full((len(ts), N_STATE), np.nan)

# -------------------------
# 4) 观测映射 + 似然：先给通用接口
#    你后面接论文数据时，把 obs_map / nll 部分按 count/ratio/cytokine细化
# -------------------------
@dataclass
class SubjectData:
    ts: np.ndarray
    y0: np.ndarray
    # 观测：列表形式更灵活，每个观测一个 dict
    # 例如 {"kind":"count","name":"Th2","t_idx":..., "value":..., "sigma":...}
    obs: List[Dict]

def obs_map(sim: np.ndarray, ob: Dict) -> float:
    """把模拟轨迹映射到某一个观测值（单个标量）"""
    t_idx = int(ob["t_idx"])
    kind = ob["kind"]

    if kind == "state":  # 直接观测某个 state（最简）
        name = ob["name"]
        return float(sim[t_idx, IDX[name]])

    if kind == "ratio":
        # ratio = A / (A + B) or A/B：你按论文/数据定义
        num = ob["num"]
        den = ob["den"]
        a = sim[t_idx, IDX[num]]
        b = sim[t_idx, IDX[den]]
        return float(a / (a + b + 1e-12))

    if kind == "cytokine":
        # 如果 cytokine 就是 state，等同 "state"
        name = ob["name"]
        return float(sim[t_idx, IDX[name]])

    raise ValueError(f"Unknown kind: {kind}")

def nll_one(pred: float, ob: Dict) -> float:
    """单个观测的负对数似然"""
    y = float(ob["value"])
    sigma = float(ob.get("sigma", 1.0))

    # 最基础：高斯（你后面可以替换成 lognormal / negbin / beta）
    return 0.5 * ((pred - y) / sigma) ** 2 + np.log(sigma + 1e-12)

def total_nll(logtheta: np.ndarray, spec: ParamSpec, dataset: List[SubjectData], verbose: bool = False) -> float:
    p = unpack_params(logtheta, spec)

    tot = 0.0
    for i, subj in enumerate(dataset):
        if verbose:
            print(f"处理受试者 {i+1}/{len(dataset)}")
        sim = simulate(subj.ts, subj.y0, p, verbose=verbose)
        if np.isnan(sim).any():
            if verbose:
                print(f"  警告: 受试者 {i+1} 的模拟结果包含NaN，返回大惩罚值")
            return 1e50  # 数值爆炸惩罚

        for ob in subj.obs:
            pred = obs_map(sim, ob)
            tot += nll_one(pred, ob)
        
        if verbose:
            print(f"  受试者 {i+1} 处理完成，当前总NLL={tot:.4g}")

    return float(tot)

# -------------------------
# 5) 多起点优化（Scipy minimize + L-BFGS-B）
# -------------------------
def fit_multistart(
    spec: ParamSpec,
    dataset: List[SubjectData],
    n_starts: int = 30,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)

    best_x = None
    best_f = np.inf

    print(f"开始多起点优化: {n_starts} 个起点, {len(dataset)} 个受试者")
    
    # 用于跟踪目标函数调用次数
    call_count = [0]  # 使用列表以便在闭包中修改
    
    for k in range(n_starts):
        print(f"\n{'='*60}")
        print(f"起点 {k+1}/{n_starts}")
        print(f"{'='*60}")
        x0 = rng.uniform(spec.lb, spec.ub)  # 在 log-parameter 空间采样
        call_count[0] = 0  # 重置计数器
        
        # 创建一个包装函数来控制verbose和跟踪调用次数
        def objective(z):
            call_count[0] += 1
            if verbose and call_count[0] % 10 == 0:  # 每10次调用打印一次
                print(f"    目标函数调用次数: {call_count[0]}")
            return total_nll(z, spec, dataset, verbose=(verbose and k == 0 and call_count[0] <= 3))  # 只在第一个起点的前几次调用时详细打印
        
        try:
            res = minimize(
                fun=objective,
                x0=x0,
                method="L-BFGS-B",
                bounds=list(zip(spec.lb, spec.ub)),
                options={"maxiter": 500},
            )
            if res.fun < best_f:
                best_f = float(res.fun)
                best_x = np.array(res.x, dtype=float)
                print(f"[最佳] 新的最佳结果: f={res.fun:.4g} (迭代次数={res.nit}, 函数调用={call_count[0]})")
            else:
                print(f"  当前结果: f={res.fun:.4g} (迭代次数={res.nit}, 函数调用={call_count[0]}, 最佳={best_f:.4g})")
            print(f"  优化状态: success={res.success}, message={res.message if hasattr(res, 'message') else 'N/A'}")
        except Exception as e:
            print(f"  [错误] 优化过程出错: {type(e).__name__}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()

    assert best_x is not None
    print(f"\n{'='*60}")
    print(f"多起点优化完成！最佳NLL: {best_f:.4g}")
    print(f"{'='*60}")
    return best_x, best_f


# -------------------------
# 6) demo：用“假数据”跑通管线（你接真实数据时替换 dataset）
# -------------------------
def demo_dataset() -> List[SubjectData]:
    # 这里只是演示格式：一个受试者，观测 mDC 在 t=0,10 的值
    ts = np.linspace(0, 10, 21)
    y0 = np.ones(N_STATE)

    dataset = [
        SubjectData(
            ts=ts,
            y0=y0,
            obs=[
                {"kind": "state", "name": "mDC", "t_idx": 0,  "value": 1.0, "sigma": 0.5},
                {"kind": "state", "name": "mDC", "t_idx": 20, "value": 1.2, "sigma": 0.5},
            ],
        )
    ]
    return dataset


# -------------------------
# 7) 测试积分：直接运行积分并绘图
# -------------------------
def get_initial_conditions() -> np.ndarray:
    """
    根据参考图片设置合理的初始条件
    
    Returns:
    --------
    y0 : 初始条件数组，形状为 (N_STATE,)
    """
    y0 = np.zeros(N_STATE, dtype=float)
    
    # 根据参考图片设置初始值
    # 第一张图（细胞计数）：
    y0[IDX["Antigen"]] = 0.0      # Antigen初始为0
    y0[IDX["nDC"]] = 35.0         # nDC约30-35
    y0[IDX["mDC"]] = 2.0          # mDC初始为2
    y0[IDX["pDC"]] = 2.0           # pDC约0-5，取2
    y0[IDX["naive_CD4"]] = 170.0  # naive CD4 T约150-180，取170
    y0[IDX["act_CD4"]] = 200.0      # active CD4 T初始为0
    y0[IDX["Th2"]] = 0.3          # Th2约0.1-0.5，取0.3
    y0[IDX["iTreg"]] = 50.0       # iTreg约0-50，取25
    y0[IDX["CD4_CTL"]] = 1.0      # CD4 CTL初始为0
    y0[IDX["nTreg"]] = 2.0       # nTreg约10-30，取20
    y0[IDX["TFH"]] = 5.0         # TFH约10-20，取15
    y0[IDX["CD56_NK"]] = 50.0          # NK约50
    y0[IDX["CD16_NK"]] = 200.0     # act_NK约180-200，取190
    y0[IDX["Naive_B"]] = 92.0     # Naive B约84-92，取88
    y0[IDX["Act_B"]] = 80.0       # Act_B约80
    y0[IDX["TD_Plasma"]] = 2.0      # TD-Plasma初始为0
    y0[IDX["TI_Plasma"]] = 0.0      # TI-Plasma初始为0
    y0[IDX["IgG4"]] = 0.0         # IgG4初始为0
    
    # 第二张图（细胞因子），初始值都较低
    y0[IDX["GMCSF"]] = 3e+3        # GMCSF初始较低
    y0[IDX["IL_33"]] = 0.0        # IL_33初始较低
    y0[IDX["IL_6"]] = 2e+4         # IL_6初始较低
    y0[IDX["IL_12"]] = 0.0        # IL_12初始较低
    y0[IDX["IL_15"]] = 9e+4        # IL_15初始较低
    y0[IDX["IL_7"]] = 3.5e+5         # IL_7初始较低
    y0[IDX["IFN1"]] = 0.0         # IFN1初始较低
    y0[IDX["IL_1"]] = 3e+3         # IL_1初始较低
    y0[IDX["IL_2"]] = 8e+3         # IL_2初始较低
    y0[IDX["IL_4"]] = 2.5e+2         # IL_4初始较低
    y0[IDX["IL_10"]] = 6e+3        # IL_10初始较低
    y0[IDX["TGFbeta"]] = 1e+1      # TGFbeta初始较低
    y0[IDX["IFN_g"]] = 2e+5        # IFN_g初始较低
    
    return y0

def test_integration(t_end: float = 300.0, n_points: int = 300, verbose: bool = True):
    """
    测试ODE积分是否能正常运行，并绘制16个变量的时间序列图
    
    Parameters:
    -----------
    t_end : 积分结束时间
    n_points : 时间点数量
    verbose : 是否打印详细信息
    """
    print("="*60)
    print("测试ODE积分")
    print("="*60)
    
    # 创建时间点（时间步为1，从0到300，共301个点）
    ts = np.arange(0, int(t_end) + 1, 1.0)  # 0, 1, 2, ..., 300
    
    # 设置初始条件（根据参考图片设置合理初值）
    y0 = get_initial_conditions()
    print(f"初始条件: 根据参考图片设置")
    print(f"  主要细胞类型:")
    print(f"    nDC: {y0[IDX['nDC']]:.1f}, naive_CD4: {y0[IDX['naive_CD4']]:.1f}, CD56_NK: {y0[IDX['CD56_NK']]:.1f}")
    print(f"    CD16_NK: {y0[IDX['CD16_NK']]:.1f}, Naive_B: {y0[IDX['Naive_B']]:.1f}, Act_B: {y0[IDX['Act_B']]:.1f}")
    print(f"  细胞因子: 初始值均为0")
    print(f"时间范围: [0, {t_end}], 时间点数: {n_points}")
    
    # 设置默认参数（所有参数为1.0）
    p = {name: 1.0 for name in ALL_PARAMS}
    print(f"参数设置: 所有参数 = 1.0 (默认值)")
    print()
    
    # 运行积分
    print("开始积分...")
    result = simulate(ts, y0, p, verbose=verbose)
    
    # 检查结果
    if np.isnan(result).any():
        print("\n[错误] 积分失败：结果包含NaN值")
        return None
    
    print(f"\n[成功] 积分成功！结果形状: {result.shape}")
    print(f"  时间点数: {len(ts)}, 状态变量数: {N_STATE}")
    
    # 绘制16个变量的时间序列图（按照论文图片顺序）
    print("\n绘制16个变量的时间序列图...")
    plot_results(ts, result)
    
    return result

def plot_results(ts: np.ndarray, result: np.ndarray):
    """
    绘制16个变量的时间序列图（按照论文图片顺序）
    
    Parameters:
    -----------
    ts : 时间点数组
    result : 积分结果，形状为 (n_time, n_state)
    """
    # 按照论文图片的顺序定义要绘制的变量
    plot_order = [
        "nDC",           # 1
        "mDC",           # 2
        "pDC",           # 3
        "naive_CD4",     # 4
        "act_CD4",       # 5
        "Th2",           # 6
        "iTreg",         # 7
        "CD4_CTL",       # 8
        "nTreg",         # 9
        "TFH",           # 10
        "CD56_NK",            # 11
        "CD16_NK",        # 12
        "Naive_B",       # 13
        "Act_B",         # 14
        "TD_Plasma",       # 15
        "TI_Plasma",       # 16
    ]
    
    n_vars = len(plot_order)
    
    # 创建子图：4行4列，缩小尺寸
    fig, axes = plt.subplots(4, 4, figsize=(14, 12))
    fig.suptitle('ODE Integration Results - 16 Variables', fontsize=16, fontweight='bold', y=0.995)
    
    axes = axes.flatten()
    
    for i, var_name in enumerate(plot_order):
        if var_name not in IDX:
            print(f"Warning: Variable '{var_name}' not found in STATE_NAMES, skipping...")
            continue
            
        ax = axes[i]
        var_idx = IDX[var_name]
        
        # 使用更好看的颜色（深蓝紫色）
        ax.plot(ts, result[:, var_idx], color='#2E86AB', linewidth=2.5)
        # 去掉X轴和Y轴标签，只保留标题
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(var_name, fontsize=11, fontweight='bold', pad=8)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # 为顶部标题留出空间
    
    # 保存图片
    output_file = 'ode_integration_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  图片已保存: {output_file}")
    
    # 显示图片
    plt.show()
    print("  图片已显示")


# -------------------------
# 8) 参数优化：根据目标值优化未知参数
# -------------------------
def get_known_parameters() -> Dict[str, float]:
    """
    从表格中提取已知参数及其值（使用Value 1列）
    
    Returns:
    --------
    known_params : 已知参数字典
    """
    # 从表格中提取的已知参数（Value 1列）
    # 注意：参数名需要与代码中的ALL_PARAMS列表匹配
    known_params = {
        # 第一个表格（32个参数，按mean排序）
        "k_IFN_g_CD4_CTL_f": 22978.01,
        "k_GMCSF_Th2_Antigen_f": 1.69,
        "k_IL_10_nTreg_f": 1.21,
        "k_IL_4_Th2_Antigen_f": 1.2,
        "k_TI_IS_B_d": 1.0,  # k_TI_IS_B_cells_d
        "k_Naive_B_Antigen_f": 0.96,
        "k_IL_6_mDC_f": 0.83,
        "k_IL_6_TFH_f": 0.57,
        "k_IL_12_mDC_f": 0.51,
        "k_IL_2_act_CD4_Antigen_f": 0.3,
        "k_IL_33_pDC_f": 0.3,
        "k_IL_6_pDC_f": 0.28,
        "k_Act_B_Antigen_pro_f": 0.25,  # k_Act_B_cells_Antigen_f
        "k_mDC_GMCSF_f": 0.23,
        "k_mDC_GMCSF_d": 0.23,
        "k_nDC_f": 0.19,
        "k_act_CD4_IL_7_d": 0.13,
        "k_act_CD4_mDC_f": 0.13,
        "k_IFN_g_act_NK_f": 0.12,
        "k_naive_CD4_IL_7_f": 0.12,
        "k_IL_2_d": 0.11,
        "k_TD_IS_B_d": 0.09,  # k_TD_IS_B_cells_d
        "k_IL_10_iTreg_f": 0.08,
        "k_pro_act_NK_IL_12_f": 0.07,
        "k_GMCSF_act_NK_f": 0.06,
        "k_mDC_Antigen_f": 0.06,
        "k_pDC_Antigen_f": 0.06,
        "k_pDC_f": 0.06,
        "k_mDC_d": 0.06,
        "k_mDC_f": 0.06,
        "k_IL_1_mDC_f": 0.06,
        "k_CD4_f": 0.05,
        "k_naive_CD4_IL_15_f": 0.05,
        "k_pDC_d": 2.0,
        
        # 第二个表格（37个参数）
        "k_Act_B_Antigen_f": 1.69,
        "k_iTreg_mDC_f": 1.21,
        "k_NK_f": 1.2,
        "k_IgG4_TD_f": 0.0,  # k_IgG4_TD_IS_B_cells_f - HC中IgG4不产生
        "k_IgG4_TI_f": 0.0,  # k_IgG4_TI_IS_B_cells_f - HC中IgG4不产生
        "k_TFH_f": 1.0,
        "k_act_NK_IL_12_f": 0.96,
        "k_GMCSF_d": 0.83,
        "k_iTreg_d": 0.57,
        "k_nTreg_d": 0.51,
        "k_TFH_mDC_Antigen_f": 0.3,
        "k_nTreg_mDC_f": 0.3,
        "k_IL_6_d": 0.28,
        "k_act_CD4_CTL_antigen_f": 0.25,
        "k_IL_10_d": 0.23,
        "k_act_NK_d": 0.23,
        "k_IL_4_d": 0.19,
        "k_CD4_CTL_f": 0.13,  # k_CD4_CTL_CD4_CTL_f
        "k_act_NK_IFN1_f": 0.13,
        "k_IFN1_pDC_f": 0.12,
        "k_Th2_f": 0.12,
        "k_act_CD4_IL_4_d": 0.12,
        "k_IL_12_d": 0.11,
        "k_iTreg_TGFbeta_f": 0.11,
        "k_TGFbeta_iTreg_f": 0.09,
        "k_CD4_CTL_d": 0.08,
        "k_IFN_g_d": 0.07,
        "k_IFN1_d": 0.06,
        "k_TFH_IFN1_f": 0.06,
        "k_TFH_IL_6_f": 0.06,
        "k_TGFbeta_CD4_CTL_f": 0.06,
        "k_TGFbeta_d": 0.06,
        "k_Th2_IL_4_f": 0.06,
        "k_act_CD4_IFN1_d": 0.06,
        "k_act_CD4_IL_15_d": 0.06,
        "k_act_NK_IFN1_d": 0.06,
        "k_act_NK_IFN_g_d": 0.06,
        
        # 第三个表格（35个参数）
        "k_iTreg_mDC_d": 0.06,
        "k_mDC_GMCSF_d": 0.06,
        "k_naive_CD4_IL_15_d": 0.06,
        "k_Act_B_d": 0.06,  # k_Act_B_cells_d
        "k_TGFbeta_nTreg_f": 0.06,
        "k_act_CD4_IFN1_f": 0.06,
        "k_TI_base_f": 0.06,  # k_TI_IS_B_cells_base_f
        "k_TD_IL_4_f": 0.06,  # k_TD_IS_B_cells_IL_4_f
        "k_naive_CD4_d": 0.05,
        "k_IL_15_d": 0.05,
        "k_Naive_B_d": 0.05,  # k_Naive_B_cells_d
        "k_act_NK_base_f": 0.05,
        "k_act_CD4_d": 0.05,
        "k_IL_7_d": 0.05,
        "k_CD4_CTL_d": 0.05,  # 重复，使用上面的值
        "k_Act_B_f": 0.05,  # k_Act_B_Act_B_f
        "k_nTreg_f": 0.05,
        "k_IL_15_f": 0.05,
        "k_TGFbeta_nTreg_f": 0.05,  # 重复
        "k_IL_7_f": 0.05,
        "k_naive_CD4_IL_7_d": 0.05,
        "k_Act_B_d": 0.05,  # 重复
        "k_NK_d": 0.05,
        "k_nDC_d": 0.05,
        "k_act_CD4_mDC_m": 0.05,
        "k_IL_2_act_CD4_f": 0.05,
        "k_act_NK_IL_12_m": 0.05,
        "k_naive_CD4_IL_15_d": 0.05,  # 重复
        "k_act_CD4_IL_15_d": 0.05,  # 重复
        "k_IL_15_Antigen_f": 0.05,
        "k_act_CD4_IL_15_f": 0.05,
        "k_act_CD4_f": 0.05,
        "k_NK_m": 0.05,
        "k_CD4_m": 0.05,
        "k_nDC_m": 0.05,
        "k_TFH_m": 0.05,
        "k_act_CD4_m": 0.05,
        "k_nTreg_m": 0.05,
        "k_TD_m": 0.05,  # k_TD_IS_B_cells_TD_IS_B_cells_m
        "k_mDC_m": 0.05,
        "k_act_NK_m": 0.05,
        "k_iTreg_m": 0.05,
        "k_CD4_CTL_m": 0.05,  # k_CD4_CTL_CD4_CTL_m
        "k_Th2_TGFbeta_m": 0.05,
        "k_iTreg_TGFbeta_m": 0.05,
        "k_act_NK_IFN1_m": 0.05,
        "k_TGFbeta_nTreg_mDC_m": 0.05,
        "k_TFH_IFN1_m": 0.05,
        "k_mDC_IL_10_m": 0.05,
        "k_mDC_GMCSF_m": 0.05,
        "k_iTreg_IL_10_m": 0.05,
        "k_act_CD4_IL_2_m": 0.05,
        "k_act_CD4_IFN1_m": 0.05,
        "k_act_NK_IL_2_m": 0.05,
        "k_act_NK_IFN_g_m": 0.05,
        "k_IL_10_nTreg_mDC_m": 0.05,
        "k_act_CD4_IL_15_m": 0.05,
        "k_naive_CD4_IL_7_m": 0.05,
        "k_naive_CD4_IL_15_m": 0.05,
        "k_Th2_IL_33_f": 0.05,
        "k_Th2_IL_10_m": 0.05,
        "k_iTreg_IL_1_m": 0.05,
        "k_act_CD4_CTL_basal_f": 0.05,
        "k_act_CD4_IL_7_m": 0.05,
        "k_act_CD4_IL_33_d": 0.05,
        "k_act_CD4_IL_33_f": 0.05,
        "k_TGFbeta_nTreg_mDC_f": 0.05,
        "k_Th2_IL_4_f": 0.05,  # 重复
        "k_TFH_mDC_f": 0.05,
        "k_Th2_IL_10_f": 0.05,
        "k_iTreg_IL_10_f": 0.05,
        "k_Th2_TGFbeta_f": 0.05,
        "k_act_CD4_IL_2_f": 0.05,
        "k_act_NK_IL_2_f": 0.05,
        "k_act_NK_IL_12_d": 0.05,
        "k_act_NK_base_d": 0.05,
        "k_act_NK_IFN1_d": 0.05,  # 重复
        "k_act_NK_IFN_g_f": 0.05,
        "k_Th2_d": 0.05,
        "k_TFH_IL_6_d": 0.05,
        "k_IL_33_d": 0.05,
        "k_IL_1_d": 0.05,
        "k_IL_15_d": 0.05,  # 重复
        "k_IL_7_d": 0.05,  # 重复
        "k_Naive_B_d": 0.05,  # 重复
        "k_act_CD4_d": 0.05,  # 重复
        "k_IL_7_d": 0.05,  # 重复
        "k_iTreg_mDC_d": 0.05,  # 重复
        "k_CD4_CTL_d": 0.05,  # 重复
        "k_Act_B_f": 0.05,  # 重复
        "k_nTreg_f": 0.05,  # 重复
        "k_IL_15_f": 0.05,  # 重复
        "k_TGFbeta_nTreg_f": 0.05,  # 重复
        "k_IL_7_f": 0.05,  # 重复
        "k_naive_CD4_IL_7_d": 0.05,  # 重复
        "k_Act_B_d": 0.05,  # 重复
        "k_NK_d": 0.05,  # 重复
        "k_nDC_d": 0.05,  # 重复
        
        # 第四个表格（13个参数）
        "k_TD_IS_B_TD_IS_B_m": 1.09,  # k_TD_IS_B_cells_TD_IS_B_cells_m
        "k_TFH_m": 0.28,
        "k_Th2_m": 0.28,
        "k_TI_IS_B_TI_IS_B_m": 0.18,  # k_TI_IS_B_cells_TI_IS_B_cells_m
        "k_TFH_nTreg_m": 0.16,
        "k_nTreg_mDC_m": 0.16,
        "k_IFN1_CD4_CTL_m": 0.06,
        "k_TFH_IFN1_m": 0.06,  # 重复
        "k_TGFbeta_nTreg_mDC_m": 0.06,  # 重复
        "k_act_NK_IFN1_m": 0.06,  # 重复
        "k_iTreg_TGFbeta_m": 0.06,  # 重复
        "k_Th2_TGFbeta_m": 0.04,
        "k_CD4_CTL_CD4_CTL_m": 0.03,
        
        # 从第三个表格的详细数据中提取更多参数（使用Value 1列）
        "k_TI_IFN_g_f": 24750.2,
        "k_TI_IL_10_f": 7547.31,
        "k_iTreg_f": 117927.9,
        "k_IL_7_f": 67190.33,  # 覆盖之前的值
        "k_IL_15_Antigen_f": 21639.64,
        "k_naive_CD4_IL_7_m": 14800.21,
        "k_mDC_IL_10_m": 12749.46,
        "k_act_CD4_IL_7_m": 880.36,
        "k_pro_act_NK_IL_12_m": 709.93,
        "k_act_NK_IL_12_m": 583.55,
        "k_act_NK_IL_2_m": 580.0,
        "k_act_CD4_IL_2_m": 573.6,
        "k_Th2_IL_33_m": 516.94,
        "k_act_CD4_IL_15_m": 472.39,
        "k_act_NK_IFN_g_m": 466.53,
        "k_naive_CD4_IL_15_m": 461.37,
        "k_Th2_IL_4_m": 429.92,
        "k_Th2_IL_12_m": 419.03,
        "k_mDC_GMCSF_m": 418.36,
        "k_Th2_IL_10_m": 326.73,
        "k_iTreg_IL_10_m": 270.21,
        "k_TFH_IL_6_m": 207.64,
        "k_act_NK_m": 105.32,
        "k_act_CD4_m": 100.75,
        "k_CD4_m": 86.78,
        "k_Naive_B_m": 67.2,
        "k_iTreg_IL_1_m": 59.33,
        "k_Act_B_m": 39.82,
        "k_act_CD4_mDC_m": 35.85,
        "k_NK_m": 33.59,
        "k_iTreg_m": 13.85,
        "k_nDC_m": 7.24,
        "k_mDC_m": 2.54,
        "k_nTreg_m": 7.0,
        "k_pDC_m": 2.0,
        "k_IL_10_nTreg_mDC_m": 0.0,
    }
    
    # 只保留在ALL_PARAMS中存在的参数
    filtered_params = {}
    for param_name, param_value in known_params.items():
        if param_name in ALL_PARAMS:
            filtered_params[param_name] = param_value
    
    return filtered_params

def get_hc_friendly_parameters() -> Dict[str, float]:
    """
    生成HC（健康对照）稳态友好的参数字典 params_init
    
    HC参数初始化规则（按优先级）：
    
    1️⃣ 抗原依赖激活参数（HC下必须关闭）
       - 所有描述抗原诱导激活、分化、成熟、类切换的参数（如 naive→activated、B→plasma、plasma→IgG4）
       - 初值设为 0.0，确保其在 RHS 中严格乘以 A(t)，因为 A(t)=0 时应完全关闭
    
    2️⃣ IgG4 产生相关参数
       - 所有 IgG4 产生速率参数（TD/TI plasma → IgG4）初值设为 0.0
    
    3️⃣ Basal / Homeostatic 生成参数（维持稳态的基本参数）
       - 所有非抗原依赖的 basal 生成项、背景分泌项
       - 根据目标稳态值精心设计，一般范围 1e-4 ~ 1e-2（推荐 1e-3）
    
    4️⃣ 死亡 / 清除 / 抑制参数
       - 所有细胞死亡、细胞因子清除、抑制项参数
       - 初值设为 0.1（如无更可靠信息）
    
    5️⃣ Hill / 饱和常数（K值，_m 后缀）
       - 若为细胞数相关，设为 50–200；
       - 若为 cytokine 浓度相关，设为 1e3–1e5（与目标量级同阶）
    
    6️⃣ 已知可靠参数（来自论文 Supplementary Table）
       - 若某参数在文献中明确给出且其生物学意义不应在 HC 下关闭，则使用文献值覆盖以上默认规则
       - 文献值优先级最高
    
    目标：在 A(t)=0 条件下，ODE 系统通过长时间积分可以收敛到一个低 IgG4、低 cytokine、
          细胞群稳定的稳态（HC-like steady state）
    
    目标 HC 稳态值：
    {
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
        "CD56_NK": 50.0,
        "CD16_NK": 200.0,
        "Naive_B": 88.0,
        "Act_B": 95.0,
        "TD_Plasma": 2.0,
        "TI_Plasma": 0.0,
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
    """
    
    # 初始化参数字典
    params = {}
    
    # ========== 第零步：获取已知文献值作为基础 ==========
    # 但我们会在后面用 HC 友好的规则覆盖它们
    literature_params = get_known_parameters()
    params.update(literature_params)
    
    # ========== 第一步：规则1️⃣ - 抗原依赖激活参数 → 0.0（优先级最高） ==========
    # 包含 'Antigen' 且以 '_f' 结尾的参数必须为 0.0（确保A(t)=0时完全关闭）
    # 这一步会覆盖文献值（因为 HC 下必须关闭）
    antigen_dependent_activations = [
        # nDC → mDC/pDC 分化
        "k_mDC_Antigen_f",
        "k_mDC_GMCSF_f",  # GMCSF驱动的分化也依赖Antigen
        "k_pDC_Antigen_f",
        # naive_CD4 相关
        "k_IL_15_Antigen_f",
        # Th2 分化诱导
        "k_GMCSF_Th2_Antigen_f",
        "k_IL_4_Th2_Antigen_f",
        # Act_B 相关
        "k_Act_B_Antigen_f",
        "k_Naive_B_Antigen_f",
        "k_Act_B_Antigen_pro_f",
        # CD4_CTL 相关
        "k_act_CD4_CTL_antigen_f",
        # TFH 相关
        "k_TFH_mDC_Antigen_f",
        # IL_2 相关
        "k_IL_2_act_CD4_Antigen_f",
    ]
    for param_name in antigen_dependent_activations:
        if param_name in ALL_PARAMS:
            params[param_name] = 0.0
    
    # ========== 第二步：规则2️⃣ - IgG4 产生参数 → 0.0 ==========
    # HC下IgG4完全不产生（会覆盖文献值）
    igg4_params = [
        "k_IgG4_TI_f",
        "k_IgG4_TD_f",
    ]
    for param_name in igg4_params:
        if param_name in ALL_PARAMS:
            params[param_name] = 0.0
    
    # ========== 第三步：规则3️⃣ - 细胞自我增殖参数 → 0.0（会覆盖文献值） ==========
    # 在 HC 稳态下，细胞数量保持稳定，logistic 增殖项应该为 0.0
    # 细胞的维持由 basal 激活项驱动，而不是自我增殖
    proliferation_params = {
        "k_nDC_f": 0.0,
        "k_mDC_f": 0.0,
        "k_pDC_f": 0.0,
        "k_CD4_f": 0.0,
        "k_act_CD4_f": 0.0,
        "k_Th2_f": 0.0,
        "k_iTreg_f": 0.0,
        "k_nTreg_f": 0.0,
        "k_CD4_CTL_f": 0.0,
        "k_NK_f": 0.0,
        "k_act_NK_f": 0.0,
        "k_Naive_B_f": 0.0,
        "k_Act_B_f": 0.0,
        "k_TD_f": 0.0,
        "k_TI_f": 0.0,
        "k_TFH_f": 0.0,
    }
    for param_name, value in proliferation_params.items():
        if param_name in ALL_PARAMS:
            params[param_name] = value
    
    # ========== 第四步：规则4️⃣+5️⃣ - Basal/Homeostatic 参数，精心设计以达到目标稳态 ==========
    # 这些参数是根据稳态平衡方程 d/dt=0 反向计算的
    # 稳态条件：生成 = 清除（对于每个物种）
    
    # ---- 细胞因子 basal 生成参数 ----
    # 基于目标稳态值和清除速率反推生成速率
    # 假设在稳态时 d[cytokine]/dt = production_rate - k_d * [cytokine] = 0
    # 则 production_rate = k_d * [cytokine]_steady_state
    
    # IL_7: 目标 3.5e5，假设 k_d=0.1，则需要 basal_f ≈ 0.1 * 3.5e5 = 3.5e4 或更高的倍数
    params["k_IL_7_f"] = 3.5e-2  # IL_7 的 basal 生成，使之收敛到 3.5e5
    
    # IL_15: 目标 1.0e5，类似计算
    params["k_IL_15_f"] = 1.0e-2  # IL_15 的 basal 生成，使之收敛到 1.0e5
    
    # IL_6: 目标 3.5e4，主要来自 mDC 和 TFH
    # 在稳态时：k_IL_6_mDC_f * mDC + k_IL_6_pDC_f * pDC + k_IL_6_TFH_f * TFH = k_d * IL_6_ss
    # mDC=20, pDC=2, TFH=18, IL_6_ss=3.5e4, k_d=0.1
    # 假设分别贡献：mDC 60%, pDC 10%, TFH 30%
    # k_IL_6_mDC_f * 20 * 0.6 = 0.1 * 3.5e4 => k_IL_6_mDC_f ≈ 1.05e-1
    params["k_IL_6_mDC_f"] = 1.0e-1   # mDC 主要生成 IL-6
    params["k_IL_6_pDC_f"] = 5.0e-3   # pDC 辅助生成
    params["k_IL_6_TFH_f"] = 5.0e-3   # TFH 辅助生成
    
    # IL_12: 目标 1.5e4，主要来自 mDC
    # k_IL_12_mDC_f * mDC = 0.1 * 1.5e4 => k_IL_12_mDC_f ≈ 7.5e-2
    params["k_IL_12_mDC_f"] = 7.5e-2  # mDC 生成 IL-12
    
    # IL_2: 目标 1.2e4，来自 act_CD4
    # k_IL_2_act_CD4_f * act_CD4 = 0.1 * 1.2e4 => k_IL_2_act_CD4_f ≈ 3e-2 （act_CD4=400）
    params["k_IL_2_act_CD4_f"] = 3.0e-2  # act_CD4 生成 IL-2
    
    # IL_10: 目标 8.0e3，来自 iTreg 和 nTreg
    # k_IL_10_iTreg_f * iTreg + k_IL_10_nTreg_f * nTreg = 0.1 * 8.0e3
    # iTreg=250, nTreg=20，假设 iTreg 贡献更多
    params["k_IL_10_iTreg_f"] = 2.0e-2  # iTreg 主要生成 IL-10
    params["k_IL_10_nTreg_f"] = 3.0e-3  # nTreg 辅助生成
    
    # IL_4: 目标 1.0e2，来自 Th2
    # k_IL_4_Th2_f * Th2 = 0.1 * 1.0e2 => k_IL_4_Th2_f ≈ 3.33e-1 （Th2=0.30）
    params["k_IL_4_Th2_f"] = 3.0e-1   # Th2 生成 IL-4
    
    # IL_1: 目标 2.5e3，来自 mDC
    # k_IL_1_mDC_f * mDC = 0.1 * 2.5e3 => k_IL_1_mDC_f ≈ 1.25e-1
    params["k_IL_1_mDC_f"] = 1.25e-1  # mDC 生成 IL-1
    
    # GMCSF: 目标 3.0e3，来自 Th2 和 CD16_NK
    # k_GMCSF_Th2_f * Th2 + k_GMCSF_act_NK_f * CD16_NK = 0.1 * 3.0e3 （需考虑清除与生成平衡）
    # Th2=0.30, CD16_NK=200，假设分别贡献
    params["k_GMCSF_Th2_f"] = 1.0e0   # Th2 生成 GMCSF
    params["k_GMCSF_act_NK_f"] = 1.5e-2  # CD16_NK 生成 GMCSF
    
    # TGFbeta: 目标 1.0e2，来自 iTreg, nTreg, CD4_CTL
    # k_TGFbeta_iTreg_f * iTreg + k_TGFbeta_nTreg_f * nTreg + k_TGFbeta_CD4_CTL_f * CD4_CTL = 0.1 * 1.0e2
    params["k_TGFbeta_iTreg_f"] = 3.0e-2  # iTreg 主要生成 TGF-β
    params["k_TGFbeta_nTreg_f"] = 4.0e-3  # nTreg 辅助生成
    params["k_TGFbeta_CD4_CTL_f"] = 8.0e-2  # CD4_CTL 生成 TGF-β
    
    # IFN-gamma: 目标 2.0e5，来自 CD4_CTL 和 CD16_NK
    # k_IFN_g_CD4_CTL_f * CD4_CTL + k_IFN_g_act_NK_f * CD16_NK = 0.1 * 2.0e5
    # CD4_CTL=1, CD16_NK=200，CD16_NK 应贡献更多
    params["k_IFN_g_CD4_CTL_f"] = 1.0e2   # CD4_CTL 生成 IFN-γ （小数量贡献）
    params["k_IFN_g_act_NK_f"] = 1.0e-1  # CD16_NK 主要生成 IFN-γ
    
    # IFN-I: 目标 0.0，设为极小值（防止无限增长）
    params["k_IFN1_pDC_f"] = 1.0e-5  # pDC 极小生成 IFN-I
    
    # IL_33: 目标 0.0，设为极小值（防止无限增长）
    params["k_IL_33_pDC_f"] = 1.0e-5  # pDC 极小生成 IL-33
    
    # ---- 细胞相关的 basal 生成参数 ----
    # 这些参数控制细胞的基础生成或激活（未被抗原驱动的部分）
    
    # nDC 自我增殖: HC 下应该为 0（稳态维持）
    params["k_nDC_f"] = 0.0  # nDC 自我增殖关闭
    
    # mDC 自我增殖: HC 下应该为 0（稳态维持）
    params["k_mDC_f"] = 0.0  # mDC 自我增殖关闭
    
    # pDC 自我增殖: HC 下应该为 0（稳态维持）
    params["k_pDC_f"] = 0.0  # pDC 自我增殖关闭
    
    # CD4 相关
    params["k_CD4_f"] = 0.0  # naive_CD4 logistic 增殖关闭
    params["k_act_CD4_f"] = 0.0  # act_CD4 logistic 增殖关闭
    params["k_act_CD4_mDC_f"] = 5.0e-3  # naive_CD4 → act_CD4 由 mDC 激活
    params["k_act_CD4_IL_2_f"] = 2.0e-3  # IL-2 诱导的激活（较弱）
    
    # IL_15 和 IL_7 对 CD4 激活的促进（辅助增殖）
    params["k_naive_CD4_IL_15_f"] = 5.0e-3  # IL-15 增强 naive_CD4 增殖
    params["k_naive_CD4_IL_7_f"] = 3.0e-3   # IL-7 增强 naive_CD4 增殖
    params["k_act_CD4_IL_15_f"] = 5.0e-3   # IL-15 增强 act_CD4 增殖
    params["k_act_CD4_IL_7_f"] = 3.0e-3    # IL-7 增强 act_CD4 增殖
    
    # Th2 分化和增殖
    params["k_Th2_f"] = 0.0  # Th2 logistic 增殖关闭，由分化驱动
    params["k_Th2_IL_4_f"] = 1.0e-2  # IL-4 诱导 Th2 分化
    params["k_Th2_IL_33_f"] = 1.5e-2  # IL-33 诱导 Th2 分化
    
    # iTreg 分化和增殖
    params["k_iTreg_f"] = 0.0  # iTreg logistic 增殖关闭，由分化驱动
    params["k_iTreg_TGFbeta_f"] = 2.0e-2  # TGF-β 诱导 iTreg 分化
    params["k_iTreg_IL_10_f"] = 1.0e-2   # IL-10 增强 iTreg
    params["k_iTreg_mDC_f"] = 1.5e-2   # mDC 诱导 iTreg 分化
    
    # nTreg 增殖和维持
    params["k_nTreg_f"] = 0.0  # nTreg logistic 增殖关闭
    params["k_nTreg_mDC_f"] = 1.0e-2  # mDC 维持 nTreg
    
    # CD4_CTL 分化和增殖
    params["k_CD4_CTL_f"] = 0.0  # CD4_CTL logistic 增殖关闭
    params["k_act_CD4_CTL_basal_f"] = 5.0e-3  # 小基础分化速率
    
    # TFH 分化和增殖
    params["k_TFH_f"] = 0.0  # TFH logistic 增殖关闭，由分化驱动
    params["k_TFH_mDC_f"] = 5.0e-3  # mDC 诱导 TFH 分化
    params["k_TFH_IL_6_f"] = 3.0e-3  # IL-6 诱导 TFH 分化
    params["k_TFH_IFN1_f"] = 1.0e-3  # IFN-I 弱化诱导
    
    # CD56_NK 相关
    params["k_NK_f"] = 0.0  # CD56_NK logistic 增殖关闭
    params["k_act_NK_f"] = 0.0  # CD16_NK logistic 增殖关闭
    params["k_act_NK_base_f"] = 2.0e-1  # CD56_NK 的基础激活速率
    params["k_act_NK_IL_12_f"] = 1.5e-2  # IL-12 诱导 CD56_NK 激活
    params["k_act_NK_IL_2_f"] = 1.0e-2   # IL-2 诱导 CD56_NK 激活
    params["k_act_NK_IFN1_f"] = 5.0e-3   # IFN-I 诱导 CD56_NK 激活
    params["k_act_NK_IFN_g_f"] = 3.0e-3  # IFN-γ 自我增强
    
    # B 细胞相关
    params["k_Naive_B_f"] = 0.0  # Naive_B logistic 增殖关闭
    params["k_Act_B_f"] = 0.0   # Act_B logistic 增殖关闭
    params["k_Act_B_basal_f"] = 1.0e-2  # Naive_B → Act_B 的基础激活速率
    params["k_TD_f"] = 0.0     # TD_Plasma logistic 增殖关闭
    params["k_TI_f"] = 0.0     # TI_Plasma logistic 增殖关闭
    
    # Plasma cell 分化参数
    params["k_TD_base_f"] = 1.0e-3  # Act_B → TD_Plasma 的基础分化速率
    params["k_TI_base_f"] = 1.0e-4  # Act_B → TI_Plasma 的基础分化速率（极小，HC下TI_IS_B接近0）
    params["k_TD_IL_4_f"] = 5.0e-3  # IL-4 增强 TD 分化
    params["k_TI_IFN_g_f"] = 5.0e-3  # IFN-γ 增强 TI 分化
    params["k_TI_IL_10_f"] = 3.0e-3  # IL-10 增强 TI 分化
    
    # IgG4 清除参数（IgG4 在 HC 下应接近0）
    params["k_IgG4_d"] = 0.1  # IgG4 清除速率
    
    # ========== 第五步：规则6️⃣ - 死亡/清除参数 → 0.1 ==========
    # 所有尚未设置的 _d 参数（死亡、清除）
    
    # 处理所有其他 _d 参数
    for param_name in ALL_PARAMS:
        if param_name.endswith('_d') and param_name not in params:
            params[param_name] = 0.1
    
    # ========== 第六步：规则7️⃣ - Hill/饱和常数（_m 参数）==========
    # 细胞数相关的常数：50-200
    # Cytokine浓度相关的常数：1e3-1e5
    
    # 处理所有 _m 参数
    for param_name in ALL_PARAMS:
        if param_name.endswith('_m') and param_name not in params:
            # 默认判断：包含cytokine关键字的视为cytokine相关
            if any(cyto in param_name for cyto in ['IL_', 'IFN', 'GMCSF', 'TGFbeta']):
                params[param_name] = 1e4
            else:
                params[param_name] = 100
    
    # ========== 第七步：规则8️⃣ - 填充所有尚未设置的参数 ==========
    # 使用保守的默认值 1e-3
    for param_name in ALL_PARAMS:
        if param_name not in params:
            # 检查是否是 _f, _d, _m 参数
            if param_name.endswith('_f'):
                params[param_name] = 1e-3  # basal/其他生成参数
            elif param_name.endswith('_d'):
                params[param_name] = 0.1  # 死亡/清除参数
            elif param_name.endswith('_m'):
                params[param_name] = 100  # 默认为细胞数相关常数
            else:
                params[param_name] = 1e-3  # 其他未知参数
    
    return params

def setup_target_values():
    """
    根据参考图片设置目标值
    A图（细胞计数）在t=200的值，B图（细胞因子）在t=200的值
    HC阶段（t=100）应该等于初始值（稳态值）
    
    Returns:
    --------
    target_hc : dict，HC阶段（t=100）的目标值
    target_igg4 : dict，IgG4阶段（t=200）的目标值
    """
    # A图16个变量在t=200的值（从左到右，从上到下）
    # nDC, mDC, pDC, naive_CD4, act_CD4, Th2, iTreg, CD4_CTL, nTreg, TFH, CD56_NK, CD16_NK, Naive_B, Act_B, TD_Plasma, TI_Plasma
    a_values_t200 = [0, 20, 4, 75, 500, 0.2, 45, 2, 5, 20, 25, 310, 94, 90, 10, 0.1]
    a_vars = ["nDC", "mDC", "pDC", "naive_CD4", "act_CD4", "Th2", "iTreg", "CD4_CTL", 
              "nTreg", "TFH", "CD56_NK", "CD16_NK", "Naive_B", "Act_B", "TD_Plasma", "TI_Plasma"]
    
    # B图13个变量在t=200的值（从左到右，从上到下）
    # GMCSF, IL_33, IL_6, IL_12, IL_15, IL_7, IFN1, IL_1, IL_2, IL_4, IL_10, TGFbeta, IFN_g
    b_values_t200 = [5.25e3, 3e4, 6e4, 2e4, 1.4e5, 1e5, 1e0, 6e3, 1.5e4, 1.8e3, 1e4, 1e2, 1e6]
    b_vars = ["GMCSF", "IL_33", "IL_6", "IL_12", "IL_15", "IL_7", "IFN1", "IL_1", 
              "IL_2", "IL_4", "IL_10", "TGFbeta", "IFN_g"]
    
    # 构建目标值字典（t=200）
    target_igg4 = {}
    for var, val in zip(a_vars, a_values_t200):
        target_igg4[var] = val
    for var, val in zip(b_vars, b_values_t200):
        target_igg4[var] = val
    
    # HC阶段（t=100）应该等于初始值（稳态值）
    # 使用初始条件作为HC阶段的目标值
    y0 = get_initial_conditions()
    target_hc = {}
    for var_name in STATE_NAMES:
        if var_name != "Antigen":  # Antigen在t=100时应该是0
            target_hc[var_name] = y0[IDX[var_name]]
    target_hc["Antigen"] = 0.0
    
    return target_hc, target_igg4

def compute_loss(p: Dict[str, float], target_hc: Dict[str, float], target_igg4: Dict[str, float], 
                 y0: np.ndarray, verbose: bool = False) -> float:
    """
    计算损失函数：ODE预测值与目标值之间的L2 loss
    
    Parameters:
    -----------
    p : 参数字典
    target_hc : HC阶段（t=100）的目标值
    target_igg4 : IgG4阶段（t=200）的目标值
    y0 : 初始条件
    verbose : 是否打印详细信息
    
    Returns:
    --------
    loss : L2损失
    """
    # 积分到t=200
    ts = np.array([0.0, 100.0, 200.0])
    result = simulate(ts, y0, p, verbose=False)
    
    if result is None or np.isnan(result).any():
        return 1e10  # 返回很大的损失值
    
    loss = 0.0
    
    # HC阶段（t=100，索引为1）的损失
    for var_name, target_val in target_hc.items():
        if var_name in IDX:
            pred_val = result[1, IDX[var_name]]
            loss += (pred_val - target_val) ** 2
    
    # IgG4阶段（t=200，索引为2）的损失
    for var_name, target_val in target_igg4.items():
        if var_name in IDX:
            pred_val = result[2, IDX[var_name]]
            loss += (pred_val - target_val) ** 2
    
    return loss

def optimize_parameters(unknown_params: List[str], fixed_params: Dict[str, float] = None, 
                       verbose: bool = True) -> Tuple[Dict[str, float], float]:
    """
    优化未知参数
    
    Parameters:
    -----------
    unknown_params : 要优化的参数名列表（这些参数初始值设为1.0）
    fixed_params : 锁定的参数及其值（如果为None，则所有不在unknown_params中的参数都设为1.0）
    verbose : 是否打印详细信息
    
    Returns:
    --------
    best_params : 优化后的参数字典
    final_loss : 最终损失值
    """
    # 设置目标值
    target_hc, target_igg4 = setup_target_values()
    y0 = get_initial_conditions()
    
    
    # 初始化所有参数
    p_init = {name: 1.0 for name in ALL_PARAMS}
    p_init.update(fixed_params)  # 更新固定参数
    
    if verbose:
        print(f"要优化的参数数量: {len(unknown_params)}")
        print(f"固定参数数量: {len(fixed_params)}")
        print()
    
    # 创建进度条
    if verbose:
        pbar = tqdm(total=OPTIMIZATION_CONFIG['progress_bar_total'], desc="Optimizing", unit="iter", ncols=80)
    else:
        pbar = None
    
    # 参数边界（log空间）
    lb = OPTIMIZATION_CONFIG['log_param_lb']
    ub = OPTIMIZATION_CONFIG['log_param_ub']
    
    # 初始值（log空间，所有未知参数初始为log(1.0)=0）
    x0 = np.zeros(len(unknown_params))
    
    # 定义损失计算函数（返回标量）
    def objective_func(log_params_np: np.ndarray) -> float:
        """目标函数（用于scipy.optimize.minimize）"""
        params_dict = p_init.copy()
        for i, param_name in enumerate(unknown_params):
            # 应用边界约束
            log_val = np.clip(log_params_np[i], lb, ub)
            params_dict[param_name] = np.exp(log_val)
        return compute_loss(params_dict, target_hc, target_igg4, y0, verbose=False)
    
    # 定义有限差分梯度计算函数
    def gradient_func(log_params_np: np.ndarray) -> np.ndarray:
        """使用有限差分计算梯度"""
        eps = 1e-5
        grad = np.zeros(len(unknown_params))
        loss_base = objective_func(log_params_np)
        
        for i in range(len(unknown_params)):
            log_params_plus = log_params_np.copy()
            log_params_plus[i] += eps
            loss_plus = objective_func(log_params_plus)
            grad[i] = (loss_plus - loss_base) / eps
        
        return grad
    
    # 参数边界约束
    bounds = [(lb, ub) for _ in range(len(unknown_params))]
    
    # 使用scipy.optimize.minimize进行优化
    scipy_method = OPTIMIZATION_CONFIG['scipy_method']
    
    if verbose:
        print(f"使用scipy优化器: {scipy_method}")
    
    # 定义回调函数用于进度跟踪
    call_count = [0]
    def callback_func(xk):
        call_count[0] += 1
        if pbar is not None and call_count[0] <= OPTIMIZATION_CONFIG['progress_bar_total']:
            loss_val = objective_func(xk)
            pbar.set_postfix({"loss": f"{loss_val:.4e}"})
            pbar.update(1)
    
    # 优化
    result = minimize(
        objective_func,
        x0,
        method=scipy_method,
        jac=gradient_func if scipy_method != 'Nelder-Mead' else None,  # Nelder-Mead不使用梯度
        bounds=bounds if scipy_method == 'L-BFGS-B' else None,  # 仅L-BFGS-B支持边界
        options={
            'maxiter': OPTIMIZATION_CONFIG['max_iter'],
            'ftol': OPTIMIZATION_CONFIG['scipy_ftol'],
            'gtol': OPTIMIZATION_CONFIG['scipy_gtol'],
        },
        callback=callback_func,
    )
    
    if pbar is not None:
        pbar.close()
    
    # 提取最优参数
    best_log_params = result.x
    best_loss = result.fun
    best_params_dict = p_init.copy()
    for i, param_name in enumerate(unknown_params):
        log_val = np.clip(best_log_params[i], lb, ub)
        best_params_dict[param_name] = np.exp(log_val)
    
    if verbose:
        print(f"\n优化完成")
        print(f"最终损失: {best_loss:.4e}")
        print(f"优化迭代次数: {result.nit}")
        print(f"函数评估次数: {result.nfev}")
        
        # 打印参数变化
        print("\n参数变化（log空间 -> 原始空间）:")
        initial_log = np.zeros(len(unknown_params))
        final_log = best_log_params
        for i, param_name in enumerate(unknown_params):
            initial_val = np.exp(initial_log[i])
            final_val = best_params_dict[param_name]
            change = final_val - initial_val
            print(f"  {param_name}: {initial_val:.6e} -> {final_val:.6e} (变化: {change:+.6e})")
        print()
    
    return best_params_dict, best_loss


# ============================================================================
# 旧模拟代码(使用HC_init) - 已禁用，现在使用下方的antigen刺激模拟
# ============================================================================

"""
if __name__ == '__main__':
    # 导入HC初值和参数
    from utils import HC_init, HC_param
    
    print("="*60)
    print("HC稳态ODE求解")
    print("="*60)
    
    # 获取HC初值和参数
    hc_init_dict = HC_init()
    hc_params = HC_param()
    
    # 转换初值字典为数组
    y0 = np.array([hc_init_dict[state_name] for state_name in STATE_NAMES])
    
    print(f"初值数组长度: {len(y0)}")
    print(f"参数字典大小: {len(hc_params)}")
    
    # 运行ODE模拟
    t_end = SIMULATION_CONFIG['t_end']
    time_step = SIMULATION_CONFIG['time_step']
    ts = np.arange(0, int(t_end) + time_step, time_step)
    
    print(f"\n运行ODE模拟: t=0 到 t={t_end}s，时间步={time_step}s...")
    result = simulate(ts, y0, hc_params, verbose=True)
    
    if result is not None and not np.all(np.isnan(result)):
        print("\n模拟完成，开始绘图...")
        
        # 绘制结果
        import matplotlib.pyplot as plt
        
        # 绘制细胞种群
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle('HC稳态模拟 - 细胞种群', fontsize=14, fontweight='bold')
        
        # 第一行：树突状细胞
        axes[0, 0].plot(ts, result[:, STATE_NAMES.index('nDC')], 'b-', linewidth=2)
        axes[0, 0].set_ylabel('nDC (cells/mL)', fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(ts, result[:, STATE_NAMES.index('mDC')], 'r-', linewidth=2)
        axes[0, 1].set_ylabel('mDC (cells/mL)', fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 第二行：T细胞
        axes[1, 0].plot(ts, result[:, STATE_NAMES.index('act_CD4')], 'g-', linewidth=2, label='act_CD4')
        axes[1, 0].plot(ts, result[:, STATE_NAMES.index('iTreg')], 'orange', linewidth=2, label='iTreg')
        axes[1, 0].set_ylabel('Count (cells/mL)', fontsize=10)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(ts, result[:, STATE_NAMES.index('TFH')], 'purple', linewidth=2)
        axes[1, 1].set_ylabel('TFH (cells/mL)', fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        # 第三行：B细胞和IgG4
        axes[2, 0].plot(ts, result[:, STATE_NAMES.index('Naive_B')], 'cyan', linewidth=2, label='Naive_B')
        axes[2, 0].plot(ts, result[:, STATE_NAMES.index('Act_B')], 'brown', linewidth=2, label='Act_B')
        axes[2, 0].set_ylabel('Count (cells/mL)', fontsize=10)
        axes[2, 0].set_xlabel('Time (s)', fontsize=10)
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        axes[2, 1].plot(ts, result[:, STATE_NAMES.index('IgG4')], 'k-', linewidth=2)
        axes[2, 1].set_ylabel('IgG4 (ng/mL)', fontsize=10)
        axes[2, 1].set_xlabel('Time (s)', fontsize=10)
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('hc_cells.png', dpi=150, bbox_inches='tight')
        print("已保存: hc_cells.png")
        
        # 绘制细胞因子
        fig, axes = plt.subplots(3, 3, figsize=(15, 10))
        fig.suptitle('HC稳态模拟 - 细胞因子', fontsize=14, fontweight='bold')
        
        cytokines = ['IL_7', 'IL_15', 'IL_6', 'IL_12', 'IL_2', 'IL_10', 'IFN_g', 'TGFbeta', 'IL_4']
        for idx, cytokine in enumerate(cytokines):
            if cytokine in STATE_NAMES:
                ax = axes[idx // 3, idx % 3]
                cy_idx = STATE_NAMES.index(cytokine)
                ax.plot(ts, result[:, cy_idx], linewidth=2)
                ax.set_ylabel(f'{cytokine} (pg/mL)', fontsize=10)
                ax.set_xlabel('Time (s)', fontsize=10)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('hc_cytokines.png', dpi=150, bbox_inches='tight')
        print("已保存: hc_cytokines.png")
        
        plt.show()
    else:
        print("模拟失败，无法绘图")
    
    # 如果需要运行优化，取消下面的注释
    '''
    # 你要拟合的参数名（先少量跑通，再扩到 Table7 的全部）
    # 注意：这些名字必须在 rhs() 里被引用到
    param_names = [
        "k_IFN_g_CD4_CTL_f",
        "k_IFN_g_CD4_CTL_f",
        "k_GMCSF_Th2_f",
        "k_GMCSF_Th2_Antigen_f",
        "k_IL_10_nTreg_f",
        "k_IL_4_Th2_Antigen_f",
        "k_TI_IS_B_d",
        "k_Naive_B_Antigen_f",
        "k_IL_6_mDC_f",
        "k_IL_6_TFH_f",
        "k_IL_12_mDC_f",
        "k_IL_2_act_CD4_Antigen_f",
        "k_IL_33_pDC_f",
        "k_IL_6_pDC_f",
        "k_TI_IS_B_TI_IS_B_f",
        "k_Act_B_Act_B_Antigen_f",
        "k_mDC_GMCSF_f",
        "k_mDC_GMCSF_d",
        "k_nDC_f",
        "k_act_CD4_IL_7_d",
        "k_act_CD4_mDC_f",
        "k_IFN_g_act_NK_f",
        "k_naive_CD4_IL_7_f",
        "k_IL_2_d",
        "k_TD_IS_B_d",
        "k_IL_10_iTreg_f",
        "k_pro_act_NK_IL_12_f",
        "k_GMCSF_act_NK_f",
        "k_mDC_Antigen_f",
        "k_pDC_Antigen_f",
        "k_pDC_f",
        "k_mDC_d",
        "k_mDC_f",
        "k_IL_1_mDC_f",
        "k_CD4_f",
        "k_naive_CD4_IL_15_f",
        "k_pDC_d",
        "k_Act_B_Antigen_f",
        "k_iTreg_mDC_f",
        "k_NK_f",
        "k_IgG4_TD_IS_B_f",
        "k_IgG4_TI_IS_B_f",
        "k_TFH_f",
        "k_act_NK_IL_12_f",
        "k_GMCSF_d",
        "k_iTreg_d",
        "k_nTreg_d",
        "k_TFH_mDC_Antigen_f",
        "k_nTreg_mDC_f",
        "k_IL_6_d",
        "k_act_CD4_CTL_antigen_f",
        "k_IL_10_d",
        "k_act_NK_d",
        "k_IL_4_d",
        "k_CD4_CTL_CD4_CTL_f",
        "k_act_NK_IFN1_f",
        "k_IFN1_pDC_f",
        "k_Th2_f",
        "k_act_CD4_IL_4_d",
        "k_TD_IS_B_Act_B_f",
        "k_iTreg_f",
        "k_naive_CD4_d",
        "k_IL_12_d",
        "k_IFN1_d",
        "k_TGFbeta_d",
        "k_IL_33_d",
        "k_IL_15_d",
        "k_Naive_B_d",
        "k_act_NK_base_f",
        "k_act_CD4_d",
        "k_IL_7_d",
        "k_iTreg_mDC_d",
        "k_CD4_CTL_d",
        "k_Act_B_Act_B_f",
        "k_nTreg_f",
        "k_IL_15_f",
        "k_TGFbeta_nTreg_f",
        "k_IL_7_f",
        "k_naive_CD4_IL_7_d",
        "k_Act_B_Act_B_d",
        "k_NK_d",
        "k_nDC_d",
        "k_act_CD4_mDC_m",
        "k_IL_2_act_CD4_f",
        "k_act_NK_IL_12_m",
        "k_naive_CD4_IL_15_d",
        "k_act_CD4_IL_15_d",
        "k_IL_15_Antigen_f",
        "k_IL_2_act_CD4_Antigen_f2",
        "k_act_CD4_IL_15_f",
        "k_act_CD4_f",
        "k_NK_m",
        "k_CD4_m",
        "k_nDC_m",
        "k_TFH_m",
        "k_act_CD4_m",
        "k_nTreg_m",
        "k_TD_IS_B_TD_IS_B_m",
        "k_mDC_m",
        "k_act_NK_m",
        "k_iTreg_m",
        "k_CD4_CTL_CD4_CTL_m",
        "k_Th2_TGFbeta_m",
        "k_iTreg_TGFbeta_m",
        "k_act_NK_IFN1_m",
        "k_TGFbeta_nTreg_mDC_m",
        "k_TFH_IFN1_m",
        "k_mDC_IL_10_m",
        "k_mDC_GMCSF_m",
        "k_iTreg_IL_10_m",
        "k_act_CD4_IL_2_m",
        "k_act_CD4_IFN1_m",
        "k_act_NK_IL_2_m",
        "k_act_NK_IFN_g_m",
        "k_IL_10_nTreg_mDC_m",
        "k_act_CD4_IL_15_m",
        "k_naive_CD4_IL_7_m",
        "k_naive_CD4_IL_15_m",
        "k_Th2_IL_33_f",
        "k_Th2_IL_10_m",
        "k_iTreg_IL_1_m",
        "k_act_CD4_CTL_basal_f",
        "k_act_CD4_IL_7_m",
        "k_act_CD4_IL_33_d",
        "k_act_CD4_IL_33_f",
        "k_TGFbeta_nTreg_mDC_f",
        "k_Th2_IL_4_f",
        "k_TFH_mDC_f",
        "k_Th2_IL_10_f",
        "k_iTreg_IL_10_f",
        "k_Th2_TGFbeta_f",
        "k_act_CD4_CTL_antigen_f2",
        "k_act_CD4_IL_2_f",
        "k_act_NK_IL_2_f",
        "k_act_NK_IL_12_d",
        "k_act_NK_base_d",
        "k_act_NK_IFN1_d",
        "k_act_NK_IFN_g_f2",
        "k_Th2_d2",
        "k_TFH_IL_6_d",
        "k_IL_12_mDC_f2",
        "k_mDC_f2",
        "k_IL_7_f2",
        "k_naive_CD4_IL_7_f2",
        "k_naive_CD4_IL_15_f2",
        "k_TFH_mDC_Antigen_f2",
        "k_act_CD4_mDC_f2",
        "k_TFH_f2",
        "k_TFH_m2",
        "k_act_CD4_CTL_antigen_f3",
        "k_act_CD4_mDC_m2",
        "k_mDC_d2",
        "k_mDC_m2",
        "k_nDC_f2",
        "k_act_NK_IL_12_f2",
        "k_pro_act_NK_IL_12_f2",
        "k_Th2_IL_4_f2",
        "k_Th2_IL_33_f2",
        "k_Th2_IL_10_m2",
        "k_TFH_IFN1_m2",
        "k_Th2_TGFbeta_m2",
        "k_IL_12_d2",
        "k_NK_d2",
        "k_act_NK_IL_2_f2",
        "k_act_NK_base_f2",
        "k_act_NK_IFN_g_f3",
        "k_act_NK_d2",
        "k_act_NK_IL_12_m2",
        "k_act_NK_IL_12_f3",
        "k_IL_12_mDC_f3",
        "k_mDC_m3",
        "k_nDC_f3",
        "k_mDC_f3",
        "k_IL_7_f3",
        "k_naive_CD4_IL_7_f3",
        "k_naive_CD4_IL_15_f3",
        "k_TFH_mDC_f2",
        "k_TFH_IL_6_f2",
        "k_TFH_mDC_Antigen_f3",
        "k_act_CD4_mDC_f3",
        "k_TFH_f3",
        "k_TFH_m3",
        "k_act_CD4_CTL_antigen_f4",
        "k_act_CD4_mDC_m3",
        "k_mDC_d3",
        "k_TGFbeta_d2",
        "k_Th2_TGFbeta_m3",
        "k_CD4_CTL_CD4_CTL_m2",
    ]
    # log空间边界：先宽一点；后面用 Table7 的尺度收紧
    lb = np.log(np.full(len(param_names), 1e-8))
    ub = np.log(np.full(len(param_names), 1e6))
    spec = ParamSpec(names=param_names, lb=lb, ub=ub)

    dataset = demo_dataset()

    best_logtheta, best_f = fit_multistart(spec, dataset, n_starts=5, seed=0)
    best_params = unpack_params(best_logtheta, spec)

    print("Best NLL:", best_f)
    print("Best params:", best_params)
    '''
"""


# ============================================================================
# Antigen刺激模拟：从HC_bl出发，t=100-150时antigen线性上升
# ============================================================================

if __name__ == '__main__':
    from scipy.integrate import odeint
    from utils import HC_bl, IgG_param, rhs_hc, STATE_NAMES
    
    print("=" * 80)
    print("Antigen刺激模拟：HC_bl + IgG_param，t=100-150时antigen从0线性上升到1")
    print("=" * 80)
    
    # 加载初始条件和参数
    y0_dict = HC_bl()
    p = IgG_param()
    y0 = np.array([y0_dict[name] for name in STATE_NAMES])
    
    print(f"\n初始条件: HC_bl (稳态)")
    print(f"参数: IgG_param (Antigen相关项已降至1e-1)")
    print(f"时间范围: t=0 到 t=300s")
    print(f"Antigen刺激: t=100-150s 线性上升 0→1\n")

    # 诊断：在初始点计算 RHS 残差，检查“稳态误差”是否足够小
    dy0 = rhs_hc(0.0, y0, p)
    max_abs_resid = np.max(np.abs(dy0))
    print(f"初始点 RHS 最大残差: {max_abs_resid:.3e}\n")
    
    # 定义时间相关的Antigen函数
    def antigen_stimulus(t):
        """Antigen随时间变化：t∈[100,150]时从0线性上升到1"""
        if t < 100:
            return 0.0
        elif t <= 150:
            return (t - 100) / 50.0  # 线性从0到1
        else:
            return 1.0

    def antigen_stimulus_derivative(t):
        """对应刺激函数的导数，保证State中的Antigen与驱动匹配"""
        if 100 <= t <= 150:
            return 1.0 / 50.0
        return 0.0
    
    # 修改RHS：直接将 y[0] 置为外部给定的 antigen(t)，并将其导数设为0
    # 这样其余方程在每个时间点都能“看到”实时的 Antigen，而不会受积分误差漂移
    def rhs_with_antigen(y, t):
        antigen = antigen_stimulus(t)
        y_forced = y.copy()
        y_forced[0] = antigen  # 强制当前时刻的 Antigen

        dydt = rhs_hc(t, y_forced, p)
        dydt[0] = antigen_stimulus_derivative(t)
        return dydt
    
    # 求解ODE
    t = np.linspace(0, 300, 3001)
    print("求解ODE系统...")
    y_solution = odeint(rhs_with_antigen, y0, t, rtol=1e-8, atol=1e-10)
    print("[OK] 求解完成\n")
    
    # ====================================================================
    # Combined Figure: 细胞计数（上）+ 细胞因子（下）
    # ====================================================================
    print("生成综合图表...")
    
    cells_order = [
        "nDC", "mDC", "pDC", "naive_CD4",
        "act_CD4", "Th2", "iTreg", "CD4_CTL",
        "nTreg", "TFH", "CD56_NK", "CD16_NK",
        "Naive_B", "Act_B", "TD_Plasma", "TI_Plasma"
    ]
    
    cytokine_order = [
        "GMCSF", "IL_33", "IL_6", "IL_12",
        "IL_15", "IL_7", "IFN1", "IL_1",
        "IL_2", "IL_4", "IL_10", "TGFbeta",
        "IFN_g", "Antigen"
    ]
    
    # 创建8x4的图表（共32个子图，最后两个隐藏）
    fig, axes = plt.subplots(8, 4, figsize=(16, 20))
    axes = axes.flatten()
    
    # 绘制细胞计数（前16个子图）
    for plot_idx, var_name in enumerate(cells_order):
        ax = axes[plot_idx]
        state_idx = IDX[var_name]
        ax.plot(t, y_solution[:, state_idx], linewidth=1.5, color='steelblue')
        
        # 标记antigen刺激区间
        ax.axvspan(100, 150, alpha=0.15, color='red')
        
        ax.set_title(var_name, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Cell count', fontsize=8)
        ax.tick_params(labelsize=7)
    
    # 绘制细胞因子（后16个子图中的14个）
    for plot_idx, var_name in enumerate(cytokine_order):
        ax = axes[16 + plot_idx]
        state_idx = IDX[var_name]
        ax.plot(t, y_solution[:, state_idx], linewidth=1.5, color='#2E86AB')
        
        # 标记antigen刺激区间
        ax.axvspan(100, 150, alpha=0.15, color='red')
        
        ax.set_title(var_name, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Concentration [pmol/mL]', fontsize=8)
        ax.tick_params(labelsize=7)
    
    # 隐藏最后两个空子图（只用14个细胞因子）
    axes[-2].set_visible(False)
    axes[-1].set_visible(False)
    
    plt.suptitle('ODE Simulation: HC baseline vs IgG4 response (Antigen stimulus: t=100-150s)', 
                 fontsize=13, fontweight='bold', y=0.9995)
    plt.tight_layout()
    
    output_path = 'antigen_stimulus.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] 已保存: {output_path}\n")
    print("[OK] 显示轨线图\n")
    plt.show()
    
    print("=" * 80)
    print("完成！")
    print("=" * 80)
