# 使用pytorch梯度下降优化求解data driven ode的废案
from __future__ import annotations
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional
from scipy.integrate import solve_ivp
from scipy.optimize import minimize  # 保留用于其他函数
try:
    import torch
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Please install it with: pip install torch")

try:
    from torchdiffeq import odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    print("Warning: torchdiffeq not available. Please install it with: pip install torchdiffeq")

# ============================================================================
# 超参数配置区域 - 方便调整
# ============================================================================
OPTIMIZATION_CONFIG = {
    'max_iter': 100,              # 优化最大迭代次数
    'log_param_lb': -10,          # log参数空间下界（对应约1e-4）
    'log_param_ub': 10,           # log参数空间上界（对应约1e4）
    'progress_bar_total': 100,    # 进度条显示的总迭代次数
    # PyTorch优化器选项
    'optimizer': 'LBFGS',           # 优化器类型: 'Adam', 'LBFGS', 'SGD'
    'lr': 1.0,                     # 学习率（LBFGS使用）
    'weight_decay': 0.0,           # 权重衰减（L2正则化）
    # LBFGS特定选项
    'max_eval': 20,                # LBFGS每次迭代的最大函数评估次数
    'history_size': 10,            # LBFGS历史大小
    'line_search_fn': 'strong_wolfe',        # LBFGS线搜索函数: None, 'strong_wolfe'
}

SIMULATION_CONFIG = {
    't_end': 300.0,               # 模拟结束时间（秒）
    'time_step': 1.0,             # 时间步长（秒）
}

# 全局精度设置
DTYPE = torch.float64 if TORCH_AVAILABLE else np.float64

# ODE求解器配置 - PyTorch 自动微分专用
# 使用 torchdiffeq 的 scipy_solver 包装器，通过 BDF 算法处理刚性问题
ODE_SOLVER_CONFIG = {
    'backend': 'torchdiffeq',     # 固定使用 torchdiffeq（支持自动微分）
    'method': 'scipy_solver',     # scipy_solver 包装器（支持 SciPy 的所有求解器）
    'rtol': 1e-2,                 # 相对容差（为了性能稍微放宽）
    'atol': 1e-2,                 # 绝对容差（为了性能稍微放宽）
    'step_size': None,            # 自适应步长控制
    'dtype': DTYPE,               # 使用 float64（64 位浮点数）确保精度
    'solver': 'BDF',              # 使用 BDF（隐式方法，专为刚性问题设计）
}

# ============================================================================

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib


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

# -------------------------
# 0) 约定：state 顺序必须固定
# -------------------------
STATE_NAMES = [
    "Antigen", "nDC", "mDC", "GMCSF", "pDC",
    "IL_33", "IL_6", "IL_12", "IL_15", "IL_7", "IFN1", "IL_1", "IL_2", "IL_4", "IL_10", "TGFbeta", "IFN_g",
    "naive_CD4", "act_CD4", "Th2", "iTreg", "CD4_CTL", "nTreg", "TFH",
    "NK", "act_NK",
    "Naive_B", "Act_B", "TD_IS_B", "TI_IS_B",
    "IgG4",
]
N_STATE = len(STATE_NAMES)
IDX = {n: i for i, n in enumerate(STATE_NAMES)}

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
    NK        = y[IDX["NK"]]
    act_NK    = y[IDX["act_NK"]]
    Naive_B   = y[IDX["Naive_B"]]
    Act_B     = y[IDX["Act_B"]]
    TD_IS_B   = y[IDX["TD_IS_B"]]
    TI_IS_B   = y[IDX["TI_IS_B"]]
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
        + p["k_GMCSF_act_NK_f"] * act_NK
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
        - p["k_act_NK_IL_12_d"] * NK * IL_12 / (IL_12 + p["k_act_NK_IL_12_m"])
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
        - p["k_act_NK_IFN1_d"] * NK * IFN1 / (IFN1 + p["k_act_NK_IFN1_m"])
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
        - p["k_act_NK_IL_2_d"] * NK * IL_2 / (IL_2 + p["k_act_NK_IL_2_m"])
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
        + p["k_IFN_g_act_NK_f"] * act_NK
        - p["k_act_NK_IFN_g_d"] * NK * IFN_g / (IFN_g + p["k_act_NK_IFN_g_m"])
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
        p["k_NK_f"] * NK * (1 - NK / p["k_NK_m"])
        - p["k_act_NK_base_f"] * NK
        - p["k_act_NK_IL_12_f"] * NK * IL_12 / (IL_12 + p["k_act_NK_IL_12_m"])
        - p["k_act_NK_IL_2_f"] * NK * IL_2 / (IL_2 + p["k_act_NK_IL_2_m"])
        - p["k_act_NK_IFN1_f"] * NK * IFN1 / (IFN1 + p["k_act_NK_IFN1_m"])
        - p["k_act_NK_IFN_g_f"] * NK * IFN_g / (IFN_g + p["k_act_NK_IFN_g_m"])
        - p["k_NK_d"] * NK
    )

    dact_NK = (
        p["k_act_NK_base_f"] * NK
        + p["k_act_NK_IL_12_f"] * NK * IL_12 / (IL_12 + p["k_act_NK_IL_12_m"])
        + p["k_act_NK_IL_2_f"] * NK * IL_2 / (IL_2 + p["k_act_NK_IL_2_m"])
        + p["k_act_NK_IFN1_f"] * NK * IFN1 / (IFN1 + p["k_act_NK_IFN1_m"])
        + p["k_act_NK_IFN_g_f"] * NK * IFN_g / (IFN_g + p["k_act_NK_IFN_g_m"])
        + p["k_act_NK_f"] * act_NK * (1 - act_NK / p["k_act_NK_m"])
        - p["k_act_NK_d"] * act_NK
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
        + p["k_TD_f"] * TD_IS_B * (1 - TD_IS_B / p["k_TD_m"])
        - p["k_TD_d"] * TD_IS_B
    )

    dTI_IS_B = (
        p["k_TI_base_f"] * Act_B
        + p["k_TI_IFN_g_f"] * Act_B * IFN_g
        + p["k_TI_IL_10_f"] * Act_B * IL_10
        + p["k_TI_f"] * TI_IS_B * (1 - TI_IS_B / p["k_TI_m"])
        - p["k_TI_d"] * TI_IS_B
    )

    dIgG4 = (
        p["k_IgG4_TI_f"] * 1e8 * TI_IS_B
        + p["k_IgG4_TD_f"] * 1e8 * TD_IS_B
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
# 2.5) PyTorch 版本的 RHS（支持自动微分和梯度回传）
# -------------------------
def rhs_torch(t: float, y: torch.Tensor, p: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    ODE系统右侧（PyTorch版本，支持自动微分）
    
    Parameters:
    -----------
    t : 时间标量
    y : 状态向量（torch.Tensor, shape=(N_STATE,), dtype=float64）
    p : 参数字典，值为torch.Tensor（支持requires_grad）
    
    Returns:
    --------
    dy : 导数向量（torch.Tensor, dtype=float64）
    """
    # 解包 states
    Antigen   = y[IDX["Antigen"]]
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
    NK        = y[IDX["NK"]]
    act_NK    = y[IDX["act_NK"]]
    Naive_B   = y[IDX["Naive_B"]]
    Act_B     = y[IDX["Act_B"]]
    TD_IS_B   = y[IDX["TD_IS_B"]]
    TI_IS_B   = y[IDX["TI_IS_B"]]
    IgG4      = y[IDX["IgG4"]]

    # Antigen 根据时间变化：0-100s为0，100-150s线性增长到1，150s后保持1
    if t < 100.0:
        Antigen_val = torch.tensor(0.0, dtype=y.dtype, device=y.device)
    elif t < 150.0:
        Antigen_val = (t - 100.0) / 50.0
        Antigen_val = torch.tensor(Antigen_val, dtype=y.dtype, device=y.device)
    else:
        Antigen_val = torch.tensor(1.0, dtype=y.dtype, device=y.device)

    # ---- Antigen导数 ----
    if t < 100.0:
        dAntigen = torch.tensor(0.0, dtype=y.dtype, device=y.device)
    elif t < 150.0:
        dAntigen = torch.tensor(1.0 / 50.0, dtype=y.dtype, device=y.device)
    else:
        dAntigen = torch.tensor(0.0, dtype=y.dtype, device=y.device)

    dnDC = (
        p["k_nDC_f"] * nDC * (1 - nDC / p["k_nDC_m"])
        - p["k_mDC_Antigen_f"] * Antigen_val * nDC * (p["k_mDC_IL_10_m"] / (p["k_mDC_IL_10_m"] + IL_10))
        - p["k_mDC_GMCSF_f"] * Antigen_val * nDC * (GMCSF / (GMCSF + p["k_mDC_GMCSF_m"])) * (p["k_mDC_IL_10_m"] / (p["k_mDC_IL_10_m"] + IL_10))
        - p["k_pDC_Antigen_f"] * nDC * Antigen_val
        - p["k_nDC_d"] * nDC
    )

    dmDC = (
        p["k_mDC_Antigen_f"] * Antigen_val * nDC * (p["k_mDC_IL_10_m"] / (p["k_mDC_IL_10_m"] + IL_10))
        + p["k_mDC_GMCSF_f"] * Antigen_val * nDC * (GMCSF / (GMCSF + p["k_mDC_GMCSF_m"])) * (p["k_mDC_IL_10_m"] / (p["k_mDC_IL_10_m"] + IL_10))
        + p["k_mDC_f"] * mDC * (1 - mDC / p["k_mDC_m"])
        - p["k_mDC_d"] * mDC
    )

    dGMCSF = (
        - p["k_mDC_GMCSF_d"] * Antigen_val * nDC * (GMCSF / (GMCSF + p["k_mDC_GMCSF_m"])) * (p["k_mDC_IL_10_m"] / (p["k_mDC_IL_10_m"] + IL_10))
        + p["k_GMCSF_Th2_f"] * Th2
        + p["k_GMCSF_Th2_Antigen_f"] * Th2 * Antigen_val
        + p["k_GMCSF_act_NK_f"] * act_NK
        - p["k_GMCSF_d"] * GMCSF
    )

    dpDC = (
        p["k_pDC_Antigen_f"] * nDC * Antigen_val
        + p["k_pDC_f"] * pDC * (1 - pDC / p["k_pDC_m"])
        - p["k_pDC_d"] * pDC
    )

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
        - p["k_act_NK_IL_12_d"] * NK * IL_12 / (IL_12 + p["k_act_NK_IL_12_m"])
        - p["k_IL_12_d"] * IL_12
    )

    dIL_15 = (
        p["k_IL_15_f"]
        + p["k_IL_15_Antigen_f"] * Antigen_val
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
        - p["k_act_NK_IFN1_d"] * NK * IFN1 / (IFN1 + p["k_act_NK_IFN1_m"])
        - p["k_IFN1_d"] * IFN1
    )

    dIL_1 = (
        p["k_IL_1_mDC_f"] * mDC
        - p["k_IL_1_d"] * IL_1
    )

    dIL_2 = (
        p["k_IL_2_act_CD4_f"] * act_CD4
        + p["k_IL_2_act_CD4_Antigen_f"] * act_CD4 * Antigen_val
        - p["k_act_CD4_IL_2_d"] * naive_CD4 * IL_2 / (p["k_act_CD4_IL_2_m"] + IL_2)
        - p["k_act_NK_IL_2_d"] * NK * IL_2 / (IL_2 + p["k_act_NK_IL_2_m"])
        - p["k_IL_2_d"] * IL_2
    )

    dIL_4 = (
        p["k_IL_4_Th2_f"] * Th2
        + p["k_IL_4_Th2_Antigen_f"] * Th2 * Antigen_val
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
        + p["k_IFN_g_act_NK_f"] * act_NK
        - p["k_act_NK_IFN_g_d"] * NK * IFN_g / (IFN_g + p["k_act_NK_IFN_g_m"])
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
        - p["k_act_CD4_CTL_antigen_f"] * act_CD4 * Antigen_val
        - p["k_act_CD4_IFN1_f"] * act_CD4 * IFN1 / (p["k_IFN1_CD4_CTL_m"] + IFN1)
        - p["k_TFH_mDC_f"] * act_CD4
        - p["k_TFH_mDC_Antigen_f"] * act_CD4 * Antigen_val
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
        + p["k_act_CD4_CTL_antigen_f"] * act_CD4 * Antigen_val
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
        + p["k_TFH_mDC_Antigen_f"] * act_CD4 * Antigen_val
        + p["k_TFH_IFN1_f"] * act_CD4 * IFN1 / (p["k_TFH_IFN1_m"] + IFN1)
        + p["k_TFH_IL_6_f"] * act_CD4 * IL_6 / (p["k_TFH_IL_6_m"] + IL_6)
        + p["k_TFH_f"] * TFH * (1 - TFH / p["k_TFH_m"])
        - p["k_TFH_d"] * TFH
    )

    dNK = (
        p["k_NK_f"] * NK * (1 - NK / p["k_NK_m"])
        - p["k_act_NK_base_f"] * NK
        - p["k_act_NK_IL_12_f"] * NK * IL_12 / (IL_12 + p["k_act_NK_IL_12_m"])
        - p["k_act_NK_IL_2_f"] * NK * IL_2 / (IL_2 + p["k_act_NK_IL_2_m"])
        - p["k_act_NK_IFN1_f"] * NK * IFN1 / (IFN1 + p["k_act_NK_IFN1_m"])
        - p["k_act_NK_IFN_g_f"] * NK * IFN_g / (IFN_g + p["k_act_NK_IFN_g_m"])
        - p["k_NK_d"] * NK
    )

    dact_NK = (
        p["k_act_NK_base_f"] * NK
        + p["k_act_NK_IL_12_f"] * NK * IL_12 / (IL_12 + p["k_act_NK_IL_12_m"])
        + p["k_act_NK_IL_2_f"] * NK * IL_2 / (IL_2 + p["k_act_NK_IL_2_m"])
        + p["k_act_NK_IFN1_f"] * NK * IFN1 / (IFN1 + p["k_act_NK_IFN1_m"])
        + p["k_act_NK_IFN_g_f"] * NK * IFN_g / (IFN_g + p["k_act_NK_IFN_g_m"])
        + p["k_act_NK_f"] * act_NK * (1 - act_NK / p["k_act_NK_m"])
        - p["k_act_NK_d"] * act_NK
    )

    dNaive_B = (
        p["k_Naive_B_f"] * Naive_B * (1 - Naive_B / p["k_Naive_B_m"])
        + p["k_Naive_B_Antigen_f"] * Naive_B * Antigen_val * (1 - Naive_B / p["k_Naive_B_m"])
        - p["k_Act_B_basal_f"] * Naive_B
        - p["k_Act_B_Antigen_f"] * Naive_B * Antigen_val
        - p["k_Naive_B_d"] * Naive_B
    )

    dAct_B = (
        p["k_Act_B_basal_f"] * Naive_B
        + p["k_Act_B_Antigen_f"] * Naive_B * Antigen_val
        + p["k_Act_B_f"] * Act_B * (1 - Act_B / p["k_Act_B_m"])
        + p["k_Act_B_Antigen_pro_f"] * Act_B * Antigen_val * (1 - Act_B / p["k_Act_B_m"])
        - p["k_Act_B_d"] * Act_B
    )

    dTD_IS_B = (
        p["k_TD_base_f"] * Act_B
        + p["k_TD_IL_4_f"] * Act_B * IL_4
        + p["k_TD_f"] * TD_IS_B * (1 - TD_IS_B / p["k_TD_m"])
        - p["k_TD_d"] * TD_IS_B
    )

    dTI_IS_B = (
        p["k_TI_base_f"] * Act_B
        + p["k_TI_IFN_g_f"] * Act_B * IFN_g
        + p["k_TI_IL_10_f"] * Act_B * IL_10
        + p["k_TI_f"] * TI_IS_B * (1 - TI_IS_B / p["k_TI_m"])
        - p["k_TI_d"] * TI_IS_B
    )

    dIgG4 = (
        p["k_IgG4_TI_f"] * 1e8 * TI_IS_B
        + p["k_IgG4_TD_f"] * 1e8 * TD_IS_B
        - p["k_IgG4_d"] * IgG4
    )

    # 合并所有导数为torch.Tensor，维持梯度
    dy = torch.stack([
        dAntigen, dnDC, dmDC, dGMCSF, dpDC,
        dIL_33, dIL_6, dIL_12, dIL_15, dIL_7, dIFN1, dIL_1, dIL_2, dIL_4, dIL_10, dTGFbeta, dIFN_g,
        dnaive_CD4, dact_CD4, dTh2, diTreg, dCD4_CTL, dnTreg, dTFH,
        dNK, dact_NK,
        dNaive_B, dAct_B, dTD_IS_B, dTI_IS_B,
        dIgG4
    ], dim=0)
    
    return dy

# -------------------------
# 3) 仿真：支持多受试者（不同 y0）+ 指定输出时间点
# -------------------------
def simulate_torchdiffeq(ts: np.ndarray, y0: np.ndarray, p: Dict[str, float], verbose: bool = False) -> np.ndarray:
    """
    使用 torchdiffeq 求解ODE系统（对刚性问题有更好支持）
    
    Parameters:
    -----------
    ts : 时间点数组
    y0 : 初始条件
    p : 参数字典
    verbose : 是否打印详细信息
    """
    if not TORCHDIFFEQ_AVAILABLE:
        if verbose:
            print("  错误: torchdiffeq不可用，请安装: pip install torchdiffeq")
        return np.full((len(ts), N_STATE), np.nan)
    
    try:
        # 获取求解器配置
        method = ODE_SOLVER_CONFIG['method']
        rtol = ODE_SOLVER_CONFIG['rtol']
        atol = ODE_SOLVER_CONFIG['atol']
        
        # 转换为torch张量（使用float64精度）
        t_torch = torch.from_numpy(ts).to(DTYPE)
        y0_torch = torch.from_numpy(y0).to(DTYPE)
        
        if verbose:
            print(f"  使用 torchdiffeq 求解: 方法={method}, rtol={rtol}, atol={atol}")
            print(f"  时间跨度=[{ts[0]:.2f}, {ts[-1]:.2f}], 输出点数={len(ts)}")
        
        # 定义ODE函数包装器（适配torchdiffeq）
        def ode_func(t, y):
            # 转换为numpy计算
            y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
            t_scalar = float(t) if isinstance(t, torch.Tensor) else float(t)
            
            # 调用原始RHS函数
            dy_np = rhs(t_scalar, y_np, p)
            
            # 转换回torch（使用指定精度）
            dy_torch = torch.from_numpy(dy_np).to(DTYPE)
            return dy_torch
        
        # 使用 torchdiffeq 求解
        solution = odeint(
            ode_func,
            y0_torch,
            t_torch,
            method=method,
            rtol=rtol,
            atol=atol,
        )
        
        # 转换回numpy
        result = solution.detach().cpu().numpy()
        
        if verbose:
            print(f"  ODE求解完成 ({method}): 成功")
        
        return result  # (T, N_STATE)
        
    except Exception as e:
        if verbose:
            print(f"  ODE求解异常 (torchdiffeq): {type(e).__name__}: {e}")
        return np.full((len(ts), N_STATE), np.nan)


def simulate_torch_grad(ts: torch.Tensor, y0: torch.Tensor, p: Dict[str, torch.Tensor], verbose: bool = False) -> torch.Tensor:
    """
    使用 torchdiffeq 求解ODE系统（支持自动微分和梯度回传）
    
    Parameters:
    -----------
    ts : 时间点数组（torch.Tensor, float64）
    y0 : 初始条件（torch.Tensor, float64，支持requires_grad）
    p : 参数字典，值为torch.Tensor（支持requires_grad）
    verbose : 是否打印详细信息
    
    Returns:
    --------
    solution : torch.Tensor, shape=(T, N_STATE)，支持梯度回传
    """
    if not TORCHDIFFEQ_AVAILABLE:
        raise ImportError("torchdiffeq is required for gradient-based optimization. Install with: pip install torchdiffeq")
    
    try:
        method = ODE_SOLVER_CONFIG['method']
        rtol = ODE_SOLVER_CONFIG['rtol']
        atol = ODE_SOLVER_CONFIG['atol']
        
        if verbose:
            print(f"  [torch_grad] 使用 torchdiffeq 求解 (方法={method}, 支持梯度)")
        
        # 定义ODE函数包装器（使用torch版本RHS）
        def ode_func_torch(t_scalar, y_torch):
            """ODE右侧函数，接收和返回torch.Tensor，支持梯度"""
            # t可能是Tensor或float，转成float
            t_float = float(t_scalar) if isinstance(t_scalar, torch.Tensor) else float(t_scalar)
            # 调用torch版本的RHS
            dy_torch = rhs_torch(t_float, y_torch, p)
            return dy_torch
        
        # 使用 torchdiffeq 求解（保持梯度链）
        # 为 scipy_solver 准备 options
        solver_options = {}
        if method == 'scipy_solver' and 'solver' in ODE_SOLVER_CONFIG:
            solver_options = {'method': ODE_SOLVER_CONFIG['solver']}
        
        solution = odeint(
            ode_func_torch,
            y0,  # 初始条件，可能requires_grad=True
            ts,  # 时间点
            method=method,
            rtol=rtol,
            atol=atol,
            options=solver_options if solver_options else None,
        )
        
        if verbose:
            print(f"  [torch_grad] ODE求解完成, shape={solution.shape}, requires_grad={solution.requires_grad}")
        
        return solution  # (T, N_STATE)，支持梯度
        
    except Exception as e:
        print(f"  [ERROR] torchdiffeq求解失败: {type(e).__name__}: {e}")
        raise

def simulate_scipy(ts: np.ndarray, y0: np.ndarray, p: Dict[str, float], verbose: bool = False, try_bdf_on_fail: bool = True) -> np.ndarray:
    """
    使用 scipy.integrate.solve_ivp 求解ODE系统
    
    Parameters:
    -----------
    ts : 时间点数组
    y0 : 初始条件
    p : 参数字典
    verbose : 是否打印详细信息
    try_bdf_on_fail : 如果LSODA失败，是否尝试BDF方法（对刚性问题更有效）
    """
    methods_to_try = [ODE_SOLVER_CONFIG['scipy_method']]
    if try_bdf_on_fail and ODE_SOLVER_CONFIG['scipy_method'] != 'BDF':
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
                rtol=ODE_SOLVER_CONFIG['scipy_rtol'],
                atol=ODE_SOLVER_CONFIG['scipy_atol'],
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


def simulate(ts: np.ndarray, y0: np.ndarray, p: Dict[str, float], verbose: bool = False, try_bdf_on_fail: bool = True) -> np.ndarray:
    """
    求解ODE系统 - 根据配置自动选择求解器后端
    
    Parameters:
    -----------
    ts : 时间点数组
    y0 : 初始条件
    p : 参数字典
    verbose : 是否打印详细信息
    try_bdf_on_fail : 如果主方法失败，是否尝试备用方法（对刚性问题更有效）
    """
    ts = np.asarray(ts, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    assert y0.shape == (N_STATE,)

    if verbose:
        backend = ODE_SOLVER_CONFIG['backend']
        print(f"  开始求解ODE (后端: {backend}): t_span=[{ts[0]:.2f}, {ts[-1]:.2f}], 时间点数={len(ts)}")

    # 根据配置选择求解器后端
    backend = ODE_SOLVER_CONFIG.get('backend', 'scipy')
    
    if backend == 'torchdiffeq' and TORCHDIFFEQ_AVAILABLE:
        return simulate_torchdiffeq(ts, y0, p, verbose)
    else:
        # 默认使用scipy
        if backend == 'torchdiffeq' and not TORCHDIFFEQ_AVAILABLE:
            if verbose:
                print("  警告: 配置为torchdiffeq但不可用，切换到scipy")
        return simulate_scipy(ts, y0, p, verbose, try_bdf_on_fail)

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
    y0[IDX["NK"]] = 50.0          # NK约50
    y0[IDX["act_NK"]] = 200.0     # act_NK约180-200，取190
    y0[IDX["Naive_B"]] = 92.0     # Naive B约84-92，取88
    y0[IDX["Act_B"]] = 80.0       # Act_B约80
    y0[IDX["TD_IS_B"]] = 2.0      # TD-Plasma初始为0
    y0[IDX["TI_IS_B"]] = 0.0      # TI-Plasma初始为0
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
    print(f"    nDC: {y0[IDX['nDC']]:.1f}, naive_CD4: {y0[IDX['naive_CD4']]:.1f}, NK: {y0[IDX['NK']]:.1f}")
    print(f"    act_NK: {y0[IDX['act_NK']]:.1f}, Naive_B: {y0[IDX['Naive_B']]:.1f}, Act_B: {y0[IDX['Act_B']]:.1f}")
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
        "NK",            # 11
        "act_NK",        # 12
        "Naive_B",       # 13
        "Act_B",         # 14
        "TD_IS_B",       # 15
        "TI_IS_B",       # 16
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
        "k_IgG4_TD_f": 1.0,  # k_IgG4_TD_IS_B_cells_f
        "k_IgG4_TI_f": 1.0,  # k_IgG4_TI_IS_B_cells_f
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
    # nDC, mDC, pDC, naive_CD4, act_CD4, Th2, iTreg, CD4_CTL, nTreg, TFH, NK, act_NK, Naive_B, Act_B, TD_IS_B, TI_IS_B
    a_values_t200 = [0, 20, 4, 75, 500, 0.2, 45, 2, 5, 20, 25, 310, 94, 90, 10, 0.1]
    a_vars = ["nDC", "mDC", "pDC", "naive_CD4", "act_CD4", "Th2", "iTreg", "CD4_CTL", 
              "nTreg", "TFH", "NK", "act_NK", "Naive_B", "Act_B", "TD_IS_B", "TI_IS_B"]
    
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
    
    # 添加 Antigen 和 IgG4（都为0）
    target_igg4["Antigen"] = 0.0
    target_igg4["IgG4"] = 0.0
    
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
    优化未知参数（使用PyTorch自动微分）
    
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
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for optimization. Please install it with: pip install torch")
    
    # 设置目标值
    target_hc, target_igg4 = setup_target_values()
    y0_np = get_initial_conditions()
    
    # 计算归一化参数：(x - HC) / (IgG - HC)
    # 只使用 HC 和 IgG4 都有的变量
    common_vars = set(target_hc.keys()) & set(target_igg4.keys())
    common_vars = sorted(list(common_vars))  # 排序确保顺序一致
    
    # 构建只包含共同变量的向量
    hc_values_list = [target_hc[var] for var in common_vars]
    igg4_values_list = [target_igg4[var] for var in common_vars]
    
    hc_values = torch.tensor(hc_values_list, dtype=DTYPE)
    igg4_values = torch.tensor(igg4_values_list, dtype=DTYPE)
    
    # 计算分母：IgG - HC
    norm_denominator = igg4_values - hc_values
    
    # 对 y0 进行归一化：(y0 - HC) / (IgG - HC)
    y0_normalized = np.zeros_like(y0_np)
    for i, var_name in enumerate(STATE_NAMES):
        if var_name in common_vars:
            var_idx = common_vars.index(var_name)
            hc_val = hc_values[var_idx].item()
            denom_val = norm_denominator[var_idx].item()
            if abs(denom_val) > 1e-10:
                y0_normalized[i] = (y0_np[i] - hc_val) / denom_val
            else:
                y0_normalized[i] = 0.0
    
    y0_torch = torch.from_numpy(y0_normalized).to(DTYPE)  # 转为torch，指定精度
    
    if verbose:
        print(f"共同变量数: {len(common_vars)}")
    
    # 所有参数都参与优化（不再区分已知和未知）
    # 初始化所有参数为1.0
    p_init_np = {name: 1.0 for name in ALL_PARAMS}
    
    if verbose:
        print(f"要优化的参数总数: {len(ALL_PARAMS)}")
        print(f"优化器: {OPTIMIZATION_CONFIG['optimizer']}")
        print(f"\n初始条件 y0:")
        print(f"  shape: {y0_torch.shape}")
        print(f"  min: {y0_torch.min().item():.4e}, max: {y0_torch.max().item():.4e}, mean: {y0_torch.mean().item():.4e}")
        print(f"  非零元素数: {(y0_torch != 0).sum().item()}/{len(y0_torch)}")
        print()
    
    # 初始化可优化参数为torch.Tensor（log空间，所有参数）
    # 初始值 1e-4 对应 log(1e-4) ≈ -9.21
    log_init_value = np.log(1e-4)
    log_params_tensor = torch.full((len(ALL_PARAMS),), log_init_value, requires_grad=True, dtype=DTYPE)
    
    # 参数边界（log空间）
    lb = OPTIMIZATION_CONFIG['log_param_lb']
    ub = OPTIMIZATION_CONFIG['log_param_ub']
    
    # 时间点数组（四个目标时间点：t=50, 100, 200, 250）
    ts_numpy = np.array([0.0, 50.0, 100.0, 200.0, 250.0])
    ts_torch = torch.from_numpy(ts_numpy).to(DTYPE)
    
    # 定义损失计算函数（支持自动微分，包含归一化）
    def compute_loss_with_grad(log_params_tensor_input: torch.Tensor) -> torch.Tensor:
        """
        计算损失（支持自动微分，包含输入输出归一化）
        
        Parameters:
        -----------
        log_params_tensor_input : log空间的参数张量，shape=(n_params,)
        
        Returns:
        --------
        loss : 标量torch.Tensor，支持梯度回传
        """
        # 应用边界约束
        log_params_clipped = torch.clamp(log_params_tensor_input, min=lb, max=ub)
        
        # 转换为原始空间
        params_exp = torch.exp(log_params_clipped)
        
        # 构建完整的参数字典（所有参数都参与优化）
        p_torch = {}
        for i, name in enumerate(ALL_PARAMS):
            p_torch[name] = params_exp[i]
        
        try:
            # 使用torch版本的ODE求解（支持梯度）
            solution = simulate_torch_grad(ts_torch, y0_torch, p_torch, verbose=False)
            
            # 计算损失（使用线性归一化：(x - HC) / (IgG - HC)）
            # 四个时间点的目标：t=50(idx=1)->HC, t=100(idx=2)->HC, t=200(idx=3)->IgG, t=250(idx=4)->IgG
            loss_terms = []
            
            time_indices_and_targets = [
                (1, target_hc),      # t=50, 目标=HC (归一化为0)
                (2, target_hc),      # t=100, 目标=HC (归一化为0)
                (3, target_igg4),    # t=200, 目标=IgG (归一化为1)
                (4, target_igg4),    # t=250, 目标=IgG (归一化为1)
            ]
            
            for time_idx, target_dict in time_indices_and_targets:
                pred_list = []
                target_list = []
                
                for var_name, target_val in target_dict.items():
                    if var_name in IDX:
                        pred_val = solution[time_idx, IDX[var_name]]
                        pred_list.append(pred_val)
                        target_list.append(target_val)
                
                if pred_list:
                    pred = torch.stack(pred_list)
                    target = torch.tensor(target_list, dtype=DTYPE, device=solution.device)
                    
                    # 归一化：(x - HC) / (IgG - HC)
                    # 预测值归一化
                    pred_norm = (pred - hc_values) / (norm_denominator + 1e-10)
                    # 目标值归一化（HC->0，IgG->1）
                    if target_dict == target_hc:
                        target_norm = torch.zeros_like(pred)  # HC 目标归一化为 0
                    else:
                        target_norm = torch.ones_like(pred)   # IgG 目标归一化为 1
                    
                    loss = torch.sum((pred_norm - target_norm) ** 2)
                    loss_terms.append(loss)
            
            # 汇总损失
            if loss_terms:
                loss = torch.stack(loss_terms).sum()
            else:
                loss = torch.tensor(0.0, dtype=DTYPE, device=solution.device)
            
            return loss
            
        except Exception as e:
            # ODE求解失败，返回大的惩罚值
            if verbose:
                print(f"  ODE求解异常: {type(e).__name__}: {e}")
            return torch.tensor(1e10, dtype=DTYPE)
    
    # 创建优化器
    optimizer_name = OPTIMIZATION_CONFIG['optimizer'].lower()
    if optimizer_name == 'adam':
        optimizer = optim.Adam([log_params_tensor], lr=OPTIMIZATION_CONFIG['lr'], 
                               weight_decay=OPTIMIZATION_CONFIG['weight_decay'])
    elif optimizer_name == 'lbfgs':
        optimizer = optim.LBFGS([log_params_tensor], 
                                lr=OPTIMIZATION_CONFIG['lr'],
                                max_iter=OPTIMIZATION_CONFIG['max_iter'],
                                max_eval=OPTIMIZATION_CONFIG['max_eval'],
                                history_size=OPTIMIZATION_CONFIG['history_size'],
                                line_search_fn=OPTIMIZATION_CONFIG['line_search_fn'])
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD([log_params_tensor], lr=OPTIMIZATION_CONFIG['lr'],
                             weight_decay=OPTIMIZATION_CONFIG['weight_decay'])
    else:
        raise ValueError(f"Unknown optimizer: {OPTIMIZATION_CONFIG['optimizer']}")
    
    # 优化循环
    best_loss = float('inf')
    best_log_params = log_params_tensor.detach().clone()
    loss_history = []
    
    # 创建进度条
    if verbose:
        pbar = tqdm(total=OPTIMIZATION_CONFIG['max_iter'], desc="Optimizing (Autograd)", unit="iter", ncols=100)
    else:
        pbar = None
    
    for iteration in range(OPTIMIZATION_CONFIG['max_iter']):
        # 清空梯度
        optimizer.zero_grad()
        
        # 前向传播计算损失（自动建立计算图）
        loss = compute_loss_with_grad(log_params_tensor)
        
        # 反向传播计算梯度
        loss.backward()
        
        # 更新进度条的信息
        if pbar is not None:
            if log_params_tensor.grad is not None:
                grad_norm = torch.norm(log_params_tensor.grad).item()
                non_zero_grads = (torch.abs(log_params_tensor.grad) > 1e-12).sum().item()
                pbar.set_postfix({'loss': f'{loss.item():.2e}', 'grad_norm': f'{grad_norm:.2e}'})
            else:
                pbar.set_postfix({'loss': f'{loss.item():.2e}', 'grad': 'None'})
        
        # 应用梯度裁剪
        if log_params_tensor.grad is not None:
            torch.nn.utils.clip_grad_norm_([log_params_tensor], max_norm=10.0)
        
        # 优化器步进
        if optimizer_name == 'lbfgs':
            def closure():
                optimizer.zero_grad()
                loss = compute_loss_with_grad(log_params_tensor)
                loss.backward()
                return loss
            
            loss = optimizer.step(closure)
        else:
            optimizer.step()
        
        # 应用边界约束
        with torch.no_grad():
            log_params_tensor.clamp_(lb, ub)
        
        # 记录最佳结果
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        if loss_val < best_loss:
            best_loss = loss_val
            best_log_params = log_params_tensor.detach().clone()
        
        # 更新进度条
        if pbar is not None:
            pbar.set_postfix({"loss": f"{loss_val:.4e}", "best": f"{best_loss:.4e}"})
            pbar.update(1)
        
        # 检查收敛
        if iteration > 10 and len(loss_history) > 10:
            recent_loss = np.array(loss_history[-10:])
            if np.std(recent_loss) < 1e-10 * np.mean(recent_loss):
                if verbose:
                    print(f"\n收敛于迭代 {iteration}")
                break
    
    if pbar is not None:
        pbar.close()
    
    # 构建最优参数字典
    best_params_dict = {}
    best_log_params_np = best_log_params.detach().cpu().numpy()
    for i, param_name in enumerate(ALL_PARAMS):
        best_params_dict[param_name] = np.exp(np.clip(best_log_params_np[i], lb, ub))
    
    if verbose:
        print(f"\n最终损失: {best_loss:.4e}")
        print(f"优化迭代次数: {iteration + 1}")
        print(f"损失历史范围: [{min(loss_history):.4e}, {max(loss_history):.4e}]")
        
        # 打印参数变化
        print("\n参数变化（log空间 -> 原始空间，显示非1.0的）:")
        initial_log = np.zeros(len(ALL_PARAMS))
        for i, param_name in enumerate(ALL_PARAMS):
            initial_val = np.exp(initial_log[i])
            final_val = best_params_dict[param_name]
            if abs(final_val - 1.0) > 1e-6:  # 只显示明显改变的
                change_ratio = final_val / initial_val if initial_val != 0 else float('inf')
                print(f"  {param_name}: {initial_val:.6e} -> {final_val:.6e} (倍数: {change_ratio:.4f}x)")
        print()
    
    return best_params_dict, best_loss


if __name__ == "__main__":
    # 测试积分功能：时间范围0-300s，时间步为1
    # test_integration(t_end=300.0, n_points=301, verbose=True)
    
    print("="*60)
    print("参数优化（所有参数，使用归一化）")
    print("="*60)
    print(f"总参数数量: {len(ALL_PARAMS)}")
    print()
    
    # 优化所有参数（不再使用 unknown_params 和 known_params 的分法）
    best_params, final_loss = optimize_parameters(
        unknown_params=ALL_PARAMS,  # 所有参数
        fixed_params={},             # 无固定参数
        verbose=True
    )
    
    print(f"\n优化完成，最终损失: {final_loss:.4e}")
    
    # 打印优化后的未知参数及其值
    print("\n" + "="*60)
    print("优化后的未知参数:")
    print("="*60)
    for param_name in unknown_params:
        param_value = best_params[param_name]
        print(f"  {param_name}: {param_value:.6e}")
    
    # 使用优化后的参数运行完整模拟并绘图
    print("\n" + "="*60)
    print("使用优化后的参数进行模拟并绘图")
    print("="*60)
    
    # 获取初始条件（稳态）
    y0 = get_initial_conditions()
    
    # 运行完整模拟
    t_end = SIMULATION_CONFIG['t_end']
    time_step = SIMULATION_CONFIG['time_step']
    ts = np.arange(0, int(t_end) + time_step, time_step)
    
    print(f"运行模拟: t=0 到 t={t_end}s，时间步=1s...")
    result = simulate(ts, y0, best_params, verbose=True)
    
    if result is not None:
        print("模拟完成，开始绘图...")
        plot_results(ts, result)
    else:
        print("模拟失败，无法绘图")
    
    # 如果需要运行优化，取消下面的注释
    """
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
    """
