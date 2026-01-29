# ============================================================================
# RHS线性化：将所有31个变量的RHS转换成关于参数的线性方程形式
# 前提：所有变量值已知（从target数据获取），Ag(t)已知
# 目标：RHS = 0 的线性方程组
# ============================================================================

# ============================================================================
# 线性化规则
# ============================================================================
# 1. 增殖项 k_f * X * (1 - X/k_m)：
#    展开：k_f * X - k_f * X^2 / k_m
#    定义：k_f_div_m = k_f / k_m
#    线性化：k_f * X - k_f_div_m * X^2
#
# 2. 希尔方程项 k_f * X * Y / (Y + k_m)：
#    当X, Y已知时，定义组合参数：k_f_eff = k_f * Y / (Y + k_m)
#    注意：k_f_eff依赖于k_m，需要先估计k_m或通过迭代求解
#    线性化：k_f_eff * X
#
# 3. 复杂乘积项 k_f * X * (k_m1/(k_m1+Y1)) * (k_m2/(k_m2+Y2))：
#    定义组合参数：k_f_eff = k_f * (k_m1/(k_m1+Y1)) * (k_m2/(k_m2+Y2))
#    线性化：k_f_eff * X

# ============================================================================
# 需要定义的新参数（组合参数）
# ============================================================================

# 类型1：增殖项中的 k_f / k_m
NEW_PARAMS_DIV = [
    :k_nDC_f_div_m,           # = k_nDC_f / k_nDC_m
    :k_mDC_f_div_m,           # = k_mDC_f / k_mDC_m
    :k_pDC_f_div_m,           # = k_pDC_f / k_pDC_m
    :k_CD4_f_div_m,           # = k_CD4_f / k_CD4_m
    :k_act_CD4_f_div_m,       # = k_act_CD4_f / k_act_CD4_m
    :k_Th2_f_div_m,           # = k_Th2_f / k_Th2_m
    :k_iTreg_f_div_m,         # = k_iTreg_f / k_iTreg_m
    :k_CD4_CTL_f_div_m,       # = k_CD4_CTL_f / k_CD4_CTL_m
    :k_nTreg_f_div_m,         # = k_nTreg_f / k_nTreg_m
    :k_TFH_f_div_m,           # = k_TFH_f / k_TFH_m
    :k_NK_f_div_m,            # = k_NK_f / k_NK_m
    :k_act_NK_f_div_m,        # = k_act_NK_f / k_act_NK_m
    :k_Naive_B_f_div_m,       # = k_Naive_B_f / k_Naive_B_m
    :k_Act_B_f_div_m,         # = k_Act_B_f / k_Act_B_m
    :k_TD_f_div_m,            # = k_TD_f / k_TD_m
    :k_TI_f_div_m,            # = k_TI_f / k_TI_m
]

# 类型2：希尔方程项的组合参数（这些依赖于变量值，需要在每个时间点重新计算）
# 注意：这些参数依赖于k_m值，需要先估计k_m或通过迭代求解

# ============================================================================
# 1. nDC 的 RHS 线性化
# ============================================================================
# 原始项：
#   + k_nDC_f * nDC * (1 - nDC/k_nDC_m)
#   - k_mDC_Antigen_f * Ag(t) * nDC * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10))
#   - k_mDC_GMCSF_f * Ag(t) * nDC * (GMCSF / (GMCSF + k_mDC_GMCSF_m)) * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10))
#   - k_pDC_Antigen_f * nDC * Ag(t)
#   - k_nDC_d * nDC
#
# 线性化（nDC, IL_10, GMCSF, Ag(t) 已知）：
#   + k_nDC_f * nDC - k_nDC_f_div_m * nDC^2
#   - k_mDC_Antigen_IL10_eff * Ag(t) * nDC
#     其中 k_mDC_Antigen_IL10_eff = k_mDC_Antigen_f * k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10)
#   - k_mDC_GMCSF_IL10_eff * Ag(t) * nDC
#     其中 k_mDC_GMCSF_IL10_eff = k_mDC_GMCSF_f * (GMCSF / (GMCSF + k_mDC_GMCSF_m)) * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10))
#   - k_pDC_Antigen_f * Ag(t) * nDC
#   - k_nDC_d * nDC
#
# 新参数：
#   - k_mDC_Antigen_IL10_eff（依赖于k_mDC_Antigen_f和k_mDC_IL_10_m）
#   - k_mDC_GMCSF_IL10_eff（依赖于k_mDC_GMCSF_f, k_mDC_GMCSF_m, k_mDC_IL_10_m）

function rhs_nDC_linearized(nDC, IL_10, GMCSF, Ag_t, params)
    """
    nDC的线性化RHS
    
    参数：
    - nDC, IL_10, GMCSF, Ag_t: 已知变量值
    - params: 参数字典，包含：
        k_nDC_f, k_nDC_f_div_m, k_nDC_d,
        k_mDC_Antigen_IL10_eff, k_mDC_GMCSF_IL10_eff,
        k_pDC_Antigen_f
    """
    return (
        + params[:k_nDC_f] * nDC
        - params[:k_nDC_f_div_m] * nDC^2
        - params[:k_mDC_Antigen_IL10_eff] * Ag_t * nDC
        - params[:k_mDC_GMCSF_IL10_eff] * Ag_t * nDC
        - params[:k_pDC_Antigen_f] * Ag_t * nDC
        - params[:k_nDC_d] * nDC
    )
end

# ============================================================================
# 2. mDC 的 RHS 线性化
# ============================================================================
# 原始项：
#   + k_mDC_Antigen_f * Ag(t) * nDC * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10))
#   + k_mDC_GMCSF_f * Ag(t) * nDC * (GMCSF / (GMCSF + k_mDC_GMCSF_m)) * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10))
#   + k_mDC_f * mDC * (1 - mDC / k_mDC_m)
#   - k_mDC_d * mDC
#
# 线性化（mDC, nDC, IL_10, GMCSF, Ag(t) 已知）：
#   + k_mDC_Antigen_IL10_eff * Ag(t) * nDC  (复用nDC中的定义)
#   + k_mDC_GMCSF_IL10_eff * Ag(t) * nDC  (复用nDC中的定义)
#   + k_mDC_f * mDC - k_mDC_f_div_m * mDC^2
#   - k_mDC_d * mDC

function rhs_mDC_linearized(mDC, nDC, IL_10, GMCSF, Ag_t, params)
    return (
        + params[:k_mDC_Antigen_IL10_eff] * Ag_t * nDC
        + params[:k_mDC_GMCSF_IL10_eff] * Ag_t * nDC
        + params[:k_mDC_f] * mDC
        - params[:k_mDC_f_div_m] * mDC^2
        - params[:k_mDC_d] * mDC
    )
end

# ============================================================================
# 3. pDC 的 RHS 线性化
# ============================================================================
# 原始项：
#   + k_pDC_Antigen_f * nDC * Ag(t)
#   + k_pDC_f * pDC * (1 - pDC / k_pDC_m)
#   - k_pDC_d * pDC
#
# 线性化：
#   + k_pDC_Antigen_f * Ag(t) * nDC
#   + k_pDC_f * pDC - k_pDC_f_div_m * pDC^2
#   - k_pDC_d * pDC

function rhs_pDC_linearized(pDC, nDC, Ag_t, params)
    return (
        + params[:k_pDC_Antigen_f] * Ag_t * nDC
        + params[:k_pDC_f] * pDC
        - params[:k_pDC_f_div_m] * pDC^2
        - params[:k_pDC_d] * pDC
    )
end

# ============================================================================
# 4. GMCSF 的 RHS 线性化
# ============================================================================
# 原始项：
#   + k_GMCSF_Th2_f * Th2
#   + k_GMCSF_Th2_Antigen_f * Th2 * Ag(t)
#   + k_GMCSF_act_NK_f * act_NK
#   - k_mDC_GMCSF_d * Ag(t) * nDC * (GMCSF / (GMCSF + k_mDC_GMCSF_m)) * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10))
#   - k_GMCSF_d * GMCSF
#
# 线性化：
#   + k_GMCSF_Th2_f * Th2
#   + k_GMCSF_Th2_Antigen_f * Th2 * Ag(t)
#   + k_GMCSF_act_NK_f * act_NK
#   - k_mDC_GMCSF_d_eff * Ag(t) * nDC
#     其中 k_mDC_GMCSF_d_eff = k_mDC_GMCSF_d * (GMCSF / (GMCSF + k_mDC_GMCSF_m)) * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10))
#   - k_GMCSF_d * GMCSF

function rhs_GMCSF_linearized(GMCSF, Th2, act_NK, nDC, IL_10, Ag_t, params)
    return (
        + params[:k_GMCSF_Th2_f] * Th2
        + params[:k_GMCSF_Th2_Antigen_f] * Th2 * Ag_t
        + params[:k_GMCSF_act_NK_f] * act_NK
        - params[:k_mDC_GMCSF_d_eff] * Ag_t * nDC
        - params[:k_GMCSF_d] * GMCSF
    )
end

# ============================================================================
# 5. IL_33 的 RHS 线性化
# ============================================================================
# 原始项：
#   + k_IL_33_pDC_f * pDC
#   - k_act_CD4_IL_33_d * act_CD4 * IL_33 / (k_Th2_IL_33_m + IL_33)
#   - k_IL_33_d * IL_33
#
# 线性化：
#   + k_IL_33_pDC_f * pDC
#   - k_act_CD4_IL_33_d_eff * act_CD4
#     其中 k_act_CD4_IL_33_d_eff = k_act_CD4_IL_33_d * IL_33 / (k_Th2_IL_33_m + IL_33)
#   - k_IL_33_d * IL_33

function rhs_IL_33_linearized(IL_33, pDC, act_CD4, params)
    return (
        + params[:k_IL_33_pDC_f] * pDC
        - params[:k_act_CD4_IL_33_d_eff] * act_CD4
        - params[:k_IL_33_d] * IL_33
    )
end

# ============================================================================
# 6. IL_6 的 RHS 线性化
# ============================================================================
# 原始项：
#   + k_IL_6_pDC_f * pDC
#   + k_IL_6_mDC_f * mDC
#   + k_IL_6_TFH_f * TFH * (k_TFH_nTreg_m / (nTreg + k_TFH_nTreg_m))
#   - k_TFH_IL_6_d * act_CD4 * IL_6 / (k_TFH_IL_6_m + IL_6)
#   - k_IL_6_d * IL_6
#
# 线性化：
#   + k_IL_6_pDC_f * pDC
#   + k_IL_6_mDC_f * mDC
#   + k_IL_6_TFH_eff * TFH
#     其中 k_IL_6_TFH_eff = k_IL_6_TFH_f * (k_TFH_nTreg_m / (nTreg + k_TFH_nTreg_m))
#   - k_TFH_IL_6_d_eff * act_CD4
#     其中 k_TFH_IL_6_d_eff = k_TFH_IL_6_d * IL_6 / (k_TFH_IL_6_m + IL_6)
#   - k_IL_6_d * IL_6

function rhs_IL_6_linearized(IL_6, pDC, mDC, TFH, nTreg, act_CD4, params)
    return (
        + params[:k_IL_6_pDC_f] * pDC
        + params[:k_IL_6_mDC_f] * mDC
        + params[:k_IL_6_TFH_eff] * TFH
        - params[:k_TFH_IL_6_d_eff] * act_CD4
        - params[:k_IL_6_d] * IL_6
    )
end

# ============================================================================
# 7. IL_12 的 RHS 线性化
# ============================================================================
# 原始项：
#   + k_IL_12_mDC_f * mDC
#   - k_act_NK_IL_12_d * NK * IL_12 / (IL_12 + k_act_NK_IL_12_m)
#   - k_IL_12_d * IL_12
#
# 线性化：
#   + k_IL_12_mDC_f * mDC
#   - k_act_NK_IL_12_d_eff * NK
#     其中 k_act_NK_IL_12_d_eff = k_act_NK_IL_12_d * IL_12 / (IL_12 + k_act_NK_IL_12_m)
#   - k_IL_12_d * IL_12

function rhs_IL_12_linearized(IL_12, mDC, NK, params)
    return (
        + params[:k_IL_12_mDC_f] * mDC
        - params[:k_act_NK_IL_12_d_eff] * NK
        - params[:k_IL_12_d] * IL_12
    )
end

# ============================================================================
# 8. IL_15 的 RHS 线性化
# ============================================================================
# 原始项：
#   + k_IL_15_f
#   + k_IL_15_Antigen_f * Ag(t)
#   - k_naive_CD4_IL_15_d * naive_CD4 * IL_15 / (k_naive_CD4_IL_15_m + IL_15)
#   - k_act_CD4_IL_15_d * act_CD4 * IL_15 / (k_act_CD4_IL_15_m + IL_15)
#   - k_IL_15_d * IL_15
#
# 线性化：
#   + k_IL_15_f
#   + k_IL_15_Antigen_f * Ag(t)
#   - k_naive_CD4_IL_15_d_eff * naive_CD4
#     其中 k_naive_CD4_IL_15_d_eff = k_naive_CD4_IL_15_d * IL_15 / (k_naive_CD4_IL_15_m + IL_15)
#   - k_act_CD4_IL_15_d_eff * act_CD4
#     其中 k_act_CD4_IL_15_d_eff = k_act_CD4_IL_15_d * IL_15 / (k_act_CD4_IL_15_m + IL_15)
#   - k_IL_15_d * IL_15

function rhs_IL_15_linearized(IL_15, naive_CD4, act_CD4, Ag_t, params)
    return (
        + params[:k_IL_15_f]
        + params[:k_IL_15_Antigen_f] * Ag_t
        - params[:k_naive_CD4_IL_15_d_eff] * naive_CD4
        - params[:k_act_CD4_IL_15_d_eff] * act_CD4
        - params[:k_IL_15_d] * IL_15
    )
end

# ============================================================================
# 9. IL_7 的 RHS 线性化
# ============================================================================
# 原始项：
#   + k_IL_7_f
#   - k_naive_CD4_IL_7_d * naive_CD4 * IL_7 / (k_naive_CD4_IL_7_m + IL_7)
#   - k_act_CD4_IL_7_d * act_CD4 * IL_7 / (k_act_CD4_IL_7_m + IL_7)
#   - k_IL_7_d * IL_7
#
# 线性化：
#   + k_IL_7_f
#   - k_naive_CD4_IL_7_d_eff * naive_CD4
#     其中 k_naive_CD4_IL_7_d_eff = k_naive_CD4_IL_7_d * IL_7 / (k_naive_CD4_IL_7_m + IL_7)
#   - k_act_CD4_IL_7_d_eff * act_CD4
#     其中 k_act_CD4_IL_7_d_eff = k_act_CD4_IL_7_d * IL_7 / (k_act_CD4_IL_7_m + IL_7)
#   - k_IL_7_d * IL_7

function rhs_IL_7_linearized(IL_7, naive_CD4, act_CD4, params)
    return (
        + params[:k_IL_7_f]
        - params[:k_naive_CD4_IL_7_d_eff] * naive_CD4
        - params[:k_act_CD4_IL_7_d_eff] * act_CD4
        - params[:k_IL_7_d] * IL_7
    )
end

# ============================================================================
# 10. IFN1 的 RHS 线性化
# ============================================================================
# 原始项：
#   + k_IFN1_pDC_f * pDC
#   - k_act_CD4_IFN1_d * act_CD4 * IFN1 / (k_IFN1_CD4_CTL_m + IFN1)
#   - k_act_NK_IFN1_d * NK * IFN1 / (IFN1 + k_act_NK_IFN1_m)
#   - k_IFN1_d * IFN1
#
# 线性化：
#   + k_IFN1_pDC_f * pDC
#   - k_act_CD4_IFN1_d_eff * act_CD4
#     其中 k_act_CD4_IFN1_d_eff = k_act_CD4_IFN1_d * IFN1 / (k_IFN1_CD4_CTL_m + IFN1)
#   - k_act_NK_IFN1_d_eff * NK
#     其中 k_act_NK_IFN1_d_eff = k_act_NK_IFN1_d * IFN1 / (IFN1 + k_act_NK_IFN1_m)
#   - k_IFN1_d * IFN1

function rhs_IFN1_linearized(IFN1, pDC, act_CD4, NK, params)
    return (
        + params[:k_IFN1_pDC_f] * pDC
        - params[:k_act_CD4_IFN1_d_eff] * act_CD4
        - params[:k_act_NK_IFN1_d_eff] * NK
        - params[:k_IFN1_d] * IFN1
    )
end

# ============================================================================
# 11. IL_1 的 RHS 线性化
# ============================================================================
# 原始项：
#   + k_IL_1_mDC_f * mDC
#   - k_IL_1_d * IL_1
#
# 线性化（已经是线性的）：
#   + k_IL_1_mDC_f * mDC
#   - k_IL_1_d * IL_1

function rhs_IL_1_linearized(IL_1, mDC, params)
    return (
        + params[:k_IL_1_mDC_f] * mDC
        - params[:k_IL_1_d] * IL_1
    )
end

# ============================================================================
# 12. IL_2 的 RHS 线性化
# ============================================================================
# 原始项：
#   + k_IL_2_act_CD4_f * act_CD4
#   + k_IL_2_act_CD4_Antigen_f * act_CD4 * Ag(t)
#   - k_act_CD4_IL_2_d * naive_CD4 * IL_2 / (k_act_CD4_IL_2_m + IL_2)
#   - k_act_NK_IL_2_d * NK * IL_2 / (IL_2 + k_act_NK_IL_2_m)
#   - k_IL_2_d * IL_2
#
# 线性化：
#   + k_IL_2_act_CD4_f * act_CD4
#   + k_IL_2_act_CD4_Antigen_f * act_CD4 * Ag(t)
#   - k_act_CD4_IL_2_d_eff * naive_CD4
#     其中 k_act_CD4_IL_2_d_eff = k_act_CD4_IL_2_d * IL_2 / (k_act_CD4_IL_2_m + IL_2)
#   - k_act_NK_IL_2_d_eff * NK
#     其中 k_act_NK_IL_2_d_eff = k_act_NK_IL_2_d * IL_2 / (IL_2 + k_act_NK_IL_2_m)
#   - k_IL_2_d * IL_2

function rhs_IL_2_linearized(IL_2, act_CD4, naive_CD4, NK, Ag_t, params)
    return (
        + params[:k_IL_2_act_CD4_f] * act_CD4
        + params[:k_IL_2_act_CD4_Antigen_f] * act_CD4 * Ag_t
        - params[:k_act_CD4_IL_2_d_eff] * naive_CD4
        - params[:k_act_NK_IL_2_d_eff] * NK
        - params[:k_IL_2_d] * IL_2
    )
end

# ============================================================================
# 13. IL_4 的 RHS 线性化
# ============================================================================
# 原始项：
#   + k_IL_4_Th2_f * Th2
#   + k_IL_4_Th2_Antigen_f * Th2 * Ag(t)
#   - k_act_CD4_IL_4_d * act_CD4 * IL_4 / (k_Th2_IL_4_m + IL_4)
#   - k_IL_4_d * IL_4
#
# 线性化：
#   + k_IL_4_Th2_f * Th2
#   + k_IL_4_Th2_Antigen_f * Th2 * Ag(t)
#   - k_act_CD4_IL_4_d_eff * act_CD4
#     其中 k_act_CD4_IL_4_d_eff = k_act_CD4_IL_4_d * IL_4 / (k_Th2_IL_4_m + IL_4)
#   - k_IL_4_d * IL_4

function rhs_IL_4_linearized(IL_4, Th2, act_CD4, Ag_t, params)
    return (
        + params[:k_IL_4_Th2_f] * Th2
        + params[:k_IL_4_Th2_Antigen_f] * Th2 * Ag_t
        - params[:k_act_CD4_IL_4_d_eff] * act_CD4
        - params[:k_IL_4_d] * IL_4
    )
end

# ============================================================================
# 14. IL_10 的 RHS 线性化
# ============================================================================
# 原始项：
#   + k_IL_10_iTreg_f * iTreg
#   + k_IL_10_nTreg_f * nTreg * mDC / (k_IL_10_nTreg_mDC_m + mDC)
#   - k_iTreg_mDC_d * act_CD4 * IL_10 / (k_iTreg_IL_10_m + IL_10)
#   - k_IL_10_d * IL_10
#
# 线性化：
#   + k_IL_10_iTreg_f * iTreg
#   + k_IL_10_nTreg_eff * nTreg
#     其中 k_IL_10_nTreg_eff = k_IL_10_nTreg_f * mDC / (k_IL_10_nTreg_mDC_m + mDC)
#   - k_iTreg_mDC_d_eff * act_CD4
#     其中 k_iTreg_mDC_d_eff = k_iTreg_mDC_d * IL_10 / (k_iTreg_IL_10_m + IL_10)
#   - k_IL_10_d * IL_10

function rhs_IL_10_linearized(IL_10, iTreg, nTreg, mDC, act_CD4, params)
    return (
        + params[:k_IL_10_iTreg_f] * iTreg
        + params[:k_IL_10_nTreg_eff] * nTreg
        - params[:k_iTreg_mDC_d_eff] * act_CD4
        - params[:k_IL_10_d] * IL_10
    )
end

# ============================================================================
# 15. TGFbeta 的 RHS 线性化
# ============================================================================
# 原始项：
#   + k_TGFbeta_iTreg_f * iTreg
#   + k_TGFbeta_CD4_CTL_f * CD4_CTL
#   + k_TGFbeta_nTreg_f * nTreg * mDC / (k_TGFbeta_nTreg_mDC_m + mDC)
#   - k_iTreg_mDC_d * act_CD4 * TGFbeta / (k_iTreg_TGFbeta_m + TGFbeta)
#   - k_TGFbeta_d * TGFbeta
#
# 线性化：
#   + k_TGFbeta_iTreg_f * iTreg
#   + k_TGFbeta_CD4_CTL_f * CD4_CTL
#   + k_TGFbeta_nTreg_eff * nTreg
#     其中 k_TGFbeta_nTreg_eff = k_TGFbeta_nTreg_f * mDC / (k_TGFbeta_nTreg_mDC_m + mDC)
#   - k_iTreg_mDC_d_TGFbeta_eff * act_CD4
#     其中 k_iTreg_mDC_d_TGFbeta_eff = k_iTreg_mDC_d * TGFbeta / (k_iTreg_TGFbeta_m + TGFbeta)
#   - k_TGFbeta_d * TGFbeta

function rhs_TGFbeta_linearized(TGFbeta, iTreg, CD4_CTL, nTreg, mDC, act_CD4, params)
    return (
        + params[:k_TGFbeta_iTreg_f] * iTreg
        + params[:k_TGFbeta_CD4_CTL_f] * CD4_CTL
        + params[:k_TGFbeta_nTreg_eff] * nTreg
        - params[:k_iTreg_mDC_d_TGFbeta_eff] * act_CD4
        - params[:k_TGFbeta_d] * TGFbeta
    )
end

# ============================================================================
# 16. IFN_g 的 RHS 线性化
# ============================================================================
# 原始项：
#   + k_IFN_g_CD4_CTL_f * CD4_CTL
#   + k_IFN_g_act_NK_f * act_NK
#   - k_act_NK_IFN_g_d * NK * IFN_g / (IFN_g + k_act_NK_IFN_g_m)
#   - k_IFN_g_d * IFN_g
#
# 线性化：
#   + k_IFN_g_CD4_CTL_f * CD4_CTL
#   + k_IFN_g_act_NK_f * act_NK
#   - k_act_NK_IFN_g_d_eff * NK
#     其中 k_act_NK_IFN_g_d_eff = k_act_NK_IFN_g_d * IFN_g / (IFN_g + k_act_NK_IFN_g_m)
#   - k_IFN_g_d * IFN_g

function rhs_IFN_g_linearized(IFN_g, CD4_CTL, act_NK, NK, params)
    return (
        + params[:k_IFN_g_CD4_CTL_f] * CD4_CTL
        + params[:k_IFN_g_act_NK_f] * act_NK
        - params[:k_act_NK_IFN_g_d_eff] * NK
        - params[:k_IFN_g_d] * IFN_g
    )
end

# ============================================================================
# 17. naive_CD4 的 RHS 线性化
# ============================================================================
# 原始项：
#   + k_CD4_f * naive_CD4 * (1 - naive_CD4 / k_CD4_m)
#   + k_naive_CD4_IL_15_f * naive_CD4 * (1 - naive_CD4 / k_CD4_m) * IL_15 / (k_naive_CD4_IL_15_m + IL_15)
#   + k_naive_CD4_IL_7_f * naive_CD4 * (1 - naive_CD4 / k_CD4_m) * IL_7 / (k_naive_CD4_IL_7_m + IL_7)
#   - k_act_CD4_mDC_f * naive_CD4 * mDC / (k_act_CD4_mDC_m + mDC)
#   - k_act_CD4_IL_2_f * naive_CD4 * IL_2 / (k_act_CD4_IL_2_m + IL_2)
#   - k_naive_CD4_d * naive_CD4
#
# 线性化：
#   + k_CD4_f * naive_CD4 - k_CD4_f_div_m * naive_CD4^2
#   + k_naive_CD4_IL_15_eff * naive_CD4 - k_naive_CD4_IL_15_eff_div_m * naive_CD4^2
#     其中 k_naive_CD4_IL_15_eff = k_naive_CD4_IL_15_f * IL_15 / (k_naive_CD4_IL_15_m + IL_15)
#         k_naive_CD4_IL_15_eff_div_m = k_naive_CD4_IL_15_eff / k_CD4_m = k_naive_CD4_IL_15_f * IL_15 / ((k_naive_CD4_IL_15_m + IL_15) * k_CD4_m)
#   + k_naive_CD4_IL_7_eff * naive_CD4 - k_naive_CD4_IL_7_eff_div_m * naive_CD4^2
#     其中 k_naive_CD4_IL_7_eff = k_naive_CD4_IL_7_f * IL_7 / (k_naive_CD4_IL_7_m + IL_7)
#         k_naive_CD4_IL_7_eff_div_m = k_naive_CD4_IL_7_eff / k_CD4_m
#   - k_act_CD4_mDC_f_eff * naive_CD4
#     其中 k_act_CD4_mDC_f_eff = k_act_CD4_mDC_f * mDC / (k_act_CD4_mDC_m + mDC)
#   - k_act_CD4_IL_2_f_eff * naive_CD4
#     其中 k_act_CD4_IL_2_f_eff = k_act_CD4_IL_2_f * IL_2 / (k_act_CD4_IL_2_m + IL_2)
#   - k_naive_CD4_d * naive_CD4

function rhs_naive_CD4_linearized(naive_CD4, IL_15, IL_7, mDC, IL_2, params)
    return (
        + params[:k_CD4_f] * naive_CD4
        - params[:k_CD4_f_div_m] * naive_CD4^2
        + params[:k_naive_CD4_IL_15_eff] * naive_CD4
        - params[:k_naive_CD4_IL_15_eff_div_m] * naive_CD4^2
        + params[:k_naive_CD4_IL_7_eff] * naive_CD4
        - params[:k_naive_CD4_IL_7_eff_div_m] * naive_CD4^2
        - params[:k_act_CD4_mDC_f_eff] * naive_CD4
        - params[:k_act_CD4_IL_2_f_eff] * naive_CD4
        - params[:k_naive_CD4_d] * naive_CD4
    )
end

# ============================================================================
# 18. act_CD4 的 RHS 线性化（最复杂）
# ============================================================================
# 原始项很多，需要仔细处理
# 由于项数太多，我将分部分处理

function rhs_act_CD4_linearized(act_CD4, naive_CD4, mDC, IL_2, IL_15, IL_7, IL_4, IL_33, TGFbeta, IL_10, IL_12, IL_1, IFN1, IL_6, Ag_t, params)
    """
    act_CD4的线性化RHS
    
    原始项：
    + k_act_CD4_mDC_f * naive_CD4 * mDC / (k_act_CD4_mDC_m + mDC)
    + k_act_CD4_IL_2_f * naive_CD4 * IL_2 / (k_act_CD4_IL_2_m + IL_2)
    + k_act_CD4_f * act_CD4 * (1 - act_CD4 / k_act_CD4_m)
    + k_act_CD4_IL_15_f * act_CD4 * (1 - act_CD4 / k_act_CD4_m) * IL_15 / (k_act_CD4_IL_15_m + IL_15)
    + k_act_CD4_IL_7_f * act_CD4 * (1 - act_CD4 / k_act_CD4_m) * IL_7 / (k_act_CD4_IL_7_m + IL_7)
    - act_CD4 * k_Th2_f * (k_Th2_TGFbeta_m / (k_Th2_TGFbeta_m + TGFbeta)) * (k_Th2_IL_10_m / (k_Th2_IL_10_m + IL_10)) * (k_Th2_IL_12_m / (k_Th2_IL_12_m + IL_12))
    - act_CD4 * k_Th2_IL_4_f * IL_4 / (k_Th2_IL_4_m + IL_4) * (k_Th2_TGFbeta_m / (k_Th2_TGFbeta_m + TGFbeta)) * (k_Th2_IL_10_m / (k_Th2_IL_10_m + IL_10)) * (k_Th2_IL_12_m / (k_Th2_IL_12_m + IL_12))
    - act_CD4 * k_Th2_IL_33_f * IL_33 / (k_Th2_IL_33_m + IL_33) * (k_Th2_TGFbeta_m / (k_Th2_TGFbeta_m + TGFbeta)) * (k_Th2_IL_10_m / (k_Th2_IL_10_m + IL_10)) * (k_Th2_IL_12_m / (k_Th2_IL_12_m + IL_12))
    - act_CD4 * k_iTreg_mDC_f * k_iTreg_TGFbeta_f * TGFbeta / (k_iTreg_TGFbeta_m + TGFbeta) * (k_iTreg_IL_1_m / (k_iTreg_IL_1_m + IL_1))
    - act_CD4 * k_iTreg_mDC_f * k_iTreg_IL_10_f * IL_10 / (k_iTreg_IL_10_m + IL_10) * (k_iTreg_IL_1_m / (k_iTreg_IL_1_m + IL_1))
    - k_act_CD4_CTL_basal_f * act_CD4
    - k_act_CD4_CTL_antigen_f * act_CD4 * Ag(t)
    - k_act_CD4_IFN1_f * act_CD4 * IFN1 / (k_IFN1_CD4_CTL_m + IFN1)
    - k_TFH_mDC_f * act_CD4
    - k_TFH_mDC_Antigen_f * act_CD4 * Ag(t)
    - k_TFH_IFN1_f * act_CD4 * IFN1 / (k_TFH_IFN1_m + IFN1)
    - k_TFH_IL_6_f * act_CD4 * IL_6 / (k_TFH_IL_6_m + IL_6)
    - k_act_CD4_d * act_CD4
    """
    # 从naive_CD4转换的项
    from_naive = (
        + params[:k_act_CD4_mDC_f_eff] * naive_CD4
        + params[:k_act_CD4_IL_2_f_eff] * naive_CD4
    )
    
    # 基础增殖项
    act_CD4_prod = (
        + params[:k_act_CD4_f] * act_CD4
        - params[:k_act_CD4_f_div_m] * act_CD4^2
    )
    
    # IL_15相关的增殖项
    act_CD4_IL15_prod = (
        + params[:k_act_CD4_IL_15_eff] * act_CD4
        - params[:k_act_CD4_IL_15_eff_div_m] * act_CD4^2
    )
    
    # IL_7相关的增殖项
    act_CD4_IL7_prod = (
        + params[:k_act_CD4_IL_7_eff] * act_CD4
        - params[:k_act_CD4_IL_7_eff_div_m] * act_CD4^2
    )
    
    # 转换为Th2的项（包含多个希尔方程的乘积）
    to_Th2 = (
        - params[:k_Th2_f_eff] * act_CD4
        - params[:k_Th2_IL_4_eff] * act_CD4
        - params[:k_Th2_IL_33_eff] * act_CD4
    )
    
    # 转换为iTreg的项
    to_iTreg = (
        - params[:k_iTreg_TGFbeta_eff] * act_CD4
        - params[:k_iTreg_IL_10_eff] * act_CD4
    )
    
    # 转换为CD4_CTL的项
    to_CD4_CTL = (
        - params[:k_act_CD4_CTL_basal_f] * act_CD4
        - params[:k_act_CD4_CTL_antigen_f] * act_CD4 * Ag_t
        - params[:k_act_CD4_IFN1_f_eff] * act_CD4
    )
    
    # 转换为TFH的项
    to_TFH = (
        - params[:k_TFH_mDC_f] * act_CD4
        - params[:k_TFH_mDC_Antigen_f] * act_CD4 * Ag_t
        - params[:k_TFH_IFN1_f_eff] * act_CD4
        - params[:k_TFH_IL_6_f_eff] * act_CD4
    )
    
    # 降解项
    degradation = - params[:k_act_CD4_d] * act_CD4
    
    return (from_naive + act_CD4_prod + act_CD4_IL15_prod + act_CD4_IL7_prod + 
            to_Th2 + to_iTreg + to_CD4_CTL + to_TFH + degradation)
end

# ============================================================================
# 19. Th2 的 RHS 线性化
# ============================================================================
# 原始项：
#   + act_CD4 * k_Th2_f * (k_Th2_TGFbeta_m / (k_Th2_TGFbeta_m + TGFbeta)) * (k_Th2_IL_10_m / (k_Th2_IL_10_m + IL_10)) * (k_Th2_IL_12_m / (k_Th2_IL_12_m + IL_12))
#   + act_CD4 * k_Th2_IL_4_f * IL_4 / (k_Th2_IL_4_m + IL_4) * (k_Th2_TGFbeta_m / (k_Th2_TGFbeta_m + TGFbeta)) * (k_Th2_IL_10_m / (k_Th2_IL_10_m + IL_10)) * (k_Th2_IL_12_m / (k_Th2_IL_12_m + IL_12))
#   + act_CD4 * k_Th2_IL_33_f * IL_33 / (k_Th2_IL_33_m + IL_33) * (k_Th2_TGFbeta_m / (k_Th2_TGFbeta_m + TGFbeta)) * (k_Th2_IL_10_m / (k_Th2_IL_10_m + IL_10)) * (k_Th2_IL_12_m / (k_Th2_IL_12_m + IL_12))
#   + k_Th2_f * Th2 * (1 - Th2 / k_Th2_m)
#   - k_Th2_d * Th2
#
# 线性化：
#   + k_Th2_f_eff * act_CD4
#     其中 k_Th2_f_eff = k_Th2_f * (k_Th2_TGFbeta_m / (k_Th2_TGFbeta_m + TGFbeta)) * (k_Th2_IL_10_m / (k_Th2_IL_10_m + IL_10)) * (k_Th2_IL_12_m / (k_Th2_IL_12_m + IL_12))
#   + k_Th2_IL_4_eff * act_CD4
#     其中 k_Th2_IL_4_eff = k_Th2_IL_4_f * IL_4 / (k_Th2_IL_4_m + IL_4) * (k_Th2_TGFbeta_m / (k_Th2_TGFbeta_m + TGFbeta)) * (k_Th2_IL_10_m / (k_Th2_IL_10_m + IL_10)) * (k_Th2_IL_12_m / (k_Th2_IL_12_m + IL_12))
#   + k_Th2_IL_33_eff * act_CD4
#     其中 k_Th2_IL_33_eff = k_Th2_IL_33_f * IL_33 / (k_Th2_IL_33_m + IL_33) * (k_Th2_TGFbeta_m / (k_Th2_TGFbeta_m + TGFbeta)) * (k_Th2_IL_10_m / (k_Th2_IL_10_m + IL_10)) * (k_Th2_IL_12_m / (k_Th2_IL_12_m + IL_12))
#   + k_Th2_f * Th2 - k_Th2_f_div_m * Th2^2
#   - k_Th2_d * Th2

function rhs_Th2_linearized(Th2, act_CD4, TGFbeta, IL_10, IL_12, IL_4, IL_33, params)
    return (
        + params[:k_Th2_f_eff] * act_CD4
        + params[:k_Th2_IL_4_eff] * act_CD4
        + params[:k_Th2_IL_33_eff] * act_CD4
        + params[:k_Th2_f] * Th2
        - params[:k_Th2_f_div_m] * Th2^2
        - params[:k_Th2_d] * Th2
    )
end

# ============================================================================
# 20. iTreg 的 RHS 线性化
# ============================================================================
# 原始项：
#   + act_CD4 * k_iTreg_mDC_f * k_iTreg_TGFbeta_f * TGFbeta / (k_iTreg_TGFbeta_m + TGFbeta) * (k_iTreg_IL_1_m / (k_iTreg_IL_1_m + IL_1))
#   + act_CD4 * k_iTreg_mDC_f * k_iTreg_IL_10_f * IL_10 / (k_iTreg_IL_10_m + IL_10) * (k_iTreg_IL_1_m / (k_iTreg_IL_1_m + IL_1))
#   + k_iTreg_f * iTreg * (1 - iTreg / k_iTreg_m)
#   - k_iTreg_d * iTreg
#
# 线性化：
#   + k_iTreg_TGFbeta_eff * act_CD4
#     其中 k_iTreg_TGFbeta_eff = k_iTreg_mDC_f * k_iTreg_TGFbeta_f * TGFbeta / (k_iTreg_TGFbeta_m + TGFbeta) * (k_iTreg_IL_1_m / (k_iTreg_IL_1_m + IL_1))
#   + k_iTreg_IL_10_eff * act_CD4
#     其中 k_iTreg_IL_10_eff = k_iTreg_mDC_f * k_iTreg_IL_10_f * IL_10 / (k_iTreg_IL_10_m + IL_10) * (k_iTreg_IL_1_m / (k_iTreg_IL_1_m + IL_1))
#   + k_iTreg_f * iTreg - k_iTreg_f_div_m * iTreg^2
#   - k_iTreg_d * iTreg

function rhs_iTreg_linearized(iTreg, act_CD4, TGFbeta, IL_10, IL_1, params)
    return (
        + params[:k_iTreg_TGFbeta_eff] * act_CD4
        + params[:k_iTreg_IL_10_eff] * act_CD4
        + params[:k_iTreg_f] * iTreg
        - params[:k_iTreg_f_div_m] * iTreg^2
        - params[:k_iTreg_d] * iTreg
    )
end

# ============================================================================
# 21. CD4_CTL 的 RHS 线性化
# ============================================================================
# 原始项：
#   + k_act_CD4_CTL_basal_f * act_CD4
#   + k_act_CD4_CTL_antigen_f * act_CD4 * Ag(t)
#   + k_act_CD4_IFN1_f * act_CD4 * IFN1 / (k_IFN1_CD4_CTL_m + IFN1)
#   + k_CD4_CTL_f * CD4_CTL * (1 - CD4_CTL / k_CD4_CTL_m)
#   - k_CD4_CTL_d * CD4_CTL
#
# 线性化：
#   + k_act_CD4_CTL_basal_f * act_CD4
#   + k_act_CD4_CTL_antigen_f * act_CD4 * Ag(t)
#   + k_act_CD4_IFN1_f_eff * act_CD4
#     其中 k_act_CD4_IFN1_f_eff = k_act_CD4_IFN1_f * IFN1 / (k_IFN1_CD4_CTL_m + IFN1)
#   + k_CD4_CTL_f * CD4_CTL - k_CD4_CTL_f_div_m * CD4_CTL^2
#   - k_CD4_CTL_d * CD4_CTL

function rhs_CD4_CTL_linearized(CD4_CTL, act_CD4, IFN1, Ag_t, params)
    return (
        + params[:k_act_CD4_CTL_basal_f] * act_CD4
        + params[:k_act_CD4_CTL_antigen_f] * act_CD4 * Ag_t
        + params[:k_act_CD4_IFN1_f_eff] * act_CD4
        + params[:k_CD4_CTL_f] * CD4_CTL
        - params[:k_CD4_CTL_f_div_m] * CD4_CTL^2
        - params[:k_CD4_CTL_d] * CD4_CTL
    )
end

# ============================================================================
# 22. nTreg 的 RHS 线性化
# ============================================================================
# 原始项：
#   + k_nTreg_mDC_f * nTreg * (1 - nTreg / k_nTreg_m) * mDC / (k_nTreg_mDC_m + mDC)
#   - k_nTreg_d * nTreg
#
# 线性化：
#   + k_nTreg_mDC_f_eff * nTreg - k_nTreg_mDC_f_eff_div_m * nTreg^2
#     其中 k_nTreg_mDC_f_eff = k_nTreg_mDC_f * mDC / (k_nTreg_mDC_m + mDC)
#         k_nTreg_mDC_f_eff_div_m = k_nTreg_mDC_f_eff / k_nTreg_m = k_nTreg_mDC_f * mDC / ((k_nTreg_mDC_m + mDC) * k_nTreg_m)
#   - k_nTreg_d * nTreg

function rhs_nTreg_linearized(nTreg, mDC, params)
    return (
        + params[:k_nTreg_mDC_f_eff] * nTreg
        - params[:k_nTreg_mDC_f_eff_div_m] * nTreg^2
        - params[:k_nTreg_d] * nTreg
    )
end

# ============================================================================
# 23. TFH 的 RHS 线性化
# ============================================================================
# 原始项：
#   + k_TFH_mDC_f * act_CD4
#   + k_TFH_mDC_Antigen_f * act_CD4 * Ag(t)
#   + k_TFH_IFN1_f * act_CD4 * IFN1 / (k_TFH_IFN1_m + IFN1)
#   + k_TFH_IL_6_f * act_CD4 * IL_6 / (k_TFH_IL_6_m + IL_6)
#   + k_TFH_f * TFH * (1 - TFH / k_TFH_m)
#   - k_TFH_d * TFH
#
# 线性化：
#   + k_TFH_mDC_f * act_CD4
#   + k_TFH_mDC_Antigen_f * act_CD4 * Ag(t)
#   + k_TFH_IFN1_f_eff * act_CD4
#     其中 k_TFH_IFN1_f_eff = k_TFH_IFN1_f * IFN1 / (k_TFH_IFN1_m + IFN1)
#   + k_TFH_IL_6_f_eff * act_CD4
#     其中 k_TFH_IL_6_f_eff = k_TFH_IL_6_f * IL_6 / (k_TFH_IL_6_m + IL_6)
#   + k_TFH_f * TFH - k_TFH_f_div_m * TFH^2
#   - k_TFH_d * TFH

function rhs_TFH_linearized(TFH, act_CD4, IFN1, IL_6, Ag_t, params)
    return (
        + params[:k_TFH_mDC_f] * act_CD4
        + params[:k_TFH_mDC_Antigen_f] * act_CD4 * Ag_t
        + params[:k_TFH_IFN1_f_eff] * act_CD4
        + params[:k_TFH_IL_6_f_eff] * act_CD4
        + params[:k_TFH_f] * TFH
        - params[:k_TFH_f_div_m] * TFH^2
        - params[:k_TFH_d] * TFH
    )
end

# ============================================================================
# 24. NK 的 RHS 线性化
# ============================================================================
# 原始项：
#   + k_NK_f * NK * (1 - NK / k_NK_m)
#   - k_act_NK_base_f * NK
#   - k_act_NK_IL_12_f * NK * IL_12 / (IL_12 + k_act_NK_IL_12_m)
#   - k_act_NK_IL_2_f * NK * IL_2 / (IL_2 + k_act_NK_IL_2_m)
#   - k_act_NK_IFN1_f * NK * IFN1 / (IFN1 + k_act_NK_IFN1_m)
#   - k_act_NK_IFN_g_f * NK * IFN_g / (IFN_g + k_act_NK_IFN_g_m)
#   - k_NK_d * NK
#
# 线性化：
#   + k_NK_f * NK - k_NK_f_div_m * NK^2
#   - k_act_NK_base_f * NK
#   - k_act_NK_IL_12_f_eff * NK
#     其中 k_act_NK_IL_12_f_eff = k_act_NK_IL_12_f * IL_12 / (IL_12 + k_act_NK_IL_12_m)
#   - k_act_NK_IL_2_f_eff * NK
#     其中 k_act_NK_IL_2_f_eff = k_act_NK_IL_2_f * IL_2 / (IL_2 + k_act_NK_IL_2_m)
#   - k_act_NK_IFN1_f_eff * NK
#     其中 k_act_NK_IFN1_f_eff = k_act_NK_IFN1_f * IFN1 / (IFN1 + k_act_NK_IFN1_m)
#   - k_act_NK_IFN_g_f_eff * NK
#     其中 k_act_NK_IFN_g_f_eff = k_act_NK_IFN_g_f * IFN_g / (IFN_g + k_act_NK_IFN_g_m)
#   - k_NK_d * NK

function rhs_NK_linearized(NK, IL_12, IL_2, IFN1, IFN_g, params)
    return (
        + params[:k_NK_f] * NK
        - params[:k_NK_f_div_m] * NK^2
        - params[:k_act_NK_base_f] * NK
        - params[:k_act_NK_IL_12_f_eff] * NK
        - params[:k_act_NK_IL_2_f_eff] * NK
        - params[:k_act_NK_IFN1_f_eff] * NK
        - params[:k_act_NK_IFN_g_f_eff] * NK
        - params[:k_NK_d] * NK
    )
end

# ============================================================================
# 25. act_NK 的 RHS 线性化
# ============================================================================
# 原始项：
#   + k_act_NK_base_f * NK
#   + k_act_NK_IL_12_f * NK * IL_12 / (IL_12 + k_act_NK_IL_12_m)
#   + k_act_NK_IL_2_f * NK * IL_2 / (IL_2 + k_act_NK_IL_2_m)
#   + k_act_NK_IFN1_f * NK * IFN1 / (IFN1 + k_act_NK_IFN1_m)
#   + k_act_NK_IFN_g_f * NK * IFN_g / (IFN_g + k_act_NK_IFN_g_m)
#   + k_act_NK_f * act_NK * (1 - act_NK / k_act_NK_m)
#   - k_act_NK_d * act_NK
#
# 线性化：
#   + k_act_NK_base_f * NK
#   + k_act_NK_IL_12_f_eff * NK  (复用NK中的定义)
#   + k_act_NK_IL_2_f_eff * NK  (复用NK中的定义)
#   + k_act_NK_IFN1_f_eff * NK  (复用NK中的定义)
#   + k_act_NK_IFN_g_f_eff * NK  (复用NK中的定义)
#   + k_act_NK_f * act_NK - k_act_NK_f_div_m * act_NK^2
#   - k_act_NK_d * act_NK

function rhs_act_NK_linearized(act_NK, NK, IL_12, IL_2, IFN1, IFN_g, params)
    return (
        + params[:k_act_NK_base_f] * NK
        + params[:k_act_NK_IL_12_f_eff] * NK
        + params[:k_act_NK_IL_2_f_eff] * NK
        + params[:k_act_NK_IFN1_f_eff] * NK
        + params[:k_act_NK_IFN_g_f_eff] * NK
        + params[:k_act_NK_f] * act_NK
        - params[:k_act_NK_f_div_m] * act_NK^2
        - params[:k_act_NK_d] * act_NK
    )
end

# ============================================================================
# 26. Naive_B 的 RHS 线性化
# ============================================================================
# 原始项：
#   + k_Naive_B_f * Naive_B * (1 - Naive_B / k_Naive_B_m)
#   + k_Naive_B_Antigen_f * Naive_B * Ag(t) * (1 - Naive_B / k_Naive_B_m)
#   - k_Act_B_basal_f * Naive_B
#   - k_Act_B_Antigen_f * Naive_B * Ag(t)
#   - k_Naive_B_d * Naive_B
#
# 线性化：
#   + k_Naive_B_f * Naive_B - k_Naive_B_f_div_m * Naive_B^2
#   + k_Naive_B_Antigen_f * Ag(t) * Naive_B - k_Naive_B_Antigen_f_div_m * Ag(t) * Naive_B^2
#     其中 k_Naive_B_Antigen_f_div_m = k_Naive_B_Antigen_f / k_Naive_B_m
#   - k_Act_B_basal_f * Naive_B
#   - k_Act_B_Antigen_f * Ag(t) * Naive_B
#   - k_Naive_B_d * Naive_B

function rhs_Naive_B_linearized(Naive_B, Ag_t, params)
    return (
        + params[:k_Naive_B_f] * Naive_B
        - params[:k_Naive_B_f_div_m] * Naive_B^2
        + params[:k_Naive_B_Antigen_f] * Ag_t * Naive_B
        - params[:k_Naive_B_Antigen_f_div_m] * Ag_t * Naive_B^2
        - params[:k_Act_B_basal_f] * Naive_B
        - params[:k_Act_B_Antigen_f] * Ag_t * Naive_B
        - params[:k_Naive_B_d] * Naive_B
    )
end

# ============================================================================
# 27. Act_B 的 RHS 线性化
# ============================================================================
# 原始项：
#   + k_Act_B_basal_f * Naive_B
#   + k_Act_B_Antigen_f * Naive_B * Ag(t)
#   + k_Act_B_f * Act_B * (1 - Act_B / k_Act_B_m)
#   + k_Act_B_Antigen_pro_f * Act_B * Ag(t) * (1 - Act_B / k_Act_B_m)
#   - k_Act_B_d * Act_B
#
# 线性化：
#   + k_Act_B_basal_f * Naive_B
#   + k_Act_B_Antigen_f * Naive_B * Ag(t)
#   + k_Act_B_f * Act_B - k_Act_B_f_div_m * Act_B^2
#   + k_Act_B_Antigen_pro_f * Ag(t) * Act_B - k_Act_B_Antigen_pro_f_div_m * Ag(t) * Act_B^2
#     其中 k_Act_B_Antigen_pro_f_div_m = k_Act_B_Antigen_pro_f / k_Act_B_m
#   - k_Act_B_d * Act_B

function rhs_Act_B_linearized(Act_B, Naive_B, Ag_t, params)
    return (
        + params[:k_Act_B_basal_f] * Naive_B
        + params[:k_Act_B_Antigen_f] * Naive_B * Ag_t
        + params[:k_Act_B_f] * Act_B
        - params[:k_Act_B_f_div_m] * Act_B^2
        + params[:k_Act_B_Antigen_pro_f] * Ag_t * Act_B
        - params[:k_Act_B_Antigen_pro_f_div_m] * Ag_t * Act_B^2
        - params[:k_Act_B_d] * Act_B
    )
end

# ============================================================================
# 28. TD_IS_B 的 RHS 线性化
# ============================================================================
# 原始项：
#   + k_TD_base_f * Act_B
#   + k_TD_IL_4_f * Act_B * IL_4
#   + k_TD_f * TD_IS_B * (1 - TD_IS_B / k_TD_m)
#   - k_TD_d * TD_IS_B
#
# 线性化：
#   + k_TD_base_f * Act_B
#   + k_TD_IL_4_f * Act_B * IL_4
#   + k_TD_f * TD_IS_B - k_TD_f_div_m * TD_IS_B^2
#   - k_TD_d * TD_IS_B

function rhs_TD_IS_B_linearized(TD_IS_B, Act_B, IL_4, params)
    return (
        + params[:k_TD_base_f] * Act_B
        + params[:k_TD_IL_4_f] * Act_B * IL_4
        + params[:k_TD_f] * TD_IS_B
        - params[:k_TD_f_div_m] * TD_IS_B^2
        - params[:k_TD_d] * TD_IS_B
    )
end

# ============================================================================
# 29. TI_IS_B 的 RHS 线性化
# ============================================================================
# 原始项：
#   + k_TI_base_f * Act_B
#   + k_TI_IFN_g_f * Act_B * IFN_g
#   + k_TI_IL_10_f * Act_B * IL_10
#   + k_TI_f * TI_IS_B * (1 - TI_IS_B / k_TI_m)
#   - k_TI_d * TI_IS_B
#
# 线性化：
#   + k_TI_base_f * Act_B
#   + k_TI_IFN_g_f * Act_B * IFN_g
#   + k_TI_IL_10_f * Act_B * IL_10
#   + k_TI_f * TI_IS_B - k_TI_f_div_m * TI_IS_B^2
#   - k_TI_d * TI_IS_B

function rhs_TI_IS_B_linearized(TI_IS_B, Act_B, IFN_g, IL_10, params)
    return (
        + params[:k_TI_base_f] * Act_B
        + params[:k_TI_IFN_g_f] * Act_B * IFN_g
        + params[:k_TI_IL_10_f] * Act_B * IL_10
        + params[:k_TI_f] * TI_IS_B
        - params[:k_TI_f_div_m] * TI_IS_B^2
        - params[:k_TI_d] * TI_IS_B
    )
end

# ============================================================================
# 30. IgG4 的 RHS 线性化
# ============================================================================
# 原始项：
#   + k_IgG4_TI_f * TI_IS_B
#   + k_IgG4_TD_f * TD_IS_B
#   - k_IgG4_d * IgG4
#
# 线性化（已经是线性的）：
#   + k_IgG4_TI_f * TI_IS_B
#   + k_IgG4_TD_f * TD_IS_B
#   - k_IgG4_d * IgG4

function rhs_IgG4_linearized(IgG4, TI_IS_B, TD_IS_B, params)
    return (
        + params[:k_IgG4_TI_f] * TI_IS_B
        + params[:k_IgG4_TD_f] * TD_IS_B
        - params[:k_IgG4_d] * IgG4
    )
end

# ============================================================================
# 总结：所有需要定义的组合参数列表
# ============================================================================

# 增殖项相关的组合参数（16个）
COMBINED_PARAMS_DIV = [
    :k_nDC_f_div_m,
    :k_mDC_f_div_m,
    :k_pDC_f_div_m,
    :k_CD4_f_div_m,
    :k_act_CD4_f_div_m,
    :k_Th2_f_div_m,
    :k_iTreg_f_div_m,
    :k_CD4_CTL_f_div_m,
    :k_nTreg_f_div_m,
    :k_TFH_f_div_m,
    :k_NK_f_div_m,
    :k_act_NK_f_div_m,
    :k_Naive_B_f_div_m,
    :k_Act_B_f_div_m,
    :k_TD_f_div_m,
    :k_TI_f_div_m,
    :k_Naive_B_Antigen_f_div_m,      # = k_Naive_B_Antigen_f / k_Naive_B_m
    :k_Act_B_Antigen_pro_f_div_m,   # = k_Act_B_Antigen_pro_f / k_Act_B_m
    :k_naive_CD4_IL_15_eff_div_m,   # = k_naive_CD4_IL_15_eff / k_CD4_m
    :k_naive_CD4_IL_7_eff_div_m,    # = k_naive_CD4_IL_7_eff / k_CD4_m
    :k_act_CD4_IL_15_eff_div_m,     # = k_act_CD4_IL_15_eff / k_act_CD4_m
    :k_act_CD4_IL_7_eff_div_m,      # = k_act_CD4_IL_7_eff / k_act_CD4_m
    :k_nTreg_mDC_f_eff_div_m,       # = k_nTreg_mDC_f_eff / k_nTreg_m
]

# 希尔方程项相关的组合参数（这些依赖于变量值，需要在每个时间点重新计算）
# 注意：这些参数依赖于k_m值，需要先估计k_m或通过迭代求解
COMBINED_PARAMS_HILL = [
    :k_mDC_Antigen_IL10_eff,         # = k_mDC_Antigen_f * k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10)
    :k_mDC_GMCSF_IL10_eff,           # = k_mDC_GMCSF_f * (GMCSF / (GMCSF + k_mDC_GMCSF_m)) * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10))
    :k_mDC_GMCSF_d_eff,              # = k_mDC_GMCSF_d * (GMCSF / (GMCSF + k_mDC_GMCSF_m)) * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10))
    :k_act_CD4_IL_33_d_eff,          # = k_act_CD4_IL_33_d * IL_33 / (k_Th2_IL_33_m + IL_33)
    :k_IL_6_TFH_eff,                 # = k_IL_6_TFH_f * (k_TFH_nTreg_m / (nTreg + k_TFH_nTreg_m))
    :k_TFH_IL_6_d_eff,               # = k_TFH_IL_6_d * IL_6 / (k_TFH_IL_6_m + IL_6)
    :k_act_NK_IL_12_d_eff,           # = k_act_NK_IL_12_d * IL_12 / (IL_12 + k_act_NK_IL_12_m)
    :k_naive_CD4_IL_15_d_eff,        # = k_naive_CD4_IL_15_d * IL_15 / (k_naive_CD4_IL_15_m + IL_15)
    :k_act_CD4_IL_15_d_eff,          # = k_act_CD4_IL_15_d * IL_15 / (k_act_CD4_IL_15_m + IL_15)
    :k_naive_CD4_IL_7_d_eff,         # = k_naive_CD4_IL_7_d * IL_7 / (k_naive_CD4_IL_7_m + IL_7)
    :k_act_CD4_IL_7_d_eff,           # = k_act_CD4_IL_7_d * IL_7 / (k_act_CD4_IL_7_m + IL_7)
    :k_act_CD4_IFN1_d_eff,           # = k_act_CD4_IFN1_d * IFN1 / (k_IFN1_CD4_CTL_m + IFN1)
    :k_act_NK_IFN1_d_eff,            # = k_act_NK_IFN1_d * IFN1 / (IFN1 + k_act_NK_IFN1_m)
    :k_act_CD4_IL_2_d_eff,           # = k_act_CD4_IL_2_d * IL_2 / (k_act_CD4_IL_2_m + IL_2)
    :k_act_NK_IL_2_d_eff,            # = k_act_NK_IL_2_d * IL_2 / (IL_2 + k_act_NK_IL_2_m)
    :k_act_CD4_IL_4_d_eff,           # = k_act_CD4_IL_4_d * IL_4 / (k_Th2_IL_4_m + IL_4)
    :k_IL_10_nTreg_eff,              # = k_IL_10_nTreg_f * mDC / (k_IL_10_nTreg_mDC_m + mDC)
    :k_iTreg_mDC_d_eff,              # = k_iTreg_mDC_d * IL_10 / (k_iTreg_IL_10_m + IL_10)
    :k_TGFbeta_nTreg_eff,            # = k_TGFbeta_nTreg_f * mDC / (k_TGFbeta_nTreg_mDC_m + mDC)
    :k_iTreg_mDC_d_TGFbeta_eff,      # = k_iTreg_mDC_d * TGFbeta / (k_iTreg_TGFbeta_m + TGFbeta)
    :k_act_NK_IFN_g_d_eff,           # = k_act_NK_IFN_g_d * IFN_g / (IFN_g + k_act_NK_IFN_g_m)
    :k_naive_CD4_IL_15_eff,          # = k_naive_CD4_IL_15_f * IL_15 / (k_naive_CD4_IL_15_m + IL_15)
    :k_naive_CD4_IL_7_eff,           # = k_naive_CD4_IL_7_f * IL_7 / (k_naive_CD4_IL_7_m + IL_7)
    :k_act_CD4_mDC_f_eff,            # = k_act_CD4_mDC_f * mDC / (k_act_CD4_mDC_m + mDC)
    :k_act_CD4_IL_2_f_eff,           # = k_act_CD4_IL_2_f * IL_2 / (k_act_CD4_IL_2_m + IL_2)
    :k_act_CD4_IL_15_eff,            # = k_act_CD4_IL_15_f * IL_15 / (k_act_CD4_IL_15_m + IL_15)
    :k_act_CD4_IL_7_eff,             # = k_act_CD4_IL_7_f * IL_7 / (k_act_CD4_IL_7_m + IL_7)
    :k_Th2_f_eff,                    # = k_Th2_f * (k_Th2_TGFbeta_m / (k_Th2_TGFbeta_m + TGFbeta)) * (k_Th2_IL_10_m / (k_Th2_IL_10_m + IL_10)) * (k_Th2_IL_12_m / (k_Th2_IL_12_m + IL_12))
    :k_Th2_IL_4_eff,                 # = k_Th2_IL_4_f * IL_4 / (k_Th2_IL_4_m + IL_4) * (k_Th2_TGFbeta_m / (k_Th2_TGFbeta_m + TGFbeta)) * (k_Th2_IL_10_m / (k_Th2_IL_10_m + IL_10)) * (k_Th2_IL_12_m / (k_Th2_IL_12_m + IL_12))
    :k_Th2_IL_33_eff,                # = k_Th2_IL_33_f * IL_33 / (k_Th2_IL_33_m + IL_33) * (k_Th2_TGFbeta_m / (k_Th2_TGFbeta_m + TGFbeta)) * (k_Th2_IL_10_m / (k_Th2_IL_10_m + IL_10)) * (k_Th2_IL_12_m / (k_Th2_IL_12_m + IL_12))
    :k_iTreg_TGFbeta_eff,            # = k_iTreg_mDC_f * k_iTreg_TGFbeta_f * TGFbeta / (k_iTreg_TGFbeta_m + TGFbeta) * (k_iTreg_IL_1_m / (k_iTreg_IL_1_m + IL_1))
    :k_iTreg_IL_10_eff,              # = k_iTreg_mDC_f * k_iTreg_IL_10_f * IL_10 / (k_iTreg_IL_10_m + IL_10) * (k_iTreg_IL_1_m / (k_iTreg_IL_1_m + IL_1))
    :k_act_CD4_IFN1_f_eff,           # = k_act_CD4_IFN1_f * IFN1 / (k_IFN1_CD4_CTL_m + IFN1)
    :k_TFH_IFN1_f_eff,               # = k_TFH_IFN1_f * IFN1 / (k_TFH_IFN1_m + IFN1)
    :k_TFH_IL_6_f_eff,               # = k_TFH_IL_6_f * IL_6 / (k_TFH_IL_6_m + IL_6)
    :k_act_NK_IL_12_f_eff,           # = k_act_NK_IL_12_f * IL_12 / (IL_12 + k_act_NK_IL_12_m)
    :k_act_NK_IL_2_f_eff,            # = k_act_NK_IL_2_f * IL_2 / (IL_2 + k_act_NK_IL_2_m)
    :k_act_NK_IFN1_f_eff,            # = k_act_NK_IFN1_f * IFN1 / (IFN1 + k_act_NK_IFN1_m)
    :k_act_NK_IFN_g_f_eff,           # = k_act_NK_IFN_g_f * IFN_g / (IFN_g + k_act_NK_IFN_g_m)
    :k_nTreg_mDC_f_eff,              # = k_nTreg_mDC_f * mDC / (k_nTreg_mDC_m + mDC)
]

# ============================================================================
# 反解原始参数的方法
# ============================================================================
# 对于增殖项：k_f_div_m = k_f / k_m
#   - 如果已知 k_f 和 k_f_div_m，则 k_m = k_f / k_f_div_m
#
# 对于希尔方程项的组合参数，需要通过优化或迭代方法反解：
#   例如：k_f_eff = k_f * Y / (Y + k_m)
#   - 如果已知 k_f_eff 和 Y，需要估计 k_m 或通过优化求解
#   - 或者固定 k_m，然后 k_f = k_f_eff * (Y + k_m) / Y

# ============================================================================
# 检查清单
# ============================================================================
# 已完成的变量（1-30）：
#   1. nDC ✓
#   2. mDC ✓
#   3. pDC ✓
#   4. GMCSF ✓
#   5. IL_33 ✓
#   6. IL_6 ✓
#   7. IL_12 ✓
#   8. IL_15 ✓
#   9. IL_7 ✓
#   10. IFN1 ✓
#   11. IL_1 ✓
#   12. IL_2 ✓
#   13. IL_4 ✓
#   14. IL_10 ✓
#   15. TGFbeta ✓
#   16. IFN_g ✓
#   17. naive_CD4 ✓
#   18. act_CD4 ✓（部分，需要补充完整）
#   19. Th2 ✓
#   20. iTreg ✓
#   21. CD4_CTL ✓
#   22. nTreg ✓
#   23. TFH ✓
#   24. NK ✓
#   25. act_NK ✓
#   26. Naive_B ✓
#   27. Act_B ✓
#   28. TD_IS_B ✓
#   29. TI_IS_B ✓
#   30. IgG4 ✓

# 注意：act_CD4 的线性化函数需要补充完整，因为它包含很多项

# ============================================================================
# 生成线性方程组：将HC状态和IgG状态的变量值代入31个线性化RHS方程
# ============================================================================

# 定义每个方程使用的参数列表（从函数实现中提取）
const EQUATION_PARAMS = Dict(
    1 => [:k_nDC_f, :k_nDC_f_div_m, :k_nDC_d, :k_mDC_Antigen_IL10_eff, :k_mDC_GMCSF_IL10_eff, :k_pDC_Antigen_f],
    2 => [:k_mDC_Antigen_IL10_eff, :k_mDC_GMCSF_IL10_eff, :k_mDC_f, :k_mDC_f_div_m, :k_mDC_d],
    3 => [:k_pDC_Antigen_f, :k_pDC_f, :k_pDC_f_div_m, :k_pDC_d],
    4 => [:k_GMCSF_Th2_f, :k_GMCSF_Th2_Antigen_f, :k_GMCSF_act_NK_f, :k_mDC_GMCSF_d_eff, :k_GMCSF_d],
    5 => [:k_IL_33_pDC_f, :k_act_CD4_IL_33_d_eff, :k_IL_33_d],
    6 => [:k_IL_6_pDC_f, :k_IL_6_mDC_f, :k_IL_6_TFH_eff, :k_TFH_IL_6_d_eff, :k_IL_6_d],
    7 => [:k_IL_12_mDC_f, :k_act_NK_IL_12_d_eff, :k_IL_12_d],
    8 => [:k_IL_15_f, :k_IL_15_Antigen_f, :k_naive_CD4_IL_15_d_eff, :k_act_CD4_IL_15_d_eff, :k_IL_15_d],
    9 => [:k_IL_7_f, :k_naive_CD4_IL_7_d_eff, :k_act_CD4_IL_7_d_eff, :k_IL_7_d],
    10 => [:k_IFN1_pDC_f, :k_act_CD4_IFN1_d_eff, :k_act_NK_IFN1_d_eff, :k_IFN1_d],
    11 => [:k_IL_1_mDC_f, :k_IL_1_d],
    12 => [:k_IL_2_act_CD4_f, :k_IL_2_act_CD4_Antigen_f, :k_act_CD4_IL_2_d_eff, :k_act_NK_IL_2_d_eff, :k_IL_2_d],
    13 => [:k_IL_4_Th2_f, :k_IL_4_Th2_Antigen_f, :k_act_CD4_IL_4_d_eff, :k_IL_4_d],
    14 => [:k_IL_10_iTreg_f, :k_IL_10_nTreg_eff, :k_iTreg_mDC_d_eff, :k_IL_10_d],
    15 => [:k_TGFbeta_iTreg_f, :k_TGFbeta_CD4_CTL_f, :k_TGFbeta_nTreg_eff, :k_iTreg_mDC_d_TGFbeta_eff, :k_TGFbeta_d],
    16 => [:k_IFN_g_CD4_CTL_f, :k_IFN_g_act_NK_f, :k_act_NK_IFN_g_d_eff, :k_IFN_g_d],
    17 => [:k_CD4_f, :k_CD4_f_div_m, :k_naive_CD4_IL_15_eff, :k_naive_CD4_IL_15_eff_div_m, :k_naive_CD4_IL_7_eff, :k_naive_CD4_IL_7_eff_div_m, :k_act_CD4_mDC_f_eff, :k_act_CD4_IL_2_f_eff, :k_naive_CD4_d],
    18 => [:k_act_CD4_mDC_f_eff, :k_act_CD4_IL_2_f_eff, :k_act_CD4_f, :k_act_CD4_f_div_m, :k_act_CD4_IL_15_eff, :k_act_CD4_IL_15_eff_div_m, :k_act_CD4_IL_7_eff, :k_act_CD4_IL_7_eff_div_m, :k_Th2_f_eff, :k_Th2_IL_4_eff, :k_Th2_IL_33_eff, :k_iTreg_TGFbeta_eff, :k_iTreg_IL_10_eff, :k_act_CD4_CTL_basal_f, :k_act_CD4_CTL_antigen_f, :k_act_CD4_IFN1_f_eff, :k_TFH_mDC_f, :k_TFH_mDC_Antigen_f, :k_TFH_IFN1_f_eff, :k_TFH_IL_6_f_eff, :k_act_CD4_d],
    19 => [:k_Th2_f_eff, :k_Th2_IL_4_eff, :k_Th2_IL_33_eff, :k_Th2_f, :k_Th2_f_div_m, :k_Th2_d],
    20 => [:k_iTreg_TGFbeta_eff, :k_iTreg_IL_10_eff, :k_iTreg_f, :k_iTreg_f_div_m, :k_iTreg_d],
    21 => [:k_act_CD4_CTL_basal_f, :k_act_CD4_CTL_antigen_f, :k_act_CD4_IFN1_f_eff, :k_CD4_CTL_f, :k_CD4_CTL_f_div_m, :k_CD4_CTL_d],
    22 => [:k_nTreg_mDC_f_eff, :k_nTreg_mDC_f_eff_div_m, :k_nTreg_d],
    23 => [:k_TFH_mDC_f, :k_TFH_mDC_Antigen_f, :k_TFH_IFN1_f_eff, :k_TFH_IL_6_f_eff, :k_TFH_f, :k_TFH_f_div_m, :k_TFH_d],
    24 => [:k_NK_f, :k_NK_f_div_m, :k_act_NK_base_f, :k_act_NK_IL_12_f_eff, :k_act_NK_IL_2_f_eff, :k_act_NK_IFN1_f_eff, :k_act_NK_IFN_g_f_eff, :k_NK_d],
    25 => [:k_act_NK_base_f, :k_act_NK_IL_12_f_eff, :k_act_NK_IL_2_f_eff, :k_act_NK_IFN1_f_eff, :k_act_NK_IFN_g_f_eff, :k_act_NK_f, :k_act_NK_f_div_m, :k_act_NK_d],
    26 => [:k_Naive_B_f, :k_Naive_B_f_div_m, :k_Naive_B_Antigen_f, :k_Naive_B_Antigen_f_div_m, :k_Act_B_basal_f, :k_Act_B_Antigen_f, :k_Naive_B_d],
    27 => [:k_Act_B_basal_f, :k_Act_B_Antigen_f, :k_Act_B_f, :k_Act_B_f_div_m, :k_Act_B_Antigen_pro_f, :k_Act_B_Antigen_pro_f_div_m, :k_Act_B_d],
    28 => [:k_TD_base_f, :k_TD_IL_4_f, :k_TD_f, :k_TD_f_div_m, :k_TD_d],
    29 => [:k_TI_base_f, :k_TI_IFN_g_f, :k_TI_IL_10_f, :k_TI_f, :k_TI_f_div_m, :k_TI_d],
    30 => [:k_IgG4_TI_f, :k_IgG4_TD_f, :k_IgG4_d],
)

# HC状态（健康稳态）的变量值
const HC_STATE = Dict(
    :Antigen => 0.0,
    :nDC => 30.0,
    :mDC => 0.0,
    :pDC => 0.0,
    :naive_CD4 => 180.0,
    :act_CD4 => 300.0,
    :Th2 => 0.3,
    :iTreg => 50.0,
    :CD4_CTL => 1.0,
    :nTreg => 10.0,
    :TFH => 20.0,
    :NK => 50.0,
    :act_NK => 200.0,
    :Naive_B => 86.0,
    :Act_B => 80.0,
    :TD_IS_B => 1.5,
    :TI_IS_B => 1.0,
    :GMCSF => 3e3,
    :IL_33 => 1e0,
    :IL_6 => 1.0e4,
    :IL_12 => 1e0,
    :IL_15 => 1.0e4,
    :IL_7 => 3.5e5,
    :IFN1 => 1e0,
    :IL_1 => 3.0e3,
    :IL_2 => 5.0e3,
    :IL_4 => 2.5e2,
    :IL_10 => 4.0e3,
    :TGFbeta => 5.0e1,
    :IFN_g => 2.0e5,
    :IgG4 => 0.0,
)

# IgG状态的变量值
const IgG_STATE = Dict(
    :Antigen => 1.0,
    :nDC => 0.0,
    :mDC => 20.0,
    :pDC => 4.0,
    :naive_CD4 => 80.0,
    :act_CD4 => 500.0,
    :Th2 => 0.3,
    :iTreg => 40.0,
    :CD4_CTL => 3.0,
    :nTreg => 10.0,
    :TFH => 20.0,
    :NK => 25.0,
    :act_NK => 310.0,
    :Naive_B => 94.0,
    :Act_B => 85.0,
    :TD_IS_B => 10.0,
    :TI_IS_B => 0.1,
    :GMCSF => 5.2e3,
    :IL_33 => 4.0e4,
    :IL_6 => 6.0e4,
    :IL_12 => 2.0e4,
    :IL_15 => 1.4e5,
    :IL_7 => 1.0e5,
    :IFN1 => 1.0,
    :IL_1 => 7.0e3,
    :IL_2 => 1.3e4,
    :IL_4 => 1.8e3,
    :IL_10 => 9.0e3,
    :TGFbeta => 5.0e1,
    :IFN_g => 1.0e6,
    :IgG4 => 140.0,
)


# ============================================================================
# HC状态（健康稳态）的线性方程组
# ============================================================================
# 将HC状态的变量值代入31个线性化RHS方程，得到 RHS = 0 的线性方程组
# 所有项都显示，即使系数为0也写成 0.0 * 参数 的形式
const HC_LINEAR_SYSTEM = [
    "Eq1: nDC: + 30.0 * k_nDC_f - 900.0 * k_nDC_f_div_m - 30.0 * k_nDC_d + 0.0 * k_mDC_Antigen_IL10_eff + 0.0 * k_mDC_GMCSF_IL10_eff + 0.0 * k_pDC_Antigen_f = 0",
    "Eq2: mDC: + 0.0 * k_mDC_Antigen_IL10_eff + 0.0 * k_mDC_GMCSF_IL10_eff + 0.0 * k_mDC_f + 0.0 * k_mDC_f_div_m + 0.0 * k_mDC_d = 0",
    "Eq3: pDC: + 0.0 * k_pDC_Antigen_f + 0.0 * k_pDC_f + 0.0 * k_pDC_f_div_m + 0.0 * k_pDC_d = 0",
    "Eq4: GMCSF: + 0.3 * k_GMCSF_Th2_f + 0.0 * k_GMCSF_Th2_Antigen_f + 200.0 * k_GMCSF_act_NK_f + 0.0 * k_mDC_GMCSF_d_eff - 3000.0 * k_GMCSF_d = 0",
    "Eq5: IL_33: + 0.0 * k_IL_33_pDC_f - 300.0 * k_act_CD4_IL_33_d_eff - 1.0 * k_IL_33_d = 0",
    "Eq6: IL_6: + 0.0 * k_IL_6_pDC_f + 0.0 * k_IL_6_mDC_f + 20.0 * k_IL_6_TFH_eff - 300.0 * k_TFH_IL_6_d_eff - 10000.0 * k_IL_6_d = 0",
    "Eq7: IL_12: + 0.0 * k_IL_12_mDC_f - 50.0 * k_act_NK_IL_12_d_eff - 1.0 * k_IL_12_d = 0",
    "Eq8: IL_15: + 1.0 * k_IL_15_f + 0.0 * k_IL_15_Antigen_f - 180.0 * k_naive_CD4_IL_15_d_eff - 300.0 * k_act_CD4_IL_15_d_eff - 10000.0 * k_IL_15_d = 0",
    "Eq9: IL_7: + 1.0 * k_IL_7_f - 180.0 * k_naive_CD4_IL_7_d_eff - 300.0 * k_act_CD4_IL_7_d_eff - 350000.0 * k_IL_7_d = 0",
    "Eq10: IFN1: + 0.0 * k_IFN1_pDC_f - 300.0 * k_act_CD4_IFN1_d_eff - 50.0 * k_act_NK_IFN1_d_eff - 1.0 * k_IFN1_d = 0",
    "Eq11: IL_1: + 0.0 * k_IL_1_mDC_f - 3000.0 * k_IL_1_d = 0",
    "Eq12: IL_2: + 300.0 * k_IL_2_act_CD4_f + 0.0 * k_IL_2_act_CD4_Antigen_f - 180.0 * k_act_CD4_IL_2_d_eff - 50.0 * k_act_NK_IL_2_d_eff - 5000.0 * k_IL_2_d = 0",
    "Eq13: IL_4: + 0.3 * k_IL_4_Th2_f + 0.0 * k_IL_4_Th2_Antigen_f - 300.0 * k_act_CD4_IL_4_d_eff - 250.0 * k_IL_4_d = 0",
    "Eq14: IL_10: + 50.0 * k_IL_10_iTreg_f + 10.0 * k_IL_10_nTreg_eff - 300.0 * k_iTreg_mDC_d_eff - 4000.0 * k_IL_10_d = 0",
    "Eq15: TGFbeta: + 50.0 * k_TGFbeta_iTreg_f + 1.0 * k_TGFbeta_CD4_CTL_f + 10.0 * k_TGFbeta_nTreg_eff - 300.0 * k_iTreg_mDC_d_TGFbeta_eff - 50.0 * k_TGFbeta_d = 0",
    "Eq16: IFN_g: + 1.0 * k_IFN_g_CD4_CTL_f + 200.0 * k_IFN_g_act_NK_f - 50.0 * k_act_NK_IFN_g_d_eff - 200000.0 * k_IFN_g_d = 0",
    "Eq17: naive_CD4: + 180.0 * k_CD4_f - 32400.0 * k_CD4_f_div_m + 180.0 * k_naive_CD4_IL_15_eff - 32400.0 * k_naive_CD4_IL_15_eff_div_m + 180.0 * k_naive_CD4_IL_7_eff - 32400.0 * k_naive_CD4_IL_7_eff_div_m - 180.0 * k_act_CD4_mDC_f_eff - 180.0 * k_act_CD4_IL_2_f_eff - 180.0 * k_naive_CD4_d = 0",
    "Eq18: act_CD4: + 180.0 * k_act_CD4_mDC_f_eff + 180.0 * k_act_CD4_IL_2_f_eff + 300.0 * k_act_CD4_f - 90000.0 * k_act_CD4_f_div_m + 300.0 * k_act_CD4_IL_15_eff - 90000.0 * k_act_CD4_IL_15_eff_div_m + 300.0 * k_act_CD4_IL_7_eff - 90000.0 * k_act_CD4_IL_7_eff_div_m - 300.0 * k_Th2_f_eff - 300.0 * k_Th2_IL_4_eff - 300.0 * k_Th2_IL_33_eff - 300.0 * k_iTreg_TGFbeta_eff - 300.0 * k_iTreg_IL_10_eff - 300.0 * k_act_CD4_CTL_basal_f + 0.0 * k_act_CD4_CTL_antigen_f - 300.0 * k_act_CD4_IFN1_f_eff - 300.0 * k_TFH_mDC_f + 0.0 * k_TFH_mDC_Antigen_f - 300.0 * k_TFH_IFN1_f_eff - 300.0 * k_TFH_IL_6_f_eff - 300.0 * k_act_CD4_d = 0",
    "Eq19: Th2: + 300.0 * k_Th2_f_eff + 300.0 * k_Th2_IL_4_eff + 300.0 * k_Th2_IL_33_eff + 0.3 * k_Th2_f - 0.09 * k_Th2_f_div_m - 0.3 * k_Th2_d = 0",
    "Eq20: iTreg: + 300.0 * k_iTreg_TGFbeta_eff + 300.0 * k_iTreg_IL_10_eff + 50.0 * k_iTreg_f - 2500.0 * k_iTreg_f_div_m - 50.0 * k_iTreg_d = 0",
    "Eq21: CD4_CTL: + 300.0 * k_act_CD4_CTL_basal_f + 0.0 * k_act_CD4_CTL_antigen_f + 300.0 * k_act_CD4_IFN1_f_eff + 1.0 * k_CD4_CTL_f - 1.0 * k_CD4_CTL_f_div_m - 1.0 * k_CD4_CTL_d = 0",
    "Eq22: nTreg: + 10.0 * k_nTreg_mDC_f_eff - 100.0 * k_nTreg_mDC_f_eff_div_m - 10.0 * k_nTreg_d = 0",
    "Eq23: TFH: + 300.0 * k_TFH_mDC_f + 0.0 * k_TFH_mDC_Antigen_f + 300.0 * k_TFH_IFN1_f_eff + 300.0 * k_TFH_IL_6_f_eff + 20.0 * k_TFH_f - 400.0 * k_TFH_f_div_m - 20.0 * k_TFH_d = 0",
    "Eq24: NK: + 50.0 * k_NK_f - 2500.0 * k_NK_f_div_m - 50.0 * k_act_NK_base_f - 50.0 * k_act_NK_IL_12_f_eff - 50.0 * k_act_NK_IL_2_f_eff - 50.0 * k_act_NK_IFN1_f_eff - 50.0 * k_act_NK_IFN_g_f_eff - 50.0 * k_NK_d = 0",
    "Eq25: act_NK: + 50.0 * k_act_NK_base_f + 50.0 * k_act_NK_IL_12_f_eff + 50.0 * k_act_NK_IL_2_f_eff + 50.0 * k_act_NK_IFN1_f_eff + 50.0 * k_act_NK_IFN_g_f_eff + 200.0 * k_act_NK_f - 40000.0 * k_act_NK_f_div_m - 200.0 * k_act_NK_d = 0",
    "Eq26: Naive_B: + 86.0 * k_Naive_B_f - 7396.0 * k_Naive_B_f_div_m + 0.0 * k_Naive_B_Antigen_f + 0.0 * k_Naive_B_Antigen_f_div_m - 86.0 * k_Act_B_basal_f + 0.0 * k_Act_B_Antigen_f - 86.0 * k_Naive_B_d = 0",
    "Eq27: Act_B: + 86.0 * k_Act_B_basal_f + 0.0 * k_Act_B_Antigen_f + 80.0 * k_Act_B_f - 6400.0 * k_Act_B_f_div_m + 0.0 * k_Act_B_Antigen_pro_f + 0.0 * k_Act_B_Antigen_pro_f_div_m - 80.0 * k_Act_B_d = 0",
    "Eq28: TD_IS_B: + 80.0 * k_TD_base_f + 20000.0 * k_TD_IL_4_f + 1.5 * k_TD_f - 2.25 * k_TD_f_div_m - 1.5 * k_TD_d = 0",
    "Eq29: TI_IS_B: + 80.0 * k_TI_base_f + 1.6e7 * k_TI_IFN_g_f + 320000.0 * k_TI_IL_10_f + 1.0 * k_TI_f - 1.0 * k_TI_f_div_m - 1.0 * k_TI_d = 0",
    "Eq30: IgG4: + 1.0 * k_IgG4_TI_f + 1.5 * k_IgG4_TD_f + 0.0 * k_IgG4_d = 0",
]

# ============================================================================
# IgG状态的线性方程组
# ============================================================================
# 将IgG状态的变量值代入31个线性化RHS方程，得到 RHS = 0 的线性方程组
# 所有项都显示，即使系数为0也写成 0.0 * 参数 的形式
const IgG_LINEAR_SYSTEM = [
    "Eq1: nDC: + 0.0 * k_nDC_f + 0.0 * k_nDC_f_div_m + 0.0 * k_nDC_d + 0.0 * k_mDC_Antigen_IL10_eff + 0.0 * k_mDC_GMCSF_IL10_eff + 0.0 * k_pDC_Antigen_f = 0",
    "Eq2: mDC: + 0.0 * k_mDC_Antigen_IL10_eff + 0.0 * k_mDC_GMCSF_IL10_eff + 20.0 * k_mDC_f - 400.0 * k_mDC_f_div_m - 20.0 * k_mDC_d = 0",
    "Eq3: pDC: + 0.0 * k_pDC_Antigen_f + 4.0 * k_pDC_f - 16.0 * k_pDC_f_div_m - 4.0 * k_pDC_d = 0",
    "Eq4: GMCSF: + 0.3 * k_GMCSF_Th2_f + 0.3 * k_GMCSF_Th2_Antigen_f + 310.0 * k_GMCSF_act_NK_f + 0.0 * k_mDC_GMCSF_d_eff - 5200.0 * k_GMCSF_d = 0",
    "Eq5: IL_33: + 4.0 * k_IL_33_pDC_f - 500.0 * k_act_CD4_IL_33_d_eff - 40000.0 * k_IL_33_d = 0",
    "Eq6: IL_6: + 4.0 * k_IL_6_pDC_f + 20.0 * k_IL_6_mDC_f + 20.0 * k_IL_6_TFH_eff - 500.0 * k_TFH_IL_6_d_eff - 60000.0 * k_IL_6_d = 0",
    "Eq7: IL_12: + 20.0 * k_IL_12_mDC_f - 25.0 * k_act_NK_IL_12_d_eff - 20000.0 * k_IL_12_d = 0",
    "Eq8: IL_15: + 1.0 * k_IL_15_f + 1.0 * k_IL_15_Antigen_f - 80.0 * k_naive_CD4_IL_15_d_eff - 500.0 * k_act_CD4_IL_15_d_eff - 140000.0 * k_IL_15_d = 0",
    "Eq9: IL_7: + 1.0 * k_IL_7_f - 80.0 * k_naive_CD4_IL_7_d_eff - 500.0 * k_act_CD4_IL_7_d_eff - 100000.0 * k_IL_7_d = 0",
    "Eq10: IFN1: + 4.0 * k_IFN1_pDC_f - 500.0 * k_act_CD4_IFN1_d_eff - 25.0 * k_act_NK_IFN1_d_eff - 1.0 * k_IFN1_d = 0",
    "Eq11: IL_1: + 20.0 * k_IL_1_mDC_f - 7000.0 * k_IL_1_d = 0",
    "Eq12: IL_2: + 500.0 * k_IL_2_act_CD4_f + 500.0 * k_IL_2_act_CD4_Antigen_f - 80.0 * k_act_CD4_IL_2_d_eff - 25.0 * k_act_NK_IL_2_d_eff - 13000.0 * k_IL_2_d = 0",
    "Eq13: IL_4: + 0.3 * k_IL_4_Th2_f + 0.3 * k_IL_4_Th2_Antigen_f - 500.0 * k_act_CD4_IL_4_d_eff - 1800.0 * k_IL_4_d = 0",
    "Eq14: IL_10: + 40.0 * k_IL_10_iTreg_f + 10.0 * k_IL_10_nTreg_eff - 500.0 * k_iTreg_mDC_d_eff - 9000.0 * k_IL_10_d = 0",
    "Eq15: TGFbeta: + 40.0 * k_TGFbeta_iTreg_f + 3.0 * k_TGFbeta_CD4_CTL_f + 10.0 * k_TGFbeta_nTreg_eff - 500.0 * k_iTreg_mDC_d_TGFbeta_eff - 50.0 * k_TGFbeta_d = 0",
    "Eq16: IFN_g: + 3.0 * k_IFN_g_CD4_CTL_f + 310.0 * k_IFN_g_act_NK_f - 25.0 * k_act_NK_IFN_g_d_eff - 1.0e6 * k_IFN_g_d = 0",
    "Eq17: naive_CD4: + 80.0 * k_CD4_f - 6400.0 * k_CD4_f_div_m + 80.0 * k_naive_CD4_IL_15_eff - 6400.0 * k_naive_CD4_IL_15_eff_div_m + 80.0 * k_naive_CD4_IL_7_eff - 6400.0 * k_naive_CD4_IL_7_eff_div_m - 80.0 * k_act_CD4_mDC_f_eff - 80.0 * k_act_CD4_IL_2_f_eff - 80.0 * k_naive_CD4_d = 0",
    "Eq18: act_CD4: + 80.0 * k_act_CD4_mDC_f_eff + 80.0 * k_act_CD4_IL_2_f_eff + 500.0 * k_act_CD4_f - 250000.0 * k_act_CD4_f_div_m + 500.0 * k_act_CD4_IL_15_eff - 250000.0 * k_act_CD4_IL_15_eff_div_m + 500.0 * k_act_CD4_IL_7_eff - 250000.0 * k_act_CD4_IL_7_eff_div_m - 500.0 * k_Th2_f_eff - 500.0 * k_Th2_IL_4_eff - 500.0 * k_Th2_IL_33_eff - 500.0 * k_iTreg_TGFbeta_eff - 500.0 * k_iTreg_IL_10_eff - 500.0 * k_act_CD4_CTL_basal_f - 500.0 * k_act_CD4_CTL_antigen_f - 500.0 * k_act_CD4_IFN1_f_eff - 500.0 * k_TFH_mDC_f - 500.0 * k_TFH_mDC_Antigen_f - 500.0 * k_TFH_IFN1_f_eff - 500.0 * k_TFH_IL_6_f_eff - 500.0 * k_act_CD4_d = 0",
    "Eq19: Th2: + 500.0 * k_Th2_f_eff + 500.0 * k_Th2_IL_4_eff + 500.0 * k_Th2_IL_33_eff + 0.3 * k_Th2_f - 0.09 * k_Th2_f_div_m - 0.3 * k_Th2_d = 0",
    "Eq20: iTreg: + 500.0 * k_iTreg_TGFbeta_eff + 500.0 * k_iTreg_IL_10_eff + 40.0 * k_iTreg_f - 1600.0 * k_iTreg_f_div_m - 40.0 * k_iTreg_d = 0",
    "Eq21: CD4_CTL: + 500.0 * k_act_CD4_CTL_basal_f + 500.0 * k_act_CD4_CTL_antigen_f + 500.0 * k_act_CD4_IFN1_f_eff + 3.0 * k_CD4_CTL_f - 9.0 * k_CD4_CTL_f_div_m - 3.0 * k_CD4_CTL_d = 0",
    "Eq22: nTreg: + 10.0 * k_nTreg_mDC_f_eff - 100.0 * k_nTreg_mDC_f_eff_div_m - 10.0 * k_nTreg_d = 0",
    "Eq23: TFH: + 500.0 * k_TFH_mDC_f + 500.0 * k_TFH_mDC_Antigen_f + 500.0 * k_TFH_IFN1_f_eff + 500.0 * k_TFH_IL_6_f_eff + 20.0 * k_TFH_f - 400.0 * k_TFH_f_div_m - 20.0 * k_TFH_d = 0",
    "Eq24: NK: + 25.0 * k_NK_f - 625.0 * k_NK_f_div_m - 25.0 * k_act_NK_base_f - 25.0 * k_act_NK_IL_12_f_eff - 25.0 * k_act_NK_IL_2_f_eff - 25.0 * k_act_NK_IFN1_f_eff - 25.0 * k_act_NK_IFN_g_f_eff - 25.0 * k_NK_d = 0",
    "Eq25: act_NK: + 25.0 * k_act_NK_base_f + 25.0 * k_act_NK_IL_12_f_eff + 25.0 * k_act_NK_IL_2_f_eff + 25.0 * k_act_NK_IFN1_f_eff + 25.0 * k_act_NK_IFN_g_f_eff + 310.0 * k_act_NK_f - 96100.0 * k_act_NK_f_div_m - 310.0 * k_act_NK_d = 0",
    "Eq26: Naive_B: + 94.0 * k_Naive_B_f - 8836.0 * k_Naive_B_f_div_m + 94.0 * k_Naive_B_Antigen_f - 8836.0 * k_Naive_B_Antigen_f_div_m - 94.0 * k_Act_B_basal_f - 94.0 * k_Act_B_Antigen_f - 94.0 * k_Naive_B_d = 0",
    "Eq27: Act_B: + 94.0 * k_Act_B_basal_f + 94.0 * k_Act_B_Antigen_f + 85.0 * k_Act_B_f - 7225.0 * k_Act_B_f_div_m + 85.0 * k_Act_B_Antigen_pro_f - 7225.0 * k_Act_B_Antigen_pro_f_div_m - 85.0 * k_Act_B_d = 0",
    "Eq28: TD_IS_B: + 85.0 * k_TD_base_f + 153000.0 * k_TD_IL_4_f + 10.0 * k_TD_f - 100.0 * k_TD_f_div_m - 10.0 * k_TD_d = 0",
    "Eq29: TI_IS_B: + 85.0 * k_TI_base_f + 8.5e7 * k_TI_IFN_g_f + 765000.0 * k_TI_IL_10_f + 0.1 * k_TI_f - 0.01 * k_TI_f_div_m - 0.1 * k_TI_d = 0",
    "Eq30: IgG4: + 0.1 * k_IgG4_TI_f + 10.0 * k_IgG4_TD_f - 140.0 * k_IgG4_d = 0",
]

# =========================================================================
# 从线性系统字符串构造系数矩阵 A，并用 SVD 求零空间 V（解 Au = 0）
# =========================================================================

using LinearAlgebra

function build_coefficient_matrix(equations::Vector{String}, param_list::Vector{Symbol})
    m = length(equations)
    n = length(param_list)
    A = zeros(Float64, m, n)

    for (i, eq) in enumerate(equations)
        for (j, p) in enumerate(param_list)
            pat = "([+-]?\\s*\\d+(?:\\.\\d+)?(?:e[+-]?\\d+)?)\\s*\\*\\s*" * String(p)
            mth = match(Regex(pat, "i"), eq)
            if mth !== nothing
                A[i, j] = parse(Float64, replace(mth.captures[1], " " => ""))
            end
        end
    end

    return A
end

function nullspace_via_svd(A::AbstractMatrix; rtol::Real=1e-10)
    F = svd(A)
    s = F.S
    Vfull = F.V

    smax = isempty(s) ? 0.0 : maximum(s)
    tol = rtol * smax
    idx = findall(si -> si ≤ tol, s)

    if isempty(idx)
        jmin = argmin(s)
        return Vfull[:, jmin:jmin], s
    end

    return Vfull[:, idx], s
end

# 参数列顺序：使用EQUATION_PARAMS里出现过的所有参数，去重后排序
const ALL_PARAMS = sort!(collect(Set(vcat(values(EQUATION_PARAMS)...))))

# 60个方程：HC(30) + IgG(30)
const ALL_EQUATIONS = vcat(HC_LINEAR_SYSTEM, IgG_LINEAR_SYSTEM)

const A = build_coefficient_matrix(ALL_EQUATIONS, ALL_PARAMS)
const V, Svals = nullspace_via_svd(A)

"""
V: A 的零空间基（每一列是一组可行的线性化参数 u，使得 A*u ≈ 0）
Svals: A 的奇异值（用于检查零空间维数/数值稳定性）
ALL_PARAMS: V 各行对应的参数名
"""

# =========================================================================
# 生成随机线性组合系数，得到多组零空间解 u
# 并提供 u -> p 的反解函数（从线性化参数回到原参数空间）
# =========================================================================

using Random
using Catalyst
using DifferentialEquations
using CSV
using DataFrames

"""
根据零空间基 V 生成随机解：u = V * c。

参数：
- V: (n×k) 零空间基
- n_samples: 需要生成的解的个数
- rng: 随机数发生器（便于复现实验）
- dist: :normal 或 :uniform

返回：
- C: (k×n_samples) 线性组合系数
- U: (n×n_samples) 对应的解向量 u
"""
function sample_nullspace_solutions(V::AbstractMatrix; n_samples::Int=100, rng::AbstractRNG=MersenneTwister(1), dist::Symbol=:normal)
    k = size(V, 2)

    C = if dist == :normal
        randn(rng, k, n_samples)
    elseif dist == :uniform
        2 .* rand(rng, k, n_samples) .- 1
    else
        error("Unsupported dist=$(dist). Use :normal or :uniform")
    end

    U = V * C
    return C, U
end

"""
将线性化参数向量 u（按 ALL_PARAMS 顺序）反解回原参数空间 p。

说明：
- u 的分量对应的是线性化后“组合参数/原始参数”的混合体（例如 k_f_div_m = k_f/k_m）。
- 该反解只能在信息充分时完成：
  - 对于 *_div_m 类型：如果同时给出了 k_f 和 k_f_div_m，则可得 k_m = k_f / k_f_div_m。
  - 对于希尔型组合参数 *_eff：通常依赖多个原始参数与变量值，单靠 u 不能唯一反解；这里只做“能反解的部分”。

返回：
- p::Dict{Symbol,Float64}: 已反解出的原始参数（只包含可确定的那些）
- p_missing::Vector{Symbol}: 无法唯一反解、仍缺失的原始参数名
"""
function u_to_p(u::AbstractVector, param_names::Vector{Symbol})
    @assert length(u) == length(param_names)

    # 先把 u 映射成一个字典（仍然处在线性化/组合参数空间）
    u_dict = Dict{Symbol,Float64}(param_names[i] => float(u[i]) for i in eachindex(param_names))

    p = Dict{Symbol,Float64}()

    # 1) 直接存在于 u 中、且本身就是“原始参数”的，先原样放进 p
    #    这里我们用一个保守规则：
    #    - 名字不含 "_div_m" 且不含 "_eff" 的，先当作原始参数保留
    for (k, v) in u_dict
        ks = String(k)
        if !occursin("_div_m", ks) && !occursin("_eff", ks)
            p[k] = v
        end
    end

    # 2) 处理 *_div_m：若已知 k_x_f 与 k_x_f_div_m，则反推出 k_x_m
    for (k, vdiv) in u_dict
        ks = String(k)
        if endswith(ks, "_f_div_m")
            base = ks[1:end-length("_f_div_m")]
            kf_sym = Symbol(base * "_f")
            km_sym = Symbol(base * "_m")

            if haskey(u_dict, kf_sym)
                kf = u_dict[kf_sym]
                if vdiv != 0.0
                    p[kf_sym] = kf
                    p[k] = vdiv
                    p[km_sym] = kf / vdiv
                end
            end
        end

        # 兼容类似 *_eff_div_m（例如 k_act_CD4_IL_15_eff_div_m），它对应 eff / m
        if endswith(ks, "_eff_div_m")
            base = ks[1:end-length("_eff_div_m")]
            keff_sym = Symbol(base * "_eff")
            km_sym = Symbol(base * "_m")
            if haskey(u_dict, keff_sym)
                keff = u_dict[keff_sym]
                if vdiv != 0.0
                    p[keff_sym] = keff
                    p[k] = vdiv
                    p[km_sym] = keff / vdiv
                end
            end
        end
    end

    # 3) 标记目前无法唯一反解的组合参数（主要是 *_eff）
    p_missing = Symbol[]
    for (k, _) in u_dict
        ks = String(k)
        if occursin("_eff", ks)
            # 组合参数本身我们也放进 p（作为“已知的组合参数值”），但原始参数可能缺失
            if !haskey(p, k)
                p[k] = u_dict[k]
            end
        end
    end

    return p, p_missing
end

"""
返回 utils.py 中 param_optimized_candidates()[0] 对应的参数字典（作为固定 k_m 等超参数来源）。

注意：这里把数值直接硬编码成 Julia Dict，避免跨语言调用。
"""
function fixed_params_from_utils_candidate1()
    return Dict{Symbol,Float64}(
        :k_Act_B_Antigen_f => 2.0924090242e-02,
        :k_Act_B_Antigen_pro_f => 2.1482623678e-02,
        :k_Act_B_basal_f => 2.6824138668e-01,
        :k_Act_B_d => 7.7585462390e-01,
        :k_Act_B_f => 3.4889876477e-02,
        :k_Act_B_m => 1.0591926356e+02,
        :k_CD4_CTL_d => 5.0000000000e+00,
        :k_CD4_CTL_f => 3.8613443623e-01,
        :k_CD4_CTL_m => 6.3355391665e+00,
        :k_CD4_f => 4.0729063652e-02,
        :k_CD4_m => 4.7218208665e+01,
        :k_GMCSF_Th2_Antigen_f => 1.5894652437e-01,
        :k_GMCSF_Th2_f => 6.5532387033e-01,
        :k_GMCSF_act_NK_f => 2.4697826467e+00,
        :k_GMCSF_d => 2.5850477173e+00,
        :k_IFN1_CD4_CTL_m => 5.3233700019e+05,
        :k_IFN1_d => 4.9033346646e-01,
        :k_IFN1_pDC_f => 4.7443497345e-02,
        :k_IFN_g_CD4_CTL_f => 3.0411280404e-02,
        :k_IFN_g_act_NK_f => 1.3564310118e+01,
        :k_IFN_g_d => 1.6091510125e+00,
        :k_IL_10_d => 2.1724514623e+00,
        :k_IL_10_iTreg_f => 3.6202125193e-01,
        :k_IL_10_nTreg_f => 1.1964324541e+01,
        :k_IL_10_nTreg_mDC_m => 2.9881344476e+05,
        :k_IL_12_d => 6.5453472805e-02,
        :k_IL_12_mDC_f => 3.0488999368e-01,
        :k_IL_15_Antigen_f => 2.4850683260e-02,
        :k_IL_15_d => 2.2741200218e-01,
        :k_IL_15_f => 7.3315140446e-01,
        :k_IL_1_d => 3.6069563136e+00,
        :k_IL_1_mDC_f => 7.3896326698e-01,
        :k_IL_2_act_CD4_Antigen_f => 3.1164101926e-01,
        :k_IL_2_act_CD4_f => 1.5736909957e-02,
        :k_IL_2_d => 1.1883558498e+00,
        :k_IL_33_d => 1.4910603514e-01,
        :k_IL_33_pDC_f => 1.2844848683e+01,
        :k_IL_4_Th2_Antigen_f => 1.1708045048e-01,
        :k_IL_4_Th2_f => 1.6054510281e+00,
        :k_IL_4_d => 3.0519339338e+00,
        :k_IL_6_TFH_f => 1.1737947803e+00,
        :k_IL_6_d => 1.1566469991e-01,
        :k_IL_6_mDC_f => 6.2792477319e+00,
        :k_IL_6_pDC_f => 3.4174953895e+00,
        :k_IL_7_d => 2.5604073180e+00,
        :k_IL_7_f => 2.1432891117e-02,
        :k_IgG4_TD_f => 1.7687246861e+01,
        :k_IgG4_TI_f => 6.7046470826e-01,
        :k_IgG4_d => 1.1977497633e+00,
        :k_NK_d => 3.4147258450e-01,
        :k_NK_f => 1.4498876048e-02,
        :k_NK_m => 2.6392583219e+01,
        :k_Naive_B_Antigen_f => 9.9378713223e+00,
        :k_Naive_B_d => 1.8606774865e-01,
        :k_Naive_B_f => 2.3792252675e-01,
        :k_Naive_B_m => 1.5892448739e+02,
        :k_TD_IL_4_f => 1.0000000000e-02,
        :k_TD_base_f => 1.0000000000e-02,
        :k_TD_d => 5.0000000000e+00,
        :k_TD_f => 3.5753372580e-01,
        :k_TD_m => 2.3506102902e+01,
        :k_TFH_IFN1_f => 1.6227686725e-02,
        :k_TFH_IFN1_m => 3.3568897564e+05,
        :k_TFH_IL_6_d => 4.6159384653e+00,
        :k_TFH_IL_6_f => 2.1034530885e-01,
        :k_TFH_IL_6_m => 3.8763582310e+05,
        :k_TFH_d => 4.2699133816e-01,
        :k_TFH_f => 2.9238672155e-02,
        :k_TFH_m => 4.3374911086e+01,
        :k_TFH_mDC_Antigen_f => 1.8926920096e+00,
        :k_TFH_mDC_f => 3.4417821908e-01,
        :k_TFH_nTreg_m => 1.4968238388e+04,
        :k_TGFbeta_CD4_CTL_f => 1.9942459728e-01,
        :k_TGFbeta_d => 7.9100278929e-01,
        :k_TGFbeta_iTreg_f => 1.3500020513e+00,
        :k_TGFbeta_nTreg_f => 2.0585048734e-02,
        :k_TGFbeta_nTreg_mDC_m => 1.3492513554e+04,
        :k_TI_IFN_g_f => 1.0000000000e-02,
        :k_TI_IL_10_f => 1.0000000000e-02,
        :k_TI_base_f => 1.0000000000e-02,
        :k_TI_d => 5.0000000000e+00,
        :k_TI_f => 1.0000000000e-02,
        :k_TI_m => 5.0000000000e-01,
        :k_Th2_IL_10_m => 1.0149499756e+04,
        :k_Th2_IL_12_m => 1.0058275360e+04,
        :k_Th2_IL_33_f => 4.8063028271e+00,
        :k_Th2_IL_33_m => 1.0000000000e+06,
        :k_Th2_IL_4_f => 5.8454488948e-02,
        :k_Th2_IL_4_m => 2.0838911827e+05,
        :k_Th2_TGFbeta_m => 9.2088827181e+03,
        :k_Th2_d => 4.4680327353e-01,
        :k_Th2_f => 1.2551725630e-02,
        :k_Th2_m => 2.2373879849e+00,
        :k_act_CD4_CTL_antigen_f => 1.6591260632e-02,
        :k_act_CD4_CTL_basal_f => 1.0000000000e-02,
        :k_act_CD4_IFN1_d => 7.3145202277e-02,
        :k_act_CD4_IFN1_f => 6.6379740267e-02,
        :k_act_CD4_IL_15_d => 6.1584310369e-02,
        :k_act_CD4_IL_15_f => 9.6373680906e-02,
        :k_act_CD4_IL_15_m => 1.9497927629e+04,
        :k_act_CD4_IL_2_d => 1.8742297237e+00,
        :k_act_CD4_IL_2_f => 1.7356438199e+01,
        :k_act_CD4_IL_2_m => 1.6173152569e+04,
        :k_act_CD4_IL_33_d => 9.1008907000e-01,
        :k_act_CD4_IL_33_f => 1.3996014958e+00,
        :k_act_CD4_IL_4_d => 1.7352481354e+00,
        :k_act_CD4_IL_7_d => 1.2632253792e-01,
        :k_act_CD4_IL_7_f => 3.3758999699e-02,
        :k_act_CD4_IL_7_m => 5.0695237377e+05,
        :k_act_CD4_d => 4.2099820843e-01,
        :k_act_CD4_f => 2.5446825675e-02,
        :k_act_CD4_m => 5.8595963095e+02,
        :k_act_CD4_mDC_f => 3.9261565995e-02,
        :k_act_CD4_mDC_m => 1.5601863947e+04,
        :k_act_NK_IFN1_d => 3.3616664572e+00,
        :k_act_NK_IFN1_f => 1.7402257010e-02,
        :k_act_NK_IFN1_m => 2.8438641315e+03,
        :k_act_NK_IFN_g_d => 8.2119536958e-01,
        :k_act_NK_IFN_g_f => 8.4062080594e-01,
        :k_act_NK_IFN_g_m => 2.0069806084e+03,
        :k_act_NK_IL_12_d => 3.0057716704e+00,
        :k_act_NK_IL_12_f => 1.3785657464e+00,
        :k_act_NK_IL_12_m => 3.1662596178e+04,
        :k_act_NK_IL_2_d => 4.3052877245e+00,
        :k_act_NK_IL_2_f => 2.6048281470e+00,
        :k_act_NK_IL_2_m => 2.0506727288e+04,
        :k_act_NK_base_f => 1.3848625586e-01,
        :k_act_NK_d => 4.0481537246e-01,
        :k_act_NK_f => 2.0509636386e-02,
        :k_act_NK_m => 1.1225536460e+03,
        :k_iTreg_IL_10_f => 3.7572223192e-01,
        :k_iTreg_IL_10_m => 3.7930724139e+05,
        :k_iTreg_IL_1_m => 2.1689112469e+04,
        :k_iTreg_TGFbeta_f => 1.1913538260e+01,
        :k_iTreg_TGFbeta_m => 4.0363527241e+05,
        :k_iTreg_d => 2.4148039148e+00,
        :k_iTreg_f => 2.6661567999e+00,
        :k_iTreg_m => 4.9331668328e+01,
        :k_iTreg_mDC_d => 1.0572025463e+00,
        :k_iTreg_mDC_f => 2.7469165138e-02,
        :k_mDC_Antigen_f => 4.4140914271e-02,
        :k_mDC_GMCSF_d => 1.3945753929e+00,
        :k_mDC_GMCSF_f => 1.7265801546e+00,
        :k_mDC_GMCSF_m => 7.2361007289e+05,
        :k_mDC_IL_10_m => 1.1988137728e+05,
        :k_mDC_d => 2.2395216716e-01,
        :k_mDC_f => 6.3563710587e-01,
        :k_mDC_m => 5.2371132281e+01,
        :k_nDC_d => 3.0566225513e-01,
        :k_nDC_f => 1.0759263285e-01,
        :k_nDC_m => 1.4844504491e+00,
        :k_nTreg_d => 7.0447499091e-02,
        :k_nTreg_m => 7.1375423781e+00,
        :k_nTreg_mDC_f => 2.9135749007e-01,
        :k_nTreg_mDC_m => 9.1224966666e+05,
        :k_naive_CD4_IL_15_d => 1.1154764168e+00,
        :k_naive_CD4_IL_15_f => 9.2398440122e-01,
        :k_naive_CD4_IL_15_m => 3.2294380371e+03,
        :k_naive_CD4_IL_7_d => 1.6614678511e-01,
        :k_naive_CD4_IL_7_f => 1.2426446811e-01,
        :k_naive_CD4_IL_7_m => 4.3854577123e+04,
        :k_naive_CD4_d => 2.7741986150e-01,
        :k_pDC_Antigen_f => 1.8627642329e+01,
        :k_pDC_d => 1.4238701693e+00,
        :k_pDC_f => 4.8519067219e+00,
        :k_pDC_m => 5.4338686962e+00,
    )
end

"""
通用反解：
- 输入：线性化参数向量 u（按 param_names 顺序），以及一个“状态字典 state”用来提供稳态下的 Y 值。
- 输出：原始参数字典 p_full（尽最大可能从 u + 固定的 k_m 反解出 k_f），以及仍无法确定的参数列表。

反解策略：
1) 增殖项：k_f 与 k_f_div_m 已知 -> k_m = k_f / (k_f_div_m)
2) 希尔项：已知 k_eff，固定 k_m -> 反解 k_f
3) 复杂乘积项：已知 k_eff，固定所有 k_m -> 反解 k_f（或其中一个速率常数）

注意：该函数会优先使用 u 中已有的原始参数值；若缺失则尝试用固定参数补齐。
"""
function u_to_p_full(u::AbstractVector, param_names::Vector{Symbol}; state::Dict{Symbol,Float64}, fixed::Dict{Symbol,Float64}=fixed_params_from_utils_candidate1())
    @assert length(u) == length(param_names)

    u_dict = Dict{Symbol,Float64}(param_names[i] => float(u[i]) for i in eachindex(param_names))

    # 最终输出参数（尽量覆盖 utils.py 的 ALL_PARAMS）
    p = Dict{Symbol,Float64}()
    missing = Symbol[]

    # 帮助函数：从 u / fixed / state 取值
    get_u(sym) = get(u_dict, sym, NaN)
    get_fixed(sym) = get(fixed, sym, NaN)
    get_state(sym) = get(state, sym, NaN)

    # 先把 fixed 全部作为默认值灌入（保证有 k_m 可用）
    for (k, v) in fixed
        p[k] = v
    end

    # 再用 u 里“直接出现的原始参数”覆盖（不含 _div_m / _eff 的）
    for (k, v) in u_dict
        ks = String(k)
        if !occursin("_div_m", ks) && !occursin("_eff", ks)
            p[k] = v
        end
    end

    # -----------------------------
    # A) 增殖/自抑制项：k_m = k_f / (k_f_div_m)
    # -----------------------------
    for (k, vdiv) in u_dict
        ks = String(k)
        if endswith(ks, "_f_div_m") && isfinite(vdiv) && vdiv != 0.0
            base = ks[1:end-length("_f_div_m")]
            kf_sym = Symbol(base * "_f")
            km_sym = Symbol(base * "_m")

            kf = get(u_dict, kf_sym, get(p, kf_sym, NaN))
            if isfinite(kf)
                p[kf_sym] = kf
                p[k] = vdiv
                p[km_sym] = kf / vdiv
            end
        end
    end

    # -----------------------------
    # B) 希尔项：k_eff = k_f * Y/(Y+k_m)  ->  k_f = k_eff * (Y+k_m)/Y
    # 这里把每一个在 projection.jl 里出现的 *_eff 映射回 utils.py 的 k_f
    # -----------------------------

    # 1) act_CD4 促进 CD4_CTL：k_act_CD4_IFN1_f
    if haskey(u_dict, :k_act_CD4_IFN1_f_eff)
        keff = u_dict[:k_act_CD4_IFN1_f_eff]
        Y = get_state(:IFN1)
        km = get(p, :k_IFN1_CD4_CTL_m, get_fixed(:k_IFN1_CD4_CTL_m))
        if isfinite(keff) && isfinite(Y) && Y != 0.0 && isfinite(km)
            p[:k_act_CD4_IFN1_f] = keff * (Y + km) / Y
        else
            push!(missing, :k_act_CD4_IFN1_f)
        end
    end

    # 2) act_CD4 -> TFH：k_TFH_IFN1_f, k_TFH_IL_6_f
    if haskey(u_dict, :k_TFH_IFN1_f_eff)
        keff = u_dict[:k_TFH_IFN1_f_eff]
        Y = get_state(:IFN1)
        km = get(p, :k_TFH_IFN1_m, get_fixed(:k_TFH_IFN1_m))
        if isfinite(keff) && isfinite(Y) && Y != 0.0 && isfinite(km)
            p[:k_TFH_IFN1_f] = keff * (Y + km) / Y
        else
            push!(missing, :k_TFH_IFN1_f)
        end
    end

    if haskey(u_dict, :k_TFH_IL_6_f_eff)
        keff = u_dict[:k_TFH_IL_6_f_eff]
        Y = get_state(:IL_6)
        km = get(p, :k_TFH_IL_6_m, get_fixed(:k_TFH_IL_6_m))
        if isfinite(keff) && isfinite(Y) && Y != 0.0 && isfinite(km)
            p[:k_TFH_IL_6_f] = keff * (Y + km) / Y
        else
            push!(missing, :k_TFH_IL_6_f)
        end
    end

    # 3) NK 激活：k_act_NK_IL_12_f, k_act_NK_IL_2_f, k_act_NK_IFN1_f, k_act_NK_IFN_g_f
    if haskey(u_dict, :k_act_NK_IL_12_f_eff)
        keff = u_dict[:k_act_NK_IL_12_f_eff]
        Y = get_state(:IL_12)
        km = get(p, :k_act_NK_IL_12_m, get_fixed(:k_act_NK_IL_12_m))
        if isfinite(keff) && isfinite(Y) && Y != 0.0 && isfinite(km)
            p[:k_act_NK_IL_12_f] = keff * (Y + km) / Y
        else
            push!(missing, :k_act_NK_IL_12_f)
        end
    end

    if haskey(u_dict, :k_act_NK_IL_2_f_eff)
        keff = u_dict[:k_act_NK_IL_2_f_eff]
        Y = get_state(:IL_2)
        km = get(p, :k_act_NK_IL_2_m, get_fixed(:k_act_NK_IL_2_m))
        if isfinite(keff) && isfinite(Y) && Y != 0.0 && isfinite(km)
            p[:k_act_NK_IL_2_f] = keff * (Y + km) / Y
        else
            push!(missing, :k_act_NK_IL_2_f)
        end
    end

    if haskey(u_dict, :k_act_NK_IFN1_f_eff)
        keff = u_dict[:k_act_NK_IFN1_f_eff]
        Y = get_state(:IFN1)
        km = get(p, :k_act_NK_IFN1_m, get_fixed(:k_act_NK_IFN1_m))
        if isfinite(keff) && isfinite(Y) && Y != 0.0 && isfinite(km)
            p[:k_act_NK_IFN1_f] = keff * (Y + km) / Y
        else
            push!(missing, :k_act_NK_IFN1_f)
        end
    end

    if haskey(u_dict, :k_act_NK_IFN_g_f_eff)
        keff = u_dict[:k_act_NK_IFN_g_f_eff]
        Y = get_state(:IFN_g)
        km = get(p, :k_act_NK_IFN_g_m, get_fixed(:k_act_NK_IFN_g_m))
        if isfinite(keff) && isfinite(Y) && Y != 0.0 && isfinite(km)
            p[:k_act_NK_IFN_g_f] = keff * (Y + km) / Y
        else
            push!(missing, :k_act_NK_IFN_g_f)
        end
    end

    # -----------------------------
    # C) 复杂乘积项（示例：Th2 分化三项）
    # 结构：k_eff = k_f * Π_i g_i(state, k_m_i)
    # -> 固定所有 k_m_i 后，k_f = k_eff / Π_i g_i
    # -----------------------------

    # Th2 分化公共调制项（不含 IL4/IL33 那个额外的 IL/(m+IL) ）
    function th2_mod(state::Dict{Symbol,Float64}, p::Dict{Symbol,Float64})
        TGFbeta = state[:TGFbeta]
        IL_10 = state[:IL_10]
        IL_12 = state[:IL_12]
        a = p[:k_Th2_TGFbeta_m] / (p[:k_Th2_TGFbeta_m] + TGFbeta)
        b = p[:k_Th2_IL_10_m] / (p[:k_Th2_IL_10_m] + IL_10)
        c = p[:k_Th2_IL_12_m] / (p[:k_Th2_IL_12_m] + IL_12)
        return a * b * c
    end

    if haskey(u_dict, :k_Th2_f_eff)
        keff = u_dict[:k_Th2_f_eff]
        mod = th2_mod(state, p)
        if isfinite(keff) && isfinite(mod) && mod != 0.0
            p[:k_Th2_f] = keff / mod
        else
            push!(missing, :k_Th2_f)
        end
    end

    if haskey(u_dict, :k_Th2_IL_4_eff)
        keff = u_dict[:k_Th2_IL_4_eff]
        mod = th2_mod(state, p) * (state[:IL_4] / (p[:k_Th2_IL_4_m] + state[:IL_4]))
        if isfinite(keff) && isfinite(mod) && mod != 0.0
            p[:k_Th2_IL_4_f] = keff / mod
        else
            push!(missing, :k_Th2_IL_4_f)
        end
    end

    if haskey(u_dict, :k_Th2_IL_33_eff)
        keff = u_dict[:k_Th2_IL_33_eff]
        mod = th2_mod(state, p) * (state[:IL_33] / (p[:k_Th2_IL_33_m] + state[:IL_33]))
        if isfinite(keff) && isfinite(mod) && mod != 0.0
            p[:k_Th2_IL_33_f] = keff / mod
        else
            push!(missing, :k_Th2_IL_33_f)
        end
    end

    # -----------------------------
    # D) iTreg 相关复杂项：
    # k_iTreg_TGFbeta_eff = k_iTreg_mDC_f * k_iTreg_TGFbeta_f * TGFbeta/(m+TGFbeta) * (k_iTreg_IL_1_m/(k_iTreg_IL_1_m+IL_1))
    # k_iTreg_IL_10_eff   = k_iTreg_mDC_f * k_iTreg_IL_10_f   * IL_10/(m+IL_10)       * (k_iTreg_IL_1_m/(k_iTreg_IL_1_m+IL_1))
    # 由于它们是多个 k_f 的乘积，这里按你的设定：固定除了一个速率常数以外的其它项。
    # 这里选择“反解 k_iTreg_TGFbeta_f / k_iTreg_IL_10_f”，并使用 fixed 中的 k_iTreg_mDC_f。
    # -----------------------------

    itreg_mod_IL1 = p[:k_iTreg_IL_1_m] / (p[:k_iTreg_IL_1_m] + state[:IL_1])

    if haskey(u_dict, :k_iTreg_TGFbeta_eff)
        keff = u_dict[:k_iTreg_TGFbeta_eff]
        denom = p[:k_iTreg_mDC_f] * (state[:TGFbeta] / (p[:k_iTreg_TGFbeta_m] + state[:TGFbeta])) * itreg_mod_IL1
        if isfinite(keff) && isfinite(denom) && denom != 0.0
            p[:k_iTreg_TGFbeta_f] = keff / denom
        else
            push!(missing, :k_iTreg_TGFbeta_f)
        end
    end

    if haskey(u_dict, :k_iTreg_IL_10_eff)
        keff = u_dict[:k_iTreg_IL_10_eff]
        denom = p[:k_iTreg_mDC_f] * (state[:IL_10] / (p[:k_iTreg_IL_10_m] + state[:IL_10])) * itreg_mod_IL1
        if isfinite(keff) && isfinite(denom) && denom != 0.0
            p[:k_iTreg_IL_10_f] = keff / denom
        else
            push!(missing, :k_iTreg_IL_10_f)
        end
    end

    # -----------------------------
    # E) mDC / nDC 与 IL10/GMCSF/Antigen 的复杂项
    # -----------------------------

    # k_mDC_Antigen_IL10_eff = k_mDC_Antigen_f * k_mDC_IL_10_m/(k_mDC_IL_10_m + IL_10)
    if haskey(u_dict, :k_mDC_Antigen_IL10_eff)
        keff = u_dict[:k_mDC_Antigen_IL10_eff]
        mod = p[:k_mDC_IL_10_m] / (p[:k_mDC_IL_10_m] + state[:IL_10])
        if isfinite(keff) && isfinite(mod) && mod != 0.0
            p[:k_mDC_Antigen_f] = keff / mod
        else
            push!(missing, :k_mDC_Antigen_f)
        end
    end

    # k_mDC_GMCSF_IL10_eff = k_mDC_GMCSF_f * (GMCSF/(GMCSF+k_mDC_GMCSF_m)) * (k_mDC_IL_10_m/(k_mDC_IL_10_m+IL_10))
    if haskey(u_dict, :k_mDC_GMCSF_IL10_eff)
        keff = u_dict[:k_mDC_GMCSF_IL10_eff]
        mod = (state[:GMCSF] / (state[:GMCSF] + p[:k_mDC_GMCSF_m])) * (p[:k_mDC_IL_10_m] / (p[:k_mDC_IL_10_m] + state[:IL_10]))
        if isfinite(keff) && isfinite(mod) && mod != 0.0
            p[:k_mDC_GMCSF_f] = keff / mod
        else
            push!(missing, :k_mDC_GMCSF_f)
        end
    end

    # k_mDC_GMCSF_d_eff = k_mDC_GMCSF_d * (GMCSF/(GMCSF+k_mDC_GMCSF_m)) * (k_mDC_IL_10_m/(k_mDC_IL_10_m+IL_10))
    if haskey(u_dict, :k_mDC_GMCSF_d_eff)
        keff = u_dict[:k_mDC_GMCSF_d_eff]
        mod = (state[:GMCSF] / (state[:GMCSF] + p[:k_mDC_GMCSF_m])) * (p[:k_mDC_IL_10_m] / (p[:k_mDC_IL_10_m] + state[:IL_10]))
        if isfinite(keff) && isfinite(mod) && mod != 0.0
            p[:k_mDC_GMCSF_d] = keff / mod
        else
            push!(missing, :k_mDC_GMCSF_d)
        end
    end

    # -----------------------------
    # F) 其它希尔项（降解项里也有 IL/(m+IL) 的 eff）
    # -----------------------------

    # k_act_CD4_IL_33_d_eff = k_act_CD4_IL_33_d * IL_33/(k_Th2_IL_33_m + IL_33)
    if haskey(u_dict, :k_act_CD4_IL_33_d_eff)
        keff = u_dict[:k_act_CD4_IL_33_d_eff]
        Y = get_state(:IL_33)
        km = get(p, :k_Th2_IL_33_m, get_fixed(:k_Th2_IL_33_m))
        mod = Y / (km + Y)
        if isfinite(keff) && isfinite(mod) && mod != 0.0
            p[:k_act_CD4_IL_33_d] = keff / mod
        else
            push!(missing, :k_act_CD4_IL_33_d)
        end
    end

    # k_act_CD4_IL_4_d_eff = k_act_CD4_IL_4_d * IL_4/(k_Th2_IL_4_m + IL_4)
    if haskey(u_dict, :k_act_CD4_IL_4_d_eff)
        keff = u_dict[:k_act_CD4_IL_4_d_eff]
        Y = get_state(:IL_4)
        km = get(p, :k_Th2_IL_4_m, get_fixed(:k_Th2_IL_4_m))
        mod = Y / (km + Y)
        if isfinite(keff) && isfinite(mod) && mod != 0.0
            p[:k_act_CD4_IL_4_d] = keff / mod
        else
            push!(missing, :k_act_CD4_IL_4_d)
        end
    end

    # k_iTreg_mDC_d_eff = k_iTreg_mDC_d * IL_10/(k_iTreg_IL_10_m + IL_10)
    if haskey(u_dict, :k_iTreg_mDC_d_eff)
        keff = u_dict[:k_iTreg_mDC_d_eff]
        Y = get_state(:IL_10)
        km = get(p, :k_iTreg_IL_10_m, get_fixed(:k_iTreg_IL_10_m))
        mod = Y / (km + Y)
        if isfinite(keff) && isfinite(mod) && mod != 0.0
            p[:k_iTreg_mDC_d] = keff / mod
        else
            push!(missing, :k_iTreg_mDC_d)
        end
    end

    # k_iTreg_mDC_d_TGFbeta_eff = k_iTreg_mDC_d * TGFbeta/(k_iTreg_TGFbeta_m + TGFbeta)
    if haskey(u_dict, :k_iTreg_mDC_d_TGFbeta_eff)
        keff = u_dict[:k_iTreg_mDC_d_TGFbeta_eff]
        Y = get_state(:TGFbeta)
        km = get(p, :k_iTreg_TGFbeta_m, get_fixed(:k_iTreg_TGFbeta_m))
        mod = Y / (km + Y)
        if isfinite(keff) && isfinite(mod) && mod != 0.0
            p[:k_iTreg_mDC_d] = keff / mod
        else
            push!(missing, :k_iTreg_mDC_d)
        end
    end

    # k_IL_10_nTreg_eff = k_IL_10_nTreg_f * mDC/(k_IL_10_nTreg_mDC_m + mDC)
    if haskey(u_dict, :k_IL_10_nTreg_eff)
        keff = u_dict[:k_IL_10_nTreg_eff]
        Y = get_state(:mDC)
        km = get(p, :k_IL_10_nTreg_mDC_m, get_fixed(:k_IL_10_nTreg_mDC_m))
        mod = Y / (km + Y)
        if isfinite(keff) && isfinite(mod) && mod != 0.0
            p[:k_IL_10_nTreg_f] = keff / mod
        else
            push!(missing, :k_IL_10_nTreg_f)
        end
    end

    # k_TGFbeta_nTreg_eff = k_TGFbeta_nTreg_f * mDC/(k_TGFbeta_nTreg_mDC_m + mDC)
    if haskey(u_dict, :k_TGFbeta_nTreg_eff)
        keff = u_dict[:k_TGFbeta_nTreg_eff]
        Y = get_state(:mDC)
        km = get(p, :k_TGFbeta_nTreg_mDC_m, get_fixed(:k_TGFbeta_nTreg_mDC_m))
        mod = Y / (km + Y)
        if isfinite(keff) && isfinite(mod) && mod != 0.0
            p[:k_TGFbeta_nTreg_f] = keff / mod
        else
            push!(missing, :k_TGFbeta_nTreg_f)
        end
    end

    # k_IL_6_TFH_eff = k_IL_6_TFH_f * (k_TFH_nTreg_m/(nTreg + k_TFH_nTreg_m))
    if haskey(u_dict, :k_IL_6_TFH_eff)
        keff = u_dict[:k_IL_6_TFH_eff]
        km = get(p, :k_TFH_nTreg_m, get_fixed(:k_TFH_nTreg_m))
        mod = km / (get_state(:nTreg) + km)
        if isfinite(keff) && isfinite(mod) && mod != 0.0
            p[:k_IL_6_TFH_f] = keff / mod
        else
            push!(missing, :k_IL_6_TFH_f)
        end
    end

    # -----------------------------
    # G) 仍未覆盖到的 *_eff 以及必要的参数
    # -----------------------------
    for (k, _) in u_dict
        ks = String(k)
        if occursin("_eff", ks)
            # 线性化空间里的组合参数也保留一下，便于debug
            p[k] = u_dict[k]
        end
    end

    return p, missing
end

"""
批量反解：输入 U（n×N，每列一个 u），输出 P_list（长度 N，每个元素是 Dict）。
"""
function batch_u_to_p_full(U::AbstractMatrix, param_names::Vector{Symbol}; state::Dict{Symbol,Float64}, fixed::Dict{Symbol,Float64}=fixed_params_from_utils_candidate1())
    N = size(U, 2)
    P_list = Vector{Dict{Symbol,Float64}}(undef, N)
    missing_list = Vector{Vector{Symbol}}(undef, N)
    for i in 1:N
        P_list[i], missing_list[i] = u_to_p_full(view(U, :, i), param_names; state=state, fixed=fixed)
    end
    return P_list, missing_list
end

# 生成 100 组随机系数，并得到 100 组解 u（每列一个解）
const C_samples, U_samples = sample_nullspace_solutions(V; n_samples=100, rng=MersenneTwister(2026), dist=:normal)

# 如果你希望只取“15维 u”（你提到的 15 组参数/15个系数），这里做一个硬性截断：
# - 当 V 的零空间维数 k >= 15 时，取前 15 列基
# - 当 k < 15 时，无法生成 15 维系数（需要回头检查 A 的秩/rtol）
const V15 = size(V, 2) >= 15 ? V[:, 1:15] : V
const C15_samples, U15_samples = sample_nullspace_solutions(V15; n_samples=100, rng=MersenneTwister(2026), dist=:normal)

# =========================================================================
# 单组参数评估：
# 1) 任选一组 u -> p
# 2) 检查 RHS(t=50) 与 RHS(t=200)
# 3) 代入 ODE，计算预测与 target 的 log10-L2 损失（与 optimize.jl 一致的 data_loss）
# =========================================================================

# 与 optimize.jl 一致的抗原输入函数
Ag_func(t) = 100.0 / (1.0 + exp(-(t - 125.0)/5.0))
@register_symbolic Ag_func(t)

# 与 optimize.jl 一致的反应网络（复用同一套反应式）
const RN = @reaction_network begin
    k_nDC_f * nDC * (1 - nDC/k_nDC_m), 0 --> nDC
    k_mDC_Antigen_f * Ag_func(t) * nDC * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10)), nDC --> 0
    k_mDC_GMCSF_f * Ag_func(t) * nDC * (GMCSF / (GMCSF + k_mDC_GMCSF_m)) * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10)), nDC --> 0
    k_pDC_Antigen_f * nDC * Ag_func(t), nDC --> 0
    k_nDC_d, nDC --> 0

    k_mDC_Antigen_f * Ag_func(t) * nDC * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10)), 0 --> mDC
    k_mDC_GMCSF_f * Ag_func(t) * nDC * (GMCSF / (GMCSF + k_mDC_GMCSF_m)) * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10)), 0 --> mDC
    k_mDC_f * mDC * (1 - mDC / k_mDC_m), 0 --> mDC
    k_mDC_d, mDC --> 0

    k_GMCSF_Th2_f * Th2, 0 --> GMCSF
    k_GMCSF_Th2_Antigen_f * Th2 * Ag_func(t), 0 --> GMCSF
    k_GMCSF_act_NK_f * act_NK, 0 --> GMCSF
    k_mDC_GMCSF_d * Ag_func(t) * nDC * (GMCSF / (GMCSF + k_mDC_GMCSF_m)) * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10)), GMCSF --> 0
    k_GMCSF_d, GMCSF --> 0

    k_pDC_Antigen_f * nDC * Ag_func(t), 0 --> pDC
    k_pDC_f * pDC * (1 - pDC / k_pDC_m), 0 --> pDC
    k_pDC_d, pDC --> 0

    k_IL_33_pDC_f * pDC, 0 --> IL_33
    k_act_CD4_IL_33_d * act_CD4 * IL_33 / (k_Th2_IL_33_m + IL_33), IL_33 --> 0
    k_IL_33_d, IL_33 --> 0

    k_IL_6_pDC_f * pDC, 0 --> IL_6
    k_IL_6_mDC_f * mDC, 0 --> IL_6
    k_IL_6_TFH_f * TFH * (k_TFH_nTreg_m / (nTreg + k_TFH_nTreg_m)), 0 --> IL_6
    k_TFH_IL_6_d * act_CD4 * IL_6 / (k_TFH_IL_6_m + IL_6), IL_6 --> 0
    k_IL_6_d, IL_6 --> 0

    k_IL_12_mDC_f * mDC, 0 --> IL_12
    k_act_NK_IL_12_d * NK * IL_12 / (IL_12 + k_act_NK_IL_12_m), IL_12 --> 0
    k_IL_12_d, IL_12 --> 0

    k_IL_15_f, 0 --> IL_15
    k_IL_15_Antigen_f * Ag_func(t), 0 --> IL_15
    k_naive_CD4_IL_15_d * naive_CD4 * IL_15 / (k_naive_CD4_IL_15_m + IL_15), IL_15 --> 0
    k_act_CD4_IL_15_d * act_CD4 * IL_15 / (k_act_CD4_IL_15_m + IL_15), IL_15 --> 0
    k_IL_15_d, IL_15 --> 0

    k_IL_7_f, 0 --> IL_7
    k_naive_CD4_IL_7_d * naive_CD4 * IL_7 / (k_naive_CD4_IL_7_m + IL_7), IL_7 --> 0
    k_act_CD4_IL_7_d * act_CD4 * IL_7 / (k_act_CD4_IL_7_m + IL_7), IL_7 --> 0
    k_IL_7_d, IL_7 --> 0

    k_IFN1_pDC_f * pDC, 0 --> IFN1
    k_act_CD4_IFN1_d * act_CD4 * IFN1 / (k_IFN1_CD4_CTL_m + IFN1), IFN1 --> 0
    k_act_NK_IFN1_d * NK * IFN1 / (IFN1 + k_act_NK_IFN1_m), IFN1 --> 0
    k_IFN1_d, IFN1 --> 0

    k_IL_1_mDC_f * mDC, 0 --> IL_1
    k_IL_1_d, IL_1 --> 0

    k_IL_2_act_CD4_f * act_CD4, 0 --> IL_2
    k_IL_2_act_CD4_Antigen_f * act_CD4 * Ag_func(t), 0 --> IL_2
    k_act_CD4_IL_2_d * naive_CD4 * IL_2 / (k_act_CD4_IL_2_m + IL_2), IL_2 --> 0
    k_act_NK_IL_2_d * NK * IL_2 / (IL_2 + k_act_NK_IL_2_m), IL_2 --> 0
    k_IL_2_d, IL_2 --> 0

    k_IL_4_Th2_f * Th2, 0 --> IL_4
    k_IL_4_Th2_Antigen_f * Th2 * Ag_func(t), 0 --> IL_4
    k_act_CD4_IL_4_d * act_CD4 * IL_4 / (k_Th2_IL_4_m + IL_4), IL_4 --> 0
    k_IL_4_d, IL_4 --> 0

    k_IL_10_iTreg_f * iTreg, 0 --> IL_10
    k_IL_10_nTreg_f * nTreg * mDC / (k_IL_10_nTreg_mDC_m + mDC), 0 --> IL_10
    k_iTreg_mDC_d * act_CD4 * IL_10 / (k_iTreg_IL_10_m + IL_10), IL_10 --> 0
    k_IL_10_d, IL_10 --> 0

    k_TGFbeta_iTreg_f * iTreg, 0 --> TGFbeta
    k_TGFbeta_CD4_CTL_f * CD4_CTL, 0 --> TGFbeta
    k_TGFbeta_nTreg_f * nTreg * mDC / (k_TGFbeta_nTreg_mDC_m + mDC), 0 --> TGFbeta
    k_iTreg_mDC_d * act_CD4 * TGFbeta / (k_iTreg_TGFbeta_m + TGFbeta), TGFbeta --> 0
    k_TGFbeta_d, TGFbeta --> 0

    k_IFN_g_CD4_CTL_f * CD4_CTL, 0 --> IFN_g
    k_IFN_g_act_NK_f * act_NK, 0 --> IFN_g
    k_act_NK_IFN_g_d * NK * IFN_g / (IFN_g + k_act_NK_IFN_g_m), IFN_g --> 0
    k_IFN_g_d, IFN_g --> 0

    k_CD4_f * naive_CD4 * (1 - naive_CD4 / k_CD4_m), 0 --> naive_CD4
    k_naive_CD4_IL_15_f * naive_CD4 * (1 - naive_CD4 / k_CD4_m) * IL_15 / (k_naive_CD4_IL_15_m + IL_15), 0 --> naive_CD4
    k_naive_CD4_IL_7_f * naive_CD4 * (1 - naive_CD4 / k_CD4_m) * IL_7 / (k_naive_CD4_IL_7_m + IL_7), 0 --> naive_CD4
    k_act_CD4_mDC_f * naive_CD4 * mDC / (k_act_CD4_mDC_m + mDC), naive_CD4 --> 0
    k_act_CD4_IL_2_f * naive_CD4 * IL_2 / (k_act_CD4_IL_2_m + IL_2), naive_CD4 --> 0
    k_naive_CD4_d, naive_CD4 --> 0

    k_act_CD4_mDC_f * naive_CD4 * mDC / (k_act_CD4_mDC_m + mDC), 0 --> act_CD4
    k_act_CD4_IL_2_f * naive_CD4 * IL_2 / (k_act_CD4_IL_2_m + IL_2), 0 --> act_CD4
    k_act_CD4_f * act_CD4 * (1 - act_CD4 / k_act_CD4_m), 0 --> act_CD4
    k_act_CD4_IL_15_f * act_CD4 * (1 - act_CD4 / k_act_CD4_m) * IL_15 / (k_act_CD4_IL_15_m + IL_15), 0 --> act_CD4
    k_act_CD4_IL_7_f * act_CD4 * (1 - act_CD4 / k_act_CD4_m) * IL_7 / (k_act_CD4_IL_7_m + IL_7), 0 --> act_CD4
    act_CD4 * k_Th2_f * (k_Th2_TGFbeta_m / (k_Th2_TGFbeta_m + TGFbeta)) * (k_Th2_IL_10_m / (k_Th2_IL_10_m + IL_10)) * (k_Th2_IL_12_m / (k_Th2_IL_12_m + IL_12)), act_CD4 --> 0
    act_CD4 * k_Th2_IL_4_f * IL_4 / (k_Th2_IL_4_m + IL_4) * (k_Th2_TGFbeta_m / (k_Th2_TGFbeta_m + TGFbeta)) * (k_Th2_IL_10_m / (k_Th2_IL_10_m + IL_10)) * (k_Th2_IL_12_m / (k_Th2_IL_12_m + IL_12)), act_CD4 --> 0
    act_CD4 * k_Th2_IL_33_f * IL_33 / (k_Th2_IL_33_m + IL_33) * (k_Th2_TGFbeta_m / (k_Th2_TGFbeta_m + TGFbeta)) * (k_Th2_IL_10_m / (k_Th2_IL_10_m + IL_10)) * (k_Th2_IL_12_m / (k_Th2_IL_12_m + IL_12)), act_CD4 --> 0
    act_CD4 * k_iTreg_mDC_f * k_iTreg_TGFbeta_f * TGFbeta / (k_iTreg_TGFbeta_m + TGFbeta) * (k_iTreg_IL_1_m / (k_iTreg_IL_1_m + IL_1)), act_CD4 --> 0
    act_CD4 * k_iTreg_mDC_f * k_iTreg_IL_10_f * IL_10 / (k_iTreg_IL_10_m + IL_10) * (k_iTreg_IL_1_m / (k_iTreg_IL_1_m + IL_1)), act_CD4 --> 0
    k_act_CD4_CTL_basal_f * act_CD4, act_CD4 --> 0
    k_act_CD4_CTL_antigen_f * act_CD4 * Ag_func(t), act_CD4 --> 0
    k_act_CD4_IFN1_f * act_CD4 * IFN1 / (k_IFN1_CD4_CTL_m + IFN1), act_CD4 --> 0
    k_TFH_mDC_f * act_CD4, act_CD4 --> 0
    k_TFH_mDC_Antigen_f * act_CD4 * Ag_func(t), act_CD4 --> 0
    k_TFH_IFN1_f * act_CD4 * IFN1 / (k_TFH_IFN1_m + IFN1), act_CD4 --> 0
    k_TFH_IL_6_f * act_CD4 * IL_6 / (k_TFH_IL_6_m + IL_6), act_CD4 --> 0
    k_act_CD4_d, act_CD4 --> 0

    act_CD4 * k_Th2_f * (k_Th2_TGFbeta_m / (k_Th2_TGFbeta_m + TGFbeta)) * (k_Th2_IL_10_m / (k_Th2_IL_10_m + IL_10)) * (k_Th2_IL_12_m / (k_Th2_IL_12_m + IL_12)), 0 --> Th2
    act_CD4 * k_Th2_IL_4_f * IL_4 / (k_Th2_IL_4_m + IL_4) * (k_Th2_TGFbeta_m / (k_Th2_TGFbeta_m + TGFbeta)) * (k_Th2_IL_10_m / (k_Th2_IL_10_m + IL_10)) * (k_Th2_IL_12_m / (k_Th2_IL_12_m + IL_12)), 0 --> Th2
    act_CD4 * k_Th2_IL_33_f * IL_33 / (k_Th2_IL_33_m + IL_33) * (k_Th2_TGFbeta_m / (k_Th2_TGFbeta_m + TGFbeta)) * (k_Th2_IL_10_m / (k_Th2_IL_10_m + IL_10)) * (k_Th2_IL_12_m / (k_Th2_IL_12_m + IL_12)), 0 --> Th2
    k_Th2_f * Th2 * (1 - Th2 / k_Th2_m), 0 --> Th2
    k_Th2_d, Th2 --> 0

    act_CD4 * k_iTreg_mDC_f * k_iTreg_TGFbeta_f * TGFbeta / (k_iTreg_TGFbeta_m + TGFbeta) * (k_iTreg_IL_1_m / (k_iTreg_IL_1_m + IL_1)), 0 --> iTreg
    act_CD4 * k_iTreg_mDC_f * k_iTreg_IL_10_f * IL_10 / (k_iTreg_IL_10_m + IL_10) * (k_iTreg_IL_1_m / (k_iTreg_IL_1_m + IL_1)), 0 --> iTreg
    k_iTreg_f * iTreg * (1 - iTreg / k_iTreg_m), 0 --> iTreg
    k_iTreg_d, iTreg --> 0

    k_act_CD4_CTL_basal_f * act_CD4, 0 --> CD4_CTL
    k_act_CD4_CTL_antigen_f * act_CD4 * Ag_func(t), 0 --> CD4_CTL
    k_act_CD4_IFN1_f * act_CD4 * IFN1 / (k_IFN1_CD4_CTL_m + IFN1), 0 --> CD4_CTL
    k_CD4_CTL_f * CD4_CTL * (1 - CD4_CTL / k_CD4_CTL_m), 0 --> CD4_CTL
    k_CD4_CTL_d, CD4_CTL --> 0

    k_nTreg_mDC_f * nTreg * (1 - nTreg / k_nTreg_m) * mDC / (k_nTreg_mDC_m + mDC), 0 --> nTreg
    k_nTreg_d, nTreg --> 0

    k_TFH_mDC_f * act_CD4, 0 --> TFH
    k_TFH_mDC_Antigen_f * act_CD4 * Ag_func(t), 0 --> TFH
    k_TFH_IFN1_f * act_CD4 * IFN1 / (k_TFH_IFN1_m + IFN1), 0 --> TFH
    k_TFH_IL_6_f * act_CD4 * IL_6 / (k_TFH_IL_6_m + IL_6), 0 --> TFH
    k_TFH_f * TFH * (1 - TFH / k_TFH_m), 0 --> TFH
    k_TFH_d, TFH --> 0

    k_NK_f * NK * (1 - NK / k_NK_m), 0 --> NK
    k_act_NK_base_f * NK, NK --> 0
    k_act_NK_IL_12_f * NK * IL_12 / (IL_12 + k_act_NK_IL_12_m), NK --> 0
    k_act_NK_IL_2_f * NK * IL_2 / (IL_2 + k_act_NK_IL_2_m), NK --> 0
    k_act_NK_IFN1_f * NK * IFN1 / (IFN1 + k_act_NK_IFN1_m), NK --> 0
    k_act_NK_IFN_g_f * NK * IFN_g / (IFN_g + k_act_NK_IFN_g_m), NK --> 0
    k_NK_d, NK --> 0

    k_act_NK_base_f * NK, 0 --> act_NK
    k_act_NK_IL_12_f * NK * IL_12 / (IL_12 + k_act_NK_IL_12_m), 0 --> act_NK
    k_act_NK_IL_2_f * NK * IL_2 / (IL_2 + k_act_NK_IL_2_m), 0 --> act_NK
    k_act_NK_IFN1_f * NK * IFN1 / (IFN1 + k_act_NK_IFN1_m), 0 --> act_NK
    k_act_NK_IFN_g_f * NK * IFN_g / (IFN_g + k_act_NK_IFN_g_m), 0 --> act_NK
    k_act_NK_f * act_NK * (1 - act_NK / k_act_NK_m), 0 --> act_NK
    k_act_NK_d, act_NK --> 0

    k_Naive_B_f * Naive_B * (1 - Naive_B / k_Naive_B_m), 0 --> Naive_B
    k_Naive_B_Antigen_f * Naive_B * Ag_func(t) * (1 - Naive_B / k_Naive_B_m), 0 --> Naive_B
    k_Act_B_basal_f * Naive_B, Naive_B --> 0
    k_Act_B_Antigen_f * Naive_B * Ag_func(t), Naive_B --> 0
    k_Naive_B_d, Naive_B --> 0

    k_Act_B_basal_f * Naive_B, 0 --> Act_B
    k_Act_B_Antigen_f * Naive_B * Ag_func(t), 0 --> Act_B
    k_Act_B_f * Act_B * (1 - Act_B / k_Act_B_m), 0 --> Act_B
    k_Act_B_Antigen_pro_f * Act_B * Ag_func(t) * (1 - Act_B / k_Act_B_m), 0 --> Act_B
    k_Act_B_d, Act_B --> 0

    k_TD_base_f * Act_B, 0 --> TD_IS_B
    k_TD_IL_4_f * Act_B * IL_4, 0 --> TD_IS_B
    k_TD_f * TD_IS_B * (1 - TD_IS_B / k_TD_m), 0 --> TD_IS_B
    k_TD_d, TD_IS_B --> 0

    k_TI_base_f * Act_B, 0 --> TI_IS_B
    k_TI_IFN_g_f * Act_B * IFN_g, 0 --> TI_IS_B
    k_TI_IL_10_f * Act_B * IL_10, 0 --> TI_IS_B
    k_TI_f * TI_IS_B * (1 - TI_IS_B / k_TI_m), 0 --> TI_IS_B
    k_TI_d, TI_IS_B --> 0

    k_IgG4_TI_f * TI_IS_B, 0 --> IgG4
    k_IgG4_TD_f * TD_IS_B, 0 --> IgG4
    k_IgG4_d, IgG4 --> 0
end

# 与 optimize.jl 保持一致的初始条件（注意：这里 Antigen 不作为物种；使用 Ag_func(t)）
const U0_MAP = [
    :nDC => 30.0,
    :mDC => 0.1,
    :pDC => 0.1,
    :naive_CD4 => 180.0,
    :act_CD4 => 300.0,
    :Th2 => 0.3,
    :iTreg => 50.0,
    :CD4_CTL => 1.0,
    :nTreg => 10.0,
    :TFH => 20.0,
    :NK => 50.0,
    :act_NK => 200.0,
    :Naive_B => 86.0,
    :Act_B => 80.0,
    :TD_IS_B => 1.5,
    :TI_IS_B => 1.0,
    :GMCSF => 3000.0,
    :IL_33 => 1.0,
    :IL_6 => 10000.0,
    :IL_12 => 1.0,
    :IL_15 => 10000.0,
    :IL_7 => 350000.0,
    :IFN1 => 1.0,
    :IL_1 => 3000.0,
    :IL_2 => 5000.0,
    :IL_4 => 250.0,
    :IL_10 => 4000.0,
    :TGFbeta => 50.0,
    :IFN_g => 200000.0,
    :IgG4 => 0.1,
]

const TSPAN = (0.0, 300.0)

"""
读取 target_data.csv，返回 target_time 与 target_data（变量名按 optimize.jl 的映射规则）。
"""
function load_target_data(path::AbstractString="target_data.csv")
    df = CSV.read(path, DataFrame, delim='\t')
    target_time = df.Time

    name_mapping = Dict(
        "CD56 NK" => "NK",
        "CD16 NK" => "act_NK",
        "TD Plasma" => "TD_IS_B",
        "TI Plasma" => "TI_IS_B",
    )

    target_data = Dict{Symbol,Vector{Float64}}()
    for col_name in names(df)
        if col_name != "Time"
            model_name = get(name_mapping, col_name, col_name)
            if model_name != "Antigen" && !haskey(target_data, Symbol(model_name))
                target_data[Symbol(model_name)] = df[!, col_name]
            end
        end
    end

    return target_time, target_data
end

"""
把 Dict{Symbol,Float64} 的参数转换为 Catalyst ODEProblem 需要的 Pair 向量。
"""
function dict_to_pairs(p::Dict{Symbol,Float64})
    return [k => v for (k, v) in p]
end

"""
计算 RHS 向量（不含 Antigen），并返回 (rhs_values, state_order)。

这里的 RHS 来自 ODEProblem 的 f：f(du, u, p, t)。
"""
function rhs_at_time(prob::ODEProblem, t::Float64, sol)
    u_t = sol(t)
    du = similar(u_t)
    prob.f(du, u_t, prob.p, t)

    # 把 rhs 按变量顺序返回（顺序与 u_t 一致）
    return du
end

"""
计算与 optimize.jl 一致的 log10-L2 data loss：
sum_{var} sum_{time} (log10(pred+eps) - log10(obs+eps))^2

注意：这里不包含 optimize.jl 的时间窗权重、smooth、steady loss，只计算“数据项”。
"""
function data_log10_l2_loss(sol, target_time, target_data; eps=1e-10)
    loss = 0.0
    for (var_sym, obs) in target_data
        try
            pred = sol[var_sym]
            loss += sum(abs2.(log10.(max.(pred, eps)) .- log10.(max.(obs, eps))))
        catch
            continue
        end
    end
    return loss
end

"""
任选一组参数（idx），检查 RHS(t=50) 与 RHS(t=200)（各30项），并计算 data_loss。

参数：
- idx: 选择第 idx 组 u（从 U15_samples 的列中选）
- state_for_inverse: 用于 u->p_full 的稳态状态（建议 IgG_STATE）

打印：
- RHS(t=50) 向量
- RHS(t=200) 向量
- data_log10_l2_loss
"""
function evaluate_one_candidate(idx::Int=1; state_for_inverse::Dict{Symbol,Float64}=IgG_STATE)
    # 1) u -> p_full
    fixed = fixed_params_from_utils_candidate1()
    u = view(U15_samples, :, idx)
    p_full, missing = u_to_p_full(u, ALL_PARAMS; state=state_for_inverse, fixed=fixed)

    # 2) 构造 ODEProblem 并求解（为了拿 RHS(t) 和 loss）
    prob = ODEProblem(RN, U0_MAP, TSPAN, dict_to_pairs(p_full))

    # 先求一条轨迹（不 saveat，rhs 用插值；loss 再用 saveat=target_time）
    sol_dense = solve(prob, QNDF(); abstol=1e-8, reltol=1e-6, verbose=false, maxiters=1_000_000)

    if sol_dense.retcode != :Success && string(sol_dense.retcode) != "Success"
        println("ODE solve failed: ", sol_dense.retcode)
        return nothing
    end

    rhs50 = rhs_at_time(prob, 50.0, sol_dense)
    rhs200 = rhs_at_time(prob, 200.0, sol_dense)

    println("RHS(t=50) = ", rhs50)
    println("RHS(t=200) = ", rhs200)

    # 3) loss：按 target_time 采样
    target_time, target_data = load_target_data()
    sol = solve(prob, QNDF(); saveat=target_time, abstol=1e-8, reltol=1e-6, verbose=false, maxiters=1_000_000)

    if sol.retcode != :Success && string(sol.retcode) != "Success"
        println("ODE solve (saveat) failed: ", sol.retcode)
        return nothing
    end

    loss = data_log10_l2_loss(sol, target_time, target_data)
    println("data log10-L2 loss = ", loss)

    return (p_full=p_full, missing=missing, rhs50=rhs50, rhs200=rhs200, loss=loss)
end

# 默认跑一次（任选第1组）
# evaluate_one_candidate(1; state_for_inverse=IgG_STATE)


