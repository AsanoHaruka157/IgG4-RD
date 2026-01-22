using Catalyst
using DifferentialEquations
using Plots
using Optimization
using OptimizationOptimJL # 包含 L-BFGS
using OptimizationBBO     # 新增：BBO 优化算法
using LineSearches        # 线搜索算法
using CSV
using DataFrames
using ForwardDiff         # 自动微分核心
using Statistics          # 统计函数，包括 var()
using QuasiMonteCarlo
using ProgressMeter       # 进度条

# ============================================================================
# 0. 全局超参数 (Hyperparameters)
# ============================================================================
# --- 采样/边界（相对初值 p0_lin 的倍数）---
STAGE1_SAMPLE_FRAC = 0.50      # 一阶段采样范围：±50%（即 [0.5x, 1.5x]）
STAGE2_LB_MULT     = 1e-2      # 二阶段优化下界：x1e-2
STAGE2_UB_MULT     = 1e2       # 二阶段优化上界：x1e2

# --- 多起点/筛选/精修 ---
N_STARTS         = 100         # 一阶段起点数量
N_TOP_CANDIDATES = 15          # 进入二阶段的候选数量
SCREEN_MAXITERS  = 30          # 一阶段粗筛每条轨迹优化步数

FINE_MAXITERS = 400            # 二阶段精修优化步数
FINE_X_ABSTOL = 1e-5
FINE_F_RELTOL = 1e-5

# --- ODE 求解器 ---
ODE_ABSTOL   = 1e-8
ODE_RELTOL   = 1e-6
ODE_MAXITERS = 1_000_000

# ============================================================================
# 1. 数据加载 (Data Loading)
# ============================================================================
# 读取制表符分隔的CSV文件
df = CSV.read("target_data.csv", DataFrame, delim='\t')
target_time = df.Time

# 变量名映射：数据文件列名 -> 模型变量名
name_mapping = Dict(
    "CD56 NK" => "NK",
    "CD16 NK" => "act_NK",
    "TD Plasma" => "TD_IS_B",
    "TI Plasma" => "TI_IS_B"
)

# 提取所有变量作为目标 (除了Time列)，避免重复
# 保持变量顺序与数据列一致（不要用 Set，否则顺序随机且可能包含非状态变量）
model_variable_names = String[]
for col_name in names(df)
    if col_name != "Time"
        model_name = get(name_mapping, col_name, col_name)
        if model_name != "Antigen" && !(model_name in model_variable_names)
            push!(model_variable_names, model_name)
        end
    end
end

# 创建目标数据字典，使用模型变量名，避免重复
target_data = Dict{Symbol, Vector{Float64}}()
for col_name in names(df)
    if col_name != "Time"
        model_name = get(name_mapping, col_name, col_name)
        # 只处理第一次遇到的映射，避免重复
        if !haskey(target_data, Symbol(model_name))
            target_data[Symbol(model_name)] = df[!, col_name]
        end
    end
end

println("数据加载完成: $(length(target_time)) 个时间点, $(length(model_variable_names)) 个变量")
println("模型变量列表: $(model_variable_names)")

# ============================================================================
# 2. 定义时间依赖函数 (Time-dependent Functions)
# ============================================================================
# 定义抗原输入函数 Ag(t) - Sigmoid函数
# 必须处理 t，且保证数学运算兼容自动微分
Ag_func(t) = 100.0 / (1.0 + exp(-(t - 125.0)/5.0))

# 注册函数，让 Catalyst 知道这是一个随时间变化的量
@register_symbolic Ag_func(t)

# ============================================================================
# 3. 模型定义 (Catalyst DSL) - Antigen作为时间函数
# ============================================================================
rn = @reaction_network begin
    # ============================================================================
    # 反应定义 (从 optimize2.py 转换)
    # ============================================================================

    # ============================================================================
    # 反应定义 - Antigen现在是时间函数 Ag_func(t)
    # ============================================================================

    # 1. nDC 反应
    k_nDC_f * nDC * (1 - nDC/k_nDC_m), 0 --> nDC
    k_mDC_Antigen_f * Ag_func(t) * nDC * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10)), nDC --> 0
    k_mDC_GMCSF_f * Ag_func(t) * nDC * (GMCSF / (GMCSF + k_mDC_GMCSF_m)) * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10)), nDC --> 0
    k_pDC_Antigen_f * nDC * Ag_func(t), nDC --> 0
    k_nDC_d, nDC --> 0

    # 3. mDC 反应
    k_mDC_Antigen_f * Ag_func(t) * nDC * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10)), 0 --> mDC
    k_mDC_GMCSF_f * Ag_func(t) * nDC * (GMCSF / (GMCSF + k_mDC_GMCSF_m)) * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10)), 0 --> mDC
    k_mDC_f * mDC * (1 - mDC / k_mDC_m), 0 --> mDC
    k_mDC_d, mDC --> 0

    # 4. GMCSF 反应
    k_GMCSF_Th2_f * Th2, 0 --> GMCSF
    k_GMCSF_Th2_Antigen_f * Th2 * Ag_func(t), 0 --> GMCSF
    k_GMCSF_act_NK_f * act_NK, 0 --> GMCSF
    k_mDC_GMCSF_d * Ag_func(t) * nDC * (GMCSF / (GMCSF + k_mDC_GMCSF_m)) * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10)), GMCSF --> 0
    k_GMCSF_d, GMCSF --> 0

    # 5. pDC 反应
    k_pDC_Antigen_f * nDC * Ag_func(t), 0 --> pDC
    k_pDC_f * pDC * (1 - pDC / k_pDC_m), 0 --> pDC
    k_pDC_d, pDC --> 0

    # 6. IL_33 反应
    k_IL_33_pDC_f * pDC, 0 --> IL_33
    k_act_CD4_IL_33_d * act_CD4 * IL_33 / (k_Th2_IL_33_m + IL_33), IL_33 --> 0
    k_IL_33_d, IL_33 --> 0

    # 7. IL_6 反应
    k_IL_6_pDC_f * pDC, 0 --> IL_6
    k_IL_6_mDC_f * mDC, 0 --> IL_6
    k_IL_6_TFH_f * TFH * (k_TFH_nTreg_m / (nTreg + k_TFH_nTreg_m)), 0 --> IL_6
    k_TFH_IL_6_d * act_CD4 * IL_6 / (k_TFH_IL_6_m + IL_6), IL_6 --> 0
    k_IL_6_d, IL_6 --> 0

    # 8. IL_12 反应
    k_IL_12_mDC_f * mDC, 0 --> IL_12
    k_act_NK_IL_12_d * NK * IL_12 / (IL_12 + k_act_NK_IL_12_m), IL_12 --> 0
    k_IL_12_d, IL_12 --> 0

    # 9. IL_15 反应
    k_IL_15_f, 0 --> IL_15
    k_IL_15_Antigen_f * Ag_func(t), 0 --> IL_15
    k_naive_CD4_IL_15_d * naive_CD4 * IL_15 / (k_naive_CD4_IL_15_m + IL_15), IL_15 --> 0
    k_act_CD4_IL_15_d * act_CD4 * IL_15 / (k_act_CD4_IL_15_m + IL_15), IL_15 --> 0
    k_IL_15_d, IL_15 --> 0

    # 10. IL_7 反应
    k_IL_7_f, 0 --> IL_7
    k_naive_CD4_IL_7_d * naive_CD4 * IL_7 / (k_naive_CD4_IL_7_m + IL_7), IL_7 --> 0
    k_act_CD4_IL_7_d * act_CD4 * IL_7 / (k_act_CD4_IL_7_m + IL_7), IL_7 --> 0
    k_IL_7_d, IL_7 --> 0

    # 11. IFN1 反应
    k_IFN1_pDC_f * pDC, 0 --> IFN1
    k_act_CD4_IFN1_d * act_CD4 * IFN1 / (k_IFN1_CD4_CTL_m + IFN1), IFN1 --> 0
    k_act_NK_IFN1_d * NK * IFN1 / (IFN1 + k_act_NK_IFN1_m), IFN1 --> 0
    k_IFN1_d, IFN1 --> 0

    # 12. IL_1 反应
    k_IL_1_mDC_f * mDC, 0 --> IL_1
    k_IL_1_d, IL_1 --> 0

    # 13. IL_2 反应
    k_IL_2_act_CD4_f * act_CD4, 0 --> IL_2
    k_IL_2_act_CD4_Antigen_f * act_CD4 * Ag_func(t), 0 --> IL_2
    k_act_CD4_IL_2_d * naive_CD4 * IL_2 / (k_act_CD4_IL_2_m + IL_2), IL_2 --> 0
    k_act_NK_IL_2_d * NK * IL_2 / (IL_2 + k_act_NK_IL_2_m), IL_2 --> 0
    k_IL_2_d, IL_2 --> 0

    # 14. IL_4 反应
    k_IL_4_Th2_f * Th2, 0 --> IL_4
    k_IL_4_Th2_Antigen_f * Th2 * Ag_func(t), 0 --> IL_4
    k_act_CD4_IL_4_d * act_CD4 * IL_4 / (k_Th2_IL_4_m + IL_4), IL_4 --> 0
    k_IL_4_d, IL_4 --> 0

    # 15. IL_10 反应
    k_IL_10_iTreg_f * iTreg, 0 --> IL_10
    k_IL_10_nTreg_f * nTreg * mDC / (k_IL_10_nTreg_mDC_m + mDC), 0 --> IL_10
    k_iTreg_mDC_d * act_CD4 * IL_10 / (k_iTreg_IL_10_m + IL_10), IL_10 --> 0
    k_IL_10_d, IL_10 --> 0

    # 16. TGFbeta 反应
    k_TGFbeta_iTreg_f * iTreg, 0 --> TGFbeta
    k_TGFbeta_CD4_CTL_f * CD4_CTL, 0 --> TGFbeta
    k_TGFbeta_nTreg_f * nTreg * mDC / (k_TGFbeta_nTreg_mDC_m + mDC), 0 --> TGFbeta
    k_iTreg_mDC_d * act_CD4 * TGFbeta / (k_iTreg_TGFbeta_m + TGFbeta), TGFbeta --> 0
    k_TGFbeta_d, TGFbeta --> 0

    # 17. IFN_g 反应
    k_IFN_g_CD4_CTL_f * CD4_CTL, 0 --> IFN_g
    k_IFN_g_act_NK_f * act_NK, 0 --> IFN_g
    k_act_NK_IFN_g_d * NK * IFN_g / (IFN_g + k_act_NK_IFN_g_m), IFN_g --> 0
    k_IFN_g_d, IFN_g --> 0

    # 18. naive_CD4 反应
    k_CD4_f * naive_CD4 * (1 - naive_CD4 / k_CD4_m), 0 --> naive_CD4
    k_naive_CD4_IL_15_f * naive_CD4 * (1 - naive_CD4 / k_CD4_m) * IL_15 / (k_naive_CD4_IL_15_m + IL_15), 0 --> naive_CD4
    k_naive_CD4_IL_7_f * naive_CD4 * (1 - naive_CD4 / k_CD4_m) * IL_7 / (k_naive_CD4_IL_7_m + IL_7), 0 --> naive_CD4
    k_act_CD4_mDC_f * naive_CD4 * mDC / (k_act_CD4_mDC_m + mDC), naive_CD4 --> 0
    k_act_CD4_IL_2_f * naive_CD4 * IL_2 / (k_act_CD4_IL_2_m + IL_2), naive_CD4 --> 0
    k_naive_CD4_d, naive_CD4 --> 0

    # 19. act_CD4 反应
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

    # 20. Th2 反应
    act_CD4 * k_Th2_f * (k_Th2_TGFbeta_m / (k_Th2_TGFbeta_m + TGFbeta)) * (k_Th2_IL_10_m / (k_Th2_IL_10_m + IL_10)) * (k_Th2_IL_12_m / (k_Th2_IL_12_m + IL_12)), 0 --> Th2
    act_CD4 * k_Th2_IL_4_f * IL_4 / (k_Th2_IL_4_m + IL_4) * (k_Th2_TGFbeta_m / (k_Th2_TGFbeta_m + TGFbeta)) * (k_Th2_IL_10_m / (k_Th2_IL_10_m + IL_10)) * (k_Th2_IL_12_m / (k_Th2_IL_12_m + IL_12)), 0 --> Th2
    act_CD4 * k_Th2_IL_33_f * IL_33 / (k_Th2_IL_33_m + IL_33) * (k_Th2_TGFbeta_m / (k_Th2_TGFbeta_m + TGFbeta)) * (k_Th2_IL_10_m / (k_Th2_IL_10_m + IL_10)) * (k_Th2_IL_12_m / (k_Th2_IL_12_m + IL_12)), 0 --> Th2
    k_Th2_f * Th2 * (1 - Th2 / k_Th2_m), 0 --> Th2
    k_Th2_d, Th2 --> 0

    # 21. iTreg 反应
    act_CD4 * k_iTreg_mDC_f * k_iTreg_TGFbeta_f * TGFbeta / (k_iTreg_TGFbeta_m + TGFbeta) * (k_iTreg_IL_1_m / (k_iTreg_IL_1_m + IL_1)), 0 --> iTreg
    act_CD4 * k_iTreg_mDC_f * k_iTreg_IL_10_f * IL_10 / (k_iTreg_IL_10_m + IL_10) * (k_iTreg_IL_1_m / (k_iTreg_IL_1_m + IL_1)), 0 --> iTreg
    k_iTreg_f * iTreg * (1 - iTreg / k_iTreg_m), 0 --> iTreg
    k_iTreg_d, iTreg --> 0

    # 22. CD4_CTL 反应
    k_act_CD4_CTL_basal_f * act_CD4, 0 --> CD4_CTL
    k_act_CD4_CTL_antigen_f * act_CD4 * Ag_func(t), 0 --> CD4_CTL
    k_act_CD4_IFN1_f * act_CD4 * IFN1 / (k_IFN1_CD4_CTL_m + IFN1), 0 --> CD4_CTL
    k_CD4_CTL_f * CD4_CTL * (1 - CD4_CTL / k_CD4_CTL_m), 0 --> CD4_CTL
    k_CD4_CTL_d, CD4_CTL --> 0

    # 23. nTreg 反应
    k_nTreg_mDC_f * nTreg * (1 - nTreg / k_nTreg_m) * mDC / (k_nTreg_mDC_m + mDC), 0 --> nTreg
    k_nTreg_d, nTreg --> 0

    # 24. TFH 反应
    k_TFH_mDC_f * act_CD4, 0 --> TFH
    k_TFH_mDC_Antigen_f * act_CD4 * Ag_func(t), 0 --> TFH
    k_TFH_IFN1_f * act_CD4 * IFN1 / (k_TFH_IFN1_m + IFN1), 0 --> TFH
    k_TFH_IL_6_f * act_CD4 * IL_6 / (k_TFH_IL_6_m + IL_6), 0 --> TFH
    k_TFH_f * TFH * (1 - TFH / k_TFH_m), 0 --> TFH
    k_TFH_d, TFH --> 0

    # 25. NK 反应
    k_NK_f * NK * (1 - NK / k_NK_m), 0 --> NK
    k_act_NK_base_f * NK, NK --> 0
    k_act_NK_IL_12_f * NK * IL_12 / (IL_12 + k_act_NK_IL_12_m), NK --> 0
    k_act_NK_IL_2_f * NK * IL_2 / (IL_2 + k_act_NK_IL_2_m), NK --> 0
    k_act_NK_IFN1_f * NK * IFN1 / (IFN1 + k_act_NK_IFN1_m), NK --> 0
    k_act_NK_IFN_g_f * NK * IFN_g / (IFN_g + k_act_NK_IFN_g_m), NK --> 0
    k_NK_d, NK --> 0

    # 26. act_NK 反应
    k_act_NK_base_f * NK, 0 --> act_NK
    k_act_NK_IL_12_f * NK * IL_12 / (IL_12 + k_act_NK_IL_12_m), 0 --> act_NK
    k_act_NK_IL_2_f * NK * IL_2 / (IL_2 + k_act_NK_IL_2_m), 0 --> act_NK
    k_act_NK_IFN1_f * NK * IFN1 / (IFN1 + k_act_NK_IFN1_m), 0 --> act_NK
    k_act_NK_IFN_g_f * NK * IFN_g / (IFN_g + k_act_NK_IFN_g_m), 0 --> act_NK
    k_act_NK_f * act_NK * (1 - act_NK / k_act_NK_m), 0 --> act_NK
    k_act_NK_d, act_NK --> 0

    # 27. Naive_B 反应
    k_Naive_B_f * Naive_B * (1 - Naive_B / k_Naive_B_m), 0 --> Naive_B
    k_Naive_B_Antigen_f * Naive_B * Ag_func(t) * (1 - Naive_B / k_Naive_B_m), 0 --> Naive_B
    k_Act_B_basal_f * Naive_B, Naive_B --> 0
    k_Act_B_Antigen_f * Naive_B * Ag_func(t), Naive_B --> 0
    k_Naive_B_d, Naive_B --> 0

    # 28. Act_B 反应
    k_Act_B_basal_f * Naive_B, 0 --> Act_B
    k_Act_B_Antigen_f * Naive_B * Ag_func(t), 0 --> Act_B
    k_Act_B_f * Act_B * (1 - Act_B / k_Act_B_m), 0 --> Act_B
    k_Act_B_Antigen_pro_f * Act_B * Ag_func(t) * (1 - Act_B / k_Act_B_m), 0 --> Act_B
    k_Act_B_d, Act_B --> 0

    # 29. TD_IS_B 反应
    k_TD_base_f * Act_B, 0 --> TD_IS_B
    k_TD_IL_4_f * Act_B * IL_4, 0 --> TD_IS_B
    k_TD_f * TD_IS_B * (1 - TD_IS_B / k_TD_m), 0 --> TD_IS_B
    k_TD_d, TD_IS_B --> 0

    # 30. TI_IS_B 反应
    k_TI_base_f * Act_B, 0 --> TI_IS_B
    k_TI_IFN_g_f * Act_B * IFN_g, 0 --> TI_IS_B
    k_TI_IL_10_f * Act_B * IL_10, 0 --> TI_IS_B
    k_TI_f * TI_IS_B * (1 - TI_IS_B / k_TI_m), 0 --> TI_IS_B
    k_TI_d, TI_IS_B --> 0

    # 31. IgG4 反应
    k_IgG4_TI_f * TI_IS_B, 0 --> IgG4
    k_IgG4_TD_f * TD_IS_B, 0 --> IgG4
    k_IgG4_d, IgG4 --> 0
end

# 初始条件 (Map 形式，清晰明了)
u0 = [
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
    :IgG4 => 0.1
]
tspan = (0.0, 300.0)

# ============================================================================
# 参数初始化函数 (±15%浮动边界)
# ============================================================================
"""
根据参数命名规则生成"线性空间"的初值，并设置±15%边界，然后在边界内随机取样
- param_symbols :: Vector{Symbol}
- target_data   :: Dict{Symbol, Vector{Float64}}
- u0            :: Vector 或 Dict（Catalyst 初始条件）
"""
function make_p0_lb_ub(param_symbols, target_data, u0_map; eps=1e-12)

    # 兼容 u0 是向量/字典/NamedTuple 的取值方式
    function get_u0(sym::Symbol)
        try
            # u0_map 是 Pair 列表，转换为字典查找
            for pair in u0_map
                if pair.first == sym
                    return pair.second
                end
            end
            return 0.0
        catch
            return 0.0
        end
    end

    # 从参数名 k_X_m 里解析出 X（尽量鲁棒）
    function parse_state_from_m(param::Symbol)
        s = String(param)
        # 形如 "k_IL_6_m" / "k_naive_CD4_IL_7_m" / "k_TFH_IFN1_m" 这种，最后一个 "_m" 前面的部分
        if endswith(s, "_m")
            core = s[3:end-2]  # 去掉 "k_" 和 "_m"
            # 对于 core 可能包含多个因子（如 naive_CD4_IL_7），这种 m 往往是"被调控对象"的半饱和；
            # 这里优先取最后一个 token 作为状态名候选（IL_7）
            toks = split(core, "_")
            # 尝试从末尾往前拼出一个能在 target_data 或 u0 里命中的变量名
            for i in length(toks):-1:1
                cand = Symbol(join(toks[i:end], "_"))
                if haskey(target_data, cand) || get_u0(cand) != 0.0
                    return cand
                end
            end
        end
        return nothing
    end

    n = length(param_symbols)
    p0_center = zeros(n)  # 中心值
    lb = zeros(n)
    ub = zeros(n)

    for (i, psym) in enumerate(param_symbols)
        name = String(psym)

        if endswith(name, "_d")
            # 降解速率
            p0_center[i] = 1e-2

        elseif occursin("_Antigen_", name)
            # 抗原相关参数
            p0_center[i] = 3e-2

        elseif endswith(name, "_pro_f")
            # 增殖速率
            p0_center[i] = 1e-3

        elseif endswith(name, "_m")
            # Michaelis常数：根据对应状态变量的尺度设置
            st = parse_state_from_m(psym)
            Sx = 1.0
            if st !== nothing
                if haskey(target_data, st)
                    Sx = maximum(target_data[st])
                else
                    Sx = get_u0(st)
                end
            end
            Sx = max(Sx, 1.0)  # 避免给到 0 尺度
            p0_center[i] = 1.0 * Sx

        elseif occursin("_base_f", name) || occursin("_basal_f", name)
            # 基础速率
            p0_center[i] = 1e-3

        elseif endswith(name, "_f")
            # 一般反应速率
            p0_center[i] = 1e-2

        else
            # 兜底：不要太激进
            p0_center[i] = 1e-2
        end

        # 设置 ±15% 边界
        p0_center[i] = max(p0_center[i], eps)
        lb[i] = p0_center[i] * 0.85
        ub[i] = p0_center[i] * 1.15
    end

    # 在边界内随机取样作为初始值（多起点优化准备）
    p0 = lb .+ rand(n) .* (ub .- lb)

    return p0, lb, ub
end

# 待优化参数的初始猜测
# 注意：Catalyst 会自动扫描反应式里的符号生成参数列表
p_init = [
    :k_act_CD4_CTL_antigen_f => 0.1,
    :k_act_CD4_CTL_basal_f => 0.1,
    :k_act_CD4_IFN1_f => 0.1,
    :k_act_CD4_IL_15_d => 0.1,
    :k_act_CD4_IL_15_f => 0.1,
    :k_act_CD4_IL_15_m => 0.1,
    :k_act_CD4_IL_2_d => 0.1,
    :k_act_CD4_IL_2_f => 0.1,
    :k_act_CD4_IL_2_m => 0.1,
    :k_act_CD4_IL_33_d => 0.1,
    :k_act_CD4_IL_4_d => 0.1,
    :k_act_CD4_IL_7_d => 0.1,
    :k_act_CD4_IL_7_f => 0.1,
    :k_act_CD4_IL_7_m => 0.1,
    :k_act_CD4_IFN1_d => 0.1,
    :k_act_CD4_d => 0.1,
    :k_act_CD4_f => 0.1,
    :k_act_CD4_m => 0.1,
    :k_act_CD4_mDC_f => 0.1,
    :k_act_CD4_mDC_m => 0.1,
    :k_act_NK_IFN1_d => 0.1,
    :k_act_NK_IFN1_f => 0.1,
    :k_act_NK_IFN1_m => 0.1,
    :k_act_NK_IFN_g_d => 0.1,
    :k_act_NK_IFN_g_f => 0.1,
    :k_act_NK_IFN_g_m => 0.1,
    :k_act_NK_IL_12_d => 0.1,
    :k_act_NK_IL_12_f => 0.1,
    :k_act_NK_IL_12_m => 0.1,
    :k_act_NK_IL_2_d => 0.1,
    :k_act_NK_IL_2_f => 0.1,
    :k_act_NK_IL_2_m => 0.1,
    :k_act_NK_base_f => 0.1,
    :k_act_NK_d => 0.1,
    :k_act_NK_f => 0.1,
    :k_act_NK_m => 0.1,
    :k_Act_B_Antigen_f => 0.1,
    :k_Act_B_Antigen_pro_f => 0.1,
    :k_Act_B_basal_f => 0.1,
    :k_Act_B_d => 0.1,
    :k_Act_B_f => 0.1,
    :k_Act_B_m => 0.1,
    :k_CD4_CTL_d => 0.1,
    :k_CD4_CTL_f => 0.1,
    :k_CD4_CTL_m => 0.1,
    :k_CD4_f => 0.1,
    :k_CD4_m => 0.1,
    :k_GMCSF_Th2_Antigen_f => 0.1,
    :k_GMCSF_Th2_f => 0.1,
    :k_GMCSF_act_NK_f => 0.1,
    :k_GMCSF_d => 0.1,
    :k_IFN1_CD4_CTL_m => 0.1,
    :k_IFN1_d => 0.1,
    :k_IFN1_pDC_f => 0.1,
    :k_IFN_g_CD4_CTL_f => 0.1,
    :k_IFN_g_act_NK_f => 0.1,
    :k_IFN_g_d => 0.1,
    :k_IL_10_d => 0.1,
    :k_IL_10_iTreg_f => 0.1,
    :k_IL_10_nTreg_f => 0.1,
    :k_IL_10_nTreg_mDC_m => 0.1,
    :k_IL_12_d => 0.1,
    :k_IL_12_mDC_f => 0.1,
    :k_IL_15_Antigen_f => 0.1,
    :k_IL_15_d => 0.1,
    :k_IL_15_f => 0.1,
    :k_IL_1_d => 0.1,
    :k_IL_1_mDC_f => 0.1,
    :k_IL_2_act_CD4_Antigen_f => 0.1,
    :k_IL_2_act_CD4_f => 0.1,
    :k_IL_2_d => 0.1,
    :k_IL_33_d => 0.1,
    :k_IL_33_pDC_f => 0.1,
    :k_IL_4_Th2_Antigen_f => 0.1,
    :k_IL_4_Th2_f => 0.1,
    :k_IL_4_d => 0.1,
    :k_IL_6_TFH_f => 0.1,
    :k_IL_6_d => 0.1,
    :k_IL_6_mDC_f => 0.1,
    :k_IL_6_pDC_f => 0.1,
    :k_IL_7_d => 0.1,
    :k_IL_7_f => 0.1,
    :k_IgG4_TI_f => 0.1,
    :k_IgG4_TD_f => 0.1,
    :k_IgG4_d => 0.1,
    :k_NK_d => 0.1,
    :k_NK_f => 0.1,
    :k_NK_m => 0.1,
    :k_Naive_B_Antigen_f => 0.1,
    :k_Naive_B_d => 0.1,
    :k_Naive_B_f => 0.1,
    :k_Naive_B_m => 0.1,
    :k_TD_IL_4_f => 0.1,
    :k_TD_base_f => 0.1,
    :k_TD_d => 0.1,
    :k_TD_f => 0.1,
    :k_TD_m => 0.1,
    :k_TFH_IFN1_f => 0.1,
    :k_TFH_IFN1_m => 0.1,
    :k_TFH_IL_6_d => 0.1,
    :k_TFH_IL_6_f => 0.1,
    :k_TFH_IL_6_m => 0.1,
    :k_TFH_mDC_Antigen_f => 0.1,
    :k_TFH_mDC_f => 0.1,
    :k_TFH_d => 0.1,
    :k_TFH_f => 0.1,
    :k_TFH_m => 0.1,
    :k_TFH_nTreg_m => 0.1,
    :k_TI_IFN_g_f => 0.1,
    :k_TI_IL_10_f => 0.1,
    :k_TI_base_f => 0.1,
    :k_TI_d => 0.1,
    :k_TI_f => 0.1,
    :k_TI_m => 0.1,
    :k_Th2_IL_10_m => 0.1,
    :k_Th2_IL_12_m => 0.1,
    :k_Th2_IL_33_f => 0.1,
    :k_Th2_IL_33_m => 0.1,
    :k_Th2_IL_4_f => 0.1,
    :k_Th2_IL_4_m => 0.1,
    :k_Th2_TGFbeta_m => 0.1,
    :k_Th2_d => 0.1,
    :k_Th2_f => 0.1,
    :k_Th2_m => 0.1,
    :k_TGFbeta_CD4_CTL_f => 0.1,
    :k_TGFbeta_d => 0.1,
    :k_TGFbeta_iTreg_f => 0.1,
    :k_TGFbeta_nTreg_f => 0.1,
    :k_TGFbeta_nTreg_mDC_m => 0.1,
    # k_act_CD4_IL_33_f 未在反应网络中使用，已删除
    :k_iTreg_IL_10_f => 0.1,
    :k_iTreg_IL_10_m => 0.1,
    :k_iTreg_IL_1_m => 0.1,
    :k_iTreg_TGFbeta_f => 0.1,
    :k_iTreg_TGFbeta_m => 0.1,
    :k_iTreg_d => 0.1,
    :k_iTreg_f => 0.1,
    :k_iTreg_m => 0.1,
    :k_iTreg_mDC_d => 0.1,
    :k_iTreg_mDC_f => 0.1,
    :k_mDC_Antigen_f => 0.1,
    :k_mDC_GMCSF_f => 0.1,
    :k_mDC_GMCSF_d => 0.1,
    :k_mDC_GMCSF_m => 0.1,
    :k_mDC_IL_10_m => 0.1,
    :k_mDC_d => 0.1,
    :k_mDC_f => 0.1,
    :k_mDC_m => 0.1,
    :k_naive_CD4_IL_15_d => 0.1,
    :k_naive_CD4_IL_15_f => 0.1,
    :k_naive_CD4_IL_15_m => 0.1,
    :k_naive_CD4_IL_7_d => 0.1,
    :k_naive_CD4_IL_7_f => 0.1,
    :k_naive_CD4_IL_7_m => 0.1,
    :k_naive_CD4_d => 0.1,
    :k_nDC_d => 0.1,
    :k_nDC_f => 0.1,
    :k_nDC_m => 0.1,
    :k_nTreg_d => 0.1,
    :k_nTreg_m => 0.1,
    :k_nTreg_mDC_f => 0.1,
    :k_nTreg_mDC_m => 0.1,
    :k_pDC_Antigen_f => 0.1,
    :k_pDC_d => 0.1,
    :k_pDC_f => 0.1,
    :k_pDC_m => 0.1,
]

# 生成 ODEProblem
# 使用 QNDF (BDF求解器) 处理极端刚性生物化学方程
oprob = ODEProblem(rn, u0, tspan, p_init)

# 提取参数名和值（用于优化）
param_symbols = [pair.first for pair in p_init]
param_values = [pair.second for pair in p_init]

# 使用±15%浮动边界初始化函数（线性空间）
p0_lin, lb_lin, ub_lin = make_p0_lb_ub(param_symbols, target_data, u0)

# 转换到对数空间用于优化（初值）
p0_log = log.(p0_lin)

# ============================================================================
# 从初值派生一阶段/二阶段的边界（对数空间）
# ============================================================================
# 一阶段采样范围：围绕初值 ±STAGE1_SAMPLE_FRAC
stage1_lb_lin = max.(p0_lin .* (1 - STAGE1_SAMPLE_FRAC), eps(Float64))
stage1_ub_lin = max.(p0_lin .* (1 + STAGE1_SAMPLE_FRAC), eps(Float64))
stage1_lb_log = log.(stage1_lb_lin)
stage1_ub_log = log.(stage1_ub_lin)

# 二阶段优化范围：初值 x1e-2 ~ x1e2
stage2_lb_lin = max.(p0_lin .* STAGE2_LB_MULT, eps(Float64))
stage2_ub_lin = max.(p0_lin .* STAGE2_UB_MULT, eps(Float64))
stage2_lb_log = log.(stage2_lb_lin)
stage2_ub_log = log.(stage2_ub_lin)

println("\n参数初始化完成 (用于派生一阶段/二阶段边界):")
println("  - 参数总数: $(length(param_symbols))")
println("  - 初值线性空间范围: [$(minimum(p0_lin)), $(maximum(p0_lin))]")
println("  - 一阶段采样范围: ±$(STAGE1_SAMPLE_FRAC*100)%")
println("  - 二阶段优化范围: x$(STAGE2_LB_MULT) ~ x$(STAGE2_UB_MULT)")

# ============================================================================
# 3. 损失函数设计 (时间窗加权 L2 Loss + 归一化曲率正则)
# ============================================================================

# 数据损失时间窗加权：50 < t < 100 段权重 1.5（用于“倒逼”该阶段更平）
WINDOW_T_LO = 50.0
WINDOW_T_HI = 100.0
WINDOW_MULT = 1.5

# smooth 正则：二阶差分曲率（归一化后乘 data_loss）
SMOOTH_EPS = 1e-3

# 生成每个采样时间点的权重向量
function time_weights(ts)
    w = ones(length(ts))
    for (i, t) in enumerate(ts)
        if (t > WINDOW_T_LO) && (t < WINDOW_T_HI)
            w[i] = WINDOW_MULT
        end
    end
    return w
end

TIME_WEIGHTS = time_weights(target_time)


# 计算平滑惩罚：在 target_time 上对每个变量的离散二阶差分做 L2
function smoothness_penalty(sol, var_syms::Vector{Symbol})
    pen = 0.0
    for v in var_syms
        # 某些符号（如 Antigen）可能不在 ODESolution 的 timeseries index 中，直接跳过
        y = try
            sol[v]
        catch
            continue
        end
        
        # 检查 y 是否包含 NaN 或 Inf
        if any(!isfinite, y)
            continue
        end
        
        n = length(y)
        if n < 3
            continue
        end
        for i in 2:(n-1)
            d2 = y[i+1] - 2y[i] + y[i-1]
            # 检查 d2 是否有效
            if isfinite(d2)
                pen += d2*d2
            end
        end
    end
    return pen
end

# 注意：不再需要 SMOOTH_0，smooth_norm 现在使用 log(smooth_raw)

# 全局标志：用于跟踪是否是第一次调用
global first_loss_call = true

# 全局变量：用于跟踪当前优化组的迭代次数和最大迭代数
global current_opt_iter = 0
global current_opt_maxiters = FINE_MAXITERS

# 辅助函数：计算稳态损失
function compute_steady_state_loss(prob, sol)
    """
    计算稳态损失：log(||rhs(t=0)||+1.1) * log(||rhs(t=250)||+1.1)
    返回：(乘积, t=0的值, t=250的值)
    """
    try
        rhs_func = prob.f
        
        # 计算 t=0 时刻的 RHS（使用初始条件）
        u0_vec = prob.u0
        du0 = similar(u0_vec)
        rhs_func(du0, u0_vec, prob.p, 0.0)
        rhs_norm_t0 = sqrt(sum(abs2, du0))
        steady_loss_t0 = log(rhs_norm_t0 + 1.1)
        
        # 计算 t=250 时刻的 RHS（使用解在 t=250 的值）
        u_t250 = sol(250.0)
        du_t250 = similar(u_t250)
        rhs_func(du_t250, u_t250, prob.p, 250.0)
        rhs_norm_t250 = sqrt(sum(abs2, du_t250))
        steady_loss_t250 = log(rhs_norm_t250 + 1.1)
        
        # 返回三个值：乘积、t=0的值、t=250的值
        return (steady_loss_t0 * steady_loss_t250, steady_loss_t0, steady_loss_t250)
    catch e
        # 如果计算失败，返回默认值（不影响总损失）
        return (1.0, 0.0, 0.0)
    end
end

function loss_function(p, constants = nothing)
    # p: 当前优化迭代的参数值向量（对数空间）
    # constants: 优化框架要求的第二个参数（这里不使用）

    # 1. 更新模型参数
    linear_params = exp.(p)  # 对数 -> 线性
    p_dict = [param_symbols[i] => linear_params[i] for i in 1:length(p)]
    new_prob = remake(oprob, p=p_dict)

    # 2. 求解 ODE
    sol = solve(new_prob, QNDF(), saveat=target_time,
                abstol=ODE_ABSTOL, reltol=ODE_RELTOL,
                verbose=false, maxiters=ODE_MAXITERS)

    if sol.retcode != :Success
        return 1e12
    end

    # 3. 数据拟合项（log10 L2）
    data_loss = 0.0
    epsilon = 1e-10
    
    # 检查是否优化进度过半
    # 注意：current_opt_iter 在 callback 中更新，可能不是每次迭代都更新，但可以提供近似值
    global current_opt_iter, current_opt_maxiters
    is_halfway = (current_opt_iter >= current_opt_maxiters / 2)
    
    # 如果过半，计算每个变量的相对误差，用于加权
    var_errors = Dict{Symbol, Float64}()
    if is_halfway
        for var_name in keys(target_data)
            try
                pred_values = sol[var_name]
                obs_values = target_data[var_name]
                
                # 计算相对误差（使用 log10 空间）
                log_pred = log10.(max.(pred_values, epsilon))
                log_obs = log10.(max.(obs_values, epsilon))
                rel_error = sum(abs2.(log_pred .- log_obs)) / (sum(abs2.(log_obs)) + epsilon)
                var_errors[var_name] = rel_error
            catch
                var_errors[var_name] = 0.0
            end
        end
        
        # 计算误差的中位数，用于确定哪些变量误差较大
        if !isempty(var_errors)
            median_error = median([v for v in values(var_errors) if isfinite(v)])
            # 误差大于中位数的变量，权重增加 2 倍
            error_threshold = max(median_error, 1e-6)
        else
            error_threshold = 1e-6
        end
    else
        error_threshold = 0.0
    end

    for var_name in keys(target_data)
        try
            pred_values = sol[var_name]
            obs_values = target_data[var_name]

            log_pred = log10.(max.(pred_values, epsilon))
            log_obs = log10.(max.(obs_values, epsilon))
            
            # 计算基础损失
            var_loss = sum(TIME_WEIGHTS .* abs2.(log_pred .- log_obs))
            
            # 如果优化进度过半且该变量误差较大，增加权重
            if is_halfway && haskey(var_errors, var_name) && var_errors[var_name] > error_threshold
                var_loss *= 2.0  # 误差较大的变量权重增加 2 倍
            end
            
            data_loss += var_loss
        catch
            continue
        end
    end

    # 4. 平滑正则：二阶差分曲率（使用 log 尺度）
    smooth_loss = smoothness_penalty(sol, collect(keys(target_data)))
    # 使用 log(smooth_raw) 作为归一化，避免除以基准值
    smooth_loss_normalized = log(max(smooth_loss, SMOOTH_EPS))

    # 5. 稳态损失：计算 t=0 和 t=250 时刻的 RHS 范数
    steady_state_loss, _, _ = compute_steady_state_loss(new_prob, sol)

    total = (smooth_loss_normalized + SMOOTH_EPS) * data_loss * steady_state_loss

    return total
end

# ============================================================================
# 4. 多起点筛选 (改为：纯采样筛选)
# ============================================================================
println("\n" * "="^70)
println("阶段 1: 高通量采样筛选 (High-Throughput Sampling Screening)")
println("="^70)

# ---------------- 配置 ----------------
# 大幅增加采样数，因为我们不再对每个点做优化，只是"看一眼"
n_starts = 1000
n_top_candidates = N_TOP_CANDIDATES # 保持不变

# ---------------- 1. 智能采样 ----------------
println("正在生成 $n_starts 组拉丁超立方采样点...")
# 依然使用 LHS，因为它比纯随机更均匀
sampler = LatinHypercubeSample()
initial_samples_matrix = QuasiMonteCarlo.sample(n_starts, stage1_lb_log, stage1_ub_log, sampler)

# ---------------- 2. 定义筛选用损失函数 ----------------
# 为筛选阶段创建一个简化的损失函数，使用更宽松的ODE参数
function screening_loss_function(p, constants = nothing)
    linear_params = exp.(p)
    p_dict = [param_symbols[i] => linear_params[i] for i in 1:length(p)]
    new_prob = remake(oprob, p=p_dict)

    sol = solve(new_prob, QNDF(), saveat=target_time,
                abstol=1e-4, reltol=1e-4,
                maxiters=10000,
                verbose=false)

    if sol.retcode != :Success
        return (1e12, 1e12, 1e12, 1e12)
    end

    # 数据拟合项
    data_loss = 0.0
    epsilon = 1e-10

    for var_name in keys(target_data)
        try
            pred_values = sol[var_name]
            obs_values = target_data[var_name]

            log_pred = log10.(max.(pred_values, epsilon))
            log_obs = log10.(max.(obs_values, epsilon))

            data_loss += sum(abs2, log_pred .- log_obs)
        catch
            continue
        end
    end

    # 平滑项：二阶差分曲率（raw）及其归一化
    smooth_loss = smoothness_penalty(sol, collect(keys(target_data)))
    smooth_norm = log(max(smooth_loss, SMOOTH_EPS))
    
    # 稳态损失
    steady_state_loss, _, _ = compute_steady_state_loss(new_prob, sol)

    total = (smooth_norm + SMOOTH_EPS) * data_loss * steady_state_loss

    return (total, data_loss, smooth_loss, smooth_norm)
end

# 为二阶段定义优化函数（使用完整的 loss_function 和 AD）
# 记录 loss 曲线（全程累计，跨 15 个候选）
loss_trace_total = Float64[]
loss_trace_data = Float64[]
loss_trace_smooth_raw = Float64[]
loss_trace_smooth_norm = Float64[]
loss_trace_group = Int[]
loss_trace_iter = Int[]

# 分解并记录一次 loss（用于 callback）
function compute_loss_terms(p_log)
    linear_params = exp.(p_log)
    p_dict = [param_symbols[i] => linear_params[i] for i in 1:length(p_log)]
    new_prob = remake(oprob, p=p_dict)

    sol = solve(new_prob, QNDF(), saveat=target_time,
                abstol=ODE_ABSTOL, reltol=ODE_RELTOL,
                verbose=false, maxiters=ODE_MAXITERS)

    if sol.retcode != :Success
        return (1e12, 1e12, 1e12, 1e12)
    end

    data_loss = 0.0
    epsv = 1e-10
    for vn in keys(target_data)
        try
            ypred = sol[vn]
            yobs = target_data[vn]
            data_loss += sum(TIME_WEIGHTS .* abs2.(log10.(max.(ypred, epsv)) .- log10.(max.(yobs, epsv))))
        catch
            continue
        end
    end

    smooth_loss = smoothness_penalty(sol, collect(keys(target_data)))
    smooth_norm = log(max(smooth_loss, SMOOTH_EPS))
    
    # 稳态损失
    steady_state_loss, _, _ = compute_steady_state_loss(new_prob, sol)

    total = (smooth_norm + SMOOTH_EPS) * data_loss * steady_state_loss
    return (total, data_loss, smooth_loss, smooth_norm)
end

# 优化函数（允许 callback 记录 loss 曲线）
optf_stage2 = OptimizationFunction(loss_function, Optimization.AutoForwardDiff())

# ---------------- 2. 并行计算 Loss ----------------
println("开始并行计算 $n_starts 组参数的初始 Loss...")
println("（此阶段不做优化，仅筛选 '潜力股'）")

# 用于存储结果的线程安全数组
# 格式: (total, data_loss, smooth_raw, smooth_norm, params)
results = Vector{Tuple{Float64, Float64, Float64, Float64, Vector{Float64}}}(undef, n_starts)

# 进度条（nohup/重定向下会刷屏；这里改为每10%打印一次摘要）
progress_step1 = max(1, n_starts ÷ 10)

Threads.@threads for i in 1:n_starts
    # 取出第 i 组参数 (对数空间)
    current_p = initial_samples_matrix[:, i]

    # 直接计算 Loss，不进行任何优化 (solve)
    # 注意：这里直接调用 screening_loss_function
    # 为了防止某些极差的参数导致 ODE 卡死，我们在 screening_loss_function 内部
    # 使用了较小的 maxiters (10000 vs 1_000_000)
    total, data_loss, smooth_raw, smooth_norm = screening_loss_function(current_p)

    results[i] = (total, data_loss, smooth_raw, smooth_norm, current_p)

    if (i % progress_step1) == 0 || i == n_starts
        # 多线程下可能会有少量重复打印，但不会像进度条那样刷屏
        pct = round(100 * i / n_starts; digits=1)
        println("筛选进度: $pct%  (i=$i/$n_starts)")
    end
end

# ---------------- 3. 排序与提取 ----------------
# 过滤掉计算失败的 (Loss 极大的)
valid_results = filter(x -> x[1] < 1e11, results)

println("\n采样完成！有效样本数: $(length(valid_results)) / $n_starts")

if isempty(valid_results)
    error("所有采样点的 ODE 求解都失败了！请检查采样范围是否合理，或放宽 ODE 求解器的容差。")
end

# 排序（按 total score）
sort!(valid_results, by=x->x[1])

# 提取前 N 名
# 元组格式: (total, data_loss, smooth_raw, smooth_norm, params)
top_candidates = valid_results[1:min(n_top_candidates, length(valid_results))]

println("\nTop $n_top_candidates 候选参数 (total = (log(smooth_raw) + $(SMOOTH_EPS)) * data_loss):")
for (i, (total, data_loss, smooth_raw, smooth_norm, _)) in enumerate(top_candidates)
    println("  #$i: total=$(total)  data=$(data_loss)  smooth_raw=$(smooth_raw)  smooth_norm=$(smooth_norm)")
end

println("\n" * "="^70)
println("阶段 2: 深度精修 (Full L-BFGS Optimization)")
println("="^70)

# 存储所有优化结果
optimized_results = Vector{Any}()

progress_step2 = max(1, n_top_candidates ÷ 10)

for (idx, cand) in enumerate(top_candidates)
    init_loss, init_data_loss, init_smooth_raw, init_smooth_norm, init_params = cand

    println("\n优化第 $idx/$(n_top_candidates) 组参数...")
    println("  初始 total: $(init_loss)")
    println("    data=$(init_data_loss)  smooth_raw=$(init_smooth_raw)  smooth_norm=$(init_smooth_norm)")
    
    # 计算初始稳态损失值
    try
        init_prob = remake(oprob, p=[param_symbols[i] => exp(init_params[i]) for i in 1:length(init_params)])
        init_sol = solve(init_prob, QNDF(), saveat=target_time, abstol=ODE_ABSTOL, reltol=ODE_RELTOL, verbose=false, maxiters=ODE_MAXITERS)
        if init_sol.retcode == :Success || string(init_sol.retcode) == "Success"
            _, init_rhs0, init_rhs250 = compute_steady_state_loss(init_prob, init_sol)
            println("    RHS稳态控制: log(||RHS0||+1.1)=$(init_rhs0)  log(||RHS250||+1.1)=$(init_rhs250)")
        end
    catch
        # 忽略错误
    end

    # 创建优化问题
    optprob = OptimizationProblem(optf_stage2, init_params, lb=stage2_lb_log, ub=stage2_ub_log)

    # 记录当前组号
    current_group = idx
    
    # 重置迭代计数器（每个优化组开始时重置）
    global current_opt_iter = 0
    global current_opt_maxiters = FINE_MAXITERS
    
    # 用于跟踪当前组内的迭代次数（从1开始）
    local_group_iter = 0
    
    # 定义 callback 记录 loss 曲线
    function loss_callback(state, args...)
        # 当前组内的迭代次数（从1开始）
        local_group_iter += 1
        # 全局迭代次数（用于记录）
        global_iter = length(loss_trace_iter) + 1
        p_curr = state.u  # 当前参数（对数空间）
        
        # 更新全局迭代计数器（用于 loss_function 中的加权逻辑）
        # 注意：这里使用当前组内的迭代次数，而不是全局迭代次数
        global current_opt_iter = local_group_iter
        
        # 计算各项 loss
        total, data, smooth, smooth_norm = compute_loss_terms(p_curr)
        
        # 保存到全局数组
        push!(loss_trace_total, total)
        push!(loss_trace_data, data)
        push!(loss_trace_smooth_raw, smooth)
        push!(loss_trace_smooth_norm, smooth_norm)
        push!(loss_trace_group, current_group)
        push!(loss_trace_iter, global_iter)
        
        return false  # 不提前终止优化
    end

    # 运行LBFGS优化（带回调）
    result = solve(optprob, OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
                   maxiters=FINE_MAXITERS,
                   x_abstol=FINE_X_ABSTOL,
                   f_reltol=FINE_F_RELTOL,
                   callback=loss_callback,
                   show_trace=false)
    
    # 打印优化器状态信息
    println("  优化器状态: $(result.retcode)")
    # 通过 callback 记录的迭代次数来估算实际迭代次数
    current_group_actual_iters = sum(loss_trace_group .== current_group)
    println("  记录到的迭代次数: $current_group_actual_iters / $(FINE_MAXITERS)")
    
    # 尝试获取优化器的内部迭代次数
    try
        if hasproperty(result, :original) && result.original !== nothing
            orig = result.original
            if hasproperty(orig, :iterations)
                println("  优化器内部迭代次数: $(orig.iterations)")
            end
        end
    catch
        # 忽略错误
    end

    # 重新计算各项分解（使用最终参数）用于打印
    final_total = result.objective
    final_data_loss = NaN
    final_smooth_raw = NaN
    final_smooth_norm = NaN

    lp = exp.(result.u)
    pd = [param_symbols[i] => lp[i] for i in 1:length(lp)]
    pr = remake(oprob, p=pd)
    s = solve(pr, QNDF(), saveat=target_time, abstol=ODE_ABSTOL, reltol=ODE_RELTOL, verbose=false, maxiters=ODE_MAXITERS)

    # 检查 ODE 求解是否成功（使用更宽松的检查，因为某些情况下 retcode 可能是其他形式）
    if s.retcode == :Success || string(s.retcode) == "Success"
        epsv = 1e-10
        dl = 0.0
        for vn in keys(target_data)
            try
                ypred = s[vn]
                yobs = target_data[vn]
                
                # 检查 ypred 是否包含 NaN 或 Inf
                if any(!isfinite, ypred)
                    # 如果包含 NaN/Inf，跳过这个变量，避免污染 dl
                    continue
                end
                
                log_pred = log10.(max.(ypred, epsv))
                log_obs = log10.(max.(yobs, epsv))
                
                # 检查计算结果是否有效
                if any(!isfinite, log_pred) || any(!isfinite, log_obs)
                    continue
                end
                
                dl += sum(TIME_WEIGHTS .* abs2.(log_pred .- log_obs))
            catch
                continue
            end
        end

        # 计算稳态损失值（在所有情况下都尝试计算）
        final_rhs0 = NaN
        final_rhs250 = NaN
        try
            _, final_rhs0, final_rhs250 = compute_steady_state_loss(pr, s)
        catch
            # 忽略错误
        end
        
        # 检查 dl 是否有效
        if !isfinite(dl) || dl == 0.0
            # 如果 dl 无效或为 0，使用优化器返回的 objective 作为参考
            println("  优化后 total: $(final_total)")
            println("    data=$(final_data_loss)  smooth_raw=$(final_smooth_raw)  smooth_norm=$(final_smooth_norm)  (warning: computed values invalid, using optimizer objective)")
            if isfinite(final_rhs0) && isfinite(final_rhs250)
                println("    RHS稳态控制: log(||RHS0||+1.1)=$(final_rhs0)  log(||RHS250||+1.1)=$(final_rhs250)")
            end
        else
            sp = smoothness_penalty(s, collect(keys(target_data)))
            
            # 检查 sp 是否有效
            if !isfinite(sp)
                println("  优化后 total: $(final_total)")
                println("    data=$(dl)  smooth_raw=NaN  smooth_norm=NaN  (warning: smoothness_penalty returned NaN/Inf)")
                if isfinite(final_rhs0) && isfinite(final_rhs250)
                    println("    RHS稳态控制: log(||RHS0||+1.1)=$(final_rhs0)  log(||RHS250||+1.1)=$(final_rhs250)")
                end
            else
                sn = log(max(sp, SMOOTH_EPS))
                
                # 检查 sn 是否有效
                if !isfinite(sn)
                    println("  优化后 total: $(final_total)")
                    println("    data=$(dl)  smooth_raw=$(sp)  smooth_norm=NaN  (warning: smooth_norm is NaN/Inf)")
                    if isfinite(final_rhs0) && isfinite(final_rhs250)
                        println("    RHS稳态控制: log(||RHS0||+1.1)=$(final_rhs0)  log(||RHS250||+1.1)=$(final_rhs250)")
                    end
                else
                    recomputed_total = (sn + SMOOTH_EPS) * dl
                    
                    # 检查 recomputed_total 是否有效
                    if !isfinite(recomputed_total)
                        println("  优化后 total: $(final_total)")
                        println("    data=$(dl)  smooth_raw=$(sp)  smooth_norm=$(sn)  (warning: recomputed_total is NaN/Inf, using optimizer objective)")
                        if isfinite(final_rhs0) && isfinite(final_rhs250)
                            println("    RHS稳态控制: log(||RHS0||+1.1)=$(final_rhs0)  log(||RHS250||+1.1)=$(final_rhs250)")
                        end
                    else
                        final_data_loss = dl
                        final_smooth_raw = sp
                        final_smooth_norm = sn
                        final_total = recomputed_total
                        
                        println("  优化后 total: $(final_total)")
                        println("    data=$(final_data_loss)  smooth_raw=$(final_smooth_raw)  smooth_norm=$(final_smooth_norm)")
                        if isfinite(final_rhs0) && isfinite(final_rhs250)
                            println("    RHS稳态控制: log(||RHS0||+1.1)=$(final_rhs0)  log(||RHS250||+1.1)=$(final_rhs250)")
                        end
                    end
                end
            end
        end
    else
        println("  优化后 total: $(final_total)")
        println("    data=NaN  smooth_raw=NaN  smooth_norm=NaN  (warning: final solve failed: $(s.retcode))")
    end

    println("  改进: $(init_loss - final_total)")

    push!(optimized_results, result)

    if (idx % progress_step2) == 0 || idx == n_top_candidates
        pct = round(100 * idx / n_top_candidates; digits=1)
        println("二阶段进度: $pct%  (idx=$idx/$n_top_candidates)")
    end
end

# 找出最终最优结果
best_result_idx = argmin([r.objective for r in optimized_results])
best_result = optimized_results[best_result_idx]

println("\n" * "="^70)
println("优化完成！")
println("="^70)
println("最佳结果来自第 $best_result_idx 组:")
println("  最终 Loss: $(best_result.objective)")
println("  参数范围（线性空间）: [$(minimum(exp.(best_result.u))), $(maximum(exp.(best_result.u)))]")

# ============================================================================
# 保存优化结果（防止后续绘图/求解报错导致结果丢失）
# ============================================================================
println("\n保存二阶段优化结果到文件...")

# 组织结果表：每行是一组精修结果
n_params = length(param_symbols)
param_names = String.(param_symbols)

df_params = DataFrame()
df_params[!, :candidate_idx] = collect(1:length(optimized_results))
df_params[!, :objective] = [r.objective for r in optimized_results]

for (j, pname) in enumerate(param_names)
    df_params[!, Symbol(pname)] = [exp(r.u[j]) for r in optimized_results]
end

CSV.write("stage2_optimized_params.tsv", df_params, delim='\t')

# 保存最佳参数（单独一份，方便直接用）
best_df = df_params[df_params.candidate_idx .== best_result_idx, :]
CSV.write("best_params.tsv", best_df, delim='\t')

# 保存 loss 曲线数据与图片
if !isempty(loss_trace_total)
    df_loss = DataFrame(
        iter = loss_trace_iter,
        group = loss_trace_group,
        total = loss_trace_total,
        data_loss = loss_trace_data,
        smooth_raw = loss_trace_smooth_raw,
        smooth_norm = loss_trace_smooth_norm,
    )
    CSV.write("loss_curve.tsv", df_loss, delim='\t')

    p_loss = plot(df_loss.iter, df_loss.total, label="total", lw=2)
    plot!(p_loss, df_loss.iter, df_loss.data_loss, label="data_loss", lw=2)
    plot!(p_loss, df_loss.iter, df_loss.smooth_raw, label="smooth_raw", lw=2)
    plot!(p_loss, df_loss.iter, df_loss.smooth_norm, label="smooth_norm", lw=2)
    xlabel!(p_loss, "iteration")
    ylabel!(p_loss, "value")
    title!(p_loss, "Loss terms across stage-2 refinements")
    savefig(p_loss, "loss_curve.png")
    println("已保存: loss_curve.tsv")
    println("已保存: loss_curve.png")
end

println("已保存: stage2_optimized_params.tsv (全部精修结果)")
println("已保存: best_params.tsv (最佳结果)")

# ============================================================================
# 5. 结果可视化 (按变量顺序分子图；叠加二阶段精修的 N_TOP_CANDIDATES 组参数)
# ============================================================================
println("\n" * "="^70)
println("生成二阶段精修参数集预测的轨线图")
println("="^70)

# 变量顺序：按数据里出现/映射后的模型变量名顺序绘制
ordered_variables = Symbol.(model_variable_names)

# 网格布局：4列
n_cols = 4
n_rows = ceil(Int, length(ordered_variables) / n_cols)

# 生成二阶段所有参数集的求解结果（与 optimized_results 一一对应）
all_solutions = Vector{Any}(undef, length(optimized_results))
for (i, result) in enumerate(optimized_results)
    linear_params = exp.(result.u)
    p_dict = [param_symbols[j] => linear_params[j] for j in 1:length(linear_params)]
    prob = remake(oprob, p=p_dict)
    all_solutions[i] = solve(prob, QNDF(), saveat=target_time, abstol=ODE_ABSTOL, reltol=ODE_RELTOL)
end

# 颜色：区分每一组参数（循环使用 tab20）
colors = palette(:tab20)

plots_array = []

for var_sym in ordered_variables
    p = plot(
        title=String(var_sym),
        titlefontsize=9, legendfontsize=6, guidefontsize=8,
        tickfontsize=7, grid=true, gridalpha=0.3
    )

    # 叠加 N_TOP_CANDIDATES 组参数的预测轨线（不同颜色）
    for (i, sol) in enumerate(all_solutions)
        plot!(
            p, sol, vars=[var_sym],
            label=(i == 1 ? "Model (N starts)" : ""),
            lw=1.2, alpha=0.7,
            color=colors[mod1(i, length(colors))]
        )
    end

    # 叠加实验数据
    if haskey(target_data, var_sym)
        scatter!(
            p, target_time, target_data[var_sym],
            label="Exp Data", color=:black,
            markersize=3, alpha=0.85,
            markerstrokewidth=0
        )
    end

    xlabel!(p, "Time")
    ylabel!(p, "Value")

    push!(plots_array, p)
end

# 填充空白子图
for i in (length(plots_array)+1):(n_rows * n_cols)
    push!(plots_array, plot(legend=false, grid=false, showaxis=false, ticks=false))
end

final_plot = plot(
    plots_array...,
    layout=(n_rows, n_cols),
    size=(1600, 300 * n_rows),
    plot_title="IgG4-RD Stage-2 Refinement: $(length(optimized_results)) Parameter Sets",
    plot_titlefontsize=14
)

savefig(final_plot, "optimization_result.png")
println("已保存完整结果图: optimization_result.png")


println("\n" * "="^70)
println("所有任务完成！")
println("="^70)