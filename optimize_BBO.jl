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
model_variable_names = Set{String}()
for col_name in names(df)
    if col_name != "Time"
        model_name = get(name_mapping, col_name, col_name)
        push!(model_variable_names, model_name)
    end
end
model_variable_names = collect(model_variable_names)

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
# 3. 损失函数设计 (对数尺度 L2 Loss)
# ============================================================================
# 新的损失函数特点：
# 1. 包含所有变量（不仅是IgG4）
# 2. 使用对数尺度损失：||log10(y) - log10(y_pred)||^2
# 3. 自动处理不同数量级的变量

# 全局标志：用于跟踪是否是第一次调用
global first_loss_call = true

function loss_function(p, constants = nothing)
    # p: 当前优化迭代的参数值向量（对数空间）
    # constants: 优化框架要求的第二个参数（这里不使用）

    # 1. 更新模型参数
    # 将对数空间的参数转换为线性空间，然后转换为符号对格式
    linear_params = exp.(p)  # 对数 -> 线性
    p_dict = [param_symbols[i] => linear_params[i] for i in 1:length(p)]
    # remake 是 SciML 的零拷贝参数更新机制，极快
    new_prob = remake(oprob, p=p_dict)

    # 2. 求解 ODE（抑制警告信息）
    # saveat=target_time: 只保存有数据的时间点，节省内存
    # 使用 QNDF (BDF求解器) 处理极端刚性问题
    # 禁用警告：随机参数中有很多NaN/dt下溢是正常的
    sol = solve(new_prob, QNDF(), saveat=target_time,
                abstol=ODE_ABSTOL, reltol=ODE_RELTOL,
                verbose=false, maxiters=ODE_MAXITERS)

    # 3. 惩罚项：如果求解失败（参数导致发散），返回巨大 Loss
    if sol.retcode != :Success
        return 1e12
    end

    # 4. 计算方差标准化的 L2 Loss (对所有变量)
    # Loss = sum over all variables and time points of ((pred - obs)^2 / var)
    # 这样可以确保不同尺度的变量对损失的贡献是平衡的

    loss = 0.0
    skipped_count = 0

    # 对每个变量计算对数尺度 L2 损失
    epsilon = 1e-10  # 避免 log10(0)
    
    for var_name in keys(target_data)
        # 使用 try-catch 来处理变量访问
        try
            pred_values = sol[var_name]
            obs_values = target_data[var_name]

            # 对数尺度 L2 损失: ||log10(y) - log10(y_pred)||^2
            # 添加 epsilon 避免 log10(0) 或 log10(负数)
            log_pred = log10.(max.(pred_values, epsilon))
            log_obs = log10.(max.(obs_values, epsilon))
            
            loss += sum(abs2, log_pred .- log_obs)
        catch
            # 如果变量不存在，跳过
            skipped_count += 1
            continue
        end
    end

    return loss
end

# ============================================================================
# 4. 多起点优化设置 (智能采样 -> 粗筛 -> 精修)
# ============================================================================
println("\n" * "="^70)
println("阶段 1: 多起点并行粗筛 (Multi-start Optimization Screening)")
println("="^70)

# ---------------- 配置 ----------------
n_starts = N_STARTS
n_top_candidates = N_TOP_CANDIDATES
# 注意：BBO 是按时间或函数评估次数控制的，不是 maxiters
# 这里我们给它一个时间限制或者评估次数限制
maxiters_bbo = 2000  # 每次尝试只跑 2000 次函数评估，非常快

# ---------------- 1. 智能采样 ----------------
# (这部分保持不变，用于生成初始种群，虽然 BBO 自带种群生成，但保留你的逻辑也可以)
println("正在生成 $n_starts 组拉丁超立方采样点...")
sampler = LatinHypercubeSample()
initial_samples_matrix = QuasiMonteCarlo.sample(n_starts, stage1_lb_log, stage1_ub_log, sampler)

# ---------------- 2. 定义优化函数 ----------------
# 【修改点 A】: 阶段1 不需要自动微分，因为我们用无导数算法
# 如果必须传 AD 参数，AutoForwardDiff 在这里不会被调用计算梯度，所以没性能损耗
# 但为了安全，二阶段我们还是需要 AD 的，所以这里定义两套
optf_stage1 = OptimizationFunction(loss_function) # 无需 AD
optf_stage2 = OptimizationFunction(loss_function, Optimization.AutoForwardDiff()) # 二阶段用

# ---------------- 3. 运行并行筛选 ----------------
println("开始并行运行 $n_starts 个初步优化任务 (算法: 差分进化 DE)...")

screen_solutions = Vector{Any}(undef, n_starts)
p = Progress(n_starts, desc="一阶段并行优化进度: ", showspeed=true)

Threads.@threads for i in 1:n_starts
    # 【修改点 B】: 使用 OptimizationBBO
    # BBO 对初值不敏感，它主要看边界，但我们可以把 initial_sample 作为种群的一部分

    # 这里的 u0 只是占位符，BBO 主要在 lb/ub 之间搜
    local_prob = OptimizationProblem(optf_stage1, initial_samples_matrix[:, i],
                                   lb=stage1_lb_log, ub=stage1_ub_log)

    # 使用差分进化算法 (Differential Evolution)
    local_sol = solve(local_prob,
                     BBO_adaptive_de_rand_1_bin_radiuslimited(),
                     maxiters=maxiters_bbo)

    screen_solutions[i] = local_sol
    next!(p)
end

# ---------------- 4. 提取并排序结果 ----------------
screen_results = Vector{Tuple{Float64, Vector{Float64}}}()
for sol in screen_solutions
    push!(screen_results, (sol.objective, copy(sol.u)))
end
sort!(screen_results, by=x->x[1])
top_candidates = screen_results[1:n_top_candidates]

println("\n" * "="^70)
println("阶段 2: 深度精修 (Full L-BFGS Optimization)")
println("="^70)

# 存储所有优化结果
optimized_results = Vector{Any}()

# 创建二阶段进度条
p2 = Progress(n_top_candidates, desc="二阶段深度优化进度: ", showspeed=true)

for (idx, (init_loss, init_params)) in enumerate(top_candidates)
    println("\n优化第 $idx/$(n_top_candidates) 组参数...")
    println("  初始 Loss: $init_loss")

    # 创建优化问题
    optprob = OptimizationProblem(optf_stage2, init_params, lb=stage2_lb_log, ub=stage2_ub_log)

    # 运行LBFGS优化
    result = solve(optprob, OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()),
                   maxiters=FINE_MAXITERS,
                   x_abstol=FINE_X_ABSTOL,
                   f_reltol=FINE_F_RELTOL)

    println("  优化后 Loss: $(result.objective)")
    println("  改进: $(init_loss - result.objective)")

    push!(optimized_results, result)
    next!(p2)
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
# 5. 结果可视化 (绘制15组参数的预测轨线)
# ============================================================================
println("\n" * "="^70)
println("生成15组参数预测的轨线图")
println("="^70)

# 定义所有30个状态变量
all_variables = [
    :nDC, :mDC, :pDC, :naive_CD4, :act_CD4, :Th2, :iTreg,
    :CD4_CTL, :nTreg, :TFH, :NK, :act_NK, :Naive_B, :Act_B, :TD_IS_B,
    :TI_IS_B, :GMCSF, :IL_33, :IL_6, :IL_12, :IL_15, :IL_7, :IFN1,
    :IL_1, :IL_2, :IL_4, :IL_10, :TGFbeta, :IFN_g, :IgG4
]

# 网格布局：4列
n_cols = 4
n_rows = ceil(Int, 31 / n_cols)  # 31 = 30状态变量 + 1个Antigen

# 生成所有15组参数的求解结果
all_solutions = []
for result in optimized_results
    linear_params = exp.(result.u)
    p_dict = [param_symbols[i] => linear_params[i] for i in 1:length(linear_params)]
    prob = remake(oprob, p=p_dict)
    sol = solve(prob, QNDF(), saveat=target_time, abstol=ODE_ABSTOL, reltol=ODE_RELTOL)
    push!(all_solutions, sol)
end

# 创建子图数组
plots_array = []

# 首先绘制Antigen时间函数
p_antigen = plot(Ag_func, 0, 300, label="Antigen Input", lw=2.0,
                 color=:red, linestyle=:dash,
                 title="Antigen (Time Function)",
                 titlefontsize=9, legendfontsize=7, guidefontsize=8,
                 tickfontsize=7, grid=true, gridalpha=0.3)
xlabel!(p_antigen, "Time")
ylabel!(p_antigen, "Value")
push!(plots_array, p_antigen)

# 绘制所有30个状态变量
for (idx, var_name) in enumerate(all_variables)
    # 创建子图
    p = plot(title=String(var_name),
             titlefontsize=9, legendfontsize=6, guidefontsize=8,
             tickfontsize=7, grid=true, gridalpha=0.3)
    
    # 绘制15组参数的预测轨线（用不同颜色/透明度）
    colors = palette(:tab10)
    for (i, sol) in enumerate(all_solutions)
        plot!(p, sol, vars=[var_name], 
              label=(i == 1 ? "Model" : ""),  # 只在第一条线上显示图例
              lw=1.0, alpha=0.6,
              color=colors[mod1(i, 10)])
    end
    
    # 如果该变量有实验数据，叠加散点
    if haskey(target_data, var_name)
        scatter!(p, target_time, target_data[var_name],
                label="Exp Data", color=:black, markersize=3, alpha=0.8,
                markerstrokewidth=0)
    end
    
    xlabel!(p, "Time")
    ylabel!(p, "Value")
    
    push!(plots_array, p)
end

# 填充空白子图
for i in (length(plots_array)+1):(n_rows * n_cols)
    push!(plots_array, plot(legend=false, grid=false, showaxis=false, ticks=false))
end

# 组合所有子图
final_plot = plot(plots_array..., layout=(n_rows, n_cols), 
                 size=(1600, 300 * n_rows),
                 plot_title="IgG4-RD Multi-Start Optimization: 15 Best Parameter Sets",
                 plot_titlefontsize=14)

savefig(final_plot, "optimization_result.png")
println("已保存完整结果图: optimization_result.png")


println("\n" * "="^70)
println("所有任务完成！")
println("="^70)