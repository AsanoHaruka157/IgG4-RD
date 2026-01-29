import Pkg
Pkg.activate(@__DIR__)

using Catalyst
using DifferentialEquations
using Plots
using Optimization
using OptimizationOptimJL  # 使用 Optim.jl 的优化器（L-BFGS）
using Optim
using LineSearches
using CSV
using DataFrames
using ForwardDiff
using LinearAlgebra                  # 用于求零空间
using Statistics

include("projection.jl")  # provides HC_STATE, IgG_STATE, and rhs_*_linearized fns

# ============================================================================
# 1. 数据加载 (Data Loading) —— 与 optimize.jl 保持一致
# ============================================================================
df = CSV.read("target_data.csv", DataFrame, delim='\t')
target_time = df.Time

name_mapping = Dict(
    "CD56 NK" => "NK",
    "CD16 NK" => "act_NK",
    "TD Plasma" => "TD_IS_B",
    "TI Plasma" => "TI_IS_B"
)

model_variable_names = String[]
for col_name in names(df)
    if col_name != "Time"
        model_name = get(name_mapping, col_name, col_name)
        if model_name != "Antigen" && !(model_name in model_variable_names)
            push!(model_variable_names, model_name)
        end
    end
end

target_data = Dict{Symbol, Vector{Float64}}()
for col_name in names(df)
    if col_name != "Time"
        model_name = get(name_mapping, col_name, col_name)
        if !haskey(target_data, Symbol(model_name))
            target_data[Symbol(model_name)] = df[!, col_name]
        end
    end
end

println("数据加载完成: $(length(target_time)) 个时间点, $(length(model_variable_names)) 个变量")

# ============================================================================
# 2. 时间依赖函数 Ag(t) —— 与 optimize.jl 一致
# ============================================================================
Ag_func(t) = 100.0 / (1.0 + exp(-(t - 125.0)/5.0))
@register_symbolic Ag_func(t)

# ============================================================================
# 3. 模型定义 (Catalyst DSL) —— 与 optimize.jl 一致
# ============================================================================
rn = @reaction_network begin
    # 1. nDC
    k_nDC_f * nDC * (1 - nDC/k_nDC_m), 0 --> nDC
    k_mDC_Antigen_f * Ag_func(t) * nDC * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10)), nDC --> 0
    k_mDC_GMCSF_f * Ag_func(t) * nDC * (GMCSF / (GMCSF + k_mDC_GMCSF_m)) * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10)), nDC --> 0
    k_pDC_Antigen_f * nDC * Ag_func(t), nDC --> 0
    k_nDC_d, nDC --> 0

    # 3. mDC
    k_mDC_Antigen_f * Ag_func(t) * nDC * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10)), 0 --> mDC
    k_mDC_GMCSF_f * Ag_func(t) * nDC * (GMCSF / (GMCSF + k_mDC_GMCSF_m)) * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10)), 0 --> mDC
    k_mDC_f * mDC * (1 - mDC / k_mDC_m), 0 --> mDC
    k_mDC_d, mDC --> 0

    # 4. GMCSF
    k_GMCSF_Th2_f * Th2, 0 --> GMCSF
    k_GMCSF_Th2_Antigen_f * Th2 * Ag_func(t), 0 --> GMCSF
    k_GMCSF_act_NK_f * act_NK, 0 --> GMCSF
    k_mDC_GMCSF_d * Ag_func(t) * nDC * (GMCSF / (GMCSF + k_mDC_GMCSF_m)) * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10)), GMCSF --> 0
    k_GMCSF_d, GMCSF --> 0

    # 5. pDC
    k_pDC_Antigen_f * nDC * Ag_func(t), 0 --> pDC
    k_pDC_f * pDC * (1 - pDC / k_pDC_m), 0 --> pDC
    k_pDC_d, pDC --> 0

    # 6. IL_33
    k_IL_33_pDC_f * pDC, 0 --> IL_33
    k_act_CD4_IL_33_d * act_CD4 * IL_33 / (k_Th2_IL_33_m + IL_33), IL_33 --> 0
    k_IL_33_d, IL_33 --> 0

    # 7. IL_6
    k_IL_6_pDC_f * pDC, 0 --> IL_6
    k_IL_6_mDC_f * mDC, 0 --> IL_6
    k_IL_6_TFH_f * TFH * (k_TFH_nTreg_m / (nTreg + k_TFH_nTreg_m)), 0 --> IL_6
    k_TFH_IL_6_d * act_CD4 * IL_6 / (k_TFH_IL_6_m + IL_6), IL_6 --> 0
    k_IL_6_d, IL_6 --> 0

    # 8. IL_12
    k_IL_12_mDC_f * mDC, 0 --> IL_12
    k_act_NK_IL_12_d * NK * IL_12 / (IL_12 + k_act_NK_IL_12_m), IL_12 --> 0
    k_IL_12_d, IL_12 --> 0

    # 9. IL_15
    k_IL_15_f, 0 --> IL_15
    k_IL_15_Antigen_f * Ag_func(t), 0 --> IL_15
    k_naive_CD4_IL_15_d * naive_CD4 * IL_15 / (k_naive_CD4_IL_15_m + IL_15), IL_15 --> 0
    k_act_CD4_IL_15_d * act_CD4 * IL_15 / (k_act_CD4_IL_15_m + IL_15), IL_15 --> 0
    k_IL_15_d, IL_15 --> 0

    # 10. IL_7
    k_IL_7_f, 0 --> IL_7
    k_naive_CD4_IL_7_d * naive_CD4 * IL_7 / (k_naive_CD4_IL_7_m + IL_7), IL_7 --> 0
    k_act_CD4_IL_7_d * act_CD4 * IL_7 / (k_act_CD4_IL_7_m + IL_7), IL_7 --> 0
    k_IL_7_d, IL_7 --> 0

    # 11. IFN1
    k_IFN1_pDC_f * pDC, 0 --> IFN1
    k_act_CD4_IFN1_d * act_CD4 * IFN1 / (k_IFN1_CD4_CTL_m + IFN1), IFN1 --> 0
    k_act_NK_IFN1_d * NK * IFN1 / (IFN1 + k_act_NK_IFN1_m), IFN1 --> 0
    k_IFN1_d, IFN1 --> 0

    # 12. IL_1
    k_IL_1_mDC_f * mDC, 0 --> IL_1
    k_IL_1_d, IL_1 --> 0

    # 13. IL_2
    k_IL_2_act_CD4_f * act_CD4, 0 --> IL_2
    k_IL_2_act_CD4_Antigen_f * act_CD4 * Ag_func(t), 0 --> IL_2
    k_act_CD4_IL_2_d * naive_CD4 * IL_2 / (k_act_CD4_IL_2_m + IL_2), IL_2 --> 0
    k_act_NK_IL_2_d * NK * IL_2 / (IL_2 + k_act_NK_IL_2_m), IL_2 --> 0
    k_IL_2_d, IL_2 --> 0

    # 14. IL_4
    k_IL_4_Th2_f * Th2, 0 --> IL_4
    k_IL_4_Th2_Antigen_f * Th2 * Ag_func(t), 0 --> IL_4
    k_act_CD4_IL_4_d * act_CD4 * IL_4 / (k_Th2_IL_4_m + IL_4), IL_4 --> 0
    k_IL_4_d, IL_4 --> 0

    # 15. IL_10
    k_IL_10_iTreg_f * iTreg, 0 --> IL_10
    k_IL_10_nTreg_f * nTreg * mDC / (k_IL_10_nTreg_mDC_m + mDC), 0 --> IL_10
    k_iTreg_mDC_d * act_CD4 * IL_10 / (k_iTreg_IL_10_m + IL_10), IL_10 --> 0
    k_IL_10_d, IL_10 --> 0

    # 16. TGFbeta
    k_TGFbeta_iTreg_f * iTreg, 0 --> TGFbeta
    k_TGFbeta_CD4_CTL_f * CD4_CTL, 0 --> TGFbeta
    k_TGFbeta_nTreg_f * nTreg * mDC / (k_TGFbeta_nTreg_mDC_m + mDC), 0 --> TGFbeta
    k_iTreg_mDC_d * act_CD4 * TGFbeta / (k_iTreg_TGFbeta_m + TGFbeta), TGFbeta --> 0
    k_TGFbeta_d, TGFbeta --> 0

    # 17. IFN_g
    k_IFN_g_CD4_CTL_f * CD4_CTL, 0 --> IFN_g
    k_IFN_g_act_NK_f * act_NK, 0 --> IFN_g
    k_act_NK_IFN_g_d * NK * IFN_g / (IFN_g + k_act_NK_IFN_g_m), IFN_g --> 0
    k_IFN_g_d, IFN_g --> 0

    # 18. naive_CD4
    k_CD4_f * naive_CD4 * (1 - naive_CD4 / k_CD4_m), 0 --> naive_CD4
    k_naive_CD4_IL_15_f * naive_CD4 * (1 - naive_CD4 / k_CD4_m) * IL_15 / (k_naive_CD4_IL_15_m + IL_15), 0 --> naive_CD4
    k_naive_CD4_IL_7_f * naive_CD4 * (1 - naive_CD4 / k_CD4_m) * IL_7 / (k_naive_CD4_IL_7_m + IL_7), 0 --> naive_CD4
    k_act_CD4_mDC_f * naive_CD4 * mDC / (k_act_CD4_mDC_m + mDC), naive_CD4 --> 0
    k_act_CD4_IL_2_f * naive_CD4 * IL_2 / (k_act_CD4_IL_2_m + IL_2), naive_CD4 --> 0
    k_naive_CD4_d, naive_CD4 --> 0

    # 19. act_CD4
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

    # 20. Th2
    act_CD4 * k_Th2_f * (k_Th2_TGFbeta_m / (k_Th2_TGFbeta_m + TGFbeta)) * (k_Th2_IL_10_m / (k_Th2_IL_10_m + IL_10)) * (k_Th2_IL_12_m / (k_Th2_IL_12_m + IL_12)), 0 --> Th2
    act_CD4 * k_Th2_IL_4_f * IL_4 / (k_Th2_IL_4_m + IL_4) * (k_Th2_TGFbeta_m / (k_Th2_TGFbeta_m + TGFbeta)) * (k_Th2_IL_10_m / (k_Th2_IL_10_m + IL_10)) * (k_Th2_IL_12_m / (k_Th2_IL_12_m + IL_12)), 0 --> Th2
    act_CD4 * k_Th2_IL_33_f * IL_33 / (k_Th2_IL_33_m + IL_33) * (k_Th2_TGFbeta_m / (k_Th2_TGFbeta_m + TGFbeta)) * (k_Th2_IL_10_m / (k_Th2_IL_10_m + IL_10)) * (k_Th2_IL_12_m / (k_Th2_IL_12_m + IL_12)), 0 --> Th2
    k_Th2_f * Th2 * (1 - Th2 / k_Th2_m), 0 --> Th2
    k_Th2_d, Th2 --> 0

    # 21. iTreg
    act_CD4 * k_iTreg_mDC_f * k_iTreg_TGFbeta_f * TGFbeta / (k_iTreg_TGFbeta_m + TGFbeta) * (k_iTreg_IL_1_m / (k_iTreg_IL_1_m + IL_1)), 0 --> iTreg
    act_CD4 * k_iTreg_mDC_f * k_iTreg_IL_10_f * IL_10 / (k_iTreg_IL_10_m + IL_10) * (k_iTreg_IL_1_m / (k_iTreg_IL_1_m + IL_1)), 0 --> iTreg
    k_iTreg_f * iTreg * (1 - iTreg / k_iTreg_m), 0 --> iTreg
    k_iTreg_d, iTreg --> 0

    # 22. CD4_CTL
    k_act_CD4_CTL_basal_f * act_CD4, 0 --> CD4_CTL
    k_act_CD4_CTL_antigen_f * act_CD4 * Ag_func(t), 0 --> CD4_CTL
    k_act_CD4_IFN1_f * act_CD4 * IFN1 / (k_IFN1_CD4_CTL_m + IFN1), 0 --> CD4_CTL
    k_CD4_CTL_f * CD4_CTL * (1 - CD4_CTL / k_CD4_CTL_m), 0 --> CD4_CTL
    k_CD4_CTL_d, CD4_CTL --> 0

    # 23. nTreg
    k_nTreg_mDC_f * nTreg * (1 - nTreg / k_nTreg_m) * mDC / (k_nTreg_mDC_m + mDC), 0 --> nTreg
    k_nTreg_d, nTreg --> 0

    # 24. TFH
    k_TFH_mDC_f * act_CD4, 0 --> TFH
    k_TFH_mDC_Antigen_f * act_CD4 * Ag_func(t), 0 --> TFH
    k_TFH_IFN1_f * act_CD4 * IFN1 / (k_TFH_IFN1_m + IFN1), 0 --> TFH
    k_TFH_IL_6_f * act_CD4 * IL_6 / (k_TFH_IL_6_m + IL_6), 0 --> TFH
    k_TFH_f * TFH * (1 - TFH / k_TFH_m), 0 --> TFH
    k_TFH_d, TFH --> 0

    # 25. NK
    k_NK_f * NK * (1 - NK / k_NK_m), 0 --> NK
    k_act_NK_base_f * NK, NK --> 0
    k_act_NK_IL_12_f * NK * IL_12 / (IL_12 + k_act_NK_IL_12_m), NK --> 0
    k_act_NK_IL_2_f * NK * IL_2 / (IL_2 + k_act_NK_IL_2_m), NK --> 0
    k_act_NK_IFN1_f * NK * IFN1 / (IFN1 + k_act_NK_IFN1_m), NK --> 0
    k_act_NK_IFN_g_f * NK * IFN_g / (IFN_g + k_act_NK_IFN_g_m), NK --> 0
    k_NK_d, NK --> 0

    # 26. act_NK
    k_act_NK_base_f * NK, 0 --> act_NK
    k_act_NK_IL_12_f * NK * IL_12 / (IL_12 + k_act_NK_IL_12_m), 0 --> act_NK
    k_act_NK_IL_2_f * NK * IL_2 / (IL_2 + k_act_NK_IL_2_m), 0 --> act_NK
    k_act_NK_IFN1_f * NK * IFN1 / (IFN1 + k_act_NK_IFN1_m), 0 --> act_NK
    k_act_NK_IFN_g_f * NK * IFN_g / (IFN_g + k_act_NK_IFN_g_m), 0 --> act_NK
    k_act_NK_f * act_NK * (1 - act_NK / k_act_NK_m), 0 --> act_NK
    k_act_NK_d, act_NK --> 0

    # 27. Naive_B
    k_Naive_B_f * Naive_B * (1 - Naive_B / k_Naive_B_m), 0 --> Naive_B
    k_Naive_B_Antigen_f * Naive_B * Ag_func(t) * (1 - Naive_B / k_Naive_B_m), 0 --> Naive_B
    k_Act_B_basal_f * Naive_B, Naive_B --> 0
    k_Act_B_Antigen_f * Naive_B * Ag_func(t), Naive_B --> 0
    k_Naive_B_d, Naive_B --> 0

    # 28. Act_B
    k_Act_B_basal_f * Naive_B, 0 --> Act_B
    k_Act_B_Antigen_f * Naive_B * Ag_func(t), 0 --> Act_B
    k_Act_B_f * Act_B * (1 - Act_B / k_Act_B_m), 0 --> Act_B
    k_Act_B_Antigen_pro_f * Act_B * Ag_func(t) * (1 - Act_B / k_Act_B_m), 0 --> Act_B
    k_Act_B_d, Act_B --> 0

    # 29. TD_IS_B
    k_TD_base_f * Act_B, 0 --> TD_IS_B
    k_TD_IL_4_f * Act_B * IL_4, 0 --> TD_IS_B
    k_TD_f * TD_IS_B * (1 - TD_IS_B / k_TD_m), 0 --> TD_IS_B
    k_TD_d, TD_IS_B --> 0

    # 30. TI_IS_B
    k_TI_base_f * Act_B, 0 --> TI_IS_B
    k_TI_IFN_g_f * Act_B * IFN_g, 0 --> TI_IS_B
    k_TI_IL_10_f * Act_B * IL_10, 0 --> TI_IS_B
    k_TI_f * TI_IS_B * (1 - TI_IS_B / k_TI_m), 0 --> TI_IS_B
    k_TI_d, TI_IS_B --> 0

    # 31. IgG4
    k_IgG4_TI_f * TI_IS_B, 0 --> IgG4
    k_IgG4_TD_f * TD_IS_B, 0 --> IgG4
    k_IgG4_d, IgG4 --> 0
end

u0 = [
    :nDC => 30.0, :mDC => 0.1, :pDC => 0.1, :naive_CD4 => 180.0, :act_CD4 => 300.0,
    :Th2 => 0.3, :iTreg => 50.0, :CD4_CTL => 1.0, :nTreg => 10.0, :TFH => 20.0,
    :NK => 50.0, :act_NK => 200.0, :Naive_B => 86.0, :Act_B => 80.0, :TD_IS_B => 1.5,
    :TI_IS_B => 1.0, :GMCSF => 3000.0, :IL_33 => 1.0, :IL_6 => 10000.0, :IL_12 => 1.0,
    :IL_15 => 10000.0, :IL_7 => 350000.0, :IFN1 => 1.0, :IL_1 => 3000.0, :IL_2 => 5000.0,
    :IL_4 => 250.0, :IL_10 => 4000.0, :TGFbeta => 50.0, :IFN_g => 200000.0, :IgG4 => 0.1
]

tspan = (0.0, 300.0)
# 用一个完整的参数初始字典（避免 MTK 报 “missing from variable map”）
p_init = [Symbol(p) => 0.1 for p in parameters(rn)]
oprob = ODEProblem(rn, u0, tspan, p_init)

# Catalyst 参数列表（用于优化变量顺序）
param_symbols = [pair.first for pair in p_init]

# 初始化（沿用 optimize.jl 的启发式）
function make_p0_lb_ub(param_symbols, target_data, u0_map; eps=1e-12)
    function get_u0(sym::Symbol)
        for pair in u0_map
            if pair.first == sym
                return pair.second
            end
        end
        return 0.0
    end

    function parse_state_from_m(param::Symbol)
        s = String(param)
        if endswith(s, "_m")
            core = s[3:end-2]
            toks = split(core, "_")
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
    p0_center = zeros(n)
    lb = zeros(n)
    ub = zeros(n)
    for (i, psym) in enumerate(param_symbols)
        name = String(psym)
        if endswith(name, "_d")
            p0_center[i] = 1e-2
        elseif occursin("_Antigen_", name)
            p0_center[i] = 3e-2
        elseif endswith(name, "_pro_f")
            p0_center[i] = 1e-3
        elseif endswith(name, "_m")
            st = parse_state_from_m(psym)
            Sx = 1.0
            if st !== nothing
                if haskey(target_data, st)
                    Sx = maximum(target_data[st])
                else
                    Sx = get_u0(st)
                end
            end
            Sx = max(Sx, 1.0)
            p0_center[i] = 1.0 * Sx
        elseif occursin("_base_f", name) || occursin("_basal_f", name)
            p0_center[i] = 1e-3
        elseif endswith(name, "_f")
            p0_center[i] = 1e-2
        else
            p0_center[i] = 1e-2
        end
        p0_center[i] = max(p0_center[i], eps)
        lb[i] = p0_center[i] * 1e-2
        ub[i] = p0_center[i] * 1e2
    end
    p0 = lb .+ rand(n) .* (ub .- lb)
    return p0, lb, ub
end

p0_lin, lb_lin, ub_lin = make_p0_lb_ub(param_symbols, target_data, u0)
p0_log = log.(p0_lin)
lb_log = log.(lb_lin)
ub_log = log.(ub_lin)

# ============================================================================
# loss: 只保留 (smooth_norm + eps) * data_loss
# ============================================================================
WINDOW_T_LO = 50.0
WINDOW_T_HI = 100.0
WINDOW_MULT = 1.5
SMOOTH_EPS = 1e-3

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

function smoothness_penalty(sol, var_syms::Vector{Symbol})
    pen = 0.0
    for v in var_syms
        y = try
            sol[v]
        catch
            continue
        end
        if any(!isfinite, y)
            continue
        end
        n = length(y)
        if n < 3
            continue
        end
        for i in 2:(n-1)
            d2 = y[i+1] - 2y[i] + y[i-1]
            if isfinite(d2)
                pen += d2*d2
            end
        end
    end
    return pen
end

ordered_variables = Symbol.(model_variable_names)

function data_loss_L2(sol, var_syms::Vector{Symbol})
    dl = 0.0
    for (vi, v) in enumerate(var_syms)
        y = try
            sol[v]
        catch
            continue
        end
        yt = target_data[v]
        for i in eachindex(yt)
            r = (y[i] - yt[i])
            dl += TIME_WEIGHTS[i] * (r*r)
        end
    end
    return dl
end

function loss_no_steady(p_log, _ = nothing)
    lin_params = exp.(p_log)
    p_dict = [param_symbols[i] => lin_params[i] for i in eachindex(param_symbols)]
    new_prob = remake(oprob, p=p_dict)
    sol = solve(new_prob, QNDF(), saveat=target_time, abstol=1e-8, reltol=1e-6, verbose=false, maxiters=1_000_000)
    if sol.retcode != :Success
        return 1e12
    end
    dl = data_loss_L2(sol, ordered_variables)
    sp = smoothness_penalty(sol, ordered_variables)
    smooth_norm = log(max(sp, SMOOTH_EPS))
    return (smooth_norm + SMOOTH_EPS) * dl
end

# ============================================================================
# 约束：RHS(t=0)=0 与 RHS(t=200)=0 （使用 projection.jl 的线性化 RHS）
# 使用线性方程组的解空间（零空间）来参数化可行域
# 这里用 HC_STATE (Antigen=0) 与 IgG_STATE (Antigen=1) 作为对应时刻的状态值
# ============================================================================

getp(pdict, k::Symbol) = pdict[k]

function combined_params_from_original(pdict, state)
    # helper for Hill fractions
    # 确保 state 中的值能够与 Dual 类型兼容
    T = typeof(first(values(pdict)))
    IL_10 = T(state[:IL_10]); GMCSF = T(state[:GMCSF]); IL_33 = T(state[:IL_33]); IL_6 = T(state[:IL_6])
    IL_12 = T(state[:IL_12]); IL_15 = T(state[:IL_15]); IL_7 = T(state[:IL_7]); IFN1 = T(state[:IFN1])
    IL_2 = T(state[:IL_2]); IL_4 = T(state[:IL_4]); TGFbeta = T(state[:TGFbeta]); IFN_g = T(state[:IFN_g])
    nTreg = T(state[:nTreg]); mDC = T(state[:mDC]); IL_1 = T(state[:IL_1])
    out = Dict{Symbol,T}()

    # div terms
    out[:k_nDC_f_div_m] = getp(pdict,:k_nDC_f) / getp(pdict,:k_nDC_m)
    out[:k_mDC_f_div_m] = getp(pdict,:k_mDC_f) / getp(pdict,:k_mDC_m)
    out[:k_pDC_f_div_m] = getp(pdict,:k_pDC_f) / getp(pdict,:k_pDC_m)
    out[:k_CD4_f_div_m] = getp(pdict,:k_CD4_f) / getp(pdict,:k_CD4_m)
    out[:k_act_CD4_f_div_m] = getp(pdict,:k_act_CD4_f) / getp(pdict,:k_act_CD4_m)
    out[:k_Th2_f_div_m] = getp(pdict,:k_Th2_f) / getp(pdict,:k_Th2_m)
    out[:k_iTreg_f_div_m] = getp(pdict,:k_iTreg_f) / getp(pdict,:k_iTreg_m)
    out[:k_CD4_CTL_f_div_m] = getp(pdict,:k_CD4_CTL_f) / getp(pdict,:k_CD4_CTL_m)
    out[:k_TFH_f_div_m] = getp(pdict,:k_TFH_f) / getp(pdict,:k_TFH_m)
    out[:k_NK_f_div_m] = getp(pdict,:k_NK_f) / getp(pdict,:k_NK_m)
    out[:k_act_NK_f_div_m] = getp(pdict,:k_act_NK_f) / getp(pdict,:k_act_NK_m)
    out[:k_Naive_B_f_div_m] = getp(pdict,:k_Naive_B_f) / getp(pdict,:k_Naive_B_m)
    out[:k_Act_B_f_div_m] = getp(pdict,:k_Act_B_f) / getp(pdict,:k_Act_B_m)
    out[:k_TD_f_div_m] = getp(pdict,:k_TD_f) / getp(pdict,:k_TD_m)
    out[:k_TI_f_div_m] = getp(pdict,:k_TI_f) / getp(pdict,:k_TI_m)
    out[:k_Naive_B_Antigen_f_div_m] = getp(pdict,:k_Naive_B_Antigen_f) / getp(pdict,:k_Naive_B_m)
    out[:k_Act_B_Antigen_pro_f_div_m] = getp(pdict,:k_Act_B_Antigen_pro_f) / getp(pdict,:k_Act_B_m)

    # Hill/eff terms
    out[:k_mDC_Antigen_IL10_eff] = getp(pdict,:k_mDC_Antigen_f) * getp(pdict,:k_mDC_IL_10_m) / (getp(pdict,:k_mDC_IL_10_m) + IL_10)
    out[:k_mDC_GMCSF_IL10_eff] = getp(pdict,:k_mDC_GMCSF_f) * (GMCSF / (GMCSF + getp(pdict,:k_mDC_GMCSF_m))) * (getp(pdict,:k_mDC_IL_10_m) / (getp(pdict,:k_mDC_IL_10_m) + IL_10))
    out[:k_mDC_GMCSF_d_eff] = getp(pdict,:k_mDC_GMCSF_d) * (GMCSF / (GMCSF + getp(pdict,:k_mDC_GMCSF_m))) * (getp(pdict,:k_mDC_IL_10_m) / (getp(pdict,:k_mDC_IL_10_m) + IL_10))

    out[:k_act_CD4_IL_33_d_eff] = getp(pdict,:k_act_CD4_IL_33_d) * IL_33 / (getp(pdict,:k_Th2_IL_33_m) + IL_33)
    out[:k_IL_6_TFH_eff] = getp(pdict,:k_IL_6_TFH_f) * (getp(pdict,:k_TFH_nTreg_m) / (nTreg + getp(pdict,:k_TFH_nTreg_m)))
    out[:k_TFH_IL_6_d_eff] = getp(pdict,:k_TFH_IL_6_d) * IL_6 / (getp(pdict,:k_TFH_IL_6_m) + IL_6)
    out[:k_act_NK_IL_12_d_eff] = getp(pdict,:k_act_NK_IL_12_d) * IL_12 / (IL_12 + getp(pdict,:k_act_NK_IL_12_m))

    out[:k_naive_CD4_IL_15_d_eff] = getp(pdict,:k_naive_CD4_IL_15_d) * IL_15 / (getp(pdict,:k_naive_CD4_IL_15_m) + IL_15)
    out[:k_act_CD4_IL_15_d_eff] = getp(pdict,:k_act_CD4_IL_15_d) * IL_15 / (getp(pdict,:k_act_CD4_IL_15_m) + IL_15)
    out[:k_naive_CD4_IL_7_d_eff] = getp(pdict,:k_naive_CD4_IL_7_d) * IL_7 / (getp(pdict,:k_naive_CD4_IL_7_m) + IL_7)
    out[:k_act_CD4_IL_7_d_eff] = getp(pdict,:k_act_CD4_IL_7_d) * IL_7 / (getp(pdict,:k_act_CD4_IL_7_m) + IL_7)

    out[:k_act_CD4_IFN1_d_eff] = getp(pdict,:k_act_CD4_IFN1_d) * IFN1 / (getp(pdict,:k_IFN1_CD4_CTL_m) + IFN1)
    out[:k_act_NK_IFN1_d_eff] = getp(pdict,:k_act_NK_IFN1_d) * IFN1 / (IFN1 + getp(pdict,:k_act_NK_IFN1_m))
    out[:k_act_CD4_IL_2_d_eff] = getp(pdict,:k_act_CD4_IL_2_d) * IL_2 / (getp(pdict,:k_act_CD4_IL_2_m) + IL_2)
    out[:k_act_NK_IL_2_d_eff] = getp(pdict,:k_act_NK_IL_2_d) * IL_2 / (IL_2 + getp(pdict,:k_act_NK_IL_2_m))
    out[:k_act_CD4_IL_4_d_eff] = getp(pdict,:k_act_CD4_IL_4_d) * IL_4 / (getp(pdict,:k_Th2_IL_4_m) + IL_4)

    out[:k_IL_10_nTreg_eff] = getp(pdict,:k_IL_10_nTreg_f) * mDC / (getp(pdict,:k_IL_10_nTreg_mDC_m) + mDC)
    out[:k_iTreg_mDC_d_eff] = getp(pdict,:k_iTreg_mDC_d) * IL_10 / (getp(pdict,:k_iTreg_IL_10_m) + IL_10)
    out[:k_TGFbeta_nTreg_eff] = getp(pdict,:k_TGFbeta_nTreg_f) * mDC / (getp(pdict,:k_TGFbeta_nTreg_mDC_m) + mDC)
    out[:k_iTreg_mDC_d_TGFbeta_eff] = getp(pdict,:k_iTreg_mDC_d) * TGFbeta / (getp(pdict,:k_iTreg_TGFbeta_m) + TGFbeta)

    out[:k_act_NK_IFN_g_d_eff] = getp(pdict,:k_act_NK_IFN_g_d) * IFN_g / (IFN_g + getp(pdict,:k_act_NK_IFN_g_m))

    out[:k_naive_CD4_IL_15_eff] = getp(pdict,:k_naive_CD4_IL_15_f) * IL_15 / (getp(pdict,:k_naive_CD4_IL_15_m) + IL_15)
    out[:k_naive_CD4_IL_7_eff] = getp(pdict,:k_naive_CD4_IL_7_f) * IL_7 / (getp(pdict,:k_naive_CD4_IL_7_m) + IL_7)
    out[:k_act_CD4_mDC_f_eff] = getp(pdict,:k_act_CD4_mDC_f) * mDC / (getp(pdict,:k_act_CD4_mDC_m) + mDC)
    out[:k_act_CD4_IL_2_f_eff] = getp(pdict,:k_act_CD4_IL_2_f) * IL_2 / (getp(pdict,:k_act_CD4_IL_2_m) + IL_2)
    out[:k_act_CD4_IL_15_eff] = getp(pdict,:k_act_CD4_IL_15_f) * IL_15 / (getp(pdict,:k_act_CD4_IL_15_m) + IL_15)
    out[:k_act_CD4_IL_7_eff] = getp(pdict,:k_act_CD4_IL_7_f) * IL_7 / (getp(pdict,:k_act_CD4_IL_7_m) + IL_7)

    sup = (getp(pdict,:k_Th2_TGFbeta_m) / (getp(pdict,:k_Th2_TGFbeta_m) + TGFbeta)) *
          (getp(pdict,:k_Th2_IL_10_m) / (getp(pdict,:k_Th2_IL_10_m) + IL_10)) *
          (getp(pdict,:k_Th2_IL_12_m) / (getp(pdict,:k_Th2_IL_12_m) + IL_12))
    out[:k_Th2_f_eff] = getp(pdict,:k_Th2_f) * sup
    out[:k_Th2_IL_4_eff] = getp(pdict,:k_Th2_IL_4_f) * (IL_4 / (getp(pdict,:k_Th2_IL_4_m) + IL_4)) * sup
    out[:k_Th2_IL_33_eff] = getp(pdict,:k_Th2_IL_33_f) * (IL_33 / (getp(pdict,:k_Th2_IL_33_m) + IL_33)) * sup

    out[:k_iTreg_TGFbeta_eff] = getp(pdict,:k_iTreg_mDC_f) * getp(pdict,:k_iTreg_TGFbeta_f) *
                               (TGFbeta / (getp(pdict,:k_iTreg_TGFbeta_m) + TGFbeta)) *
                               (getp(pdict,:k_iTreg_IL_1_m) / (getp(pdict,:k_iTreg_IL_1_m) + IL_1))
    out[:k_iTreg_IL_10_eff] = getp(pdict,:k_iTreg_mDC_f) * getp(pdict,:k_iTreg_IL_10_f) *
                             (IL_10 / (getp(pdict,:k_iTreg_IL_10_m) + IL_10)) *
                             (getp(pdict,:k_iTreg_IL_1_m) / (getp(pdict,:k_iTreg_IL_1_m) + IL_1))

    out[:k_act_CD4_IFN1_f_eff] = getp(pdict,:k_act_CD4_IFN1_f) * IFN1 / (getp(pdict,:k_IFN1_CD4_CTL_m) + IFN1)
    out[:k_TFH_IFN1_f_eff] = getp(pdict,:k_TFH_IFN1_f) * IFN1 / (getp(pdict,:k_TFH_IFN1_m) + IFN1)
    out[:k_TFH_IL_6_f_eff] = getp(pdict,:k_TFH_IL_6_f) * IL_6 / (getp(pdict,:k_TFH_IL_6_m) + IL_6)

    out[:k_act_NK_IL_12_f_eff] = getp(pdict,:k_act_NK_IL_12_f) * IL_12 / (IL_12 + getp(pdict,:k_act_NK_IL_12_m))
    out[:k_act_NK_IL_2_f_eff] = getp(pdict,:k_act_NK_IL_2_f) * IL_2 / (IL_2 + getp(pdict,:k_act_NK_IL_2_m))
    out[:k_act_NK_IFN1_f_eff] = getp(pdict,:k_act_NK_IFN1_f) * IFN1 / (IFN1 + getp(pdict,:k_act_NK_IFN1_m))
    out[:k_act_NK_IFN_g_f_eff] = getp(pdict,:k_act_NK_IFN_g_f) * IFN_g / (IFN_g + getp(pdict,:k_act_NK_IFN_g_m))

    out[:k_nTreg_mDC_f_eff] = getp(pdict,:k_nTreg_mDC_f) * mDC / (getp(pdict,:k_nTreg_mDC_m) + mDC)

    # eff_div_m terms
    out[:k_naive_CD4_IL_15_eff_div_m] = out[:k_naive_CD4_IL_15_eff] / getp(pdict,:k_CD4_m)
    out[:k_naive_CD4_IL_7_eff_div_m] = out[:k_naive_CD4_IL_7_eff] / getp(pdict,:k_CD4_m)
    out[:k_act_CD4_IL_15_eff_div_m] = out[:k_act_CD4_IL_15_eff] / getp(pdict,:k_act_CD4_m)
    out[:k_act_CD4_IL_7_eff_div_m] = out[:k_act_CD4_IL_7_eff] / getp(pdict,:k_act_CD4_m)
    out[:k_nTreg_mDC_f_eff_div_m] = out[:k_nTreg_mDC_f_eff] / getp(pdict,:k_nTreg_m)

    return out
end

function build_linearized_param_dict(pdict, state)
    # include originals
    T = typeof(first(values(pdict)))
    out = Dict{Symbol,T}()
    for (k,v) in pdict
        out[k] = v
    end
    # add combined
    for (k,v) in combined_params_from_original(pdict, state)
        out[k] = v
    end
    return out
end

function rhs_constraints_vector(pdict, state)
    pars = build_linearized_param_dict(pdict, state)
    Ag = state[:Antigen]
    T = typeof(first(values(pdict)))
    return T[
        rhs_nDC_linearized(state[:nDC], state[:IL_10], state[:GMCSF], Ag, pars),
        rhs_mDC_linearized(state[:mDC], state[:nDC], state[:IL_10], state[:GMCSF], Ag, pars),
        rhs_pDC_linearized(state[:pDC], state[:nDC], Ag, pars),
        rhs_GMCSF_linearized(state[:GMCSF], state[:Th2], state[:act_NK], state[:nDC], state[:IL_10], Ag, pars),
        rhs_IL_33_linearized(state[:IL_33], state[:pDC], state[:act_CD4], pars),
        rhs_IL_6_linearized(state[:IL_6], state[:pDC], state[:mDC], state[:TFH], state[:nTreg], state[:act_CD4], pars),
        rhs_IL_12_linearized(state[:IL_12], state[:mDC], state[:NK], pars),
        rhs_IL_15_linearized(state[:IL_15], state[:naive_CD4], state[:act_CD4], Ag, pars),
        rhs_IL_7_linearized(state[:IL_7], state[:naive_CD4], state[:act_CD4], pars),
        rhs_IFN1_linearized(state[:IFN1], state[:pDC], state[:act_CD4], state[:NK], pars),
        rhs_IL_1_linearized(state[:IL_1], state[:mDC], pars),
        rhs_IL_2_linearized(state[:IL_2], state[:act_CD4], state[:naive_CD4], state[:NK], Ag, pars),
        rhs_IL_4_linearized(state[:IL_4], state[:Th2], state[:act_CD4], Ag, pars),
        rhs_IL_10_linearized(state[:IL_10], state[:iTreg], state[:nTreg], state[:mDC], state[:act_CD4], pars),
        rhs_TGFbeta_linearized(state[:TGFbeta], state[:iTreg], state[:CD4_CTL], state[:nTreg], state[:mDC], state[:act_CD4], pars),
        rhs_IFN_g_linearized(state[:IFN_g], state[:CD4_CTL], state[:act_NK], state[:NK], pars),
        rhs_naive_CD4_linearized(state[:naive_CD4], state[:IL_15], state[:IL_7], state[:mDC], state[:IL_2], pars),
        rhs_act_CD4_linearized(state[:act_CD4], state[:naive_CD4], state[:mDC], state[:IL_2], state[:IL_15], state[:IL_7], state[:IL_4], state[:IL_33], state[:TGFbeta], state[:IL_10], state[:IL_12], state[:IL_1], state[:IFN1], state[:IL_6], Ag, pars),
        rhs_Th2_linearized(state[:Th2], state[:act_CD4], state[:TGFbeta], state[:IL_10], state[:IL_12], state[:IL_4], state[:IL_33], pars),
        rhs_iTreg_linearized(state[:iTreg], state[:act_CD4], state[:TGFbeta], state[:IL_10], state[:IL_1], pars),
        rhs_CD4_CTL_linearized(state[:CD4_CTL], state[:act_CD4], state[:IFN1], Ag, pars),
        rhs_nTreg_linearized(state[:nTreg], state[:mDC], pars),
        rhs_TFH_linearized(state[:TFH], state[:act_CD4], state[:IFN1], state[:IL_6], Ag, pars),
        rhs_NK_linearized(state[:NK], state[:IL_12], state[:IL_2], state[:IFN1], state[:IFN_g], pars),
        rhs_act_NK_linearized(state[:act_NK], state[:NK], state[:IL_12], state[:IL_2], state[:IFN1], state[:IFN_g], pars),
        rhs_Naive_B_linearized(state[:Naive_B], Ag, pars),
        rhs_Act_B_linearized(state[:Act_B], state[:Naive_B], Ag, pars),
        rhs_TD_IS_B_linearized(state[:TD_IS_B], state[:Act_B], state[:IL_4], pars),
        rhs_TI_IS_B_linearized(state[:TI_IS_B], state[:Act_B], state[:IFN_g], state[:IL_10], pars),
        rhs_IgG4_linearized(state[:IgG4], state[:TI_IS_B], state[:TD_IS_B], pars),
    ]
end

# ============================================================================
# 在线性化参数空间（projection.jl 定义的组合参数空间）中构造约束矩阵
# 线性化参数空间 p 包含：原始参数 + 组合参数（_eff, _div_m 等）
# ============================================================================

const NCONS = 60

"""
获取线性化参数空间的参数列表（原始参数 + 组合参数）
"""
function get_linearized_param_symbols()
    # 原始参数
    original_params = param_symbols
    
    # 组合参数（从 combined_params_from_original 函数中提取）
    combined_param_names = [
        # div terms
        :k_nDC_f_div_m, :k_mDC_f_div_m, :k_pDC_f_div_m, :k_CD4_f_div_m,
        :k_act_CD4_f_div_m, :k_Th2_f_div_m, :k_iTreg_f_div_m, :k_CD4_CTL_f_div_m,
        :k_TFH_f_div_m, :k_NK_f_div_m, :k_act_NK_f_div_m, :k_Naive_B_f_div_m,
        :k_Act_B_f_div_m, :k_TD_f_div_m, :k_TI_f_div_m,
        :k_Naive_B_Antigen_f_div_m, :k_Act_B_Antigen_pro_f_div_m,
        # eff terms
        :k_mDC_Antigen_IL10_eff, :k_mDC_GMCSF_IL10_eff, :k_mDC_GMCSF_d_eff,
        :k_act_CD4_IL_33_d_eff, :k_IL_6_TFH_eff, :k_TFH_IL_6_d_eff,
        :k_act_NK_IL_12_d_eff, :k_naive_CD4_IL_15_d_eff, :k_act_CD4_IL_15_d_eff,
        :k_naive_CD4_IL_7_d_eff, :k_act_CD4_IL_7_d_eff, :k_act_CD4_IFN1_d_eff,
        :k_act_NK_IFN1_d_eff, :k_act_CD4_IL_2_d_eff, :k_act_NK_IL_2_d_eff,
        :k_act_CD4_IL_4_d_eff, :k_IL_10_nTreg_eff, :k_iTreg_mDC_d_eff,
        :k_TGFbeta_nTreg_eff, :k_iTreg_mDC_d_TGFbeta_eff, :k_act_NK_IFN_g_d_eff,
        :k_naive_CD4_IL_15_eff, :k_naive_CD4_IL_7_eff, :k_act_CD4_mDC_f_eff,
        :k_act_CD4_IL_2_f_eff, :k_act_CD4_IL_15_eff, :k_act_CD4_IL_7_eff,
        :k_Th2_f_eff, :k_Th2_IL_4_eff, :k_Th2_IL_33_eff,
        :k_iTreg_TGFbeta_eff, :k_iTreg_IL_10_eff,
        :k_act_CD4_IFN1_f_eff, :k_TFH_IFN1_f_eff, :k_TFH_IL_6_f_eff,
        :k_act_NK_IL_12_f_eff, :k_act_NK_IL_2_f_eff, :k_act_NK_IFN1_f_eff,
        :k_act_NK_IFN_g_f_eff, :k_nTreg_mDC_f_eff,
        # eff_div_m terms
        :k_naive_CD4_IL_15_eff_div_m, :k_naive_CD4_IL_7_eff_div_m,
        :k_act_CD4_IL_15_eff_div_m, :k_act_CD4_IL_7_eff_div_m,
        :k_nTreg_mDC_f_eff_div_m
    ]
    
    return vcat(original_params, combined_param_names)
end

linearized_param_symbols = get_linearized_param_symbols()
N_linearized = length(linearized_param_symbols)
println("线性化参数空间维度: ", N_linearized)

"""
在线性化参数空间 p_linearized 上计算约束向量 c(p) = [RHS(t=0); RHS(t=200)]
注意：这里的 p_linearized 是线性化参数向量（包含组合参数）
"""
function constraint_linearized_space(p_linearized::AbstractVector)
    # 将向量转换为字典（线性化参数空间）
    pdict_linearized = Dict{Symbol,Float64}()
    for (i, sym) in enumerate(linearized_param_symbols)
        pdict_linearized[sym] = p_linearized[i]
    end
    
    # 从线性化参数提取原始参数部分
    pdict_original = Dict{Symbol,Float64}()
    for sym in param_symbols
        if haskey(pdict_linearized, sym)
            pdict_original[sym] = pdict_linearized[sym]
        else
            # 如果原始参数不在线性化参数中，使用默认值（后续需要改进反解逻辑）
            pdict_original[sym] = 0.1
        end
    end
    
    # 使用 build_linearized_param_dict 构建完整的参数字典
    # 注意：这里会重新计算组合参数，但我们应该使用 pdict_linearized 中的值
    # 为了保持一致性，我们先用原始参数构建，然后用线性化参数覆盖组合参数
    pars_t0 = build_linearized_param_dict(pdict_original, HC_STATE)
    pars_t200 = build_linearized_param_dict(pdict_original, IgG_STATE)
    
    # 覆盖组合参数（使用线性化参数空间中的值）
    for sym in linearized_param_symbols
        if haskey(pdict_linearized, sym) && sym ∉ param_symbols
            pars_t0[sym] = pdict_linearized[sym]
            pars_t200[sym] = pdict_linearized[sym]
        end
    end
    
    # 计算约束向量
    Ag_t0 = 0.0
    Ag_t200 = 100.0 / (1.0 + exp(-(200.0 - 125.0)/5.0))
    
    c1 = [
        rhs_nDC_linearized(HC_STATE[:nDC], HC_STATE[:IL_10], HC_STATE[:GMCSF], Ag_t0, pars_t0),
        rhs_mDC_linearized(HC_STATE[:mDC], HC_STATE[:nDC], HC_STATE[:IL_10], HC_STATE[:GMCSF], Ag_t0, pars_t0),
        rhs_pDC_linearized(HC_STATE[:pDC], HC_STATE[:nDC], Ag_t0, pars_t0),
        rhs_GMCSF_linearized(HC_STATE[:GMCSF], HC_STATE[:Th2], HC_STATE[:act_NK], HC_STATE[:nDC], HC_STATE[:IL_10], Ag_t0, pars_t0),
        rhs_IL_33_linearized(HC_STATE[:IL_33], HC_STATE[:pDC], HC_STATE[:act_CD4], pars_t0),
        rhs_IL_6_linearized(HC_STATE[:IL_6], HC_STATE[:pDC], HC_STATE[:mDC], HC_STATE[:TFH], HC_STATE[:nTreg], HC_STATE[:act_CD4], pars_t0),
        rhs_IL_12_linearized(HC_STATE[:IL_12], HC_STATE[:mDC], HC_STATE[:NK], pars_t0),
        rhs_IL_15_linearized(HC_STATE[:IL_15], HC_STATE[:naive_CD4], HC_STATE[:act_CD4], Ag_t0, pars_t0),
        rhs_IL_7_linearized(HC_STATE[:IL_7], HC_STATE[:naive_CD4], HC_STATE[:act_CD4], pars_t0),
        rhs_IFN1_linearized(HC_STATE[:IFN1], HC_STATE[:pDC], HC_STATE[:act_CD4], HC_STATE[:NK], pars_t0),
        rhs_IL_1_linearized(HC_STATE[:IL_1], HC_STATE[:mDC], pars_t0),
        rhs_IL_2_linearized(HC_STATE[:IL_2], HC_STATE[:act_CD4], HC_STATE[:naive_CD4], HC_STATE[:NK], Ag_t0, pars_t0),
        rhs_IL_4_linearized(HC_STATE[:IL_4], HC_STATE[:Th2], HC_STATE[:act_CD4], Ag_t0, pars_t0),
        rhs_IL_10_linearized(HC_STATE[:IL_10], HC_STATE[:iTreg], HC_STATE[:nTreg], HC_STATE[:mDC], HC_STATE[:act_CD4], pars_t0),
        rhs_TGFbeta_linearized(HC_STATE[:TGFbeta], HC_STATE[:iTreg], HC_STATE[:CD4_CTL], HC_STATE[:nTreg], HC_STATE[:mDC], HC_STATE[:act_CD4], pars_t0),
        rhs_IFN_g_linearized(HC_STATE[:IFN_g], HC_STATE[:CD4_CTL], HC_STATE[:act_NK], HC_STATE[:NK], pars_t0),
        rhs_naive_CD4_linearized(HC_STATE[:naive_CD4], HC_STATE[:IL_15], HC_STATE[:IL_7], HC_STATE[:mDC], HC_STATE[:IL_2], pars_t0),
        rhs_act_CD4_linearized(HC_STATE[:act_CD4], HC_STATE[:naive_CD4], HC_STATE[:mDC], HC_STATE[:IL_2], HC_STATE[:IL_15], HC_STATE[:IL_7], HC_STATE[:IL_4], HC_STATE[:IL_33], HC_STATE[:TGFbeta], HC_STATE[:IL_10], HC_STATE[:IL_12], HC_STATE[:IL_1], HC_STATE[:IFN1], HC_STATE[:IL_6], Ag_t0, pars_t0),
        rhs_Th2_linearized(HC_STATE[:Th2], HC_STATE[:act_CD4], HC_STATE[:TGFbeta], HC_STATE[:IL_10], HC_STATE[:IL_12], HC_STATE[:IL_4], HC_STATE[:IL_33], pars_t0),
        rhs_iTreg_linearized(HC_STATE[:iTreg], HC_STATE[:act_CD4], HC_STATE[:TGFbeta], HC_STATE[:IL_10], HC_STATE[:IL_1], pars_t0),
        rhs_CD4_CTL_linearized(HC_STATE[:CD4_CTL], HC_STATE[:act_CD4], HC_STATE[:IFN1], Ag_t0, pars_t0),
        rhs_nTreg_linearized(HC_STATE[:nTreg], HC_STATE[:mDC], pars_t0),
        rhs_TFH_linearized(HC_STATE[:TFH], HC_STATE[:act_CD4], HC_STATE[:IFN1], HC_STATE[:IL_6], Ag_t0, pars_t0),
        rhs_NK_linearized(HC_STATE[:NK], HC_STATE[:IL_12], HC_STATE[:IL_2], HC_STATE[:IFN1], HC_STATE[:IFN_g], pars_t0),
        rhs_act_NK_linearized(HC_STATE[:act_NK], HC_STATE[:NK], HC_STATE[:IL_12], HC_STATE[:IL_2], HC_STATE[:IFN1], HC_STATE[:IFN_g], pars_t0),
        rhs_Naive_B_linearized(HC_STATE[:Naive_B], Ag_t0, pars_t0),
        rhs_Act_B_linearized(HC_STATE[:Act_B], HC_STATE[:Naive_B], Ag_t0, pars_t0),
        rhs_TD_IS_B_linearized(HC_STATE[:TD_IS_B], HC_STATE[:Act_B], HC_STATE[:IL_4], pars_t0),
        rhs_TI_IS_B_linearized(HC_STATE[:TI_IS_B], HC_STATE[:Act_B], HC_STATE[:IFN_g], HC_STATE[:IL_10], pars_t0),
        rhs_IgG4_linearized(HC_STATE[:IgG4], HC_STATE[:TI_IS_B], HC_STATE[:TD_IS_B], pars_t0),
    ]
    
    c2 = [
        rhs_nDC_linearized(IgG_STATE[:nDC], IgG_STATE[:IL_10], IgG_STATE[:GMCSF], Ag_t200, pars_t200),
        rhs_mDC_linearized(IgG_STATE[:mDC], IgG_STATE[:nDC], IgG_STATE[:IL_10], IgG_STATE[:GMCSF], Ag_t200, pars_t200),
        rhs_pDC_linearized(IgG_STATE[:pDC], IgG_STATE[:nDC], Ag_t200, pars_t200),
        rhs_GMCSF_linearized(IgG_STATE[:GMCSF], IgG_STATE[:Th2], IgG_STATE[:act_NK], IgG_STATE[:nDC], IgG_STATE[:IL_10], Ag_t200, pars_t200),
        rhs_IL_33_linearized(IgG_STATE[:IL_33], IgG_STATE[:pDC], IgG_STATE[:act_CD4], pars_t200),
        rhs_IL_6_linearized(IgG_STATE[:IL_6], IgG_STATE[:pDC], IgG_STATE[:mDC], IgG_STATE[:TFH], IgG_STATE[:nTreg], IgG_STATE[:act_CD4], pars_t200),
        rhs_IL_12_linearized(IgG_STATE[:IL_12], IgG_STATE[:mDC], IgG_STATE[:NK], pars_t200),
        rhs_IL_15_linearized(IgG_STATE[:IL_15], IgG_STATE[:naive_CD4], IgG_STATE[:act_CD4], Ag_t200, pars_t200),
        rhs_IL_7_linearized(IgG_STATE[:IL_7], IgG_STATE[:naive_CD4], IgG_STATE[:act_CD4], pars_t200),
        rhs_IFN1_linearized(IgG_STATE[:IFN1], IgG_STATE[:pDC], IgG_STATE[:act_CD4], IgG_STATE[:NK], pars_t200),
        rhs_IL_1_linearized(IgG_STATE[:IL_1], IgG_STATE[:mDC], pars_t200),
        rhs_IL_2_linearized(IgG_STATE[:IL_2], IgG_STATE[:act_CD4], IgG_STATE[:naive_CD4], IgG_STATE[:NK], Ag_t200, pars_t200),
        rhs_IL_4_linearized(IgG_STATE[:IL_4], IgG_STATE[:Th2], IgG_STATE[:act_CD4], Ag_t200, pars_t200),
        rhs_IL_10_linearized(IgG_STATE[:IL_10], IgG_STATE[:iTreg], IgG_STATE[:nTreg], IgG_STATE[:mDC], IgG_STATE[:act_CD4], pars_t200),
        rhs_TGFbeta_linearized(IgG_STATE[:TGFbeta], IgG_STATE[:iTreg], IgG_STATE[:CD4_CTL], IgG_STATE[:nTreg], IgG_STATE[:mDC], IgG_STATE[:act_CD4], pars_t200),
        rhs_IFN_g_linearized(IgG_STATE[:IFN_g], IgG_STATE[:CD4_CTL], IgG_STATE[:act_NK], IgG_STATE[:NK], pars_t200),
        rhs_naive_CD4_linearized(IgG_STATE[:naive_CD4], IgG_STATE[:IL_15], IgG_STATE[:IL_7], IgG_STATE[:mDC], IgG_STATE[:IL_2], pars_t200),
        rhs_act_CD4_linearized(IgG_STATE[:act_CD4], IgG_STATE[:naive_CD4], IgG_STATE[:mDC], IgG_STATE[:IL_2], IgG_STATE[:IL_15], IgG_STATE[:IL_7], IgG_STATE[:IL_4], IgG_STATE[:IL_33], IgG_STATE[:TGFbeta], IgG_STATE[:IL_10], IgG_STATE[:IL_12], IgG_STATE[:IL_1], IgG_STATE[:IFN1], IgG_STATE[:IL_6], Ag_t200, pars_t200),
        rhs_Th2_linearized(IgG_STATE[:Th2], IgG_STATE[:act_CD4], IgG_STATE[:TGFbeta], IgG_STATE[:IL_10], IgG_STATE[:IL_12], IgG_STATE[:IL_4], IgG_STATE[:IL_33], pars_t200),
        rhs_iTreg_linearized(IgG_STATE[:iTreg], IgG_STATE[:act_CD4], IgG_STATE[:TGFbeta], IgG_STATE[:IL_10], IgG_STATE[:IL_1], pars_t200),
        rhs_CD4_CTL_linearized(IgG_STATE[:CD4_CTL], IgG_STATE[:act_CD4], IgG_STATE[:IFN1], Ag_t200, pars_t200),
        rhs_nTreg_linearized(IgG_STATE[:nTreg], IgG_STATE[:mDC], pars_t200),
        rhs_TFH_linearized(IgG_STATE[:TFH], IgG_STATE[:act_CD4], IgG_STATE[:IFN1], IgG_STATE[:IL_6], Ag_t200, pars_t200),
        rhs_NK_linearized(IgG_STATE[:NK], IgG_STATE[:IL_12], IgG_STATE[:IL_2], IgG_STATE[:IFN1], IgG_STATE[:IFN_g], pars_t200),
        rhs_act_NK_linearized(IgG_STATE[:act_NK], IgG_STATE[:NK], IgG_STATE[:IL_12], IgG_STATE[:IL_2], IgG_STATE[:IFN1], IgG_STATE[:IFN_g], pars_t200),
        rhs_Naive_B_linearized(IgG_STATE[:Naive_B], Ag_t200, pars_t200),
        rhs_Act_B_linearized(IgG_STATE[:Act_B], IgG_STATE[:Naive_B], Ag_t200, pars_t200),
        rhs_TD_IS_B_linearized(IgG_STATE[:TD_IS_B], IgG_STATE[:Act_B], IgG_STATE[:IL_4], pars_t200),
        rhs_TI_IS_B_linearized(IgG_STATE[:TI_IS_B], IgG_STATE[:Act_B], IgG_STATE[:IFN_g], IgG_STATE[:IL_10], pars_t200),
        rhs_IgG4_linearized(IgG_STATE[:IgG4], IgG_STATE[:TI_IS_B], IgG_STATE[:TD_IS_B], pars_t200),
    ]
    
    return vcat(c1, c2)
end

"""
构造约束矩阵（在线性化参数空间的参考点处线性化）
"""
function build_constraint_matrix_linearized_space()
    # 使用原始参数的初始值作为参考点
    p_ref_original = Float64.(p0_lin)
    
    # 计算参考点的线性化参数
    pdict_ref = Dict{Symbol,Float64}(param_symbols[i] => p_ref_original[i] for i in eachindex(param_symbols))
    pdict_linearized_ref = build_linearized_param_dict(pdict_ref, HC_STATE)
    
    # 构建参考点的线性化参数向量
    p_ref_linearized = [get(pdict_linearized_ref, sym, 0.1) for sym in linearized_param_symbols]
    
    # 计算雅可比矩阵
    A = ForwardDiff.jacobian(constraint_linearized_space, p_ref_linearized)
    c_ref = constraint_linearized_space(p_ref_linearized)
    b = c_ref - A * p_ref_linearized
    
    return A, b, p_ref_linearized
end

const A_constr_lin, b_constr_lin, p_ref_linearized = build_constraint_matrix_linearized_space()
const p_particular_lin = -A_constr_lin \ b_constr_lin
const B_null_lin = nullspace(A_constr_lin)
const NBASIS_LIN = size(B_null_lin, 2)

println("线性化参数空间约束矩阵形状: ", size(A_constr_lin), ", 零空间维度: ", NBASIS_LIN)

# ============================================================================
# 从线性化参数反解回原始参数（简化版：需要根据实际情况改进）
# ============================================================================
"""
从线性化参数 p_linearized 反解出原始参数 p_original
注意：这是一个简化实现，实际反解可能不唯一
"""
function solve_original_from_linearized(p_linearized::AbstractVector)
    pdict_lin = Dict{Symbol,Float64}(linearized_param_symbols[i] => p_linearized[i] for i in eachindex(linearized_param_symbols))
    pdict_orig = Dict{Symbol,Float64}()
    
    # 直接提取原始参数
    for sym in param_symbols
        if haskey(pdict_lin, sym)
            pdict_orig[sym] = pdict_lin[sym]
        else
            # 对于不在线性化参数中的原始参数，使用启发式反解
            # 简化版：使用默认值（后续可以改进）
            pdict_orig[sym] = 0.1
        end
    end
    
    # TODO: 实现更完整的反解逻辑，例如：
    # - 从 k_f_div_m 反解 k_f 和 k_m（需要额外约束）
    # - 从 k_f_eff 反解 k_f 和 k_m（需要额外约束）
    # 这里简化处理：假设原始参数已经在线性化参数中
    
    return pdict_orig
end

# ============================================================================
# 损失函数（使用原始参数）
# ============================================================================
function loss_with_original_params(pdict_original::Dict{Symbol,Float64})
    new_prob = remake(oprob, p=pdict_original)
    sol = solve(new_prob, QNDF(), saveat=target_time, abstol=1e-8, reltol=1e-6, verbose=false, maxiters=1_000_000)
    if sol.retcode != :Success
        return 1e12, false
    end
    dl = data_loss_L2(sol, ordered_variables)
    sp = smoothness_penalty(sol, ordered_variables)
    smooth_norm = log(max(sp, SMOOTH_EPS))
    return (smooth_norm + SMOOTH_EPS) * dl, true
end

# ============================================================================
# 网格搜索和筛选
# ============================================================================
"""
在零空间中生成网格点
"""
function generate_grid_points(n_dims, n_points_per_dim=3, range_scale=2.0)
    grid_points = Vector{Float64}[]
    ranges = [range_scale * (-1.0 + 2.0 * i / (n_points_per_dim - 1)) for i in 0:(n_points_per_dim-1)]
    
    # 使用笛卡尔积生成所有组合
    # 对于高维情况，使用递归方式生成网格点
    function generate_recursive(dims_left, current_point)
        if dims_left == 0
            push!(grid_points, copy(current_point))
            return
        end
        for val in ranges
            push!(current_point, val)
            generate_recursive(dims_left - 1, current_point)
            pop!(current_point)
        end
    end
    
    generate_recursive(n_dims, Float64[])
    
    return grid_points
end

println("开始网格搜索...")
grid_alphas = generate_grid_points(NBASIS_LIN, 3, 2.0)
println("生成了 ", length(grid_alphas), " 个网格点")

candidates = []
for (idx, alpha) in enumerate(grid_alphas)
    if idx % 100 == 0
        println("处理网格点 $idx / $(length(grid_alphas))")
    end
    
    # 计算线性化参数
    p_lin = p_particular_lin .+ B_null_lin * alpha
    
    # 检查非负性
    if any(p_lin .< 0)
        continue
    end
    
    # 反解原始参数
    pdict_orig = solve_original_from_linearized(p_lin)
    
    # 检查原始参数非负性
    if any(values(pdict_orig) .< 0)
        continue
    end
    
    # 计算损失
    loss_val, success = loss_with_original_params(pdict_orig)
    
    if success && isfinite(loss_val)
        push!(candidates, (alpha=alpha, p_lin=p_lin, p_orig=pdict_orig, loss=loss_val))
    end
end

println("找到 ", length(candidates), " 个有效候选点")

# 按损失排序，选择前15个
sort!(candidates, by=x -> x.loss)
top15 = candidates[1:min(15, length(candidates))]

println("前15个候选点的损失值:")
for (i, cand) in enumerate(top15)
    println("  $i: loss = ", cand.loss)
end

# ============================================================================
# 对前15个候选点进行梯度下降优化（使用 Optim 包）
# ============================================================================
"""
带对数惩罚的损失函数（在零空间系数 alpha 上）
"""
function loss_with_log_penalty(alpha::Vector{Float64})
    # 计算线性化参数
    p_lin = p_particular_lin .+ B_null_lin * alpha
    
    # 对数惩罚：保证非负
    log_penalty = 0.0
    for p_val in p_lin
        if p_val <= 0
            log_penalty += 1e8 * abs2(p_val)
        else
            # 软惩罚：鼓励参数不要太小
            log_penalty -= 1e-6 * log(p_val + 1e-12)
        end
    end
    
    # 反解原始参数
    pdict_orig = solve_original_from_linearized(p_lin)
    
    # 检查原始参数非负性
    for (k, v) in pdict_orig
        if v <= 0
            return 1e12 + log_penalty
        end
    end
    
    # 计算ODE损失
    loss_val, success = loss_with_original_params(pdict_orig)
    
    if !success
        return 1e12 + log_penalty
    end
    
    return loss_val + log_penalty
end

println("\n开始对前15个候选点进行优化...")
optimized_results = []

for (i, cand) in enumerate(top15)
    println("优化候选点 $i / $(length(top15))...")
    
    alpha0 = cand.alpha
    
    # 使用 L-BFGS 优化
    result = optimize(loss_with_log_penalty, alpha0, LBFGS(),
                     Optim.Options(iterations=500, g_tol=1e-6))
    
    alpha_opt = result.minimizer
    loss_opt = result.minimum
    
    # 计算最优的原始参数
    p_lin_opt = p_particular_lin .+ B_null_lin * alpha_opt
    pdict_orig_opt = solve_original_from_linearized(p_lin_opt)
    
    push!(optimized_results, (
        alpha=alpha_opt,
        p_lin=p_lin_opt,
        p_orig=pdict_orig_opt,
        loss=loss_opt,
        converged=Optim.converged(result)
    ))
    
    println("  优化后损失: ", loss_opt, ", 收敛: ", Optim.converged(result))
end

# 选择最优结果
best_result = argmin(r -> r.loss, optimized_results)
println("\n最优结果:")
println("  损失值: ", best_result.loss)
println("  收敛: ", best_result.converged)

# ============================================================================
# 用最优参数跑一次 ODE 并画图（参考 optimize.jl）\n# ============================================================================
best_p = best_result.p_orig
best_prob = remake(oprob, p=best_p)
best_sol = solve(best_prob, QNDF(), saveat=target_time, abstol=1e-8, reltol=1e-6, verbose=false, maxiters=1_000_000)

println("best_sol retcode=", best_sol.retcode)

n_cols = 5
ordered_variables_syms = Symbol.(model_variable_names)
n_rows = ceil(Int, length(ordered_variables_syms) / n_cols)
plt = plot(layout=(n_rows, n_cols), size=(1400, 900))
for (i, v) in enumerate(ordered_variables_syms)
    p = plot(target_time, target_data[v], label="target", lw=2)
    y = try
        best_sol[v]
    catch
        nothing
    end
    if y !== nothing
        plot!(p, target_time, y, label="pred", lw=2)
    end
    title!(p, String(v))
    plot!(plt, p, subplot=i)
end
savefig(plt, "optimization_constrained_result.png")
println("已保存: optimization_constrained_result.png")

