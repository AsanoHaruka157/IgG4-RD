import Pkg
Pkg.activate(@__DIR__)

using Catalyst
using DifferentialEquations
using Plots
using Optimization
using OptimizationOptimJL  # 使用 Optim.jl 的优化器
using LineSearches
using CSV
using DataFrames
using ForwardDiff
using Statistics
using LinearAlgebra
using Combinatorics

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
    # ... (保持原有的模型定义，这里省略以节省空间)
    # 实际使用时需要包含完整的模型定义
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
p_init = [Symbol(p) => 0.1 for p in parameters(rn)]
oprob = ODEProblem(rn, u0, tspan, p_init)

param_symbols = [pair.first for pair in p_init]

# ============================================================================
# 4. 定义线性化参数空间
# ============================================================================
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

# ============================================================================
# 5. 在线性化参数空间上构造约束矩阵
# ============================================================================
"""
在线性化参数空间 p 上计算约束向量 c(p) = [RHS(t=0); RHS(t=200)]
注意：这里的 p 是线性化参数向量（包含组合参数）
"""
function constraint_linearized_space(p_linearized::AbstractVector)
    # 将向量转换为字典
    pdict_linearized = Dict{Symbol,Float64}()
    for (i, sym) in enumerate(linearized_param_symbols)
        pdict_linearized[sym] = p_linearized[i]
    end
    
    # 从线性化参数反解出原始参数（简化版：假设组合参数已给定，只提取原始参数部分）
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
    pars_t0 = build_linearized_param_dict(pdict_original, HC_STATE)
    pars_t200 = build_linearized_param_dict(pdict_original, IgG_STATE)
    
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

# 构造约束矩阵（在参考点处线性化）
function build_constraint_matrix_linearized_space()
    # 使用原始参数的初始值作为参考点
    p_ref_original = [0.1 for _ in param_symbols]
    
    # 计算参考点的线性化参数
    pdict_ref = Dict{Symbol,Float64}(param_symbols[i] => p_ref_original[i] for i in eachindex(param_symbols))
    pdict_linearized_ref = build_linearized_param_dict(pdict_ref, HC_STATE)
    
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
# 6. 从线性化参数反解回原始参数（简化版：需要根据实际情况改进）
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
            # 这里需要根据实际的组合参数关系进行反解
            # 简化版：使用默认值
            pdict_orig[sym] = 0.1
        end
    end
    
    # TODO: 实现更完整的反解逻辑，例如：
    # - 从 k_f_div_m 反解 k_f 和 k_m
    # - 从 k_f_eff 反解 k_f 和 k_m
    # 这需要额外的约束或优化
    
    return pdict_orig
end

# ============================================================================
# 7. 损失函数（使用原始参数）
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
# 8. 网格搜索和筛选
# ============================================================================
"""
在零空间中生成网格点
"""
function generate_grid_points(n_dims, n_points_per_dim=3, range_scale=2.0)
    # 简化版：在每个维度上均匀采样
    grid_points = Vector{Float64}[]
    ranges = [range_scale * (-1.0 + 2.0 * i / (n_points_per_dim - 1)) for i in 0:(n_points_per_dim-1)]
    
    for combo in Iterators.product([ranges for _ in 1:n_dims]...)
        push!(grid_points, collect(combo))
    end
    
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
# 9. 对前15个候选点进行梯度下降优化
# ============================================================================
using Optim

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
            log_penalty += 1e-6 * log(p_val + 1e-12)
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
# 10. 用最优参数跑ODE并画图
# ============================================================================
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
