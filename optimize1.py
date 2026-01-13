#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IgG4-RD 参数优化脚本
目标：找到参数使系统从 HC_state 在抗原刺激后演化到 IgG_state
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import torch
from pytorch import rhs_torch, simulate_torch_grad, TORCHDIFFEQ_AVAILABLE

try:
    import nevergrad as ng
    NG_AVAILABLE = True
except ImportError:
    NG_AVAILABLE = False
    print("Warning: nevergrad not available, will fall back to scipy differential_evolution")

from scipy.optimize import differential_evolution, minimize

from utils import (
    rhs as rhs_base, HC_state, IgG_param, IgG_state,
    STATE_NAMES, IDX, ALL_PARAMS,
)

# ============================================================================
# 配置
# ============================================================================
T_END = 300.0          # 模拟总时间
T_ANTIGEN_START = 100  # 抗原刺激开始
T_ANTIGEN_END = 150    # 抗原刺激结束（线性爬升）
# 平滑抗原激活：sigmoid 中心与温度（越小越陡）
ANTIGEN_CENTER = 0.5 * (T_ANTIGEN_START + T_ANTIGEN_END)
ANTIGEN_TEMP = 5.0
T_STEADY = 200         # 稳态评估起始时间
T_EVAL_STEPS = 151     # 积分时间网格点数（Stage A 粗网格即可，进一步降采样提速）

# 归一化与约束配置
BASE_FLOOR = 1e-8
RESIDUAL_TOL = 1e-4      # RHS残差的理想上限
RESIDUAL_TARGET = 1e-3    # 多解筛选阈值（放宽以加快测试）
RESIDUAL_ALPHA = 1.0
RESIDUAL_BETA = 1e-6

# ============================================================================
# STEP 1: 准备
# ============================================================================
print("="*70)
print("IgG4 Parameter Optimization")
print("  HC_state → (antigen) → IgG_state")
print("="*70)

# 初始条件（用户提供的健康人参考值）
y0 = np.array([HC_state()[name] for name in STATE_NAMES])

# 目标状态
target_dict = IgG_state()
target_y = np.array([target_dict[name] for name in STATE_NAMES])

# 依据HC与IgG目标的量级构建每个状态的残差归一化尺度
scale_base = np.maximum(np.abs(y0), np.abs(target_y))  # 取两端最大幅值
scale_base = np.maximum(scale_base, np.ones_like(scale_base))  # 至少保证尺度为1
RESIDUAL_SCALE_VEC = RESIDUAL_ALPHA * scale_base + RESIDUAL_BETA  # 依照α、β得到最终尺度
RESIDUAL_SCALE_VEC_T = torch.tensor(RESIDUAL_SCALE_VEC, dtype=torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESIDUAL_SCALE_VEC_T = RESIDUAL_SCALE_VEC_T.to(device=DEVICE)

# 缓存固定时间网格与初始状态，避免在每次 loss_fn 中重复创建小张量
T_EVAL_T = torch.linspace(0.0, T_END, steps=T_EVAL_STEPS, dtype=torch.float64, device=DEVICE)
Y0_T = torch.tensor(y0, dtype=torch.float64, device=DEVICE)

# 参考稳态用于设置容量参数的中心值
hc_ref = HC_state()
igg_ref = IgG_state()

# 初始参数（基于文献参考值）
p_init = IgG_param()

# 常量参数torch缓存（非优化参数）
P_CONST_TORCH = {}
for k, v in p_init.items():
    P_CONST_TORCH[k] = torch.tensor(float(v), dtype=torch.float64, device=DEVICE)

# 需要优化的参数
p_names = [n for n in ALL_PARAMS if n in p_init]

# 参数重参数化：k = k0 * 10^x，其中 x 受分组约束
CARRYING_CAPACITY_MAP = {
    "k_nDC_m": "nDC",
    "k_mDC_m": "mDC",
    "k_pDC_m": "pDC",
    "k_CD4_m": "naive_CD4",
    "k_act_CD4_m": "act_CD4",
    "k_Th2_m": "Th2",
    "k_iTreg_m": "iTreg",
    "k_CD4_CTL_m": "CD4_CTL",
    "k_nTreg_m": "nTreg",
    "k_TFH_m": "TFH",
    "k_NK_m": "NK",
    "k_act_NK_m": "act_NK",
    "k_Naive_B_m": "Naive_B",
    "k_Act_B_m": "Act_B",
    "k_TD_m": "TD_IS_B",
    "k_TI_m": "TI_IS_B",
}

def carrying_reference(state_name: str) -> float:
    """根据HC/IgG目标挑选容量参数的参考量级。"""
    return max(hc_ref.get(state_name, 0.0), igg_ref.get(state_name, 0.0), 1.0)

theta_bounds = []
x0 = np.zeros(len(p_names))
for idx, name in enumerate(p_names):
    raw = max(p_init.get(name, 0.0), BASE_FLOOR)

    if name.endswith("_d"):
        lower, upper = 0.05, 5.0
    elif name.endswith("_f"):
        lower, upper = 0.01, 20.0
    elif name.endswith("_m"):
        if name in CARRYING_CAPACITY_MAP:
            state_name = CARRYING_CAPACITY_MAP[name]
            target_val = target_dict.get(state_name)
            if target_val is None or target_val <= 0:
                target_val = carrying_reference(state_name)
            target_val = max(target_val, 1.0)
            lower, upper = 0.5 * target_val, 5.0 * target_val
        else:
            lower, upper = 1e3, 1e6
    else:
        base_val = raw
        lower = max(base_val * 0.1, BASE_FLOOR)
        upper = max(base_val * 10.0, lower * 10.0)

    lower = max(lower, BASE_FLOOR)
    if upper <= lower:
        upper = lower * 10.0

    theta_bounds.append((np.log(lower), np.log(upper)))
    init_val = np.clip(raw, lower, upper)
    x0[idx] = np.log(init_val)

print(f"初始条件: HC_state (IgG4={y0[IDX['IgG4']]:.1f})")
print(f"目标: IgG_state (IgG4={target_y[IDX['IgG4']]:.1f})")
print(f"优化参数: {len(p_names)}个")

# ============================================================================
# STEP 2: ODE求解器
# ============================================================================
def antigen(t):
    """平滑的抗原输入，避免分段不光滑带来的刚性与梯度噪声。"""
    return 1.0 / (1.0 + np.exp(-(t - ANTIGEN_CENTER) / ANTIGEN_TEMP))

def rhs_wrapper(t, y, p):
    y_mod = y.copy()
    y_mod[0] = antigen(t)
    dydt = rhs_base(t, y_mod, p)
    # sigmoid 的导数：sigma'(x) = sigma(x)*(1-sigma(x)) / temp
    sigma = y_mod[0]
    dydt[0] = sigma * (1.0 - sigma) / ANTIGEN_TEMP
    return dydt

def solve_scipy(y0, p, t_eval):
    for method in ("BDF", "Radau"):
        try:
            sol = solve_ivp(lambda t, y: rhs_wrapper(t, y, p), (t_eval[0], t_eval[-1]),
                            y0, method=method, t_eval=t_eval, rtol=1.5e-5, atol=1.5e-8)
            if sol.success:
                return sol.y.T
        except Exception:
            continue
    return None
    

def solve(y0, p, t_eval):
    return solve_scipy(y0, p, t_eval)

def apply_params(x):
    p = dict(p_init)
    for i, n in enumerate(p_names):
        lb, ub = theta_bounds[i]
        theta = np.clip(x[i], lb, ub)
        p[n] = np.exp(theta)
    return p


def apply_params_torch(x_torch: torch.Tensor):
    """将log参数Tensor映射为torch参数字典（少量对象创建）。"""
    p = {}
    for i, n in enumerate(p_names):
        lb, ub = theta_bounds[i]
        theta = torch.clamp(x_torch[i], min=lb, max=ub)
        p[n] = torch.exp(theta)
    for n, v in P_CONST_TORCH.items():
        if n not in p:
            p[n] = v
    return p

def compute_residual_vectors(t_eval, sol, p):
    """计算两端时间窗口内的平均RHS残差向量。"""
    rhs_vals = np.zeros_like(sol)  # 为RHS计算预分配数组
    for idx, t in enumerate(t_eval):  # 遍历时间节点
        rhs_vals[idx] = rhs_wrapper(t, sol[idx], p)  # 针对每个时间节点计算RHS
    pre_mask = t_eval < T_ANTIGEN_START  # 标记刺激前时间段
    post_mask = t_eval > T_STEADY  # 标记稳态评估时间段
    if not pre_mask.any() or not post_mask.any():  # 若窗口为空则返回无穷惩罚
        return None, None
    pre_vec = rhs_vals[pre_mask].mean(axis=0)  # 刺激前平均残差向量
    post_vec = rhs_vals[post_mask].mean(axis=0)  # 稳态后平均残差向量
    return pre_vec, post_vec  # 返回两段残差

# ============================================================================
# STEP 3: 损失函数（Stage A：窗口平均RHS，使用 torchdiffeq 积分但无梯度）
# ============================================================================
LAST_EVAL = {"x": None, "val": None, "pre": None, "post": None, "traj": None, "t_eval": None}


def loss_fn(x, verbose=False):
    if not TORCHDIFFEQ_AVAILABLE:
        return 1e12

    x_np = np.asarray(x, dtype=float)
    x_t = torch.tensor(x_np, dtype=torch.float64, device=DEVICE)
    p_t = apply_params_torch(x_t)
    # 复用全局缓存的时间网格和初始状态，减少重复张量构造
    t_eval_t = T_EVAL_T
    y0_t = Y0_T

    try:
        with torch.no_grad():
            # 使用显式可微求解器 dopri5（即便 no_grad，这个分支更快且无 BDF 警告）
            sol_t = simulate_torch_grad(t_eval_t, y0_t, p_t, verbose=False, method_override="dopri5")
    except Exception:
        return 1e12

    # 部分 torchdiffeq 解算器可能返回较少的时间点（例如 scipy_solver）
    # 这里对齐时间网格，避免掩码与轨迹长度不一致导致的索引错误
    if sol_t.shape[0] != t_eval_t.shape[0]:
        min_steps = min(sol_t.shape[0], t_eval_t.shape[0])
        sol_t = sol_t[:min_steps]
        t_eval_t = t_eval_t[:min_steps]

    if torch.isnan(sol_t).any() or torch.isinf(sol_t).any():
        return 1e12

    # 批量计算RHS；避免 functorch 在时间分支上的数据依赖报错
    rhs_list = []
    for tt, yy in zip(t_eval_t, sol_t):
        rhs_list.append(rhs_torch(tt, yy, p_t))
    rhs_stack = torch.stack(rhs_list, dim=0)

    pre_mask = t_eval_t < T_ANTIGEN_START
    post_mask = t_eval_t > T_STEADY
    if not bool(pre_mask.any()) or not bool(post_mask.any()):
        return 1e12

    pre_vec = rhs_stack[pre_mask].mean(dim=0)
    post_vec = rhs_stack[post_mask].mean(dim=0)

    pre_scaled = pre_vec / RESIDUAL_SCALE_VEC_T
    post_scaled = post_vec / RESIDUAL_SCALE_VEC_T
    loss_t = torch.sum(pre_scaled ** 2) + torch.sum(post_scaled ** 2)
    loss = float(loss_t.cpu().item())

    LAST_EVAL["x"] = np.array(x_np, copy=True)
    LAST_EVAL["val"] = loss
    LAST_EVAL["pre"] = pre_vec.cpu().numpy()
    LAST_EVAL["post"] = post_vec.cpu().numpy()
    LAST_EVAL["traj"] = sol_t.cpu().numpy()
    LAST_EVAL["t_eval"] = t_eval_t.cpu().numpy()

    if verbose:
        pre_norm = float(pre_vec.norm().cpu().item())
        post_norm = float(post_vec.norm().cpu().item())
        print(f"Loss={loss:.2e}, pre_norm={pre_norm:.2e}, post_norm={post_norm:.2e}")

    return loss

# ============================================================================
# STEP 4: Stage A 优化（窗口平均RHS），使用 nevergrad；如缺失则退回 differential_evolution
# ============================================================================
print("\n" + "="*70)
print("Stage A: nevergrad 全局搜索 (窗口平均RHS，torch积分)")
print("="*70)

lower_bounds = np.array([b[0] for b in theta_bounds], dtype=float)
upper_bounds = np.array([b[1] for b in theta_bounds], dtype=float)

best_x = x0.copy()
best_loss = loss_fn(best_x, verbose=False)

if NG_AVAILABLE:
    parametrization = ng.p.Array(init=best_x).set_bounds(lower=lower_bounds, upper=upper_bounds)
    budget = 600  # 略减预算以加速
    optimizer = ng.optimizers.OnePlusOne(parametrization=parametrization, budget=budget, num_workers=1)
    rec = optimizer.minimize(loss_fn)
    best_x = np.array(rec.value, dtype=float)
    best_loss = loss_fn(best_x)
else:
    print("nevergrad 不可用，改用 scipy differential_evolution")
    result = differential_evolution(
        func=lambda v: loss_fn(v),
        bounds=theta_bounds,
        maxiter=200,
        popsize=10,
        polish=False,
        updating="deferred",
        workers=1,
        tol=0.03,
    )
    best_x = np.array(result.x, dtype=float)
    best_loss = result.fun

# 快速局部精修（无梯度，RHS端点损失很廉价）
polish = minimize(
    fun=lambda v: loss_fn(v),
    x0=best_x,
    method="L-BFGS-B",
    bounds=theta_bounds,
    options={"maxiter": 120, "ftol": 1.5e-10}
)
if polish.success and polish.fun < best_loss:
    best_x = np.array(polish.x, dtype=float)
    best_loss = float(polish.fun)

p_opt = apply_params(best_x)
LAST_EVAL_PRE = LAST_EVAL.get("pre")
LAST_EVAL_POST = LAST_EVAL.get("post")
sol_opt = None  # Stage A 不强制积分

# ============================================================================
# STEP 5: 验证并保存（可选轨迹求解）
# ============================================================================

DO_TRAJ_CHECK = True  # 打开轨迹复核与绘图

if DO_TRAJ_CHECK:
    t_eval = np.linspace(0, T_END, T_EVAL_STEPS)  # 与 Stage A 一致的粗网格
    sol_opt = solve(y0, p_opt, t_eval)
    y_final = sol_opt[t_eval >= T_STEADY].mean(axis=0)
    pre_vec_opt, post_vec_opt = compute_residual_vectors(t_eval, sol_opt, p_opt)
    pre_res_opt = np.linalg.norm(pre_vec_opt) if pre_vec_opt is not None else float("inf")
    post_res_opt = np.linalg.norm(post_vec_opt) if post_vec_opt is not None else float("inf")
else:
    y_final = target_y.copy()
    pre_res_opt = np.linalg.norm(LAST_EVAL_PRE) if LAST_EVAL_PRE is not None else float("inf")
    post_res_opt = np.linalg.norm(LAST_EVAL_POST) if LAST_EVAL_POST is not None else float("inf")
    t_eval = None

print("\n" + "="*70)
print("优化结果验证")
print("="*70)
print(f"{'变量':<15} {'目标':<12} {'实际':<12} {'偏差%':<10}")
print("-"*50)

key_vars = ['IgG4', 'act_CD4', 'TD Plasma', 'TI Plasma', 'Th2', 'IL_4', 'IL_10']
for v in key_vars:
    i = IDX[v]
    target = target_y[i]
    actual = y_final[i]
    if target > 0:
        dev = abs(actual - target) / target * 100
    else:
        dev = 0 if actual < 1 else float('inf')
    print(f"{v:<15} {target:<12.2e} {actual:<12.2e} {dev:<10.1f}")

print(f"前段残差范数: {pre_res_opt:.4e}")  # 输出刺激前残差范数
print(f"后段残差范数: {post_res_opt:.4e}")  # 输出稳态后残差范数
print(f"Stage A 端点损失: {best_loss:.4e}")


# ============================================================================
# STEP 6: 保存优化参数到 utils.py
# ============================================================================
print("\n" + "="*70)
print("保存优化参数...")
print("="*70)

with open('utils.py', 'r', encoding='utf-8') as f:
    content = f.read()

for signature in ('def param_optimized():', 'def param_optimized_candidates():'):
    if signature in content:
        start = content.find(signature)
        end = content.find('\ndef ', start + 1)
        if end == -1:
            end = len(content)
        content = content[:start] + content[end:]

lines = ["\ndef param_optimized_candidates():",
         '    """返回Stage A找到的候选参数列表"""',
         "    candidates = []",
         "    params = {}"]

for n in sorted(p_opt.keys()):
    lines.append(f'    params["{n}"] = {p_opt[n]:.10e}')
lines.append("    candidates.append(params)")
lines.append("    return candidates\n")

content = content.rstrip() + "\n" + "\n".join(lines)

with open('utils.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("已保存 param_optimized_candidates() 到 utils.py")

# ============================================================================
# STEP 7: 绘图（可选）
# ============================================================================
if DO_TRAJ_CHECK and sol_opt is not None and t_eval is not None:
    print("\n绘制轨迹图...")
    sol_init = solve(y0, p_init, t_eval)

    fig, axes = plt.subplots(6, 5, figsize=(16, 14))
    axes = axes.flatten()

    plot_vars = [n for n in STATE_NAMES if n != "Antigen"]
    for idx, var in enumerate(plot_vars):
        if idx >= len(axes):
            break
        ax = axes[idx]
        i = IDX[var]
        
        ax.plot(t_eval, sol_init[:, i], 'b-', alpha=0.5, label='Initial')
        ax.plot(t_eval, sol_opt[:, i], 'g-', linewidth=1.5, label='Optimized')
        ax.axhline(target_y[i], color='r', linestyle='--', alpha=0.7, label='Target')
        ax.axvline(T_ANTIGEN_START, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(T_ANTIGEN_END, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_title(var, fontsize=9)
        ax.set_xlabel('Time', fontsize=7)
        ax.tick_params(labelsize=7)

    for idx in range(len(plot_vars), len(axes)):
        axes[idx].axis('off')

    axes[0].legend(fontsize=7, loc='upper right')
    plt.tight_layout()
    plt.savefig('optimization_result.png', dpi=150)
    plt.show()

print("\n完成!")
