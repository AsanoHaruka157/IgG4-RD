#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IgG4-RD 参数优化脚本
目标：找到参数使系统从 HC_state 在抗原刺激后演化到 IgG_state
"""
import numpy as np
from scipy.optimize import basinhopping
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import torch
from pytorch import rhs_torch, simulate_torch_grad, TORCHDIFFEQ_AVAILABLE

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
T_STEADY = 200         # 稳态评估起始时间

# 归一化与约束配置
BASE_FLOOR = 1e-8
RESIDUAL_TOL = 1e-4      # RHS残差的理想上限
RESIDUAL_TARGET = 5e-5    # 多解筛选阈值
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESIDUAL_SCALE_VEC_T = torch.tensor(RESIDUAL_SCALE_VEC, dtype=torch.float64, device=DEVICE)

# 参考稳态用于设置容量参数的中心值
hc_ref = HC_state()
igg_ref = IgG_state()

# 初始参数（基于文献参考值）
p_init = IgG_param()

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
    if t < T_ANTIGEN_START:
        return 0.0
    if t <= T_ANTIGEN_END:
        return (t - T_ANTIGEN_START) / (T_ANTIGEN_END - T_ANTIGEN_START)
    return 1.0

def rhs_wrapper(t, y, p):
    y_mod = y.copy()
    y_mod[0] = antigen(t)
    dydt = rhs_base(t, y_mod, p)
    dydt[0] = 1.0 / (T_ANTIGEN_END - T_ANTIGEN_START) if T_ANTIGEN_START <= t <= T_ANTIGEN_END else 0.0
    return dydt

def solve_scipy(y0, p, t_eval):
    for method in ("BDF", "Radau"):
        try:
            sol = solve_ivp(lambda t, y: rhs_wrapper(t, y, p), (t_eval[0], t_eval[-1]),
                            y0, method=method, t_eval=t_eval, rtol=1e-5, atol=1e-8)
            if sol.success:
                return sol.y.T
        except Exception:
            continue
    return None


def solve_torch(y0, p, t_eval):
    """使用torchdiffeq求解；失败时返回None让上层回退到scipy。"""
    if not TORCHDIFFEQ_AVAILABLE:
        return None
    try:
        t_torch = torch.tensor(t_eval, dtype=torch.float64, device=DEVICE)
        y0_torch = torch.tensor(y0, dtype=torch.float64, device=DEVICE)
        p_torch = {}
        for k, v in p.items():
            val = float(v)
            if k in ("k_IgG4_TI_f", "k_IgG4_TD_f"):
                val *= 1e-8  # 抵消 rhs_torch 内部的 1e8 缩放
            p_torch[k] = torch.tensor(val, dtype=torch.float64, device=DEVICE)
        sol_t = simulate_torch_grad(t_torch, y0_torch, p_torch)
        sol_np = sol_t.detach().cpu().numpy()
        return sol_np
    except Exception as exc:
        print(f"torchdiffeq求解失败，回退scipy: {type(exc).__name__}: {exc}")
        return None


def solve(y0, p, t_eval):
    sol = solve_torch(y0, p, t_eval)
    if sol is not None:
        return sol
    return solve_scipy(y0, p, t_eval)

def apply_params(x):
    p = dict(p_init)
    for i, n in enumerate(p_names):
        lb, ub = theta_bounds[i]
        theta = np.clip(x[i], lb, ub)
        p[n] = np.exp(theta)
    return p


def apply_params_torch(x_torch: torch.Tensor):
    """将log参数Tensor映射为torch参数字典（保留requires_grad）。"""
    p = {}
    for i, n in enumerate(p_names):
        lb, ub = theta_bounds[i]
        theta = torch.clamp(x_torch[i], min=lb, max=ub)
        val = torch.exp(theta)
        if n in ("k_IgG4_TI_f", "k_IgG4_TD_f"):
            val = val * 1e-8  # torch版RHS内部有1e8缩放，保持与utils.rhs一致的实际放大倍数
        p[n] = val
    # 填充未优化的参数为常量1.0（不需要梯度）
    for n in p_init:
        if n not in p:
            p[n] = torch.tensor(float(p_init[n]), dtype=torch.float64, device=DEVICE)
    return p

def compute_residual_vectors(t_eval, sol, p):
    """计算两端时间窗口内的平均RHS残差向量。"""
    rhs_vals = np.zeros_like(sol)  # 为RHS计算预分配数组
    for idx, t in enumerate(t_eval):  # 遍历时间节点
        rhs_vals[idx] = rhs_wrapper(t, sol[idx], p)  # 针对每个时间节点计算RHS
    pre_mask = t_eval < T_ANTIGEN_START  # 标记刺激前时间段
    post_mask = t_eval >= T_STEADY  # 标记稳态评估时间段
    if not pre_mask.any() or not post_mask.any():  # 若窗口为空则返回无穷惩罚
        return None, None
    pre_vec = rhs_vals[pre_mask].mean(axis=0)  # 刺激前平均残差向量
    post_vec = rhs_vals[post_mask].mean(axis=0)  # 稳态后平均残差向量
    return pre_vec, post_vec  # 返回两段残差


def compute_residual_vectors_torch(t_eval_t: torch.Tensor, sol_t: torch.Tensor, p_t: dict):
    """torch版本的残差向量计算，用于自动微分。"""
    rhs_vals = []
    for idx, t in enumerate(t_eval_t):
        rhs_vals.append(rhs_torch(float(t), sol_t[idx], p_t))
    rhs_stack = torch.stack(rhs_vals, dim=0)
    pre_mask = t_eval_t < T_ANTIGEN_START
    post_mask = t_eval_t >= T_STEADY
    if not bool(pre_mask.any()) or not bool(post_mask.any()):
        return None, None
    pre_vec = rhs_stack[pre_mask].mean(dim=0)
    post_vec = rhs_stack[post_mask].mean(dim=0)
    return pre_vec, post_vec

# ============================================================================
# STEP 3: 损失函数（torch自动微分 + torchdiffeq）
# ============================================================================
LAST_EVAL = {"x": None, "val": None, "grad": None}


def loss_and_grad(x, verbose=False):
    """返回 (loss, grad)。默认使用torchdiffeq + autograd，失败时返回大惩罚。"""
    if not TORCHDIFFEQ_AVAILABLE:
        # 回退：无梯度版本，仅保证不崩溃
        p = apply_params(x)
        t_eval = np.linspace(0, T_END, 301)
        sol = solve_scipy(y0, p, t_eval)
        if sol is None or np.isnan(sol).any() or np.isinf(sol).any():
            return 1e12, np.zeros_like(x)
        pre_vec, post_vec = compute_residual_vectors(t_eval, sol, p)
        if pre_vec is None or post_vec is None:
            return 1e12, np.zeros_like(x)
        pre_scaled = pre_vec / RESIDUAL_SCALE_VEC
        post_scaled = post_vec / RESIDUAL_SCALE_VEC
        loss = np.sum(pre_scaled ** 2) + np.sum(post_scaled ** 2)
        return loss, np.zeros_like(x)

    x_np = np.asarray(x, dtype=float)
    x_torch = torch.tensor(x_np, dtype=torch.float64, device=DEVICE, requires_grad=True)
    p_torch = apply_params_torch(x_torch)
    t_eval_t = torch.linspace(0.0, T_END, steps=301, dtype=torch.float64, device=DEVICE)
    y0_t = torch.tensor(y0, dtype=torch.float64, device=DEVICE)

    try:
        sol_t = simulate_torch_grad(t_eval_t, y0_t, p_torch)
    except Exception:
        return 1e12, np.zeros_like(x_np)

    if torch.isnan(sol_t).any() or torch.isinf(sol_t).any() or bool((sol_t < -1e-6).any()):
        return 1e12, np.zeros_like(x_np)

    pre_vec, post_vec = compute_residual_vectors_torch(t_eval_t, sol_t, p_torch)
    if pre_vec is None or post_vec is None:
        return 1e12, np.zeros_like(x_np)
    if not (torch.isfinite(pre_vec).all() and torch.isfinite(post_vec).all()):
        return 1e12, np.zeros_like(x_np)

    pre_scaled = pre_vec / RESIDUAL_SCALE_VEC_T
    post_scaled = post_vec / RESIDUAL_SCALE_VEC_T
    loss_t = torch.sum(pre_scaled ** 2) + torch.sum(post_scaled ** 2)
    loss_val = loss_t.detach().cpu().item()
    loss_t.backward()
    grad = x_torch.grad.detach().cpu().numpy()

    if verbose:
        pre_norm = pre_vec.norm().detach().cpu().item()
        post_norm = post_vec.norm().detach().cpu().item()
        print(f"Loss={loss_val:.2e}, pre_norm={pre_norm:.2e}, post_norm={post_norm:.2e}")

    return loss_val, grad


def loss_fn(x, verbose=False):
    val, grad = loss_and_grad(x, verbose)
    LAST_EVAL["x"] = np.array(x, copy=True)
    LAST_EVAL["val"] = val
    LAST_EVAL["grad"] = grad
    return val


def loss_grad(x):
    if LAST_EVAL["x"] is not None and np.allclose(x, LAST_EVAL["x"]):
        return LAST_EVAL["grad"]
    _, grad = loss_and_grad(x)
    LAST_EVAL["x"] = np.array(x, copy=True)
    LAST_EVAL["grad"] = grad
    return grad

# ============================================================================
# STEP 4: 多起点优化（Basinhopping）
# ============================================================================
print("\n" + "="*70)
print("开始多起点Basinhopping优化...")
print("="*70)

# Basinhopping设置
minimizer_kwargs = {  # 配置局部优化器参数
    "method": "L-BFGS-B",  # 选择带约束的一阶方法
    "jac": loss_grad,  # 使用torch自动微分梯度
    "bounds": theta_bounds,  # 指定θ的范围
}

multi_results = []  # 存放满足条件的解
max_attempts = 30  # 设定最大尝试次数

for attempt in range(max_attempts):  # 逐次尝试不同初始点
    seed_val = attempt  # 固定随机种子以便复现
    print(f"尝试 {attempt + 1}/{max_attempts}，随机种子={seed_val}")  # 输出当前尝试信息
    bh_result = basinhopping(  # 调用Basinhopping全局搜索
        loss_fn, x0,
        niter=200,
        minimizer_kwargs=minimizer_kwargs,
        seed=seed_val,
        stepsize=0.5,
    )

    x_candidate = bh_result.x  # 提取候选θ
    p_candidate = apply_params(x_candidate)  # 转换为实际参数
    t_eval_dense = np.linspace(0, T_END, 601)  # 使用更细的时间网格复核
    sol_candidate = solve(y0, p_candidate, t_eval_dense)  # 计算轨迹

    if sol_candidate is None:  # 如果求解失败则跳过
        print("  轨迹求解失败，跳过")
        continue

    pre_vec, post_vec = compute_residual_vectors(t_eval_dense, sol_candidate, p_candidate)  # 评估残差向量
    if pre_vec is None or post_vec is None:  # 若窗口不完整则跳过
        print("  残差窗口无效，跳过")
        continue
    pre_norm = np.linalg.norm(pre_vec)  # 计算刺激前残差范数
    post_norm = np.linalg.norm(post_vec)  # 计算稳态后残差范数
    final_loss = loss_fn(x_candidate)  # 计算最终损失
    print(f"  损失={final_loss:.3e}，前残差={pre_norm:.3e}，后残差={post_norm:.3e}")  # 输出评估结果

    if pre_norm <= RESIDUAL_TARGET and post_norm <= RESIDUAL_TARGET:  # 检查残差是否满足要求
        multi_results.append({  # 记录解及其信息
            "result": bh_result,
            "params": p_candidate,
            "solution": sol_candidate,
            "pre_norm": pre_norm,
            "post_norm": post_norm,
            "loss": final_loss,
        })
        print(f"  已找到满足条件的第 {len(multi_results)} 组参数")  # 输出累积数量
        if len(multi_results) >= 3:  # 收集到三组后结束
            break

if len(multi_results) < 3:  # 判断是否成功收集足够解
    print("未能找到满足条件的三组参数，请考虑调整阈值或增加尝试次数")

# 选取最优解用于后续展示
if multi_results:  # 若已找到解则选出最优
    best_entry = min(multi_results, key=lambda x: x["loss"])  # 按损失排序获取最优解
    x_opt = best_entry["result"].x  # 取出最优θ
    p_opt = best_entry["params"]  # 取出最优参数
    sol_opt = best_entry["solution"]  # 取出最优轨迹
else:  # 若没有满足条件的解
    x_opt = x0  # 无解时退回初始猜测
    p_opt = apply_params(x0)  # 返回初始参数
    sol_opt = solve(y0, p_opt, np.linspace(0, T_END, 601))  # 求解初始轨迹

# ============================================================================
# STEP 5: 验证并保存
# ============================================================================

# 验证最终轨迹
t_eval = np.linspace(0, T_END, 601)  # 使用细时间网格
sol_opt = solve(y0, p_opt, t_eval)  # 计算最优轨迹
y_final = sol_opt[t_eval >= T_STEADY].mean(axis=0)  # 取稳态平均
pre_vec_opt, post_vec_opt = compute_residual_vectors(t_eval, sol_opt, p_opt)  # 记录最优解的残差向量
pre_res_opt = np.linalg.norm(pre_vec_opt) if pre_vec_opt is not None else float("inf")  # 计算前段范数
post_res_opt = np.linalg.norm(post_vec_opt) if post_vec_opt is not None else float("inf")  # 计算后段范数

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

if multi_results:  # 如果存在满足条件的解则输出摘要
    print("\n" + "="*70)
    print("多解摘要")
    print("="*70)
    for idx, entry in enumerate(multi_results, 1):  # 遍历每组解
        print(f"解#{idx}: 损失={entry['loss']:.3e}, 前残差={entry['pre_norm']:.3e}, 后残差={entry['post_norm']:.3e}")

# ============================================================================
# STEP 6: 保存优化参数到 utils.py
# ============================================================================
print("\n" + "="*70)
print("保存优化参数...")
print("="*70)

with open('utils.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 删除旧的 param_optimized 或 param_optimized_candidates
for signature in ('def param_optimized():', 'def param_optimized_candidates():'):
    if signature in content:
        start = content.find(signature)
        end = content.find('\ndef ', start + 1)
        if end == -1:
            end = len(content)
        content = content[:start] + content[end:]

lines = ["\ndef param_optimized_candidates():",  # 定义新的候选参数函数
         '    """返回Basinhopping找到的候选参数列表"""',
         "    candidates = []"]

for entry in multi_results if multi_results else [{"params": p_opt}]:  # 遍历候选参数集合
    lines.append("    params = {}")  # 为每个候选解创建字典
    for n in sorted(entry["params"].keys()):  # 保持参数名排序
        lines.append(f'    params["{n}"] = {entry["params"][n]:.10e}')  # 写入参数值
    lines.append("    candidates.append(params)")  # 将参数集加入列表
    lines.append("")  # 插入空行增强可读性

lines.append("    return candidates\n")  # 返回所有候选解

content = content.rstrip() + "\n" + "\n".join(lines)  # 将新函数拼接回utils

with open('utils.py', 'w', encoding='utf-8') as f:  # 写回utils文件
    f.write(content)

print("已保存 param_optimized_candidates() 到 utils.py")
# ============================================================================
# STEP 7: 绘图
# ============================================================================
print("\n绘制轨迹图...")

# 初始参数轨迹（对比）
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

# 清理多余的子图
for idx in range(len(plot_vars), len(axes)):
    axes[idx].axis('off')

axes[0].legend(fontsize=7, loc='upper right')
plt.tight_layout()
plt.savefig('optimization_result.png', dpi=150)
plt.show()

print("\n完成!")
