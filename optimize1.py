#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IgG4-RD 参数优化脚本（SciPy版）
目标：使用 solve_ivp (BDF) + 显式敏感度方程获取梯度，再用 L-BFGS-B 优化。
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

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
ANTIGEN_CENTER = 0.5 * (T_ANTIGEN_START + T_ANTIGEN_END)
ANTIGEN_TEMP = 5.0
T_STEADY = 200         # 稳态评估起始时间
T_EVAL_STEPS = 151     # 时间网格点数（越小越快）
T_EVAL_COARSE = 81     # 全局随机筛选用的更粗网格

# 归一化与约束配置
BASE_FLOOR = 1e-8
RESIDUAL_TOL = 1e-4      # RHS残差的理想上限（用于早停/惩罚）
RESIDUAL_TARGET = 1e-4    # 筛选阈值
RESIDUAL_ALPHA = 1.0
RESIDUAL_BETA = 1e-6

# 数值微分步长（状态/参数方向）
EPS_Y = 1e-6
EPS_THETA = 1e-6

# ============================================================================
# STEP 1: 准备
# ============================================================================
print("="*70)
print("IgG4 Parameter Optimization (SciPy BDF)")
print("  HC_state → (antigen) → IgG_state")
print("="*70)

# 初始条件与目标
HC = HC_state()
y0 = np.array([HC[name] for name in STATE_NAMES], dtype=float)

target_dict = IgG_state()
target_y = np.array([target_dict[name] for name in STATE_NAMES], dtype=float)

# 残差尺度
scale_base = np.maximum(np.abs(y0), np.abs(target_y))
scale_base = np.maximum(scale_base, np.ones_like(scale_base))
RESIDUAL_SCALE_VEC = RESIDUAL_ALPHA * scale_base + RESIDUAL_BETA

# 参考稳态
hc_ref = HC_state()
igg_ref = IgG_state()

# 初始参数（基于文献参考值）
p_init = IgG_param()

# 需要优化的参数
p_names = [n for n in ALL_PARAMS if n in p_init]

# 参数重参数化：k = exp(theta)
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
    return max(hc_ref.get(state_name, 0.0), igg_ref.get(state_name, 0.0), 1.0)

theta_bounds = []
x0 = np.zeros(len(p_names), dtype=float)
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
# STEP 2: ODE 与敏感度方程
# ============================================================================

def antigen(t: float) -> float:
    return 1.0 / (1.0 + np.exp(-(t - ANTIGEN_CENTER) / ANTIGEN_TEMP))

def rhs_wrapper(t: float, y: np.ndarray, p: dict) -> np.ndarray:
    y_mod = y.copy()
    y_mod[0] = antigen(t)
    dydt = rhs_base(t, y_mod, p)
    sigma = y_mod[0]
    dydt[0] = sigma * (1.0 - sigma) / ANTIGEN_TEMP
    return np.asarray(dydt, dtype=float)

def apply_params(theta: np.ndarray) -> dict:
    p = dict(p_init)
    for i, n in enumerate(p_names):
        lb, ub = theta_bounds[i]
        val = np.exp(np.clip(theta[i], lb, ub))
        p[n] = val
    return p

def solve_state_only(y0_in: np.ndarray, p: dict, t_eval: np.ndarray):
    """仅解状态，不带敏感度，用于快速全局筛选。"""
    try:
        sol = solve_ivp(
            fun=lambda t, y: rhs_wrapper(t, y, p),
            t_span=(t_eval[0], t_eval[-1]),
            y0=y0_in,
            method="BDF",
            t_eval=t_eval,
            rtol=2e-5,
            atol=2e-8,
        )
        return sol
    except Exception:
        return None

def jacobian_actions(t: float, y: np.ndarray, p: dict, S: np.ndarray) -> np.ndarray:
    """返回 J_y*S + J_p*dp/dtheta，S shape (n, m)。"""
    n, m = S.shape
    out = np.zeros((n, m), dtype=float)
    for j in range(m):
        v = S[:, j]
        if np.allclose(v, 0.0):
            Jy_v = np.zeros(n, dtype=float)
        else:
            y_plus = y + EPS_Y * v
            y_minus = y - EPS_Y * v
            f_plus = rhs_wrapper(t, y_plus, p)
            f_minus = rhs_wrapper(t, y_minus, p)
            Jy_v = (f_plus - f_minus) / (2.0 * EPS_Y)

        name = p_names[j]
        base = p[name]
        delta = EPS_THETA * base
        p_plus = dict(p)
        p_minus = dict(p)
        p_plus[name] = base + delta
        p_minus[name] = base - delta
        f_p = rhs_wrapper(t, y, p_plus)
        f_m = rhs_wrapper(t, y, p_minus)
        Jp_col = (f_p - f_m) / (2.0 * delta)

        out[:, j] = Jy_v + Jp_col
    return out

def augmented_ode(t: float, aug: np.ndarray, p: dict, n: int, m: int) -> np.ndarray:
    y = aug[:n]
    S = aug[n:].reshape(n, m)
    f = rhs_wrapper(t, y, p)
    GS = jacobian_actions(t, y, p, S)
    return np.concatenate([f, GS.reshape(-1)])

def solve_with_sens(theta: np.ndarray, t_eval: np.ndarray):
    p = apply_params(theta)
    n = len(y0)
    m = len(p_names)
    y_init = np.zeros((n * (m + 1),), dtype=float)
    y_init[:n] = y0

    sol = solve_ivp(
        fun=lambda t, z: augmented_ode(t, z, p, n, m),
        t_span=(t_eval[0], t_eval[-1]),
        y0=y_init,
        method="BDF",
        t_eval=t_eval,
        rtol=1e-5,
        atol=1e-8,
        jac=None,
    )
    return sol

def compute_rhs_and_grad(t_eval: np.ndarray, y_traj: np.ndarray, S_traj: np.ndarray, p: dict):
    n = y_traj.shape[1]
    m = S_traj.shape[2]
    rhs_vals = np.zeros_like(y_traj)
    df_acc_pre = np.zeros((n, m), dtype=float)
    df_acc_post = np.zeros((n, m), dtype=float)
    pre_cnt = 0
    post_cnt = 0
    for idx, t in enumerate(t_eval):
        y = y_traj[idx]
        S = S_traj[idx]
        f = rhs_wrapper(t, y, p)
        rhs_vals[idx] = f
        if t < T_ANTIGEN_START:
            df = jacobian_actions(t, y, p, S)
            df_acc_pre += df
            pre_cnt += 1
        if t >= T_STEADY:
            df = jacobian_actions(t, y, p, S)
            df_acc_post += df
            post_cnt += 1
    pre_vec = rhs_vals[t_eval < T_ANTIGEN_START].mean(axis=0) if pre_cnt > 0 else None
    post_vec = rhs_vals[t_eval >= T_STEADY].mean(axis=0) if post_cnt > 0 else None
    pre_df_mean = df_acc_pre / pre_cnt if pre_cnt > 0 else None
    post_df_mean = df_acc_post / post_cnt if post_cnt > 0 else None
    return pre_vec, post_vec, pre_df_mean, post_df_mean


def loss_no_grad(theta: np.ndarray, t_steps: int = T_EVAL_COARSE):
    """快速无梯度损失，用于全局随机筛选。"""
    t_eval = np.linspace(0.0, T_END, t_steps)
    p = apply_params(theta)
    sol = solve_state_only(y0, p, t_eval)
    if sol is None or not sol.success:
        return 1e12
    y_traj = sol.y.T
    if np.isnan(y_traj).any() or np.isinf(y_traj).any():
        return 1e12
    if np.max(np.abs(y_traj)) > 1e8:
        return 1e12
    rhs_vals = np.zeros_like(y_traj)
    for idx, t in enumerate(t_eval):
        rhs_vals[idx] = rhs_wrapper(t, y_traj[idx], p)
    pre_mask = t_eval < T_ANTIGEN_START
    post_mask = t_eval >= T_STEADY
    if not pre_mask.any() or not post_mask.any():
        return 1e12
    pre_vec = rhs_vals[pre_mask].mean(axis=0)
    post_vec = rhs_vals[post_mask].mean(axis=0)
    if np.linalg.norm(pre_vec) > 1e6 or np.linalg.norm(post_vec) > 1e6:
        return 1e12
    pre_scaled = pre_vec / RESIDUAL_SCALE_VEC
    post_scaled = post_vec / RESIDUAL_SCALE_VEC
    return float(np.sum(pre_scaled ** 2) + np.sum(post_scaled ** 2))

# ============================================================================
# STEP 3: 损失与梯度
# ============================================================================

def loss_and_grad(theta: np.ndarray):
    t_eval = np.linspace(0.0, T_END, T_EVAL_STEPS)
    sol = solve_with_sens(theta, t_eval)
    if not sol.success:
        return 1e12, np.zeros_like(theta)

    n = len(y0)
    m = len(p_names)
    y_traj = sol.y[:n, :].T
    S_traj = sol.y[n:, :].T.reshape(len(t_eval), n, m)

    if np.isnan(y_traj).any() or np.isinf(y_traj).any():
        return 1e12, np.zeros_like(theta)

    p = apply_params(theta)
    pre_vec, post_vec, pre_df, post_df = compute_rhs_and_grad(t_eval, y_traj, S_traj, p)
    if pre_vec is None or post_vec is None:
        return 1e12, np.zeros_like(theta)

    if np.linalg.norm(pre_vec) > 1e6 or np.linalg.norm(post_vec) > 1e6:
        return 1e12, np.zeros_like(theta)

    pre_scaled = pre_vec / RESIDUAL_SCALE_VEC
    post_scaled = post_vec / RESIDUAL_SCALE_VEC
    loss = float(np.sum(pre_scaled ** 2) + np.sum(post_scaled ** 2))

    grad = np.zeros_like(theta)
    for j in range(len(theta)):
        pre_term = 0.0
        post_term = 0.0
        if pre_df is not None:
            pre_term = np.sum(2.0 * pre_scaled * (pre_df[:, j] / RESIDUAL_SCALE_VEC))
        if post_df is not None:
            post_term = np.sum(2.0 * post_scaled * (post_df[:, j] / RESIDUAL_SCALE_VEC))
        grad[j] = pre_term + post_term

    return loss, grad


def loss_and_grad_steps(theta: np.ndarray, t_steps: int = T_EVAL_STEPS):
    """与 loss_and_grad 相同，但可指定时间步数用于梯度检查。"""
    t_eval = np.linspace(0.0, T_END, t_steps)
    sol = solve_with_sens(theta, t_eval)
    if not sol.success:
        return 1e12, np.zeros_like(theta)

    n = len(y0)
    m = len(p_names)
    y_traj = sol.y[:n, :].T
    S_traj = sol.y[n:, :].T.reshape(len(t_eval), n, m)

    if np.isnan(y_traj).any() or np.isinf(y_traj).any():
        return 1e12, np.zeros_like(theta)

    p = apply_params(theta)
    pre_vec, post_vec, pre_df, post_df = compute_rhs_and_grad(t_eval, y_traj, S_traj, p)
    if pre_vec is None or post_vec is None:
        return 1e12, np.zeros_like(theta)

    if np.linalg.norm(pre_vec) > 1e6 or np.linalg.norm(post_vec) > 1e6:
        return 1e12, np.zeros_like(theta)

    pre_scaled = pre_vec / RESIDUAL_SCALE_VEC
    post_scaled = post_vec / RESIDUAL_SCALE_VEC
    loss = float(np.sum(pre_scaled ** 2) + np.sum(post_scaled ** 2))

    grad = np.zeros_like(theta)
    for j in range(len(theta)):
        pre_term = 0.0
        post_term = 0.0
        if pre_df is not None:
            pre_term = np.sum(2.0 * pre_scaled * (pre_df[:, j] / RESIDUAL_SCALE_VEC))
        if post_df is not None:
            post_term = np.sum(2.0 * post_scaled * (post_df[:, j] / RESIDUAL_SCALE_VEC))
        grad[j] = pre_term + post_term

    return loss, grad


def run_gradient_check():
    """在小规模上做梯度检查（中心差分 vs. 敏感度梯度）。"""
    CHECK_STEPS = 61
    CHECK_EPS = 1e-4  # theta 空间扰动
    select_idx = list(range(min(5, len(p_names))))
    theta0 = x0.copy()
    loss_a, grad_a = loss_and_grad_steps(theta0, t_steps=CHECK_STEPS)
    print("\n[Gradient Check] loss=%.3e (steps=%d)" % (loss_a, CHECK_STEPS))
    print(f"检查参数: {[p_names[i] for i in select_idx]}")
    print(f"{'param':<18} {'grad(sens)':>14} {'grad(fd)':>14} {'rel_err':>10}")
    for j in select_idx:
        e = np.zeros_like(theta0)
        e[j] = CHECK_EPS
        lp, _ = loss_and_grad_steps(theta0 + e, t_steps=CHECK_STEPS)
        lm, _ = loss_and_grad_steps(theta0 - e, t_steps=CHECK_STEPS)
        g_fd = (lp - lm) / (2.0 * CHECK_EPS)
        g_an = grad_a[j]
        denom = max(1.0, abs(g_fd), abs(g_an))
        rel_err = abs(g_fd - g_an) / denom
        print(f"{p_names[j]:<18} {g_an:14.3e} {g_fd:14.3e} {rel_err:10.3e}")


# ============================================================================
# STEP 4: 先快速全局随机筛选，再 L-BFGS-B 精修
# ============================================================================
print("\n" + "="*70)
print("Stage A: 梯度检查 (仅检查，不做随机筛选)")
print("="*70)

# 仅做梯度检查；不进入随机筛选或优化。
DO_GRAD_CHECK = True
if DO_GRAD_CHECK:
    run_gradient_check()
    # 另外在一个“数值不大的”起点上再算一次损失与梯度（取每个参数对数区间中点）。
    theta_mid = np.array([(lb + ub) * 0.5 for (lb, ub) in theta_bounds], dtype=float)
    loss_mid, grad_mid = loss_and_grad_steps(theta_mid, t_steps=61)
    print("\n[Midpoint Grad] loss=%.3e, grad_norm=%.3e" % (loss_mid, np.linalg.norm(grad_mid)))
    import sys
    sys.exit(0)

print("\n" + "="*70)
print("Stage B: L-BFGS-B 精修（带敏感度梯度）")
print("="*70)

best_opt = None  # (loss, result, theta)
for idx, (seed_loss, seed_theta) in enumerate(best_list, 1):
    print(f"  精修起点 #{idx}: seed_loss={seed_loss:.3e}")
    res = minimize(
        fun=lambda v: loss_and_grad(v)[0],
        x0=seed_theta,
        method="L-BFGS-B",
        jac=lambda v: loss_and_grad(v)[1],
        bounds=theta_bounds,
        options={"maxiter": 200, "ftol": 1e-9, "gtol": 1e-6},
    )
    final_loss = res.fun if res.success else np.inf
    if best_opt is None or final_loss < best_opt[0]:
        best_opt = (final_loss, res, seed_theta)

opt_result = best_opt[1]
x_opt = opt_result.x if opt_result.success else best_opt[2]
p_opt = apply_params(x_opt)

print(f"优化成功: {opt_result.success}, 迭代: {opt_result.nit}, 函数调用: {opt_result.nfev}")
print(f"最终损失: {opt_result.fun if opt_result.success else np.nan:.4e}")

# ============================================================================
# STEP 5: 验证与输出
# ============================================================================

t_eval = np.linspace(0.0, T_END, 601)
sol_final = solve_with_sens(x_opt, t_eval)
y_final_traj = sol_final.y[:len(y0), :].T
p_final = apply_params(x_opt)
pre_vec_opt, post_vec_opt, _, _ = compute_rhs_and_grad(
    t_eval,
    y_final_traj,
    sol_final.y[len(y0):, :].T.reshape(len(t_eval), len(y0), len(p_names)),
    p_final,
)
pre_res_opt = np.linalg.norm(pre_vec_opt) if pre_vec_opt is not None else float("inf")
post_res_opt = np.linalg.norm(post_vec_opt) if post_vec_opt is not None else float("inf")

y_final_avg = y_final_traj[t_eval >= T_STEADY].mean(axis=0)

print("\n" + "="*70)
print("优化结果验证")
print("="*70)
print(f"{'变量':<15} {'目标':<12} {'实际':<12} {'偏差%':<10}")
print("-"*50)

key_vars = ['IgG4', 'act_CD4', 'TD Plasma', 'TI Plasma', 'Th2', 'IL_4', 'IL_10']
for v in key_vars:
    i = IDX[v]
    target = target_y[i]
    actual = y_final_avg[i]
    if target > 0:
        dev = abs(actual - target) / target * 100
    else:
        dev = 0 if actual < 1 else float('inf')
    print(f"{v:<15} {target:<12.2e} {actual:<12.2e} {dev:<10.1f}")

print(f"前段残差范数: {pre_res_opt:.4e}")
print(f"后段残差范数: {post_res_opt:.4e}")
print(f"最终损失: {opt_result.fun if opt_result.success else np.nan:.4e}")

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
        end = content.find('\n\ndef ', start + 1)
        if end == -1:
            end = len(content)
        content = content[:start] + content[end:]

lines = ["\ndef param_optimized_candidates():",
         '    """返回BDF+敏感度优化得到的候选参数列表"""',
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
DO_PLOT = True
if DO_PLOT and sol_final.success:
    print("\n绘制轨迹图...")
    sol_init = solve_ivp(lambda t, y: rhs_wrapper(t, y, p_init), (0.0, T_END), y0,
                         method="BDF", t_eval=t_eval, rtol=1e-5, atol=1e-8)

    fig, axes = plt.subplots(6, 5, figsize=(16, 14))
    axes = axes.flatten()

    plot_vars = [n for n in STATE_NAMES if n != "Antigen"]
    for idx, var in enumerate(plot_vars):
        if idx >= len(axes):
            break
        ax = axes[idx]
        i = IDX[var]
        ax.plot(t_eval, sol_init.y[i], 'b-', alpha=0.5, label='Initial')
        ax.plot(t_eval, sol_final.y[i], 'g-', linewidth=1.5, label='Optimized')
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
