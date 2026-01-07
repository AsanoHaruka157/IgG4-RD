#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IgG4-RD 参数优化脚本
目标：找到参数使系统从 HC_state 在抗原刺激后演化到 IgG_state
"""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from utils import (
    rhs_hc, HC_state, IgG_param, IgG_state,
    STATE_NAMES, IDX, ALL_PARAMS,
)

# ============================================================================
# 配置
# ============================================================================
T_END = 300.0          # 模拟总时间
T_ANTIGEN_START = 100  # 抗原刺激开始
T_ANTIGEN_END = 150    # 抗原刺激结束（线性爬升）
T_STEADY = 200         # 稳态评估起始时间

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

# 初始参数（基于文献参考值）
p_init = IgG_param()

# 需要优化的参数
p_names = [n for n in ALL_PARAMS if n in p_init]

# 参数重参数化：k = k0 * 10^x, x ∈ [-2, 2]
BASE_FLOOR = 1e-8
p_base = {n: max(p_init[n], BASE_FLOOR) for n in p_names}
bounds = [(-2.0, 2.0) for _ in p_names]
x0 = np.zeros(len(p_names))

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

def rhs(t, y, p):
    y_mod = y.copy()
    y_mod[0] = antigen(t)
    dydt = rhs_hc(t, y_mod, p)
    dydt[0] = 1.0 / (T_ANTIGEN_END - T_ANTIGEN_START) if T_ANTIGEN_START <= t <= T_ANTIGEN_END else 0.0
    return dydt

def solve(y0, p, t_eval):
    for method in ("BDF", "Radau"):
        try:
            sol = solve_ivp(lambda t, y: rhs(t, y, p), (t_eval[0], t_eval[-1]), 
                           y0, method=method, t_eval=t_eval, rtol=1e-5, atol=1e-8)
            if sol.success:
                return sol.y.T
        except:
            continue
    return None

def apply_params(x):
    p = dict(p_init)
    for i, n in enumerate(p_names):
        p[n] = p_base[n] * (10.0 ** np.clip(x[i], -2, 2))
    return p

# ============================================================================
# STEP 3: 损失函数
# ============================================================================
def loss_fn(x, verbose=False):
    p = apply_params(x)
    t_eval = np.linspace(0, T_END, 301)
    sol = solve(y0, p, t_eval)
    
    if sol is None or np.isnan(sol).any() or np.isinf(sol).any():
        return 1e12
    if (sol < -1e-6).any():
        return 1e12
    
    # 取 t > T_STEADY 的平均值作为稳态
    mask = t_eval >= T_STEADY
    y_ss = sol[mask].mean(axis=0)
    
    # 混合损失：细胞用相对误差，细胞因子和IgG4用log误差
    loss = 0.0
    eps = 1e-12
    
    # 细胞类变量（L2相对误差）
    cell_vars = ["mDC", "pDC", "naive_CD4", "act_CD4", "Th2", "iTreg", 
                 "CD4_CTL", "nTreg", "TFH", "CD56 NK", "CD16 NK",
                 "Naive_B", "Act_B", "TD Plasma", "TI Plasma"]
    for v in cell_vars:
        i = IDX[v]
        target = target_y[i]
        actual = y_ss[i]
        if target > 0:
            loss += ((actual - target) / (target + 1)) ** 2
    
    # 细胞因子和IgG4（log误差）
    log_vars = ["GMCSF", "IL_33", "IL_6", "IL_12", "IL_15", "IL_7", "IFN1",
                "IL_1", "IL_2", "IL_4", "IL_10", "TGFbeta", "IFN_g", "IgG4"]
    for v in log_vars:
        i = IDX[v]
        target = target_y[i]
        actual = y_ss[i]
        loss += (np.log10(actual + eps) - np.log10(target + eps)) ** 2
    
    if verbose:
        print(f"Loss={loss:.2e}, IgG4: {y_ss[IDX['IgG4']]:.2f} (target={target_y[IDX['IgG4']]:.2f})")
    
    return loss

# ============================================================================
# STEP 4: 优化
# ============================================================================
print("\n" + "="*70)
print("开始优化...")
print("="*70)

# 回调函数显示进度
iter_count = [0]
def callback(x):
    iter_count[0] += 1
    if iter_count[0] % 50 == 0:
        l = loss_fn(x)
        print(f"  Iter {iter_count[0]}: loss={l:.4e}")

result = minimize(
    loss_fn, x0,
    method='L-BFGS-B',
    bounds=bounds,
    callback=callback,
    options={'maxiter': 1000, 'ftol': 1e-8, 'disp': False}
)

print(f"\n优化完成!")
print(f"  迭代次数: {result.nit}")
print(f"  最终损失: {result.fun:.4e}")
print(f"  成功: {result.success}")

# ============================================================================
# STEP 5: 验证并保存
# ============================================================================
x_opt = result.x
p_opt = apply_params(x_opt)

# 验证最终轨迹
t_eval = np.linspace(0, T_END, 601)
sol_opt = solve(y0, p_opt, t_eval)
y_final = sol_opt[t_eval >= T_STEADY].mean(axis=0)

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

# ============================================================================
# STEP 6: 保存优化参数到 utils.py
# ============================================================================
print("\n" + "="*70)
print("保存优化参数...")
print("="*70)

with open('utils.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 删除旧的 param_optimized
if 'def param_optimized():' in content:
    start = content.find('def param_optimized():')
    end = content.find('\ndef ', start + 1)
    if end == -1:
        end = len(content)
    content = content[:start] + content[end:]

# 生成新函数
lines = ["\ndef param_optimized():", 
         '    """优化后的参数 (HC→IgG with antigen)"""',
         "    p = {}"]
for n in sorted(p_opt.keys()):
    lines.append(f'    p["{n}"] = {p_opt[n]:.10e}')
lines.append("    return p\n")

content = content.rstrip() + "\n" + "\n".join(lines)

with open('utils.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("已保存 param_optimized() 到 utils.py")

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
