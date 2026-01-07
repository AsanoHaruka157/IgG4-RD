"""
IgG4-RD 系统轨线分析工具
========================
从HC_state和HC_param生成轨线，采集t=1000处的稳态作为HC_bl

用法：
    python fixpoint.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from utils import HC_state, HC_param, rhs_hc, STATE_NAMES
import warnings
warnings.filterwarnings('ignore')

def generate_trajectory_and_baseline():
    """
    生成轨线，采集t=1000处的稳态
    """
    # 初始条件和参数
    y0_dict = HC_state()
    p = HC_param()
    y0 = np.array([y0_dict[name] for name in STATE_NAMES])
    
    print("=" * 80)
    print("HC_state + HC_param 轨线分析（t=0 至 t=1000s）")
    print("=" * 80)
    print(f"\n[初始条件]")
    for i, name in enumerate(STATE_NAMES[:10]):
        print(f"  {name:20s} = {y0[i]:15.6e}")
    print(f"  ... (共{len(STATE_NAMES)}个状态变量)\n")
    
    # 逐段积分，直到残差足够小或达到最大时间
    max_time = 100000.0
    t_step = 0.1
    chunk = 1000.0  # 每段积分长度（秒）
    min_time_for_check = 1000.0  # 至少跑这么久后再判停
    tol_resid = 1e-6

    print("[求解ODE系统，包含稳态判停...]")
    def rhs_wrapper(y, t):
        return rhs_hc(t, y, p)

    t_all = [0.0]
    y_all = [y0]
    current_t = 0.0
    y_current = y0

    while current_t < max_time:
        t_seg = np.linspace(current_t, min(current_t + chunk, max_time), int(chunk / t_step) + 1)
        y_seg = odeint(rhs_wrapper, y_current, t_seg, full_output=False, rtol=1e-8, atol=1e-10)

        # 追加（去掉首点避免重复）
        t_all.extend(t_seg[1:].tolist())
        y_all.extend(y_seg[1:])

        current_t = t_seg[-1]
        y_current = y_seg[-1]

        # 判停：达到最小时间且最大残差足够小
        if current_t >= min_time_for_check:
            resid = rhs_hc(current_t, y_current, p)
            max_resid = float(np.max(np.abs(resid)))
            print(f"  t={current_t:.1f}s, max|rhs|={max_resid:.3e}")
            if max_resid < tol_resid:
                print("  ✓ 残差满足阈值，提前停止积分")
                break

    t = np.array(t_all)
    y_solution = np.vstack(y_all)
    print(f"✓ 求解完成，实际积分到 t={t[-1]:.1f}s，共{len(t)}个时间点\n")
    
    # 采集最后时刻的状态作为稳态
    y_steady = y_solution[-1, :]
    final_time = t[-1]

    print(f"[t={final_time:.1f}s 稳态值]")
    print(f"{'变量':20s} {'稳态值':20s} {'与初值比':15s}")
    print(f"{'-'*55}")
    for i, name in enumerate(STATE_NAMES):
        ratio = y_steady[i] / y0[i] if y0[i] != 0 else np.nan
        print(f"{name:20s} {y_steady[i]:20.6e} {ratio:15.6e}")
    
    # 生成HC_bl函数代码并追加到utils.py
    print("\n" + "=" * 80)
    print(f"生成 HC_BL 函数（t={final_time:.1f} 处的稳态）")
    print("=" * 80)
    
    hc_bl_code = 'def HC_bl():\n    """\n    稳态值（从 HC_state 按 HC_param 运行，积分到残差收敛或 t=max_time）\n    """\n    y0 = {}\n'
    for i, name in enumerate(STATE_NAMES):
        hc_bl_code += f'    y0["{name}"] = {y_steady[i]:.15e}\n'
    hc_bl_code += '    return y0\n'
    
    # 更新或替换utils.py中的HC_bl
    with open('utils.py', 'r', encoding='utf-8') as f:
        utils_content = f.read()
    
    # 检查是否已有HC_bl
    if 'def HC_bl():' in utils_content:
        # 找到并替换现有的HC_bl函数
        start_idx = utils_content.find('def HC_bl():')
        # 找到函数结束处（下一个def或文件末尾）
        end_idx = utils_content.find('\ndef ', start_idx + 1)
        if end_idx == -1:
            end_idx = len(utils_content)
        
        utils_content = utils_content[:start_idx] + hc_bl_code + utils_content[end_idx:]
    else:
        # 追加新的HC_bl
        utils_content += '\n\n' + '# ' + '=' * 78 + '\n'
        utils_content += '# Baseline steady-state (from HC_state with HC_param at t=1000s)\n'
        utils_content += '# ' + '=' * 78 + '\n'
        utils_content += hc_bl_code
    
    with open('utils.py', 'w', encoding='utf-8') as f:
        f.write(utils_content)
    
    print("✓ HC_bl() 已更新到 utils.py\n")
    
    # 绘制尾部轨线图（仅最后 500s）：8 行 4 列
    print("=" * 80)
    print("生成轨线图（仅尾部500s）...")
    print("=" * 80)

    tail_window = 500.0
    mask_tail = t >= (t[-1] - tail_window)
    t_tail = t[mask_tail]
    y_tail = y_solution[mask_tail]

    fig, axes = plt.subplots(8, 4, figsize=(16, 20))
    axes = axes.flatten()

    for i, name in enumerate(STATE_NAMES):
        ax = axes[i]
        ax.plot(t_tail, y_tail[:, i], linewidth=1.5, color='steelblue')
        ax.axvline(x=t[-1], color='red', linestyle='--', linewidth=1, alpha=0.5, label=f't={final_time:.1f}s (HC_bl)')
        ax.set_title(f'{name}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Concentration', fontsize=8)
        ax.tick_params(labelsize=8)

    # 隐藏最后一个空子图
    axes[-1].set_visible(False)

    plt.suptitle(f'Tail trajectories (last 500s): HC_state + HC_param, HC_bl at t={final_time:.1f}s',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('fixpoint_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ 轨线图已保存到 fixpoint_analysis.png\n")
    plt.close()
    
    print("=" * 80)
    print("完成！")
    print("=" * 80)

if __name__ == '__main__':
    generate_trajectory_and_baseline()
