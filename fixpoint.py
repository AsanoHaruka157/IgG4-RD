"""
fixpoint.py

更精确地求 HC 基线稳态：直接解 dy/dt = 0 的非线性方程（对数参数化保证非负）。
结果写回 utils.py 的 HC_bl()。
"""

import numpy as np
from scipy.optimize import least_squares
from utils import HC_state, HC_param, HC_bl, rhs_hc, STATE_NAMES, IDX

def solve_steady_state(max_nfev: int = 500, tol: float = 1e-10):
    p = HC_param()

    # 初始猜测：优先用已有 HC_bl，否则用 HC_state
    try:
        y0_dict = HC_bl()
    except Exception:
        y0_dict = HC_state()
    y0 = np.array([y0_dict[n] for n in STATE_NAMES], dtype=float)

    # 对数参数化（保证非负）；Antigen 固定 0，不优化
    x0 = np.log(np.maximum(y0[1:], 1e-12))  # 30 维，跳过 Antigen

    def unpack(x):
        y = np.zeros_like(y0)
        y[0] = 0.0  # Antigen 固定为 0
        y[1:] = np.exp(x)
        return y

    def residual(x):
        y = unpack(x)
        dy = rhs_hc(0.0, y, p)
        # 去掉 Antigen 的导数项
        res = dy[1:]
        # 以相对尺度缩放，避免大尺度量级主导
        scale = np.maximum(y[1:], 1e-6)
        return res / scale

    print("开始求解 dy/dt=0 ...")
    result = least_squares(residual, x0, method="trf", xtol=tol, ftol=tol, gtol=tol, max_nfev=max_nfev)

    y_star = unpack(result.x)
    dy_star = rhs_hc(0.0, y_star, p)
    max_resid = float(np.max(np.abs(dy_star)))
    rel_resid = np.max(np.abs(dy_star[1:] / np.maximum(y_star[1:], 1e-6)))

    print(f"求解结束: status={result.status}, nfev={result.nfev}, cost={result.cost:.3e}")
    print(f"max|dy| = {max_resid:.3e}, max|dy/y| = {rel_resid:.3e}\n")

    # 打印变化较大的分量
    delta = y_star - y0
    rel_change = delta / np.maximum(np.abs(y0), 1e-12)
    pairs = sorted(zip(STATE_NAMES, rel_change, delta, y0, y_star), key=lambda x: -abs(x[1]))
    print("变化较大的前 8 项 (相对变化)")
    for name, rc, d, a, b in pairs[:8]:
        print(f"  {name:10s}: rel={rc:+.3e}, absΔ={d:+.3e}, old={a:.3e}, new={b:.3e}")

    return y_star


def write_hc_bl(y_star: np.ndarray):
    """将新稳态写入 utils.py 的 HC_bl()"""
    hc_bl_code = 'def HC_bl():\n    """\n    稳态值（直接解 dy/dt=0 得到）\n    """\n    y0 = {}\n'
    for i, name in enumerate(STATE_NAMES):
        hc_bl_code += f'    y0["{name}"] = {y_star[i]:.15e}\n'
    hc_bl_code += '    return y0\n'

    with open('utils.py', 'r', encoding='utf-8') as f:
        utils_content = f.read()

    if 'def HC_bl():' in utils_content:
        start_idx = utils_content.find('def HC_bl():')
        end_idx = utils_content.find('\ndef ', start_idx + 1)
        if end_idx == -1:
            end_idx = len(utils_content)
        utils_content = utils_content[:start_idx] + hc_bl_code + utils_content[end_idx:]
    else:
        utils_content += '\n\n' + '# ' + '=' * 78 + '\n'
        utils_content += '# Baseline steady-state solved via root finding\n'
        utils_content += '# ' + '=' * 78 + '\n'
        utils_content += hc_bl_code

    with open('utils.py', 'w', encoding='utf-8') as f:
        f.write(utils_content)
    print("✓ HC_bl() 已更新到 utils.py")


def main():
    y_star = solve_steady_state()
    write_hc_bl(y_star)
    print("完成")


if __name__ == '__main__':
    main()
