"""
Parameter perturbation experiment:
- Baseline vs IL-4 degradation rate perturbations applied at t=150
- Supports multi-scale multiplicative perturbations (up/down):
	m ∈ {10^0.3, 10^0.6, 10^0.9, 10^1.2, 10^1.5, 10^1.8, 10^2.1}
	up:   p_new = p_base * m
	down: p_new = p_base / m
"""

import numpy as np
import matplotlib.pyplot as plt

from utils import HC_bl, IgG_param, STATE_NAMES, IDX, N_STATE
from main import rhs, simulate


def run_baseline(ts: np.ndarray):
	"""Simulate baseline (no perturbation) using IgG_param and HC_bl y0."""
	y0 = np.array([HC_bl()[name] for name in STATE_NAMES], dtype=float)
	p = IgG_param()
	return simulate(ts, y0, p, verbose=False), p


def run_perturbed(ts: np.ndarray, t_switch: float, factor: float):
	"""Simulate with IL-4 degradation scaled by `factor` after t_switch."""
	y0 = np.array([HC_bl()[name] for name in STATE_NAMES], dtype=float)
	p_base = IgG_param()
	p_pert = p_base.copy()
	p_pert["k_IL_4_d"] = p_base.get("k_IL_4_d", 1.0) * factor

	ts1 = ts[ts <= t_switch]
	ts2 = ts[ts >= t_switch]

	sim1 = simulate(ts1, y0, p_base, verbose=False)
	y_switch = sim1[-1]
	sim2 = simulate(ts2, y_switch, p_pert, verbose=False)

	ts_full = np.concatenate([ts1, ts2[1:]])
	sim_full = np.vstack([sim1, sim2[1:]])
	return ts_full, sim_full


def run_perturbation_grid(ts: np.ndarray, t_switch: float, factors: np.ndarray):
	"""Run baseline + up/down perturbations for each factor."""
	base_sim, _ = run_baseline(ts)
	results = {"baseline": (ts, base_sim)}
	for f in factors:
		ts_up, sim_up = run_perturbed(ts, t_switch, f)
		ts_dn, sim_dn = run_perturbed(ts, t_switch, 1.0 / f)
		results[f"up_{f:.2f}"] = (ts_up, sim_up)
		results[f"down_{f:.2f}"] = (ts_dn, sim_dn)
	return results


def plot_comparison(result_dict, out_path: str, factors: np.ndarray):
	cells_order = [
		"nDC", "mDC", "pDC", "naive_CD4",
		"act_CD4", "Th2", "iTreg", "CD4_CTL",
		"nTreg", "TFH", "CD56 NK", "CD16 NK",
		"Naive_B", "Act_B", "TD Plasma", "TI Plasma",
	]

	cytokine_order = [
		"GMCSF", "IL_33", "IL_6", "IL_12",
		"IL_15", "IL_7", "IFN1", "IL_1",
		"IL_2", "IL_4", "IL_10", "TGFbeta",
		"IFN_g", "IgG4", "Antigen",
	]

	fig, axes = plt.subplots(8, 4, figsize=(16, 20))
	axes = axes.flatten()

	base_label = "未扰动"
	base_color = "#b0b0b0"
	# color maps for up/down sweeps
	up_colors = plt.cm.Blues(np.linspace(0.45, 0.95, len(factors)))
	dn_colors = plt.cm.Reds(np.linspace(0.45, 0.95, len(factors)))

	for idx, var in enumerate(cells_order):
		ax = axes[idx]
		i = IDX[var]
		ts_base, sim_base = result_dict["baseline"]
		ax.plot(ts_base, sim_base[:, i], color=base_color, linewidth=1.6, label=base_label)
		for k, f in enumerate(factors):
			ts_up, sim_up = result_dict[f"up_{f:.2f}"]
			ts_dn, sim_dn = result_dict[f"down_{f:.2f}"]
			ax.plot(ts_up, sim_up[:, i], color=up_colors[k], linewidth=1.2, label=f"上调x{f:.1f}" if idx == 0 else None)
			ax.plot(ts_dn, sim_dn[:, i], color=dn_colors[k], linewidth=1.2, linestyle="--", label=f"下调÷{f:.1f}" if idx == 0 else None)
		ax.axvspan(100, 150, alpha=0.12, color="red")
		ax.axvline(150, color="black", linestyle="--", linewidth=0.8)
		ax.set_title(var, fontsize=10, fontweight="bold")
		ax.grid(True, alpha=0.3)
		ax.tick_params(labelsize=7)
		ax.set_xlabel("Time (s)", fontsize=8)
		ax.set_ylabel("Count", fontsize=8)

	for idx, var in enumerate(cytokine_order):
		ax = axes[16 + idx]
		i = IDX[var]
		ts_base, sim_base = result_dict["baseline"]
		ax.plot(ts_base, sim_base[:, i], color=base_color, linewidth=1.6, label=base_label)
		for k, f in enumerate(factors):
			ts_up, sim_up = result_dict[f"up_{f:.2f}"]
			ts_dn, sim_dn = result_dict[f"down_{f:.2f}"]
			ax.plot(ts_up, sim_up[:, i], color=up_colors[k], linewidth=1.2, label=f"上调x{f:.1f}" if idx == 0 else None)
			ax.plot(ts_dn, sim_dn[:, i], color=dn_colors[k], linewidth=1.2, linestyle="--", label=f"下调÷{f:.1f}" if idx == 0 else None)
		ax.axvspan(100, 150, alpha=0.12, color="red")
		ax.axvline(150, color="black", linestyle="--", linewidth=0.8)
		ax.set_title(var, fontsize=10, fontweight="bold")
		ax.grid(True, alpha=0.3)
		ax.tick_params(labelsize=7)
		ax.set_xlabel("Time (s)", fontsize=8)
		ax.set_ylabel("Conc", fontsize=8)

	axes[-2].set_visible(False)
	axes[-1].set_visible(False)

	handles, labels = axes[0].get_legend_handles_labels()
	fig.legend(handles, labels, loc="upper right", fontsize=9)
	plt.suptitle("IL-4 degradation perturbation at t=150", fontsize=13, fontweight="bold", y=0.998)
	plt.tight_layout()
	plt.savefig(out_path, dpi=150, bbox_inches="tight")
	print(f"[OK] Figure saved: {out_path}")
	plt.show()


def main():
	t_end = 300.0
	dt = 1.0
	ts = np.arange(0.0, t_end + dt, dt)
	t_switch = 150.0
	factors = np.array([10 ** e for e in [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1]])

	result_dict = run_perturbation_grid(ts, t_switch, factors)

	# NaN check across all runs
	for key, (t_arr, sim_arr) in result_dict.items():
		if np.isnan(sim_arr).any():
			print(f"[WARN] Simulation produced NaN values for {key}; check parameter settings.")
			return

	plot_comparison(result_dict, out_path="perturb_il4_deg.png", factors=factors)


if __name__ == "__main__":
	main()
