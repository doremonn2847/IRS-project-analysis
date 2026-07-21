"""Experiment 1 — Verify the circuit model against the equivalent model.

Figures (saved to results/figures/):
  * rlc_fig1_circuit_response : |v| and arg(v) versus the varactor capacitance
    C for several loss resistances R (L1, L2 fixed at the paper values).
  * rlc_fig2_feasible_region  : the (theta, beta) curve traced by the circuit
    when only C is tuned, overlaid with the paper's Eq. (5) fit — and the much
    larger feasible region unlocked when L1, L2, C, R are all free.

Console output: grid fit of Eq. (5) constants (beta_min, k, phi) to the
circuit-generated curve — this verifies that the weeks 1-4 constants
(0.2, 1.6, 0.43*pi) are derivable from the physical circuit.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

import ieee_plot_style
from ieee_plot_style import IEEE_COLORS
from rlc_model import config
from rlc_model.circuit import reflection_coefficient, eq5_amplitude


def save(name):
    ieee_plot_style.save_figure(name, figures_dir=str(config.FIGURES_DIR))


def main():
    config.ensure_output_dirs()
    rng = np.random.default_rng(42)
    C_lo, C_hi = config.BOUNDS["C"]
    C_sweep = np.linspace(C_lo, C_hi, 20000)

    # ── Figure 1: circuit response versus C ─────────────────────────────────
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.4), sharex=True)
    colors = [IEEE_COLORS["blue"], IEEE_COLORS["green"],
              IEEE_COLORS["orange"], IEEE_COLORS["red"]]
    for R, col in zip([0.5, 1.0, 2.5, 5.0], colors):
        v = reflection_coefficient(config.L1_PAPER, config.L2_PAPER, C_sweep, R)
        ax1.plot(C_sweep * 1e12, np.abs(v), color=col, label=f"$R={R}\\,\\Omega$")
        ax2.plot(C_sweep * 1e12, np.angle(v) / np.pi, color=col)
    ax1.set_ylabel(r"Amplitude $|v_n|$")
    ax2.set_ylabel(r"Phase $\arg(v_n)/\pi$")
    ax2.set_xlabel(r"Capacitance $C_n$ (pF)")
    ieee_plot_style.format_legend(ax1, loc="lower right")
    save("rlc_fig1_circuit_response.png")
    plt.close(fig)

    # ── Fit Eq.(5) constants to the circuit curve (R = paper value) ─────────
    v = reflection_coefficient(config.L1_PAPER, config.L2_PAPER, C_sweep,
                               config.R_PAPER)
    amp, ph = np.abs(v), np.angle(v)
    order = np.argsort(ph)
    ph_s, amp_s = ph[order], amp[order]

    best = None
    for bmin in np.linspace(0.10, 0.30, 21):
        for k in np.linspace(1.0, 2.2, 25):
            for phi in np.linspace(0.35 * np.pi, 0.50 * np.pi, 31):
                err = np.mean((eq5_amplitude(ph_s, bmin, k, phi) - amp_s) ** 2)
                if best is None or err < best[0]:
                    best = (err, bmin, k, phi)
    err, bmin, k, phi = best
    print(f"[exp1] Eq.(5) fit to circuit (R={config.R_PAPER} Ohm): "
          f"beta_min={bmin:.2f}, k={k:.2f}, phi={phi/np.pi:.3f}*pi "
          f"(rmse={np.sqrt(err):.4f})")
    print(f"[exp1] Paper constants (weeks 1-4): beta_min={config.BETA_MIN}, "
          f"k={config.K_PARAM}, phi={config.PHI_PARAM/np.pi:.2f}*pi")

    # ── Figure 2: feasible region in the (theta, beta) plane ────────────────
    n_samples = 30000
    L1 = rng.uniform(*config.BOUNDS["L1"], n_samples)
    L2 = rng.uniform(*config.BOUNDS["L2"], n_samples)
    C = rng.uniform(*config.BOUNDS["C"], n_samples)
    R = rng.uniform(*config.BOUNDS["R"], n_samples)
    v_free = reflection_coefficient(L1, L2, C, R)

    fig, ax = ieee_plot_style.new_figure("single_tall")
    ax.scatter(np.angle(v_free) / np.pi, np.abs(v_free), s=1.0,
               color=IEEE_COLORS["light_gray"], alpha=0.25, edgecolors="none",
               rasterized=True, label="Free $L_1, L_2, C, R$ (RLC search space)")
    ax.plot(ph_s / np.pi, amp_s, color=IEEE_COLORS["blue"], lw=1.4,
            label=f"Only $C$ tunable ($R={config.R_PAPER}\\,\\Omega$)")
    theta_grid = np.linspace(-np.pi, np.pi, 400)
    ax.plot(theta_grid / np.pi, eq5_amplitude(theta_grid),
            color=IEEE_COLORS["red"], linestyle="--",
            label="Eq. (5) equivalent model")
    ax.set_xlabel(r"Phase $\theta/\pi$")
    ax.set_ylabel(r"Amplitude $\beta$")
    ax.set_ylim(0, 1.05)
    ieee_plot_style.format_legend(ax, loc="lower center")
    save("rlc_fig2_feasible_region.png")
    plt.close(fig)
    print("[exp1] Done.")


if __name__ == "__main__":
    main()
