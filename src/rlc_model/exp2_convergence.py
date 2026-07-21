"""Experiment 2 — PSO-RLC convergence on a single channel realization.

Figure: rlc_fig3_convergence — best-so-far rate of the component-level PSO
versus iteration, with reference levels for the AO phase-model baseline, the
varactor-only PSO, the ideal upper bound and the no-IRS lower bound.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse

import numpy as np
import matplotlib.pyplot as plt

import ieee_plot_style
from ieee_plot_style import IEEE_COLORS
from rlc_model import config
from rlc_model.system import generate_channels, rate_no_irs, ideal_upper_bound
from rlc_model.optimizers import pso_rlc, ao_phase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=float, default=498.0,
                        help="AP-user distance in meters (default 498)")
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config.ensure_output_dirs()
    rng = np.random.default_rng(args.seed)
    hd, hr, G, Phi = generate_channels(args.distance, config.N, rng)

    print("[exp2] Optimizing one channel realization "
          f"(d = {args.distance} m, N = {config.N})...")
    best, rate_rlc, hist = pso_rlc(hd, Phi, config.N, n_iter=args.iters,
                                   rng=rng, track_convergence=True)
    _, rate_c_only, _ = pso_rlc(hd, Phi, config.N, free=("C",),
                                n_iter=args.iters, rng=rng)
    _, rate_ao = ao_phase(hd, hr, G, rng=rng)
    _, rate_ideal = ideal_upper_bound(hd, Phi)
    rate_lb = rate_no_irs(hd)

    print(f"[exp2]   Ideal upper bound : {rate_ideal:.3f} bps/Hz")
    print(f"[exp2]   PSO (RLC, 4N dims): {rate_rlc:.3f} bps/Hz")
    print(f"[exp2]   PSO (C only)      : {rate_c_only:.3f} bps/Hz")
    print(f"[exp2]   AO  (Eq.5 model)  : {rate_ao:.3f} bps/Hz")
    print(f"[exp2]   No IRS            : {rate_lb:.3f} bps/Hz")

    fig, ax = ieee_plot_style.new_figure("single_tall")
    ax.plot(hist, color=IEEE_COLORS["blue"], label="PSO on $\\{L_1,L_2,C,R\\}$")
    ax.axhline(rate_ideal, color=IEEE_COLORS["black"], linestyle="-.",
               label="Ideal IRS (upper bound)")
    ax.axhline(rate_ao, color=IEEE_COLORS["green"], linestyle="-",
               label="AO: Eq.(5) phase model [1]")
    ax.axhline(rate_c_only, color=IEEE_COLORS["orange"], linestyle="--",
               label="PSO: only $C$ tunable")
    ax.axhline(rate_lb, color=IEEE_COLORS["gray"], linestyle=":",
               label="No IRS (lower bound)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Achievable Rate (bps/Hz)")
    ieee_plot_style.format_legend(ax, loc="lower right")
    ieee_plot_style.save_figure("rlc_fig3_convergence.png",
                                figures_dir=str(config.FIGURES_DIR))
    plt.close(fig)
    print("[exp2] Done.")


if __name__ == "__main__":
    main()
