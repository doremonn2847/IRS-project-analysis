"""Experiment 4 — Achievable rate versus number of IRS elements (Fig. 6 setup).

Figure: rlc_fig5_rate_vs_N. Raw averages are written to
results/data/rlc_rate_vs_N.csv.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import ieee_plot_style
from ieee_plot_style import IEEE_COLORS
from rlc_model import config
from rlc_model.system import generate_channels
from rlc_model.montecarlo import SCHEMES, SCHEME_LABELS, evaluate_schemes


def run_single_realization(distance, N_val, seed, iters):
    local_rng = np.random.default_rng(seed)
    hd, hr, G, Phi = generate_channels(distance, N_val, local_rng)
    rates = evaluate_schemes(hd, hr, G, Phi, N_val, local_rng, n_iter=iters)
    return rates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realizations", type=int, default=100)
    parser.add_argument("--distance", type=float, default=498.0)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config.ensure_output_dirs()
    rng = np.random.default_rng(args.seed)
    N_range = np.arange(10, 81, 10)

    results = {s: np.zeros(len(N_range)) for s in SCHEMES}
    print(f"[exp4] {len(N_range)} swarm sizes x {args.realizations} realizations "
          f"(d = {args.distance} m)")

    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for n_idx, N_val in enumerate(N_range):
            acc = {s: [] for s in SCHEMES}
            seeds = [rng.integers(0, 2**32 - 1) for _ in range(args.realizations)]
            futures = [
                executor.submit(run_single_realization, args.distance, int(N_val), seeds[i], args.iters)
                for i in range(args.realizations)
            ]
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                            desc=f"N = {N_val}"):
                rates = fut.result()
                for s in SCHEMES:
                    acc[s].append(rates[s])
            for s in SCHEMES:
                results[s][n_idx] = np.mean(acc[s])

    csv_path = config.DATA_DIR / "rlc_rate_vs_N.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N_elements"] + list(SCHEMES))
        for i, n in enumerate(N_range):
            writer.writerow([n] + [f"{results[s][i]:.4f}" for s in SCHEMES])
    print(f"[exp4] Data saved to {csv_path}")

    fig, ax = ieee_plot_style.new_figure("single_tall")
    style = {
        "ideal": dict(color=IEEE_COLORS["black"], linestyle="-.", marker=""),
        "pso_rlc": dict(color=IEEE_COLORS["blue"], linestyle="-", marker="o"),
        "pso_c_only": dict(color=IEEE_COLORS["orange"], linestyle="--", marker="D"),
        "ao_phase": dict(color=IEEE_COLORS["green"], linestyle="-", marker="^"),
        "no_irs": dict(color=IEEE_COLORS["gray"], linestyle=":", marker="s"),
    }
    for s in SCHEMES:
        ax.plot(N_range, results[s], markersize=4, label=SCHEME_LABELS[s],
                **style[s])
    ax.set_xlabel("Number of Reflecting Elements $N$")
    ax.set_ylabel("Achievable Rate (bps/Hz)")
    ieee_plot_style.format_legend(ax, loc="upper left")
    ieee_plot_style.save_figure("rlc_fig5_rate_vs_N.png",
                                figures_dir=str(config.FIGURES_DIR))
    plt.close(fig)
    print("[exp4] Done.")


if __name__ == "__main__":
    main()
