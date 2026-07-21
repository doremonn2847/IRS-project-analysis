"""Experiment 3 — Achievable rate versus AP-user distance (paper's Fig. 5 setup).

Figure: rlc_fig4_rate_vs_distance. Raw averages are also written to
results/data/rlc_rate_vs_distance.csv for the report tables.

Runtime scales linearly with --realizations (default 100; use 1000 for the
final report figures, ~10x longer).
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


def run_single_realization(d_user, N_val, seed, iters):
    local_rng = np.random.default_rng(seed)
    hd, hr, G, Phi = generate_channels(d_user, N_val, local_rng)
    rates = evaluate_schemes(hd, hr, G, Phi, N_val, local_rng, n_iter=iters)
    return rates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realizations", type=int, default=100)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config.ensure_output_dirs()
    rng = np.random.default_rng(args.seed)
    d_range = np.arange(482, 501, 2)

    results = {s: np.zeros(len(d_range)) for s in SCHEMES}
    print(f"[exp3] {len(d_range)} distances x {args.realizations} realizations")

    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for d_idx, d_user in enumerate(d_range):
            acc = {s: [] for s in SCHEMES}
            seeds = [rng.integers(0, 2**32 - 1) for _ in range(args.realizations)]
            futures = [
                executor.submit(run_single_realization, d_user, config.N, seeds[i], args.iters)
                for i in range(args.realizations)
            ]
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                            desc=f"d = {d_user} m"):
                rates = fut.result()
                for s in SCHEMES:
                    acc[s].append(rates[s])
            for s in SCHEMES:
                results[s][d_idx] = np.mean(acc[s])

    # CSV for report tables
    csv_path = config.DATA_DIR / "rlc_rate_vs_distance.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["d_user_m"] + list(SCHEMES))
        for i, d in enumerate(d_range):
            writer.writerow([d] + [f"{results[s][i]:.4f}" for s in SCHEMES])
    print(f"[exp3] Data saved to {csv_path}")

    # Figure
    fig, ax = ieee_plot_style.new_figure("single_tall")
    style = {
        "ideal": dict(color=IEEE_COLORS["black"], linestyle="-.", marker=""),
        "pso_rlc": dict(color=IEEE_COLORS["blue"], linestyle="-", marker="o"),
        "pso_c_only": dict(color=IEEE_COLORS["orange"], linestyle="--", marker="D"),
        "ao_phase": dict(color=IEEE_COLORS["green"], linestyle="-", marker="^"),
        "no_irs": dict(color=IEEE_COLORS["gray"], linestyle=":", marker="s"),
    }
    for s in SCHEMES:
        ax.plot(d_range, results[s], markersize=4, label=SCHEME_LABELS[s],
                **style[s])
    ax.set_xlabel("AP-User Distance $d$ (m)")
    ax.set_ylabel("Achievable Rate (bps/Hz)")
    ieee_plot_style.format_legend(ax, loc="lower left")
    ieee_plot_style.save_figure("rlc_fig4_rate_vs_distance.png",
                                figures_dir=str(config.FIGURES_DIR))
    plt.close(fig)
    print("[exp3] Done.")


if __name__ == "__main__":
    main()
