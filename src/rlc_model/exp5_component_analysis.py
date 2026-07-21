"""Experiment 5 — Influence of the components + fixed-component comparison.

Figures:
  * rlc_fig6_fixed_components : mean rate at d = 500 m when different subsets
    of {L1, L2, C, R} are optimized (the rest frozen at paper values) —
    the "comparison with some components fixed" required by the brief.
  * rlc_fig7_component_hist   : distribution of the optimized component values
    across elements and realizations (full-RLC PSO) — shows which values the
    optimizer actually selects (e.g. R pushed to its lower bound).

Data: results/data/rlc_fixed_components.csv,
      results/data/rlc_optimized_components.csv
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
from rlc_model.system import generate_channels, rate_no_irs, ideal_upper_bound
from rlc_model.optimizers import pso_rlc, ao_phase

# Fixed-component variants: label -> tuple of free components
VARIANTS = [
    ("All free\n$L_1,L_2,C,R$", ("L1", "L2", "C", "R")),
    ("$C,R$ free", ("C", "R")),
    ("$C,L_1$ free", ("C", "L1")),
    ("Only $C$ free\n(varactor)", ("C",)),
]


def run_single_realization_exp5(distance, N_val, seed, iters):
    local_rng = np.random.default_rng(seed)
    hd, hr, G, Phi = generate_channels(distance, N_val, local_rng)
    no_irs_val = rate_no_irs(hd)
    _, ideal_val = ideal_upper_bound(hd, Phi)
    _, ao_val = ao_phase(hd, hr, G, rng=local_rng)
    
    res = {
        "No IRS": no_irs_val,
        "Ideal IRS": ideal_val,
        "AO: Eq.(5)\nmodel [1]": ao_val,
        "variants": {}
    }
    
    for label, free in VARIANTS:
        best, rate, _ = pso_rlc(hd, Phi, N_val, free=free,
                                n_iter=iters, rng=local_rng)
        res["variants"][label] = (rate, best if len(free) == 4 else None)
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realizations", type=int, default=50)
    parser.add_argument("--distance", type=float, default=498.0)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config.ensure_output_dirs()
    rng = np.random.default_rng(args.seed)

    acc = {label: [] for label, _ in VARIANTS}
    acc["AO: Eq.(5)\nmodel [1]"] = []
    acc["Ideal IRS"] = []
    acc["No IRS"] = []
    opt_components = {name: [] for name in config.COMPONENT_NAMES}

    print(f"[exp5] {args.realizations} realizations at d = {args.distance} m")

    import concurrent.futures
    seeds = [rng.integers(0, 2**32 - 1) for _ in range(args.realizations)]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_single_realization_exp5, args.distance, config.N, seeds[i], args.iters)
            for i in range(args.realizations)
        ]
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                        desc="Realizations"):
            res = fut.result()
            acc["No IRS"].append(res["No IRS"])
            acc["Ideal IRS"].append(res["Ideal IRS"])
            acc["AO: Eq.(5)\nmodel [1]"].append(res["AO: Eq.(5)\nmodel [1]"])

            for label, (rate, best) in res["variants"].items():
                acc[label].append(rate)
                if best is not None:
                    for name in config.COMPONENT_NAMES:
                        opt_components[name].extend(best[name].tolist())

    # ── Bar chart: fixed-component comparison ──────────────────────────────
    order = ["Ideal IRS"] + [label for label, _ in VARIANTS] + \
            ["AO: Eq.(5)\nmodel [1]", "No IRS"]
    means = [np.mean(acc[label]) for label in order]
    colors = [IEEE_COLORS["black"], IEEE_COLORS["blue"], IEEE_COLORS["purple"],
              IEEE_COLORS["cyan"], IEEE_COLORS["orange"], IEEE_COLORS["green"],
              IEEE_COLORS["gray"]]

    fig, ax = ieee_plot_style.new_figure("double")
    bars = ax.bar(range(len(order)), means, color=colors, width=0.62)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.05, f"{m:.2f}",
                ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=7)
    ax.set_ylabel("Mean Achievable Rate (bps/Hz)")
    ax.set_ylim(0, max(means) * 1.15)
    ieee_plot_style.save_figure("rlc_fig6_fixed_components.png",
                                figures_dir=str(config.FIGURES_DIR))
    plt.close(fig)

    csv_path = config.DATA_DIR / "rlc_fixed_components.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scheme", "mean_rate_bpshz", "std_rate_bpshz"])
        for label in order:
            writer.writerow([label.replace("\n", " "),
                             f"{np.mean(acc[label]):.4f}",
                             f"{np.std(acc[label]):.4f}"])
    print(f"[exp5] Data saved to {csv_path}")

    # ── Histograms of optimized component values ───────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.4))
    for ax, name in zip(axes.flat, config.COMPONENT_NAMES):
        scale, unit = config.COMPONENT_SCALES[name]
        vals = np.array(opt_components[name]) / scale
        lo, hi = (b / scale for b in config.BOUNDS[name])
        ax.hist(vals, bins=40, range=(lo, hi), color=IEEE_COLORS["blue"],
                alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.set_xlabel(f"Optimized ${name[0]}_{{{name[1:] or 'n'}}}$ ({unit})"
                      if name.startswith("L")
                      else f"Optimized ${name}_n$ ({unit})")
        ax.set_ylabel("Count")
    ieee_plot_style.save_figure("rlc_fig7_component_hist.png",
                                figures_dir=str(config.FIGURES_DIR))
    plt.close(fig)

    csv_path = config.DATA_DIR / "rlc_optimized_components.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(config.COMPONENT_NAMES)
        for row in zip(*(opt_components[n] for n in config.COMPONENT_NAMES)):
            writer.writerow([f"{x:.6e}" for x in row])
    print(f"[exp5] Data saved to {csv_path}")
    print("[exp5] Done.")


if __name__ == "__main__":
    main()
