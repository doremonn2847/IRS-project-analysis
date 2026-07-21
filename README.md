# IRS Optimization with PSO — Electronics for IT Project

Mini-project for the **Electronics for IT** course: optimizing an
**Intelligent Reflecting Surface (IRS)** assisted wireless link with
**Particle Swarm Optimization (PSO)**, based on the practical phase-shift
model of Abeywickrama *et al.* (see `docs/Project_PhaseShift_Model.pdf`).

The project has two phases:

1. **Phase-level optimization (weeks 1–4)** — optimize the reflection phase
   θₙ of each IRS element under the equivalent model
   *vₙ = β(θₙ)·e^{jθₙ}* (Eq. 5 of the paper), and compare PSO against the
   paper's Alternating Optimization (AO).
2. **Component-level optimization (week 5, final)** — optimize the **physical
   circuit components** of every element, *L₁ₙ, L₂ₙ, Cₙ, Rₙ* (4N variables),
   mapping components → impedance *Zₙ* (Eq. 3) → reflection coefficient
   *vₙ = (Zₙ−Z₀)/(Zₙ+Z₀)* (Eq. 4) → achievable rate *R_SE* (Eq. 6).

All requirements (original + extension) are consolidated in
[`docs/requirements.md`](docs/requirements.md).

## Repository structure

```
├── docs/                      # Project briefs, reference paper, consolidated requirements
│   ├── requirements.md        # ← all requirements, old + new, structured
│   ├── Project_ElectronicIT.pdf
│   ├── Update_Project_Elect_IT.pdf   # week-5 extension brief (RLC optimization)
│   └── Project_PhaseShift_Model.pdf  # reference paper
├── src/
│   ├── ieee_plot_style.py     # shared IEEE publication-quality plot styling
│   ├── phase_model/           # phase-level scripts (weeks 1–4): PSO, DE, AO, PySwarms
│   └── rlc_model/             # component-level package (week 5, final)
│       ├── config.py          # system parameters, circuit constants, component bounds
│       ├── circuit.py         # Eq. 3 (impedance) and Eq. 4 (reflection coefficient)
│       ├── system.py          # channels, achievable rate (Eq. 6), ideal/no-IRS bounds
│       ├── optimizers.py      # vectorized PSO over {L1,L2,C,R}; AO phase baseline
│       ├── montecarlo.py      # shared scheme-comparison helper
│       ├── exp1..exp5_*.py    # experiments (see table below)
│       └── run_all.py         # run every experiment in sequence
├── results/
│   ├── figures/               # generated figures (PNG 600 dpi + vector PDF)
│   ├── data/                  # generated CSV data behind each figure
│   └── figures/legacy/        # figures from earlier weeks
├── report/                    # LaTeX report + camera-ready figures
└── requirements.txt
```

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate   # optional
pip install -r requirements.txt

# Run everything (quick pass, ~10–20 min):
python src/rlc_model/run_all.py

# Report-quality statistics (1000 realizations, hours):
python src/rlc_model/run_all.py --full
```

Every script also runs standalone and accepts `--realizations`, `--iters`,
`--seed` (see `--help`). Figures land in `results/figures/`, raw numbers in
`results/data/`.

## Week-5 experiments

| Script | Output | What it shows |
|--------|--------|---------------|
| `exp1_circuit_verification.py` | `rlc_fig1`, `rlc_fig2` | Circuit response \|vₙ\|, arg(vₙ) vs C; verifies that Eq. (5)'s constants (β_min≈0.2, k≈1.6, φ≈0.43π) are *derivable* from the physical circuit, and that freeing all four components unlocks a far larger feasible (θ, β) region than varactor-only tuning |
| `exp2_convergence.py` | `rlc_fig3` | PSO-RLC convergence on one channel realization vs AO, C-only PSO, ideal and no-IRS bounds |
| `exp3_rate_vs_distance.py` | `rlc_fig4` + CSV | Achievable rate vs AP–user distance (paper Fig. 5 setup) for all schemes |
| `exp4_rate_vs_N.py` | `rlc_fig5` + CSV | Achievable rate vs number of IRS elements (paper Fig. 6 setup) |
| `exp5_component_analysis.py` | `rlc_fig6`, `rlc_fig7` + CSV | Fixed-component comparison (all free / C,R / C,L₁ / C only) and histograms of the optimized component values |

**Compared schemes:** ideal IRS (upper bound) · PSO over {L₁,L₂,C,R}
(proposed) · PSO over C only (varactor-only hardware) · AO on the Eq. (5)
equivalent model (paper baseline) · no IRS (lower bound).

**Key result:** component-level PSO nearly reaches the ideal upper bound and
outperforms the phase-model AO, because it is not restricted to the single
β(θ) curve of a fixed circuit — it can also *reduce the loss resistance R*
and retune L₁/L₂ per element. Varactor-only PSO matches AO, confirming both
implementations are consistent.

## System model (shared by both phases)

M = 2 AP antennas, N = 40 IRS elements (default), AP–IRS distance 500 m,
Rayleigh fading with distance-dependent path loss, Pт = 36 dBm,
σ² = −94 dBm, f = 2.4 GHz, Z₀ = 377 Ω — identical to Section V of the
reference paper, so phase-level and component-level results are directly
comparable.

Component bounds (stated per the brief; rationale in
`src/rlc_model/config.py` and `docs/requirements.md`):
C ∈ [0.47, 2.35] pF (paper's varactor range) · L₁ ∈ [0.5, 5] nH ·
L₂ ∈ [0.1, 2] nH · R ∈ [0.5, 5] Ω.

## Report

The LaTeX report lives in `report/`. Course language is Vietnamese; code and
comments are in English.
