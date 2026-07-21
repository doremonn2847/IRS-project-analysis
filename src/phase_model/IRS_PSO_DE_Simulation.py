# BLOCK 1 - INSTALL & IMPORT LIBRARIES

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm # progress bar...

# Make plots look clean — IEEE publication-quality styling
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))  # so 'ieee_plot_style' (in src/) is importable
import ieee_plot_style
from ieee_plot_style import IEEE_COLORS

np.random.seed(42)

print('Libraries loaded successfully.')

# BLOCK 2 - SYSTEM PARAMETERS
# (derived directly from paper's Section V)

M = 2       # Number of AP Antennas
N = 40      # Number of IRS reflecting elements

# -- Distances --
d_AP_IRS = 500      # AP to IRS horizontal distance (meters)
d_perp = 2          # Perpendicular distance between AP-IRS line and user line (meters)

# -- Path loss model -- 
C0 = 10**(-40/10)   # Reference path loss at 1m (Linear scale)
alpha_AI = 2.2      # AP to IRS path loss exponent
alpha_IU = 2.8      # IRS to user path loss exponent
alpha_AU = 3.8      # AP to user path loss exponent

# -- Power & noise --
PT_dBm = 36         # Transmit power (dBm)
noise_dBm = -94     # Noise floor (dBm)
# Convert to linear (Watt)
PT = 10**((PT_dBm-30)/10)
sigma2 = 10**((noise_dBm-30)/10)

# -- Practical phase shift model (Eq 5 in paper) ---
beta_min = 0.2      # Minimum reflection coefficient magnitude
k_param = 1.6
phi_param = 0.43 * np.pi

# -- Simulation Settings --
N_realizations = 1000

print(f'Parameters set.')
print(f'PT = {PT*1000:.1f} mW = {PT_dBm} dBm')
print(f'sigma^2 = {sigma2:.2e} W = {noise_dBm} dBm')
print(f'SNR_max = {10*np.log10(PT/sigma2):.1f} dB (direct AP-user link)')

# BLOCK 3 - HELPER FUNCTIONS

def save_figure(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    figures_dir = os.path.join(project_dir, 'report', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    image_path = os.path.join(figures_dir, filename)
    plt.savefig(image_path, dpi=300)
    print(f'Saved: {image_path}')

def path_loss(distance, alpha):
    return np.sqrt(C0 * distance**(-alpha))

def generate_channels(d_user):
    d_IRS_user = np.sqrt((d_user - d_AP_IRS)**2 + d_perp**2)
    hd = (path_loss(d_user,    alpha_AU) *
          (np.random.randn(M)   + 1j * np.random.randn(M))   / np.sqrt(2))
    hr = (path_loss(d_IRS_user, alpha_IU) *
          (np.random.randn(N)   + 1j * np.random.randn(N))   / np.sqrt(2))
    G  = (path_loss(d_AP_IRS,  alpha_AI) *
          (np.random.randn(N, M) + 1j * np.random.randn(N, M)) / np.sqrt(2))
    return hd, hr, G

def compute_amplitude(theta):
    return (1 - beta_min) * ((np.sin(theta - phi_param) + 1) / 2)**k_param + beta_min

def build_reflection_vector(theta):
    beta = compute_amplitude(theta)
    return beta * np.exp(1j * theta)

def compute_rate(theta, hd, hr, G):
    v = build_reflection_vector(theta)
    Phi = np.diag(hr.conj()) @ G
    combined = v.conj() @ Phi + hd.conj() 
    norm_combined = np.linalg.norm(combined)
    if norm_combined < 1e-12:
        return 0.0
    signal_power = PT * norm_combined**2
    snr  = signal_power / sigma2
    rate = np.log2(1 + snr)
    return rate

# BLOCK 4 - OPTIMIZATION ALGORITHMS (PSO, DE, AO)

def pso_optimize(hd, hr, G, n_particles=30, n_iter=150, w_start=0.9, w_end=0.4, c1=2.0, c2=2.0, track_convergence=False):
    N_elem = len(hr)
    positions  = np.random.uniform(-np.pi, np.pi, (n_particles, N_elem))
    velocities = np.zeros((n_particles, N_elem))
    fitness = np.array([compute_rate(positions[i], hd, hr, G) for i in range(n_particles)])
    p_best       = positions.copy()
    p_best_score = fitness.copy()
    g_best_idx   = np.argmax(fitness)
    g_best       = positions[g_best_idx].copy()
    g_best_score = fitness[g_best_idx]
    history = [g_best_score] if track_convergence else []
    
    for t in range(n_iter):
        w = w_start - (w_start - w_end) * t / n_iter
        r1, r2 = np.random.rand(n_particles, N_elem), np.random.rand(n_particles, N_elem)
        velocities = (w * velocities + c1 * r1 * (p_best - positions) + c2 * r2 * (g_best - positions))
        positions = np.clip(positions + velocities, -np.pi, np.pi)
        fitness = np.array([compute_rate(positions[i], hd, hr, G) for i in range(n_particles)])
        
        improved = fitness > p_best_score
        p_best[improved]       = positions[improved].copy()
        p_best_score[improved] = fitness[improved]
        best_in_iter = np.argmax(fitness)
        if fitness[best_in_iter] > g_best_score:
            g_best       = positions[best_in_iter].copy()
            g_best_score = fitness[best_in_iter]
        if track_convergence:
            history.append(g_best_score)
            
    return g_best, g_best_score, history

def de_optimize(hd, hr, G, pop_size=30, n_iter=150, F=0.8, CR=0.7, track_convergence=False):
    """
    Differential Evolution (DE) Algorithm for Phase Shift Optimization.
    """
    N_elem = len(hr)
    # Initialize population randomly between -pi and pi
    population = np.random.uniform(-np.pi, np.pi, (pop_size, N_elem))
    fitness = np.array([compute_rate(population[i], hd, hr, G) for i in range(pop_size)])
    
    best_idx = np.argmax(fitness)
    best_vector = population[best_idx].copy()
    best_score = fitness[best_idx]
    history = [best_score] if track_convergence else []
    
    for t in range(n_iter):
        for i in range(pop_size):
            # 1. Mutation: Choose 3 distinct random vectors (a, b, c) different from i
            indices = list(range(pop_size))
            indices.remove(i)
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            
            mutant = a + F * (b - c)
            # Wrap around phases or clip (clipping used here for consistency)
            mutant = np.clip(mutant, -np.pi, np.pi)
            
            # 2. Crossover: Binomial
            cross_points = np.random.rand(N_elem) < CR
            # Ensure at least one dimension is inherited from the mutant
            if not np.any(cross_points):
                cross_points[np.random.randint(0, N_elem)] = True
                
            trial = np.where(cross_points, mutant, population[i])
            
            # 3. Selection
            trial_fitness = compute_rate(trial, hd, hr, G)
            if trial_fitness > fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                
                # Update global best
                if trial_fitness > best_score:
                    best_score = trial_fitness
                    best_vector = trial.copy()
                    
        if track_convergence:
            history.append(best_score)
            
    return best_vector, best_score, history

def ao_optimize(hd, hr, G, n_iter=15, use_1d_search=True):
    N_elem = len(hr)
    theta = np.random.choice([np.pi, -np.pi], N_elem)
    v = build_reflection_vector(theta)
    diag_hr_conj = np.diag(hr.conj())
    Psi = diag_hr_conj @ G @ G.conj().T @ np.diag(hr)
    hd_hat = diag_hr_conj @ G @ hd
    for iteration in range(n_iter):
        for n in range(N_elem):
            sum_val = 0.0
            for m in range(N_elem):
                if m != n:
                    sum_val += Psi[n, m] * v[m]
            phi_n = 2.0 * sum_val + 2.0 * hd_hat[n]
            if use_1d_search:
                theta_range = np.linspace(-np.pi, np.pi, 180)
                beta_range = compute_amplitude(theta_range)
                f_val = (beta_range**2) * Psi[n, n].real + beta_range * np.abs(phi_n) * np.cos(np.angle(phi_n) - theta_range)
                best_idx = np.argmax(f_val)
                theta[n] = theta_range[best_idx]
            else:
                arg_phi = np.angle(phi_n)
                lam = 0 if arg_phi >= 0 else 1
                theta_n_star = (((-1)**lam) * np.pi * (3.0*f_eval(arg_phi) - 4.0*f_eval(arg_phi + ((-1)**lam) * np.pi / 2.0) + f_eval(((-1)**lam) * np.pi)) + arg_phi * (f_eval(arg_phi) - 4.0*f_eval(arg_phi + ((-1)**lam) * np.pi / 2.0) + 3.0*f_eval(((-1)**lam) * np.pi))) / (4.0 * (f_eval(arg_phi) - 2.0 * f_eval(arg_phi + ((-1)**lam) * np.pi / 2.0) + f_eval(((-1)**lam) * np.pi)))
                theta[n] = np.clip(theta_n_star, min(arg_phi, ((-1)**lam) * np.pi), max(arg_phi, ((-1)**lam) * np.pi))
            v[n] = build_reflection_vector(np.array([theta[n]]))[0]
    return theta, compute_rate(theta, hd, hr, G)

# ── Quick test on a single channel realization ──────────────────────────────
print('\nTesting optimizers on one channel realization...')
hd_t, hr_t, G_t = generate_channels(d_user=498)
rate_no_irs = np.log2(1 + PT * np.linalg.norm(hd_t.conj())**2 / sigma2)

_, best_rate_pso, hist_pso = pso_optimize(hd_t, hr_t, G_t, track_convergence=True)
_, best_rate_de, hist_de   = de_optimize(hd_t, hr_t, G_t, track_convergence=True)
_, best_rate_ao = ao_optimize(hd_t, hr_t, G_t, use_1d_search=True)

print(f'  No IRS rate       : {rate_no_irs:.3f} bps/Hz')
print(f'  PSO best rate     : {best_rate_pso:.3f} bps/Hz')
print(f'  DE best rate      : {best_rate_de:.3f} bps/Hz')
print(f'  AO (Paper 1D) rate: {best_rate_ao:.3f} bps/Hz')

# Plot convergence
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(hist_pso, 'b-', label='PSO Convergence')
ax.plot(hist_de, 'm-', label='DE Convergence')
ax.axhline(best_rate_ao, color='g', linestyle='-.', label='AO (Paper Proposed) level')
ax.axhline(rate_no_irs, color='k', linestyle='--', label='No IRS baseline')
ax.set_xlabel('Iteration')
ax.set_ylabel('Best Rate (bps/Hz)')
ax.set_title('PSO vs DE Convergence (Single Realization)')
ax.legend()
plt.tight_layout()
save_figure('metaheuristic_convergence.png')
plt.show()

# BLOCK 5 - ACHIEVABLE RATE VS DISTANCE (FIG 5)

d_range = np.arange(482, 501, 2)
N_realizations = 1000 # Default to 100 for faster testing, change to 1000 for final

results = {
    'pso_practical': np.zeros(len(d_range)),
    'de_practical': np.zeros(len(d_range)),
    'ao_practical': np.zeros(len(d_range)),
    'ideal_upper': np.zeros(len(d_range)),
    'ideal_assumed': np.zeros(len(d_range)),
    'no_irs': np.zeros(len(d_range)),
}

print(f"\nRunning Distance Simulation: {len(d_range)} distances x {N_realizations} realizations...")

for d_idx, d_user in enumerate(tqdm(d_range, desc='Distance')):
    rates_pso, rates_de, rates_ao, rates_ideal, rates_mismat, rates_no_irs = [], [], [], [], [], []

    for _ in range(N_realizations):
        hd, hr, G = generate_channels(d_user)
        Phi = np.diag(hr.conj()) @ G

        # 1. No IRS
        rates_no_irs.append(np.log2(1 + PT * np.linalg.norm(hd.conj())**2 / sigma2))

        # 2. PSO
        _, rate_pso, _ = pso_optimize(hd, hr, G)
        rates_pso.append(rate_pso)
        
        # 3. DE
        _, rate_de, _ = de_optimize(hd, hr, G)
        rates_de.append(rate_de)

        # 4. AO
        _, rate_ao = ao_optimize(hd, hr, G)
        rates_ao.append(rate_ao)

        # 5. Ideal Upper Bound
        v_ideal = np.exp(1j * np.zeros(N))
        for _ in range(5):
            for n in range(N):
                temp = v_ideal.conj() @ Phi + hd.conj()
                contrib_n = Phi[n, :]
                temp_no_n = temp - v_ideal[n].conj() * contrib_n
                v_ideal[n] = np.exp(1j * np.angle(contrib_n @ temp_no_n.conj()))
        combined_ideal = v_ideal.conj() @ Phi + hd.conj()
        rates_ideal.append(np.log2(1 + PT * np.linalg.norm(combined_ideal)**2 / sigma2))

        # 6. Mismatch
        v_mismatch = build_reflection_vector(np.angle(v_ideal))
        combined_mm = v_mismatch.conj() @ Phi + hd.conj()
        rates_mismat.append(np.log2(1 + PT * np.linalg.norm(combined_mm)**2 / sigma2))

    results['pso_practical'][d_idx] = np.mean(rates_pso)
    results['de_practical'][d_idx]  = np.mean(rates_de)
    results['ao_practical'][d_idx]  = np.mean(rates_ao)
    results['ideal_upper'][d_idx]   = np.mean(rates_ideal)
    results['ideal_assumed'][d_idx] = np.mean(rates_mismat)
    results['no_irs'][d_idx]        = np.mean(rates_no_irs)

# Plot Figure 5
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(d_range, results['ideal_upper'],   'k-.',  label='Upper Bound: Ideal IRS')
ax.plot(d_range, results['ao_practical'],  'g-^',  markersize=4, label='AO — Practical IRS')
ax.plot(d_range, results['de_practical'],  'm-d',  markersize=4, label='DE — Practical IRS')
ax.plot(d_range, results['pso_practical'], 'b-o',  markersize=4, label='PSO — Practical IRS')
ax.plot(d_range, results['ideal_assumed'], 'r--',  label='Practical IRS with Ideal Assumption')
ax.plot(d_range, results['no_irs'],        'k-s',  markersize=4, label='Lower Bound: No IRS')

ax.set_xlabel('AP-User Horizontal Distance: d (m)')
ax.set_ylabel('Achievable Rate (bps/Hz)')
ax.set_title(f'Rate vs. Distance (N={N}, {N_realizations} realizations)')
ax.legend(loc='upper center', fontsize=8.5)
plt.tight_layout()
save_figure('achievable_rate_with_DE.png')
plt.show()

# BLOCK 7 - SUMMARY TABLE

print('\n' + '='*70)
print('SUMMARY: Rate at d=498m, N=40')
print('='*70)

idx498 = np.argmin(np.abs(d_range - 498))

rows = [
    ('Upper Bound (Ideal IRS)',            results['ideal_upper'  ][idx498]),
    ('AO — Practical Model',               results['ao_practical' ][idx498]),
    ('DE — Practical Model',               results['de_practical' ][idx498]),
    ('PSO — Practical Model',              results['pso_practical'][idx498]),
    ('Ideal Assumption on Practical HW',   results['ideal_assumed'][idx498]),
    ('No IRS (Lower Bound)',               results['no_irs'       ][idx498]),
]

print(f'  {"Scheme":<45} {"Rate (bps/Hz)":>12}')
print('-'*65)
for name, val in rows:
    print(f'  {name:<45} {val:>12.4f}')
print('='*70)

ao_val = results['ao_practical'][idx498]
de_val = results['de_practical'][idx498]
pso_val = results['pso_practical'][idx498]

print(f'\nDE reaches  {(de_val/ao_val)*100:.1f}% of the paper-proposed AO rate')
print(f'PSO reaches {(pso_val/ao_val)*100:.1f}% of the paper-proposed AO rate')