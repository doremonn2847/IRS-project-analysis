"""Optimizers for the week-5 component-level problem.

* pso_rlc      — PSO directly over the physical components {L1, L2, C, R} of
                 every element (the new requirement). Any subset of components
                 can be frozen at the paper values to study fixed-component
                 variants (e.g. varactor-only tuning: free=("C",)).
* ao_phase     — the paper's Alternating Optimization on the equivalent phase
                 model of Eq. (5). This is the comparison baseline required by
                 the brief.

The PSO search space is normalized to [0, 1]^D and decoded to physical values
per evaluation; this keeps the swarm well-conditioned despite the wildly
different component scales (nH vs pF vs Ohm).
"""
import numpy as np

from rlc_model.config import BOUNDS, PAPER_VALUES, COMPONENT_NAMES
from rlc_model.circuit import reflection_coefficient, eq5_amplitude, eq5_reflection
from rlc_model.system import rate_from_reflection


# ─────────────────────────────────────────────────────────────────────────────
#  Candidate encoding / decoding
# ─────────────────────────────────────────────────────────────────────────────

def decode_components(X, n_elements, free=COMPONENT_NAMES):
    """Map normalized particles X in [0,1]^(P, len(free)*N) to physical values.

    Returns a dict {name: (P, N) array} with frozen components filled with
    their paper values (C has no single paper value: if frozen, mid-range).
    """
    P = X.shape[0]
    comps = {}
    Xr = X.reshape(P, len(free), n_elements)
    for i, name in enumerate(free):
        lo, hi = BOUNDS[name]
        comps[name] = lo + Xr[:, i, :] * (hi - lo)
    for name in COMPONENT_NAMES:
        if name not in free:
            fixed = PAPER_VALUES[name]
            if fixed is None:  # C: no paper constant -> middle of varactor range
                lo, hi = BOUNDS[name]
                fixed = 0.5 * (lo + hi)
            comps[name] = np.full((P, n_elements), fixed)
    return comps


def components_to_rate(comps, hd, Phi):
    """Fitness: components -> Z_n (Eq.3) -> v_n (Eq.4) -> R_SE (Eq.6)."""
    v = reflection_coefficient(comps["L1"], comps["L2"], comps["C"], comps["R"])
    return rate_from_reflection(v, hd, Phi)


# ─────────────────────────────────────────────────────────────────────────────
#  PSO over component values
# ─────────────────────────────────────────────────────────────────────────────

def pso_rlc(hd, Phi, n_elements, free=COMPONENT_NAMES,
            n_particles=None, n_iter=None, w=0.7298, c1=2.05, c2=2.05,
            v_max=0.2, rng=None, track_convergence=False, polish=True):
    """Constriction-type PSO over the free component values of all elements.

    Returns (best_components, best_rate, history) where best_components is a
    dict {name: (N,) array} of physical values.
    """
    rng = np.random.default_rng() if rng is None else rng
    dim = len(free) * n_elements

    # Scale particles and iterations dynamically to handle higher dimensions (Problem 2 & 3)
    if n_particles is None:
        n_particles = max(60, min(120, int(0.6 * dim)))
    if n_iter is None:
        n_iter = max(400, min(800, int(4.0 * dim)))

    X = rng.uniform(0, 1, (n_particles, dim))
    try:
        # Warm-start particle 0 using ideal phase alignment
        from rlc_model.system import ideal_upper_bound
        v_ideal, _ = ideal_upper_bound(hd, Phi)
        target_phases = np.angle(v_ideal)
        
        C_lo, C_hi = BOUNDS["C"]
        C_sweep = np.linspace(C_lo, C_hi, 1000)
        v_sweep = reflection_coefficient(PAPER_VALUES["L1"], PAPER_VALUES["L2"], C_sweep, BOUNDS["R"][0])
        phases_sweep = np.angle(v_sweep)
        
        best_C_idx = [np.argmin(np.abs(np.arctan2(np.sin(phases_sweep - ph), np.cos(phases_sweep - ph)))) for ph in target_phases]
        C_phys = C_sweep[best_C_idx]
        
        p_reshaped = X[0].reshape(len(free), n_elements)
        for i, name in enumerate(free):
            lo, hi = BOUNDS[name]
            if name == "C":
                val_phys = C_phys
            elif name == "R":
                val_phys = np.full(n_elements, lo)  # best is lowest resistance (loss)
            elif name == "L1":
                val_phys = np.full(n_elements, PAPER_VALUES["L1"])
            elif name == "L2":
                val_phys = np.full(n_elements, PAPER_VALUES["L2"])
            p_reshaped[i, :] = np.clip((val_phys - lo) / (hi - lo), 0.0, 1.0)
    except Exception:
        pass

    V = np.zeros((n_particles, dim))

    fitness = components_to_rate(decode_components(X, n_elements, free), hd, Phi)
    p_best, p_best_score = X.copy(), fitness.copy()
    g_idx = int(np.argmax(fitness))
    g_best, g_best_score = X[g_idx].copy(), float(fitness[g_idx])

    history = [g_best_score] if track_convergence else []

    for t in range(n_iter):
        r1 = rng.uniform(size=(n_particles, dim))
        r2 = rng.uniform(size=(n_particles, dim))
        V = w * (V + c1 * r1 * (p_best - X) + c2 * r2 * (g_best - X))
        V = np.clip(V, -v_max, v_max)
        X = np.clip(X + V, 0.0, 1.0)          # physical bounds (Eq. 6-9)

        fitness = components_to_rate(decode_components(X, n_elements, free), hd, Phi)

        improved = fitness > p_best_score
        p_best[improved] = X[improved]
        p_best_score[improved] = fitness[improved]

        it_best = int(np.argmax(fitness))
        if fitness[it_best] > g_best_score:
            g_best, g_best_score = X[it_best].copy(), float(fitness[it_best])

        # Elitist restart of the global best on one random dimension
        # (same APSO-style jump-out used in the weeks 1-4 PSO).
        sigma = 0.5 * (1.0 - t / n_iter)
        cand = g_best.copy()
        cand[rng.integers(dim)] += sigma * rng.standard_normal()
        cand = np.clip(cand, 0.0, 1.0)
        cand_score = components_to_rate(
            decode_components(cand[None, :], n_elements, free), hd, Phi)[0]
        if cand_score > g_best_score:
            g_best, g_best_score = cand, float(cand_score)

        if track_convergence:
            history.append(g_best_score)

    if polish:
        try:
            from scipy.optimize import minimize
            def obj(X):
                X_clip = np.clip(X, 0.0, 1.0)
                comps = decode_components(X_clip[None, :], n_elements, free)
                return -float(components_to_rate(comps, hd, Phi)[0])
            
            def cb(X):
                if track_convergence:
                    X_clip = np.clip(X, 0.0, 1.0)
                    comps = decode_components(X_clip[None, :], n_elements, free)
                    rate = float(components_to_rate(comps, hd, Phi)[0])
                    history.append(rate)

            maxiter = 160
            res = minimize(obj, g_best, method='L-BFGS-B', bounds=[(0.0, 1.0)] * dim,
                           options={'maxiter': maxiter}, callback=cb)
            if -res.fun > g_best_score:
                g_best = np.clip(res.x, 0.0, 1.0)
                g_best_score = -res.fun
            
            if track_convergence:
                target_len = n_iter + maxiter
                while len(history) < target_len:
                    history.append(g_best_score)
        except Exception:
            pass

    best = decode_components(g_best[None, :], n_elements, free)
    best = {name: arr[0] for name, arr in best.items()}
    return best, g_best_score, history


# ─────────────────────────────────────────────────────────────────────────────
#  AO baseline on the equivalent phase model (weeks 1-4 / reference paper)
# ─────────────────────────────────────────────────────────────────────────────

def ao_phase(hd, hr, G, n_iter=15, n_grid=180, rng=None):
    """Element-wise Alternating Optimization of theta_n under Eq. (5).

    Port of the weeks 1-4 implementation (1-D grid search per element).
    Returns (theta, rate).
    """
    rng = np.random.default_rng() if rng is None else rng
    n_elements = len(hr)
    theta = rng.choice([np.pi, -np.pi], n_elements)
    v = eq5_reflection(theta)

    diag_hr_conj = np.diag(hr.conj())
    Psi = diag_hr_conj @ G @ G.conj().T @ np.diag(hr)
    hd_hat = diag_hr_conj @ G @ hd

    theta_grid = np.linspace(-np.pi, np.pi, n_grid)
    beta_grid = eq5_amplitude(theta_grid)

    for _ in range(n_iter):
        for n in range(n_elements):
            interference = Psi[n, :] @ v - Psi[n, n] * v[n]
            phi_n = 2.0 * interference + 2.0 * hd_hat[n]
            f_val = (beta_grid ** 2) * Psi[n, n].real + \
                beta_grid * np.abs(phi_n) * np.cos(np.angle(phi_n) - theta_grid)
            theta[n] = theta_grid[np.argmax(f_val)]
            v[n] = eq5_reflection(theta[n])

    Phi = diag_hr_conj @ G
    return theta, float(rate_from_reflection(v, hd, Phi))
