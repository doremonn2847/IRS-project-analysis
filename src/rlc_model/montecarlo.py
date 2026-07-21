"""Shared Monte-Carlo helper: evaluate all comparison schemes on one channel.

Schemes (keys of the returned dict):
    ideal      — ideal IRS, |v_n| = 1, free phase (upper bound)
    pso_rlc    — PSO over all components {L1, L2, C, R}  (week-5 requirement)
    pso_c_only — PSO over C only, L1/L2/R at paper values (fixed-component case)
    ao_phase   — AO on the equivalent Eq. (5) phase model (paper baseline)
    no_irs     — direct link only (lower bound)
"""
import numpy as np

from rlc_model.system import rate_no_irs, ideal_upper_bound
from rlc_model.optimizers import pso_rlc, ao_phase

SCHEMES = ("ideal", "pso_rlc", "pso_c_only", "ao_phase", "no_irs")

SCHEME_LABELS = {
    "ideal": "Ideal IRS (Upper Bound)",
    "pso_rlc": "PSO: RLC components (proposed)",
    "pso_c_only": "PSO: only $C$ tunable",
    "ao_phase": "AO: Practical IRS [1]",
    "no_irs": "No IRS (Lower Bound)",
}


def evaluate_schemes(hd, hr, G, Phi, n_elements, rng,
                     n_particles=None, n_iter=None, return_components=False):
    """Run every scheme on one channel realization; returns {scheme: rate}."""
    out = {}
    out["no_irs"] = rate_no_irs(hd)
    _, out["ideal"] = ideal_upper_bound(hd, Phi)
    best, out["pso_rlc"], _ = pso_rlc(hd, Phi, n_elements,
                                      n_particles=n_particles,
                                      n_iter=n_iter, rng=rng)
    _, out["pso_c_only"], _ = pso_rlc(hd, Phi, n_elements, free=("C",),
                                      n_particles=n_particles,
                                      n_iter=n_iter, rng=rng)
    _, out["ao_phase"] = ao_phase(hd, hr, G, rng=rng)
    if return_components:
        return out, best
    return out
